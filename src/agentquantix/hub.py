"""Talking to the Hugging Face Hub: what is trending, and what is it made of.

Step 1 of the agent. The trending list itself comes from the Hub REST API
(`sort=trendingScore`), which returns exactly the top N deterministically —
the MCP server is wired in alongside for ad-hoc lookups and the step-5
verification, but the thing the whole run keys off should not depend on a
semantic search returning the same answer twice.

Two passes, deliberately:

  * `trending()` is ONE request. It carries tags, pipeline_tag, gated flags
    and the sibling file list, which is enough to throw out most of the list
    without touching the network again.

  * `enrich()` is several requests PER MODEL and is only ever run on what
    survived the filter. On a top-100 sweep that is the difference between a
    few seconds and several minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import re

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from . import config


def api():
    return HfApi(token=config.TOKEN)


# =====================================================
# CANDIDATE
# =====================================================
@dataclass
class Candidate:
    """One trending model, progressively filled in as we learn more."""

    repo_id: str
    rank: int                       # position in the trending list, 1-based
    pipeline_tag: str | None = None
    tags: list = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    created_at: str | None = None
    gated: object = False
    private: bool = False

    # Filled by enrich()
    params: int | None = None           # total parameters, all dtypes
    source_bytes: int = 0               # what a full snapshot_download costs

    # Provenance, for the model card. None of this affects feasibility — it
    # exists so a card can credit the authors, state the real licence and
    # cite the real paper instead of hedging. The card used to close with
    # "Licensing follows the base model", which was a hedge around the fact
    # that the licence was never fetched at all.
    author: str | None = None           # the publishing user or org
    license: str | None = None          # from the Hub's "license:*" tag
    arxiv_ids: list = field(default_factory=list)   # from "arxiv:*" tags
    languages: list = field(default_factory=list)   # from "language:*" tags
    readme: str | None = None           # the source card, verbatim
    dtypes: dict = field(default_factory=dict)
    architectures: list = field(default_factory=list)
    model_type: str | None = None
    n_layers: int | None = None
    hidden_size: int | None = None
    vocab_size: int | None = None
    n_experts: int | None = None        # >0 means MoE, which adds MXFP4_MOE
    tied_embeddings: bool = False
    is_multimodal: bool = False         # has a vision tower -> mmproj
    official_gguf: str | None = None    # publisher's own GGUF repo, if any
    official_bf16_file: str | None = None
    official_bf16_bytes: int = 0
    community_ggufs: list = field(default_factory=list)

    # What WE have already published for this model. The difference between
    # "not started" and "26 of 30 done" is the difference between a 20-hour
    # job and a 90-minute one, so it has to be known before anything is ranked.
    published_files: set = field(default_factory=set)
    published_quants: set = field(default_factory=set)
    # WHICH of our repos they were found in. Usually just target_repo, but a
    # repo published under an older name shows up here too, and the user needs
    # to be told which one rather than left to guess why the sweep shrank.
    published_repos: list = field(default_factory=list)

    error: str | None = None

    # Filled by the filter
    dropped: str | None = None          # reason, or None if it survived

    @property
    def org(self):
        return self.repo_id.split("/")[0] if "/" in self.repo_id else ""

    @property
    def name(self):
        return self.repo_id.split("/")[-1]

    @property
    def url(self):
        return f"https://huggingface.co/{self.repo_id}"

    @property
    def target_repo(self):
        """Where we would publish this model's quants."""
        return f"{config.HF_NAMESPACE}/{self.name}-GGUF"


# =====================================================
# STEP 1a — the list
# =====================================================
def trending_kwargs(limit):
    """How to ask for "most trending first", across huggingface_hub versions.

    The two major versions disagree, and the 1.x form is a hard TypeError on
    0.x and vice versa:

      * 0.x wants sort="trendingScore" plus direction=-1 for descending.
      * 1.x renamed the sort keys to snake_case ("trending_score") and removed
        `direction` entirely; these ranking sorts are descending by default,
        which was verified against the live API rather than assumed.

    Detected from the signature rather than the version string, because that
    is the thing that actually determines whether the call works.
    """
    import inspect

    parameters = inspect.signature(HfApi.list_models).parameters
    if "direction" in parameters:
        return {"sort": "trendingScore", "direction": -1,
                "limit": limit, "full": True}
    return {"sort": "trending_score", "limit": limit, "full": True}


def trending(limit=None):
    """The current top-N trending models, best first.

    `full=True` is what makes the single-request filter possible: without it
    the returned objects carry neither tags nor siblings, and every candidate
    would need its own round trip just to find out it is a TTS model.
    """
    limit = limit or config.TRENDING_LIMIT
    out = []
    for rank, model in enumerate(
            api().list_models(**trending_kwargs(limit)), start=1):
        out.append(Candidate(
            repo_id=model.id,
            rank=rank,
            pipeline_tag=getattr(model, "pipeline_tag", None),
            tags=list(getattr(model, "tags", []) or []),
            downloads=getattr(model, "downloads", 0) or 0,
            likes=getattr(model, "likes", 0) or 0,
            created_at=(str(model.created_at)[:10]
                        if getattr(model, "created_at", None) else None),
            gated=getattr(model, "gated", False),
            private=bool(getattr(model, "private", False)),
        ))
    return out


def one(repo_id):
    """A Candidate for a named repo, whether or not it is trending.

    The sweep exists to work through what is trending, but the moment you want
    to re-cut an older release, test a fix, or take a request, "it has to have
    been in the last research run" is an arbitrary wall. This builds the same
    Candidate from a model_info lookup, so everything downstream — enrich,
    assess, run — is identical to the trending path.

    `rank=0` marks it as not-from-the-list; nothing ranks it against models it
    was never compared with.
    """
    model = api().model_info(repo_id, files_metadata=False)
    return Candidate(
        repo_id=model.id,
        rank=0,
        pipeline_tag=getattr(model, "pipeline_tag", None),
        tags=list(getattr(model, "tags", []) or []),
        downloads=getattr(model, "downloads", 0) or 0,
        likes=getattr(model, "likes", 0) or 0,
        created_at=(str(model.created_at)[:10]
                    if getattr(model, "created_at", None) else None),
        gated=getattr(model, "gated", False),
        private=bool(getattr(model, "private", False)),
    )


# =====================================================
# STEP 1b — the filter
# =====================================================
def _looks_derivative(candidate: Candidate):
    """Is this somebody else's weights re-cut rather than an original release?

    Three independent signals, because no single one is reliable: the Hub's
    own base_model relation tags, the quantization/adapter library tags, and
    the naming conventions the community actually uses.
    """
    lowered = {t.lower() for t in candidate.tags}

    # The Hub's own provenance tags. "base_model:finetune:X" and friends are
    # emitted from the model card's base_model field and are the strongest
    # signal available. "base_model:X" alone is ambiguous (plenty of original
    # releases cite an ancestor), so only the typed relations count.
    for tag in lowered:
        if tag.startswith("base_model:") and any(
                rel in tag for rel in
                (":finetune:", ":merge:", ":adapter:", ":quantized:")):
            return f"derivative ({tag.split(':')[1]})"

    if hit := lowered & config.DERIVATIVE_TAGS:
        return f"derivative (tag: {sorted(hit)[0]})"

    name = candidate.name.lower()
    for hint in config.DERIVATIVE_NAME_HINTS:
        if hint in name:
            return f"derivative (name: {hint.strip('-')})"
    return None


def filter_candidates(candidates):
    """Mark each candidate as kept or dropped, in place. Returns the kept ones.

    Nothing is removed from the list — the report shows what was rejected and
    why, so a model dropped by a rule the user disagrees with is visible
    rather than silently missing.
    """
    kept = []
    for candidate in candidates:
        if candidate.private:
            candidate.dropped = "private"
        elif (candidate.pipeline_tag
                and candidate.pipeline_tag not in config.ALLOWED_PIPELINE_TAGS):
            candidate.dropped = f"not text-capable ({candidate.pipeline_tag})"
        elif not candidate.pipeline_tag and "transformers" not in candidate.tags:
            # No pipeline tag AND not a transformers repo: almost always a
            # dataset-ish or research artefact with no conversion path.
            candidate.dropped = "no pipeline tag, not a transformers repo"
        elif reason := _looks_derivative(candidate):
            candidate.dropped = reason
        else:
            kept.append(candidate)
    return kept


# =====================================================
# STEP 1c — enrichment
# =====================================================
_CONFIG_PARAM_KEYS = (
    ("n_layers", ("num_hidden_layers", "n_layer", "num_layers", "n_layers")),
    ("hidden_size", ("hidden_size", "n_embd", "d_model", "hidden_dim")),
    ("vocab_size", ("vocab_size",)),
)

_EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts",
                "moe_num_experts", "num_experts_per_tok")


def _dig(cfg: dict, keys):
    """First present key, looking inside text_config/llm_config too.

    Multimodal repos put the language model's real shape one level down, and
    that is the part that gets quantized — reading only the top level would
    report a 27B vision-language model as having no layers at all.
    """
    for scope in (cfg, cfg.get("text_config") or {}, cfg.get("llm_config") or {},
                  cfg.get("language_config") or {}):
        for key in keys:
            if isinstance(scope, dict) and scope.get(key) is not None:
                return scope[key]
    return None


def _typed_tags(tags, prefix):
    """The values of the Hub's `prefix:value` tags, in order, deduplicated.

    The Hub emits `license:apache-2.0`, `arxiv:2502.16161`, `language:pt` and
    friends alongside the free-form tags. They are already in the single
    `list_models` response the sweep makes, so reading them costs nothing —
    they were simply never parsed.
    """
    seen, out = set(), []
    for tag in tags or []:
        if not isinstance(tag, str) or not tag.lower().startswith(f"{prefix}:"):
            continue
        value = tag.split(":", 1)[1].strip()
        if value and value.lower() not in seen:
            seen.add(value.lower())
            out.append(value)
    return out


def _fetch_readme(repo_id):
    """The source model's own card, verbatim, or None.

    The agent writes the quant repo's card from this rather than from what it
    remembers about the model. It is the difference between describing the
    model and inventing it — and it carries the publisher's own citation
    block, which is the only trustworthy source for one.
    """
    try:
        path = hf_hub_download(repo_id=repo_id, filename="README.md",
                               repo_type="model", token=config.TOKEN)
        return open(path, encoding="utf-8", errors="replace").read()
    except (EntryNotFoundError, HfHubHTTPError, OSError, ValueError):
        return None


def _fetch_config(repo_id):
    try:
        path = hf_hub_download(repo_id=repo_id, filename="config.json",
                               repo_type="model", token=config.TOKEN)
        return json.loads(open(path, encoding="utf-8").read())
    except (EntryNotFoundError, HfHubHTTPError, OSError, ValueError):
        return {}


def _official_gguf(candidate: Candidate, client: HfApi):
    """The publisher's own GGUF repo for this model, and its BF16/F16 file.

    This is worth real money in the pipeline: when the publisher already ships
    a BF16 GGUF, downloading it costs ONE copy of the weights, against
    safetensors PLUS a conversion output PLUS the conversion runtime. It also
    sidesteps conversion bugs entirely, since their file was built with
    whatever fork actually supports the architecture.
    """
    for suffix in ("-GGUF", "-gguf", ".gguf"):
        repo = f"{candidate.repo_id}{suffix}"
        try:
            info = client.model_info(repo, files_metadata=True)
        except Exception:
            continue
        best = None
        for sibling in info.siblings or []:
            name = sibling.rfilename
            if not name.lower().endswith(".gguf"):
                continue
            # Prefer BF16, then F16. Never a quant — a quant as the source
            # would bake its error into every file cut from it.
            rank = (0 if "bf16" in name.lower()
                    else 1 if "f16" in name.lower() else 9)
            size = getattr(sibling, "size", None) or 0
            if rank < 9 and (best is None or rank < best[0]):
                best = (rank, name, size)
        if best:
            return repo, best[1], best[2]
        return repo, None, 0
    return None, None, 0


def _community_ggufs(candidate: Candidate, client: HfApi, limit=5):
    """Other people's GGUF repos for this model.

    Not a disqualifier — the user quantizes regardless — but it tells you
    whether you would be first, which is the whole reason for watching
    trending. Matched on the model name so a differently-named org's quant of
    the same weights still shows up.
    """
    try:
        found = client.list_models(search=f"{candidate.name}-GGUF",
                                   limit=limit * 4, full=False)
    except Exception:
        return []
    out = []
    for model in found:
        if model.id.startswith(f"{config.HF_NAMESPACE}/"):
            continue          # our own repo is not competition
        if candidate.name.lower() in model.id.split("/")[-1].lower():
            out.append(model.id)
        if len(out) >= limit:
            break
    return out


_GGUF_SUFFIX = re.compile(r"[-_.]gguf$", re.I)


def _normalised(repo_name):
    """A repo's model name, with any GGUF suffix and case difference removed.

    So `Granite-4.2-3b-GGUF`, `granite-4.2-3b-gguf` and `granite-4.2-3b` all
    reduce to the same thing, and a repo published under any of those
    spellings is recognised as ours.
    """
    return _GGUF_SUFFIX.sub("", repo_name.split("/")[-1]).lower()


@lru_cache(maxsize=1)
def our_repos():
    """Every model repo in our namespace, fetched once per process.

    One request, cached, rather than a lookup per candidate: a hundred-model
    sweep asks this a hundred times and the answer never changes within it.
    Call `forget_our_repos()` after publishing something new.

    An empty tuple on failure. This feeds an optimisation — "you have already
    done some of this" — so being unable to answer must degrade to proposing
    the full sweep, never to an error.
    """
    try:
        return tuple(model.id for model in
                     api().list_models(author=config.HF_NAMESPACE, limit=1000))
    except Exception:
        return ()


def forget_our_repos():
    """Drop the cached namespace listing. Call after publishing a new repo."""
    our_repos.cache_clear()


def _already_ours(candidate: Candidate, client: HfApi):
    """What we have already published for this model, if anything.

    Without this the agent re-proposes work it has already done — and does so
    at full price, since the estimate would cover a sweep that is mostly
    finished. It also catches the genuinely valuable case: a repo left
    part-published by an interrupted run, where the remaining handful of quants
    is the cheapest work available anywhere on the list.

    The Hub listing is the authority here rather than the local status.json,
    because the status file is per-machine and the question being asked is
    "what can people already download".

    Both the predicted repo id AND the namespace listing are consulted. The
    prediction alone was too narrow: it is exactly `{namespace}/{name}-GGUF`,
    so a repo published by an older script, under a lowercase suffix, or
    renamed at any point read as "nothing published" and bought the whole
    sweep again.

    Matching is on the EXACT normalised name, never a substring: mistaking
    `granite-4.2-3b-instruct` for `granite-4.2-3b` would skip real work, which
    is a worse failure than missing a repo and redoing it.
    """
    wanted = _normalised(candidate.name)
    repo_ids = [candidate.target_repo]
    repo_ids += [repo for repo in our_repos()
                 if _normalised(repo) == wanted and repo != candidate.target_repo]

    files, found_in = set(), []
    for repo_id in repo_ids:
        try:
            listing = set(client.list_repo_files(repo_id=repo_id,
                                                 repo_type="model"))
        except Exception:
            continue                 # does not exist, or not readable
        if listing:
            files |= listing
            found_in.append(repo_id)

    candidate.published_repos = found_in
    quants = {quant for name in files if (quant := quant_of(name))}
    return files, quants


def enrich(candidate: Candidate, check_ggufs=True, want_readme=True):
    """Fill in everything the feasibility model needs. One candidate, in place.

    `want_readme` is separable because the card path wants the source README
    and the sweep does not: enrich() runs over every surviving candidate, and
    a README download each is a round trip per model for data only one of them
    will ever use.
    """
    client = api()
    try:
        info = client.model_info(candidate.repo_id, files_metadata=True)
    except Exception as e:
        candidate.error = f"{type(e).__name__}: {e}"
        return candidate

    # model_info carries the full tag list even when the candidate came from
    # `one()`, so provenance is read from the response rather than from
    # whatever the trending listing happened to include.
    tags = list(getattr(info, "tags", []) or []) or candidate.tags
    candidate.author = getattr(info, "author", None) or candidate.org
    licenses = _typed_tags(tags, "license")
    candidate.license = licenses[0] if licenses else None
    candidate.arxiv_ids = _typed_tags(tags, "arxiv")
    candidate.languages = _typed_tags(tags, "language")
    if want_readme:
        candidate.readme = _fetch_readme(candidate.repo_id)

    # The safetensors header carries an exact parameter count per dtype — far
    # better than parsing "7B" out of the name, which lies constantly (the
    # K2-Horizon 3.7B ships as "4B", and MoE names quote active, not stored).
    safetensors = getattr(info, "safetensors", None)
    if safetensors:
        candidate.params = getattr(safetensors, "total", None)
        candidate.dtypes = dict(getattr(safetensors, "parameters", {}) or {})

    weight_ext = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
    candidate.source_bytes = sum(
        (getattr(s, "size", None) or 0) for s in (info.siblings or [])
        if s.rfilename.lower().endswith(weight_ext))

    cfg = _fetch_config(candidate.repo_id)
    if cfg:
        candidate.architectures = list(cfg.get("architectures") or [])
        candidate.model_type = cfg.get("model_type")
        for attr, keys in _CONFIG_PARAM_KEYS:
            setattr(candidate, attr, _dig(cfg, keys))
        for key in _EXPERT_KEYS:
            if (value := _dig(cfg, (key,))) and key != "num_experts_per_tok":
                candidate.n_experts = value
                break
        candidate.tied_embeddings = bool(_dig(cfg, ("tie_word_embeddings",)))
        candidate.is_multimodal = bool(
            cfg.get("vision_config") or cfg.get("vision_tower")
            or cfg.get("mm_vision_tower")
            or candidate.pipeline_tag == "image-text-to-text")

    # Fall back to the config when there is no safetensors header (older repos,
    # or .bin-only uploads): 2 bytes per parameter is the bf16 assumption the
    # whole estimator runs on anyway.
    if not candidate.params and candidate.source_bytes:
        candidate.params = int(candidate.source_bytes / 2)

    if check_ggufs:
        (candidate.official_gguf, candidate.official_bf16_file,
         candidate.official_bf16_bytes) = _official_gguf(candidate, client)
        candidate.community_ggufs = _community_ggufs(candidate, client)
        (candidate.published_files,
         candidate.published_quants) = _already_ours(candidate, client)
    return candidate


# =====================================================
# STEP 5 — verification of what we published
# =====================================================
def published_files(repo_id):
    """Every file in one of our GGUF repos with its real size on the Hub.

    Used by the final check and the model card: the card's size table must
    report what is actually there, not what the estimator predicted.
    """
    try:
        info = api().model_info(repo_id, files_metadata=True)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "files": []}

    files = []
    for sibling in info.siblings or []:
        files.append({
            "name": sibling.rfilename,
            "bytes": getattr(sibling, "size", None) or 0,
            "sha": getattr(getattr(sibling, "lfs", None), "sha256", None),
        })
    files.sort(key=lambda f: f["name"])
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "downloads": getattr(info, "downloads", 0),
        "likes": getattr(info, "likes", 0),
        "last_modified": str(getattr(info, "last_modified", "") or "")[:19],
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
        "error": None,
    }


def resolve_base_model(name):
    """The original repo a quant repo was cut from, given only the model name.

    Needed when a card is written for a repo the agent did not produce — the
    hand-written scripts left no job record. Getting this wrong is worse than
    omitting it: `base_model` drives the Hub's model tree, and a bad value
    points the tree at a repo that does not exist.

    So this only answers when it is SURE: exactly one non-ours repo whose name
    matches exactly, case-insensitively.
    """
    try:
        found = list(api().list_models(search=name, limit=30, full=False))
    except Exception:
        return None
    matches = [m.id for m in found
               if m.id.split("/")[-1].lower() == name.lower()
               and not m.id.startswith(f"{config.HF_NAMESPACE}/")]
    return matches[0] if len(matches) == 1 else None


QUANT_IN_NAME = re.compile(
    r"-((?:IQ|Q|TQ)\d[A-Z0-9_]*|BF16|F16|F32|MXFP4_MOE)\.gguf$", re.I)


def quant_of(filename):
    """The quant type a published filename encodes, or None."""
    match = QUANT_IN_NAME.search(filename)
    return match.group(1).upper() if match else None
