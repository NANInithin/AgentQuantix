"""Step 5: check what actually landed on the Hub, then write the model card.

The verification is not a formality. A run can finish "successfully" with a
quant missing because llama-quantize gave up on it after three tries, or with a
file that uploaded at the wrong size because a retry raced. The card can only
ever describe files that really exist, at the sizes they really are.

There are two ways a card gets written, and they trade off differently:

  * `render()` is a template. Deterministic, no network, no model — which is
    what `aqx run` needs, because there is no LLM in that loop. Every card it
    produces is the same document with different numbers.

  * `facts()` + `validate()` is the agent path. The code computes the facts;
    the agent composes the whole document from them, table included; then the
    claims it made are checked back against the verified listing before
    anything is published.

The second is the stricter of the two, which is not the obvious way round. A
template's numbers are trusted because a template "cannot be wrong" — but
`verify()` already flags files whose size looks impossible, and nothing acts
on it. A composed card has to pass a check that a template never faces.

What the check enforces is CLAIMS, not FORM. The agent may lay the table out
however it likes, group the files however it likes, and write whatever prose
it likes. It may not name a file that does not exist, state a size that is not
the real one, or put a value in `base_model` that was not resolved — the Hub
builds its model tree from that field, and a wrong value points the tree at a
repo that does not exist.
"""

from __future__ import annotations

import re
import time

from huggingface_hub import HfApi

from . import config, hub

GB = 1024 ** 3

# Rough guidance, shown next to each quant so the card is useful to someone
# choosing a file rather than just a list of bytes.
QUANT_NOTES = {
    "BF16": "Full precision source. Every quant below is cut from this file.",
    "F16": "Full precision source.",
    "Q8_0": "Effectively lossless. Use when disk and RAM are not the constraint.",
    "Q6_K": "Near-lossless; the last stop before quality becomes measurable.",
    "Q5_K_M": "Very good quality, noticeably smaller than Q6_K.",
    "Q5_K_S": "Slightly smaller than Q5_K_M for a slight quality cost.",
    "Q4_K_M": "The usual default. Best quality-per-byte for most people.",
    "Q4_K_S": "A little smaller than Q4_K_M, a little worse.",
    "Q4_0": "Legacy round-to-nearest. Prefer Q4_K_M unless a runtime needs this.",
    "Q4_1": "Legacy. Prefer Q4_K_M.",
    "Q5_0": "Legacy. Prefer Q5_K_M.",
    "Q5_1": "Legacy. Prefer Q5_K_M.",
    "Q3_K_L": "Small, with real quality loss. Usable when RAM is tight.",
    "Q3_K_M": "Smaller again; noticeable degradation.",
    "Q3_K_S": "Aggressive. Prefer IQ3_M at a similar size.",
    "Q2_K": "Very small, heavily degraded. For experimentation.",
    "Q2_K_S": "Smaller than Q2_K, at a further quality cost.",
    "Q1_0": "Extreme. Included for completeness.",
    "Q2_0": "Extreme, group-64. Included for completeness.",
    "IQ4_XS": "Best sub-4.5bpw option; usually beats Q4_K_S at a smaller size.",
    "IQ4_NL": "Non-linear 4-bit; good on hardware without fast K-quant kernels.",
    "IQ3_M": "Strong at ~3.7bpw, clearly better than Q3_K_M.",
    "IQ3_S": "Slightly smaller than IQ3_M.",
    "IQ3_XS": "Aggressive but coherent.",
    "IQ3_XXS": "Very aggressive; the last coherent step down.",
    "IQ2_M": "The smallest size most people find usable.",
    "IQ2_S": "Below the usual usability line.",
    "IQ2_XS": "Experimental.",
    "IQ2_XXS": "Experimental.",
    "IQ1_M": "Extreme. Expect substantial degradation.",
    "IQ1_S": "Extreme. Expect substantial degradation.",
    "MXFP4_MOE": "MoE-only 4-bit microscaling format for the expert tensors.",
    "TQ1_0": "Ternary. Only meaningful for ternary-trained weights.",
    "TQ2_0": "Ternary. Only meaningful for ternary-trained weights.",
}

# Roughly ordered best-quality-first, so the card's table reads top to bottom
# as "biggest and best" down to "smallest and roughest".
QUALITY_ORDER = [
    "BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_M", "Q4_K_S", "IQ4_NL", "IQ4_XS", "Q4_1", "Q4_0", "MXFP4_MOE",
    "Q3_K_L", "Q3_K_M", "IQ3_M", "IQ3_S", "Q3_K_S", "IQ3_XS", "IQ3_XXS",
    "Q2_K", "IQ2_M", "Q2_K_S", "IQ2_S", "Q2_XS", "IQ2_XS", "IQ2_XXS",
    "Q2_0", "TQ2_0", "IQ1_M", "IQ1_S", "TQ1_0", "Q1_0",
]


def _order(quant):
    return (QUALITY_ORDER.index(quant) if quant in QUALITY_ORDER
            else len(QUALITY_ORDER))


def verify(job_or_repo, expected_quants=None, base_name=None):
    """Compare what is on the Hub against what the run was supposed to produce.

    Accepts a Job or a bare repo id. Returns a dict the caller can print and
    the card generator can consume — including the MISSING list, which is the
    part worth acting on.
    """
    if hasattr(job_or_repo, "target_repo"):
        repo_id = job_or_repo.target_repo
        # The FULL sweep, not just what this run built. A resume run that added
        # four quants to an existing repo should be checked against all thirty,
        # or the check would pass while twenty-six were still missing.
        expected_quants = (expected_quants or job_or_repo.all_quants
                           or job_or_repo.quants)
        base_name = base_name or job_or_repo.base_name
    else:
        repo_id = job_or_repo

    listing = hub.published_files(repo_id)
    if listing.get("error"):
        return {"repo_id": repo_id, "error": listing["error"], "files": []}

    files = []
    for entry in listing["files"]:
        if not entry["name"].lower().endswith(".gguf"):
            continue
        quant = hub.quant_of(entry["name"])
        files.append({
            "name": entry["name"],
            "quant": quant or ("mmproj" if entry["name"].startswith("mmproj")
                               else "?"),
            "bytes": entry["bytes"],
            "gb": round(entry["bytes"] / GB, 2),
            # A GGUF is never a few kilobytes. Anything that small is a failed
            # or truncated upload masquerading as a finished file.
            "suspect": entry["bytes"] < 1_000_000,
        })
    files.sort(key=lambda f: _order(f["quant"]))

    present = {f["quant"] for f in files}
    missing = ([q for q in expected_quants if q not in present]
               if expected_quants else [])

    return {
        "repo_id": repo_id,
        "url": listing["url"],
        "base_name": base_name or repo_id.split("/")[-1].removesuffix("-GGUF"),
        "files": files,
        "count": len(files),
        "total_gb": round(sum(f["bytes"] for f in files) / GB, 2),
        "missing": missing,
        "suspect": [f["name"] for f in files if f["suspect"]],
        "last_modified": listing["last_modified"],
        "error": None,
    }


def _source_repo(verification, job=None, assessment=None):
    """The model these quants were cut from, or None if it is not certain.

    Descending order of certainty: the job that produced the files, the
    assessment it came from, then a Hub lookup by name that only answers on an
    unambiguous match. None means "say nothing" — never a guess.
    """
    assessment = assessment or (job.assessment if job else {}) or {}
    return ((job.repo_id if job else assessment.get("repo_id"))
            or hub.resolve_base_model(verification["base_name"]))


def facts(verification, job=None, assessment=None, candidate=None):
    """Everything true about this repo, with no prose and no opinions.

    This is what the agent composes a card FROM. It exists so the agent writes
    about a model it has actually been shown rather than one it half
    remembers: the source README is in here verbatim, and so are the licence,
    the authors and the arXiv ids the Hub reports. A citation belongs to the
    publisher, and the only safe way to produce one is to copy theirs.

    `candidate` is an enriched hub.Candidate when the caller has one. Without
    it the shape and provenance fields are simply absent, which is the honest
    result — the card writer must not fill them in from memory.
    """
    assessment = assessment or (job.assessment if job else {}) or {}
    source_repo = _source_repo(verification, job, assessment)
    fork = (job.fork if job else None) or (assessment.get("fork_leads") or [None])[0]

    out = {
        "repo_id": verification["repo_id"],
        "url": verification["url"],
        "base_name": verification["base_name"],

        # The authoritative file list. Sizes are bytes as the Hub reports
        # them; `gb` is the rounded figure a card would print.
        "files": [
            {"name": f["name"], "quant": f["quant"], "bytes": f["bytes"],
             "gb": f["gb"], "note": QUANT_NOTES.get(f["quant"], ""),
             "url": (f"https://huggingface.co/{verification['repo_id']}"
                     f"/blob/main/{f['name']}")}
            for f in verification["files"]
        ],
        "count": verification["count"],
        "total_gb": verification["total_gb"],
        "missing": verification["missing"],
        "suspect": verification["suspect"],

        # None means "not established". The card must then omit base_model
        # rather than invent one.
        "source_repo": source_repo,
        "source_url": (f"https://huggingface.co/{source_repo}"
                       if source_repo else None),
        "fork": fork,
        "quantized_on": time.strftime("%Y-%m-%d"),
    }

    if candidate is not None:
        out["source"] = {
            "author": candidate.author,
            "license": candidate.license,
            "arxiv_ids": list(candidate.arxiv_ids or []),
            "languages": list(candidate.languages or []),
            "tags": [t for t in (candidate.tags or []) if ":" not in t],
            "params": candidate.params,
            "architectures": list(candidate.architectures or []),
            "model_type": candidate.model_type,
            "n_layers": candidate.n_layers,
            "hidden_size": candidate.hidden_size,
            "vocab_size": candidate.vocab_size,
            "n_experts": candidate.n_experts,
            "is_multimodal": candidate.is_multimodal,
            "downloads": candidate.downloads,
            "likes": candidate.likes,
            "created_at": candidate.created_at,
            # Verbatim, untruncated. Truncating it is how a citation block
            # gets cut in half and then reconstructed from memory.
            "readme": candidate.readme,
        }
    return out


# =====================================================
# VALIDATION
# =====================================================
# A .gguf filename anywhere in the card — link, table cell, code block.
_GGUF_IN_TEXT = re.compile(r"[\w.\-]+\.gguf", re.I)

# `base_model:` in the YAML front matter. Only the front matter counts: the
# string can legitimately appear in prose or inside a fenced example.
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)
_BASE_MODEL_LINE = re.compile(r"^base_model:\s*(.+?)\s*$", re.M)

# A size written next to a filename, e.g. "4.49 GB". Matched per table row so
# a figure can be tied to the file it claims to describe.
_SIZE_IN_ROW = re.compile(r"(\d+(?:\.\d+)?)\s*(GB|GiB|MB)\b", re.I)

_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5})\b")

# A bare quant name in prose — "Q4_K_M", "IQ3_XXS", "MXFP4_MOE" — as opposed to
# one inside a filename. Checked separately because the filename rule cannot
# see them: a card can recommend "Q8_0_00" or "Q6_K_S" without ever writing a
# .gguf, and those two do not exist. Anchored on the real llama.cpp prefixes so
# ordinary words cannot match.
_BARE_QUANT = re.compile(
    r"(?<![\w.-])((?:IQ|Q|TQ)\d[A-Z0-9_]*|MXFP4_MOE|BF16|F16|F32)(?![\w.-])")


def _front_matter(text):
    match = _FRONT_MATTER.match(text or "")
    return match.group(1) if match else ""


def validate(text, card_facts):
    """Check a composed card's CLAIMS against the verified listing.

    Returns a list of human-readable problems; empty means publishable. The
    messages are written to be handed straight back to the agent, so each one
    says what is wrong and what the real value is.

    Deliberately NOT checked: headings, ordering, table shape, tone, length,
    extra sections, extra tags. The agent owns the document. This owns the
    facts.
    """
    problems = []
    files = card_facts["files"]
    real_names = {f["name"] for f in files}

    # ---- filenames ----------------------------------------------------
    mentioned = {m for m in _GGUF_IN_TEXT.findall(text or "")}
    # Case-insensitively resolve back to the real spelling so a case slip is
    # reported as a case slip rather than as an invented file.
    lowered = {n.lower(): n for n in real_names}
    invented = sorted({m for m in mentioned if m.lower() not in lowered})
    if invented:
        problems.append(
            f"names {len(invented)} file(s) that are not in the repo: "
            f"{', '.join(invented)}. Only these exist: "
            f"{', '.join(sorted(real_names))}")

    mentioned_lower = {m.lower() for m in mentioned}
    omitted = sorted(n for n in real_names if n.lower() not in mentioned_lower)
    if omitted:
        problems.append(
            f"omits {len(omitted)} file(s) that are in the repo: "
            f"{', '.join(omitted)}. Every published file must appear.")

    # ---- quant names in prose ------------------------------------------
    # Filenames are stripped first so "Model-Q4_K_M.gguf" is judged by the
    # filename rule alone and cannot be reported twice.
    prose = _GGUF_IN_TEXT.sub(" ", text or "")
    real_quants = {f["quant"].upper() for f in files}
    unreal = sorted({q.upper() for q in _BARE_QUANT.findall(prose)}
                    - real_quants)
    if unreal:
        problems.append(
            f"recommends {len(unreal)} quant(s) this repo does not contain: "
            f"{', '.join(unreal)}. Available: "
            f"{', '.join(sorted(real_quants))}")

    # ---- sizes --------------------------------------------------------
    # Line-scoped: a size counts as a claim about a file only when it shares a
    # line with that file's name, which is what a table row looks like.
    by_name = {f["name"].lower(): f for f in files}
    for line in (text or "").splitlines():
        names_here = {m.lower() for m in _GGUF_IN_TEXT.findall(line)}
        names_here &= set(by_name)
        if len(names_here) != 1:
            continue                      # ambiguous or none; nothing to bind
        entry = by_name[next(iter(names_here))]
        for raw, unit in _SIZE_IN_ROW.findall(line):
            claimed = float(raw)
            unit = unit.upper()
            actual = (entry["bytes"] / 1024 ** 2 if unit == "MB"
                      else entry["bytes"] / GB)   # GB and GiB both mean GiB

            # Tolerance follows the precision the CARD chose, rather than a
            # fixed epsilon. "0.9 GB" and "0.93 GB" describe the same file at
            # different precisions and both are honest; "3.1 GB" is not. Half
            # of the last written decimal place is exactly the range that
            # rounds to what was written.
            decimals = len(raw.split(".")[1]) if "." in raw else 0
            tolerance = 0.5 * 10 ** -decimals * 1.01     # 1.01: float slack

            if abs(claimed - actual) > tolerance:
                problems.append(
                    f"{entry['name']}: card says {claimed:g} {unit}, "
                    f"actual size is {actual:.2f} {unit}")

    # ---- base_model ---------------------------------------------------
    declared = _BASE_MODEL_LINE.search(_front_matter(text))
    expected = card_facts.get("source_repo")
    if declared:
        value = declared.group(1).strip().strip('"\'')
        if not expected:
            problems.append(
                f"declares base_model: {value}, but the source repo was never "
                "established. Omit base_model entirely - the Hub builds its "
                "model tree from it and a wrong value points at nothing.")
        elif value != expected:
            problems.append(
                f"declares base_model: {value}, but the source is {expected}.")
    elif expected:
        problems.append(
            f"is missing `base_model: {expected}` from the front matter.")

    # ---- runtime requirement -------------------------------------------
    fork = card_facts.get("fork")
    if fork and fork.get("repo") and fork["repo"] not in (text or ""):
        problems.append(
            f"does not mention that these files need the {fork['repo']} build "
            f"at {fork.get('ref', '?')}. This architecture is not in upstream "
            "llama.cpp, so without that note the files will not run.")

    # ---- citations ------------------------------------------------------
    # An arXiv id the Hub does not report for this model is one the writer
    # produced from memory, which is the failure mode worth catching: a
    # fabricated citation on a public repo is worse than no citation.
    known = set((card_facts.get("source") or {}).get("arxiv_ids") or [])
    readme = ((card_facts.get("source") or {}).get("readme")) or ""
    for found in set(_ARXIV_ID.findall(text or "")):
        if found not in known and found not in readme:
            problems.append(
                f"cites arXiv {found}, which is not in this model's Hub tags "
                "or its source README. Cite only what the source provides.")

    return problems


def render(verification, job=None, assessment=None):
    """The README.md for a quant repo, built from the verified file listing.

    The deterministic fallback. `aqx run` has no LLM in the loop, and a run
    must never finish without a card, so this stays as the floor beneath the
    agent path.
    """
    assessment = assessment or (job.assessment if job else {}) or {}
    base_name = verification["base_name"]

    # The source repo, in descending order of certainty: the job that produced
    # these files, the assessment it came from, or a Hub lookup by name. When
    # none of those is sure, base_model is OMITTED rather than guessed — the
    # Hub builds its model tree from that field, and a wrong value points the
    # tree at a repo that does not exist.
    source_repo = (job.repo_id if job else assessment.get("repo_id")) \
        or hub.resolve_base_model(base_name)

    tags = ["gguf", "llama.cpp", "quantized"]

    front = [
        "---",
        *([f"base_model: {source_repo}"] if source_repo else []),
        "library_name: gguf",
        "pipeline_tag: text-generation",
        "tags:",
        *[f"  - {t}" for t in tags],
        "---",
        "",
    ]

    origin = (f"[{source_repo}](https://huggingface.co/{source_repo})"
              if source_repo else f"`{base_name}`")
    body = [
        f"# {base_name} GGUF",
        "",
        f"GGUF quantizations of {origin}, covering {verification['count']} files "
        f"({verification['total_gb']:.1f} GB total).",
        "",
        "## Files",
        "",
        "| File | Quant | Size | Notes |",
        "|---|---|---:|---|",
    ]
    for entry in verification["files"]:
        note = QUANT_NOTES.get(entry["quant"], "")
        if entry["quant"] == "mmproj":
            note = ("Vision projector. Download alongside any quant below to "
                    "use the image input.")
        body.append(
            f"| [{entry['name']}](https://huggingface.co/{verification['repo_id']}"
            f"/blob/main/{entry['name']}) | `{entry['quant']}` "
            f"| {entry['gb']:.2f} GB | {note} |")

    body += [
        "",
        "## Which one should I download?",
        "",
        "Pick the largest file that leaves a couple of gigabytes of headroom "
        "on the device you will run it on — the model has to fit in RAM (or "
        "VRAM, if you are offloading) alongside the KV cache and the OS.",
        "",
        "- Plenty of memory: **Q6_K** or **Q8_0**.",
        "- The usual choice: **Q4_K_M**.",
        "- Tight on memory: **IQ4_XS**, then **IQ3_M**, then **IQ2_M**.",
        "- The `IQ*` files generally beat a `Q*` file of similar size, at the "
        "cost of slightly slower inference on some hardware.",
        "",
    ]

    # `or [None]` rather than a .get() default: the default only applies when
    # the key is ABSENT, and the key is always present — it is an empty list
    # for every architecture upstream already supports. Indexing [0] into that
    # raised IndexError on the common case, which then surfaced as "the run
    # failed" long after the run had actually succeeded.
    fork = (job.fork if job else None) or (assessment.get("fork_leads") or [None])[0]
    if fork:
        body += [
            "## Runtime requirement",
            "",
            f"This architecture is not in upstream llama.cpp yet. These files "
            f"were built with [`{fork['repo']}`](https://github.com/"
            f"{fork['repo']}) at `{fork['ref']}`, and need that build (or "
            "upstream once it merges) to run.",
            "",
        ]

    body += [
        "## Usage",
        "",
        "```bash",
        f"llama-cli -hf {verification['repo_id']}:Q4_K_M -p \"Hello\"",
        "```",
        "",
        "Or download one file and point at it directly:",
        "",
        "```bash",
        f"huggingface-cli download {verification['repo_id']} "
        f"{base_name}-Q4_K_M.gguf --local-dir .",
        f"llama-cli -m {base_name}-Q4_K_M.gguf -p \"Hello\"",
        "```",
        "",
        "---",
        "",
        f"Quantized with [llama.cpp](https://github.com/ggml-org/llama.cpp) by "
        f"AgentQuantix on {time.strftime('%Y-%m-%d')}. "
        "Licensing follows the base model.",
        "",
    ]
    return "\n".join(front + body)


class CardRejected(ValueError):
    """A composed card made a claim the verified listing does not support.

    Carries the problem list so the caller can hand it back to the writer and
    let it correct the card, rather than only reporting that something failed.
    """

    def __init__(self, problems):
        self.problems = problems
        super().__init__("; ".join(problems))


def publish(verification, job=None, assessment=None, dry_run=False,
            content=None, candidate=None):
    """Push a card to the repo as README.md. Returns the text.

    With `content`, the card is the agent's own document and is validated
    against the verified listing first — an unpublishable one raises
    CardRejected rather than going up. Without it, the template renders.
    """
    if content is None:
        text = render(verification, job=job, assessment=assessment)
    else:
        text = content
        problems = validate(text, facts(verification, job=job,
                                        assessment=assessment,
                                        candidate=candidate))
        if problems:
            raise CardRejected(problems)

    if dry_run:
        return text

    api = HfApi(token=config.TOKEN)
    api.upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=verification["repo_id"],
        repo_type="model",
    )
    return text
