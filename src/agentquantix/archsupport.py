"""Can llama.cpp convert this architecture — and if not, who has a branch that can.

Two jobs:

  * The GATE. `convert_hf_to_gguf.py --print-supported-models` is the
    authoritative list of architectures the local checkout can convert, and
    `quantize.cpp` is the authoritative list of quant types it can produce.
    Both are read from the source tree rather than assumed, because the whole
    point of a fork is that its lists differ from upstream's.

  * The HUNT. When the gate says no, this looks for a branch that says yes:
    the publisher's own llama.cpp fork, or an open upstream PR adding the
    architecture. That is exactly how K2-Horizon got done by hand — upstream
    had no k2_horizon class, MBZUAI-IFM's `model/K2Horizon` branch did — and
    it is the difference between skipping the most interesting models on the
    trending list and being first to publish them.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config

GITHUB_API = "https://api.github.com"
UPSTREAM_REPO = "ggml-org/llama.cpp"


# =====================================================
# THE GATE
# =====================================================
REGISTRATION = re.compile(
    r"register\(\s*((?:[\"'][^\"']+[\"']\s*,?\s*|\n)+)\)")


def _ask_converter(llama_dir: Path):
    """Run the converter's own --print-supported-models. Returns (set, error).

    The authoritative answer when it works, because the script builds its
    registry at import time and a fork can register an architecture from a
    file nothing would think to read.

    It only works when the converter's dependencies are installed, and torch
    is not one of ours — it is an 800 MB dependency needed only for actual
    conversion, so requiring it to ASK A QUESTION would be absurd.
    """
    script = llama_dir / "convert_hf_to_gguf.py"
    if not script.exists():
        return set(), f"no convert_hf_to_gguf.py in {llama_dir}"
    try:
        done = subprocess.run(
            [sys.executable, str(script), "--print-supported-models"],
            capture_output=True, text=True, timeout=300, cwd=str(llama_dir),
        )
    except Exception as e:
        return set(), f"{type(e).__name__}: {e}"

    # The converter prints this list through its logger, which writes to
    # stderr — reading only stdout silently yields an empty set, and an empty
    # set reads as "nothing is supported", which would send every candidate
    # down the fork-hunt path.
    out = done.stdout + done.stderr
    found = {line.strip().lstrip("-").strip()
             for line in out.splitlines() if line.strip().startswith("-")}
    if found:
        return found, None

    # Surface the actual cause. "ModuleNotFoundError: No module named 'torch'"
    # is a fixable problem; "could not read the supported-architecture list"
    # is a mystery that made every model on the report look unsupported.
    last = [l for l in out.strip().splitlines() if l.strip()]
    return set(), (last[-1].strip() if last else f"exit {done.returncode}")


def _parse_registrations(llama_dir: Path):
    """Read the @ModelBase.register(...) decorators straight from the source.

    The fallback for when the converter cannot be imported. Verified against
    the real thing on a checkout where both work: the parse returns a superset
    of what --print-supported-models lists, because it also sees registrations
    behind imports the script skips.

    Good enough to answer "can this be converted here", which is all the
    research step needs — and vastly better than reporting that nothing is
    supported because an unrelated dependency is missing.
    """
    names = set()
    for directory in (llama_dir / "conversion", llama_dir):
        if not directory.is_dir():
            continue
        for source in directory.glob("*.py"):
            try:
                text = source.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for block in REGISTRATION.findall(text):
                names.update(re.findall(r"[\"']([^\"']+)[\"']", block))
    return names


def converter_status(llama_dir: Path | None = None):
    """How the architecture list was obtained, and why. For doctor/bootstrap."""
    llama_dir = Path(llama_dir or config.UPSTREAM_LLAMA)
    found, error = _ask_converter(llama_dir)
    if found:
        return {"ok": True, "source": "converter", "count": len(found),
                "error": None}
    parsed = _parse_registrations(llama_dir)
    return {"ok": bool(parsed), "source": "source-parse" if parsed else None,
            "count": len(parsed), "error": error}


def supported_architectures(llama_dir: Path | None = None):
    """Architecture class names this checkout can convert.

    Asks the converter first and falls back to reading its registration
    decorators. The fallback is what keeps `aqx research` working on a machine
    that has llama.cpp but not torch — which is every machine that has only
    ever downloaded publisher GGUFs, and was previously reported as "could not
    read the supported-architecture list" against every single candidate.
    """
    llama_dir = Path(llama_dir or config.UPSTREAM_LLAMA)
    found, _ = _ask_converter(llama_dir)
    return found or _parse_registrations(llama_dir)


def supported_quants(llama_dir: Path | None = None):
    """Quant names this checkout's llama-quantize actually offers.

    A fork on its own branch can lag or lead upstream's quant table, and
    dropping an unknown type up front beats failing three hours into a sweep.
    """
    llama_dir = Path(llama_dir or config.UPSTREAM_LLAMA)
    source = llama_dir / "tools" / "quantize" / "quantize.cpp"
    if not source.exists():                     # older layout
        source = llama_dir / "examples" / "quantize" / "quantize.cpp"
    if not source.exists():
        return set()
    text = source.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r'\{\s*"(\w+)"\s*,\s*LLAMA_FTYPE', text))


def check(candidate, llama_dir=None, supported=None):
    """Is this candidate convertible here? Returns (ok, detail).

    Multimodal repos get a second look: llama.cpp often supports the language
    tower of a vision model while the vision tower needs --mmproj, and a repo
    whose top-level architecture is unknown may still convert if its
    text_config's architecture is known.
    """
    supported = supported if supported is not None else supported_architectures(llama_dir)
    if not supported:
        return False, "could not read the supported-architecture list"
    if not candidate.architectures:
        return False, "no architectures declared in config.json"
    hits = [a for a in candidate.architectures if a in supported]
    if hits:
        return True, f"supported: {hits[0]}"
    return False, f"unsupported: {', '.join(candidate.architectures)}"


# =====================================================
# THE HUNT
# =====================================================
def _github_json(url, timeout=20):
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "AgentQuantix"}
    # Unauthenticated GitHub search allows ~10 requests per minute, which a
    # trending list with several unsupported architectures exhausts easily —
    # and every exhausted request is a fork lead silently not found. Any
    # GITHUB_TOKEN in the environment raises that to 30/min, so it is used
    # when present and never required.
    if token := (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        # Unauthenticated GitHub search is rate-limited to ~10 requests/min.
        # A miss here is a missed opportunity, never a failure — the caller
        # falls back to reporting the model as unsupported.
        return None


def _branches_matching(owner_repo, needles):
    """Branches in a repo whose name looks like it is about this architecture."""
    data = _github_json(f"{GITHUB_API}/repos/{owner_repo}/branches?per_page=100")
    if not isinstance(data, list):
        return []
    out = []
    for branch in data:
        name = branch.get("name", "")
        flat = re.sub(r"[^a-z0-9]", "", name.lower())
        if any(needle in flat for needle in needles):
            out.append(name)
    # A branch named for the model beats "master" every time, and longer, more
    # specific names beat shorter ones.
    return sorted(out, key=len, reverse=True)


def _needles(candidate):
    """Lowercased, punctuation-stripped forms of every name this arch goes by."""
    raw = [candidate.model_type or "", *candidate.architectures,
           candidate.name]
    out = set()
    for item in raw:
        if not item:
            continue
        flat = re.sub(r"[^a-z0-9]", "", item.lower())
        # Strip the boilerplate suffixes so "K2HorizonForCausalLM" and
        # "k2_horizon" both reduce to "k2horizon".
        for suffix in ("forcausallm", "forconditionalgeneration",
                       "foraudiotexttotext", "model"):
            if flat.endswith(suffix) and len(flat) > len(suffix) + 2:
                flat = flat[:-len(suffix)]
        if len(flat) >= 4:
            out.add(flat)
    return out


# A hit is cached for a week (a fork branch does not appear and vanish), a miss
# for a day (an architecture with no support today may have a PR tomorrow, and
# a miss is often just a rate-limited search rather than a real absence).
FORK_CACHE_HIT_S = 7 * 24 * 3600
FORK_CACHE_MISS_S = 24 * 3600


def _fork_cache():
    try:
        return json.loads((config.STATE_DIR / "fork-leads.json").read_text())
    except Exception:
        return {}


def _fork_cache_put(key, leads):
    try:
        cache = _fork_cache()
        cache[key] = {"leads": leads, "found_at": time.time()}
        path = config.STATE_DIR / "fork-leads.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass


def find_fork(candidate, use_cache=True):
    """Repos/branches that plausibly add support for this architecture.

    Ordered by how much they should be trusted:

      1. A llama.cpp fork owned by the publisher, on a branch named for this
         model. This is the K2-Horizon case, and it is the only kind that is
         safe to build unattended — the weights and the converter come from
         the same people.
      2. An open PR against upstream adding the architecture. Community work,
         usually correct, but it is a moving target.

    Returns a list of dicts, best first. Never raises.
    """
    needles = _needles(candidate)
    if not needles:
        return []

    # Keyed by architecture, not by model: every K2-Horizon size needs the same
    # branch, so one search answers for all of them.
    key = (candidate.architectures or [candidate.model_type or ""])[0]
    if use_cache and key:
        entry = _fork_cache().get(key)
        if entry:
            age = time.time() - entry.get("found_at", 0)
            ttl = FORK_CACHE_HIT_S if entry.get("leads") else FORK_CACHE_MISS_S
            if age < ttl:
                return entry["leads"]

    leads = []

    # ---- 1. the publisher's own fork -------------------------------------
    # The GitHub org rarely matches the Hub org exactly (Hub "IFM" is GitHub
    # "MBZUAI-IFM"), so search GitHub for llama.cpp forks whose owner looks
    # related rather than guessing a single name.
    query = f"{candidate.org} llama.cpp in:name fork:true"
    found = _github_json(
        f"{GITHUB_API}/search/repositories?q={urllib.parse.quote(query)}"
        "&per_page=10")
    for repo in (found or {}).get("items", [])[:10]:
        full_name = repo.get("full_name", "")
        if not full_name.lower().endswith("/llama.cpp"):
            continue
        for branch in _branches_matching(full_name, needles):
            leads.append({
                "kind": "publisher-fork",
                "confidence": "high",
                "repo": full_name,
                "url": repo.get("html_url"),
                "ref": branch,
                "why": f"{full_name} branch '{branch}' names this architecture",
            })

    # ---- 2. an open upstream PR ------------------------------------------
    for needle in sorted(needles, key=len, reverse=True)[:2]:
        query = f"repo:{UPSTREAM_REPO} is:pr is:open {needle}"
        found = _github_json(
            f"{GITHUB_API}/search/issues?q={urllib.parse.quote(query)}"
            "&per_page=5")
        for item in (found or {}).get("items", [])[:5]:
            leads.append({
                "kind": "upstream-pr",
                "confidence": "medium",
                "repo": UPSTREAM_REPO,
                "url": item.get("html_url"),
                "ref": f"pull/{item.get('number')}/head",
                "why": f"open PR #{item.get('number')}: {item.get('title')}",
            })

    # De-duplicate on (repo, ref) while preserving the ordering above.
    seen, unique = set(), []
    for lead in leads:
        identity = (lead["repo"], lead["ref"])
        if identity not in seen:
            seen.add(identity)
            unique.append(lead)

    if key:
        _fork_cache_put(key, unique)
    return unique
