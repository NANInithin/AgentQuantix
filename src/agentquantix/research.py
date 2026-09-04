"""Steps 1-3 end to end: what is trending, what can this box do, what should we run.

One function, `research()`, produces the whole picture. It is deliberately a
pure read — nothing here downloads weights, creates a repo or writes to the
Hub, so it is safe to run at any time, including from a scheduled trigger, and
safe to re-run when a result looks wrong.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import time

from . import archsupport, config, feasibility, hub, sysprobe


def research(limit=None, probe_disk=True, workers=8, hunt_forks=True,
             on_progress=None):
    """Trending -> filtered -> enriched -> assessed, plus the machine probe.

    `workers` parallelises enrichment, which is network-latency bound: each
    surviving candidate needs a model_info, a config.json and two GGUF-repo
    lookups, and doing 40 of those serially is minutes of pure waiting.
    """
    def progress(message):
        if on_progress:
            on_progress(message)

    started = time.time()

    progress("probing this machine...")
    sysinfo = sysprobe.probe(measure_disk=probe_disk)

    progress(f"fetching the top {limit or config.TRENDING_LIMIT} trending models...")
    candidates = hub.trending(limit)
    kept = hub.filter_candidates(candidates)
    progress(f"{len(kept)} of {len(candidates)} survived the base-model filter")

    progress(f"reading metadata for {len(kept)} candidates...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(hub.enrich, kept))

    # Read the architecture list ONCE. It costs a subprocess that imports the
    # whole converter, and it is identical for every candidate.
    progress("reading the local llama.cpp architecture support list...")
    supported = archsupport.supported_architectures()
    available_quants = archsupport.supported_quants()

    assessments = []
    for candidate in kept:
        if candidate.error:
            continue
        arch_ok, arch_detail = archsupport.check(candidate, supported=supported)

        # The fork hunt is two or three unauthenticated GitHub requests per
        # model, and GitHub rate-limits those to ~10/min — so it only runs for
        # architectures that actually need one, which is a handful at most.
        leads = []
        if not arch_ok and hunt_forks:
            progress(f"hunting for a fork that supports {candidate.repo_id}...")
            leads = archsupport.find_fork(candidate)

        assessment = feasibility.assess(
            candidate, sysinfo, arch_ok, arch_detail, fork_leads=leads)

        # Drop quant types this checkout cannot produce. Only meaningful when
        # the local build is the one that will run it; a fork build gets
        # re-checked at run time against its own table.
        if available_quants and arch_ok:
            unsupported = [q for q in assessment["all_quants"]
                           if q not in available_quants]
            if unsupported:
                # Both lists get trimmed: `quants` so the run does not attempt
                # them, `all_quants` so verification does not later report them
                # as missing from a repo that was never going to have them.
                for key in ("quants", "all_quants"):
                    assessment[key] = [q for q in assessment[key]
                                       if q in available_quants]
                assessment["warnings"].append(
                    "this llama.cpp build does not offer "
                    f"{', '.join(unsupported)} — skipped")
        assessments.append(assessment)

    assessments.sort(key=feasibility.rank_key)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - started, 1),
        "system": sysinfo,
        "trending_count": len(candidates),
        "kept_count": len(kept),
        "dropped": [{"repo_id": c.repo_id, "rank": c.rank, "reason": c.dropped}
                    for c in candidates if c.dropped],
        "errors": [{"repo_id": c.repo_id, "error": c.error}
                   for c in kept if c.error],
        "assessments": assessments,
    }


def assess_one(repo_id, sysinfo=None, hunt_forks=True, probe_disk=False):
    """Assess a single named repo, trending or not.

    Same pipeline as research(), minus the trending fetch and the base-model
    filter. The filter is deliberately skipped: it exists to stop a 100-model
    sweep wasting time on other people's finetunes, but naming a repo IS the
    judgement it was standing in for, so applying it here would only refuse
    work that was explicitly asked for.

    Raises ValueError with the Hub's reason when the repo cannot be read at
    all — a typo or a gated repo should say so, not return an empty result.
    """
    try:
        candidate = hub.one(repo_id)
    except Exception as e:
        raise ValueError(f"{repo_id}: cannot read this repo from the Hub "
                         f"({type(e).__name__}: {e})") from e

    hub.enrich(candidate)
    if candidate.error:
        raise ValueError(f"{repo_id}: {candidate.error}")

    if sysinfo is None:
        sysinfo = sysprobe.probe(measure_disk=probe_disk)

    arch_ok, arch_detail = archsupport.check(candidate)
    leads = []
    if not arch_ok and hunt_forks:
        leads = archsupport.find_fork(candidate)

    assessment = feasibility.assess(candidate, sysinfo, arch_ok, arch_detail,
                                    fork_leads=leads)

    available = archsupport.supported_quants()
    if available and arch_ok:
        unsupported = [q for q in assessment["all_quants"] if q not in available]
        if unsupported:
            for key in ("quants", "all_quants"):
                assessment[key] = [q for q in assessment[key] if q in available]
            assessment["warnings"].append(
                "this llama.cpp build does not offer "
                f"{', '.join(unsupported)} — skipped")
    return assessment


def save(result, name=None):
    """Persist a research result so `aqx run` can be handed a model by name.

    The approval itself happens interactively in whichever harness is driving,
    but the ASSESSMENT has to survive that conversation: a run started an hour
    later must use the same quant list and the same imatrix strategy that were
    shown at the gate, not a freshly computed one that may have drifted.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = name or f"{time.strftime('%Y%m%d-%H%M%S')}-research"
    path = config.REPORTS_DIR / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    latest = config.REPORTS_DIR / "latest.json"
    latest.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path


def load_latest():
    path = config.REPORTS_DIR / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find(result, repo_id):
    """The assessment for one repo id, matched loosely.

    Accepts the full "org/name", just "name", or any unambiguous suffix — at
    an approval prompt people type the short name, and refusing it because the
    org was omitted is needless friction.
    """
    wanted = repo_id.strip().lower()
    exact = [a for a in result["assessments"] if a["repo_id"].lower() == wanted]
    if exact:
        return exact[0]
    partial = [a for a in result["assessments"]
               if a["repo_id"].lower().endswith("/" + wanted)
               or wanted in a["repo_id"].lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ValueError(
            f"'{repo_id}' matches {len(partial)}: "
            + ", ".join(a["repo_id"] for a in partial[:5]))
    return None
