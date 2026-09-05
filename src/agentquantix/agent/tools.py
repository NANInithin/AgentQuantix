"""The agent's tools, defined once.

The MCP server, the built-in OpenRouter loop and the Claude Code skill all
expose exactly these, with exactly these schemas. That is the whole trick
behind harness portability: the harness supplies a model and a conversation
loop, and nothing else, so swapping Claude Code for Codex for a raw
OpenRouter key changes who is thinking, never what the agent can do.

Two rules the tool boundary enforces, regardless of harness:

  * Nothing here downloads weights, builds anything or writes to the Hub
    except `start_quantization`, and that one requires an explicit list of
    models the user named. There is no tool an over-eager model can call that
    silently starts a six-hour job.

  * Every tool returns JSON-serialisable data with the numbers already
    computed. The model's job is to explain and to ask, not to do arithmetic
    on gigabytes.
"""

from __future__ import annotations

import json

from .. import card, config, feasibility, hub, report, research, sysprobe
from ..pipeline import run as run_mod
from ..pipeline.job import Job

# =====================================================
# SCHEMAS
# =====================================================
TOOLS = [
    {
        "name": "probe_system",
        "description": (
            "Measure the machine this session is running on: CPU, RAM, GPU and "
            "VRAM, free disk on the work volume, disk throughput, whether "
            "llama.cpp is built, and which Hub transfer backends are usable. "
            "Call this before reasoning about what will fit."),
        "input_schema": {
            "type": "object",
            "properties": {
                "remeasure_disk": {
                    "type": "boolean",
                    "description": "Re-run the disk throughput measurement "
                                   "instead of using the cached value.",
                    "default": False,
                },
            },
        },
    },
    {
        "name": "research_trending",
        "description": (
            "Steps 1-3. Fetch the top trending models from the Hugging Face "
            "Hub, drop everything that is not an original text-capable base "
            "model, read each survivor's real parameter count and "
            "architecture, check it against llama.cpp's supported list "
            "(hunting for a fork when it is not supported), and estimate "
            "disk, transfer and wall-clock time for a full quant sweep on "
            "THIS machine. Read-only: downloads nothing, creates nothing. "
            "Takes a minute or two for 100 models."),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many trending models to consider "
                                   f"(default {config.TRENDING_LIMIT}).",
                },
                "hunt_forks": {
                    "type": "boolean",
                    "description": "Search GitHub for a fork or PR adding an "
                                   "architecture upstream does not support.",
                    "default": True,
                },
            },
        },
    },
    {
        "name": "get_report",
        "description": (
            "The last research result, without re-running it. Use this to "
            "answer follow-up questions instead of researching again."),
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["table", "json", "markdown"],
                    "default": "table",
                },
                "top": {"type": "integer",
                        "description": "Only the top N candidates."},
            },
        },
    },
    {
        "name": "describe_candidate",
        "description": (
            "Everything known about ONE model: the quant list, the imatrix "
            "strategy and why, the disk and time breakdown, fork leads, "
            "blockers and warnings. Works for any repo id on the Hub, whether "
            "or not it is trending and whether or not research has been run - "
            "an off-list model is assessed on the spot. This is the right "
            "tool when the user names a specific model; do NOT run "
            "research_trending to go looking for it, because the trending "
            "list is not where it will be found."),
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string",
                          "description": "Any Hub repo id, or an unambiguous "
                                         "part of one from the last report."},
            },
            "required": ["model"],
        },
    },
    {
        "name": "plan_quantization",
        "description": (
            "Show exactly what would happen for the named models — target "
            "repos, quant lists, imatrix sources, fork builds, totals — "
            "WITHOUT starting anything. This is what you show the user at the "
            "approval gate. Always call this before start_quantization."),
        "input_schema": {
            "type": "object",
            "properties": {
                "models": {"type": "array", "items": {"type": "string"}},
                "sequential": {
                    "type": "boolean",
                    "description": "Price the plan for a sequential sweep "
                                   "(one quant on disk at a time) instead of "
                                   "the default overlapped one (three). Use "
                                   "this to show the user whether a model "
                                   "blocked on disk would fit.",
                    "default": False,
                },
                "upload_workers": {
                    "type": "integer",
                    "description": "Concurrent uploads to price for. Each one "
                                   "costs a further quant of peak disk.",
                },
            },
            "required": ["models"],
        },
    },
    {
        "name": "start_quantization",
        "description": (
            "Step 4. Quantize and upload the named models. This is the only "
            "tool that spends hours, writes to the Hub and uses the disk. It "
            "requires that the user has explicitly approved these specific "
            "models in the conversation — never call it to 'save time' or "
            "because a model looks like a good idea. Runs models "
            "smallest-first, is resumable, and deletes each quant as soon as "
            "it is uploaded."),
        "input_schema": {
            "type": "object",
            "properties": {
                "models": {"type": "array", "items": {"type": "string"}},
                "user_approved": {
                    "type": "boolean",
                    "description": "True only when the user has named these "
                                   "models and said to proceed.",
                },
                "sequential": {
                    "type": "boolean",
                    "description": "Quantize, upload, delete, one at a time. "
                                   "Holds one quant on disk instead of three, "
                                   "at the cost of uploads no longer "
                                   "overlapping quantization. Use when disk is "
                                   "the binding constraint.",
                    "default": False,
                },
                "upload_workers": {
                    "type": "integer",
                    "description": "Concurrent uploads. Raising this helps an "
                                   "upload-bound run and costs one more quant "
                                   "of peak disk per stream.",
                },
                "quantize_threads": {
                    "type": "integer",
                    "description": "Threads per llama-quantize. Omit to use "
                                   "every core; set it to leave the machine "
                                   "usable during a long sweep.",
                },
            },
            "required": ["models", "user_approved"],
        },
    },
    {
        "name": "verify_published",
        "description": (
            "Step 5. List what is actually on the Hub for a published quant "
            "repo, with real sizes, flagging anything missing from the "
            "intended sweep or suspiciously small."),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string",
                         "description": "Repo id, or just the model name."},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "get_card_facts",
        "description": (
            "Everything true about a published quant repo, with no prose: the "
            "verified file listing with real names and real sizes, the "
            "resolved source repo, and the source model's authors, licence, "
            "arXiv ids, languages, shape and its README verbatim. "
            "Call this BEFORE write_model_card and compose the card from what "
            "it returns - not from what you recall about the model. Anything "
            "absent here is unestablished, and must be left out of the card "
            "rather than filled in."),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string",
                         "description": "Repo id, or just the model name."},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "write_model_card",
        "description": (
            "Step 5. Publish the README.md for a quant repo.\n\n"
            "Pass `content` with a card you composed from get_card_facts - "
            "that is the intended path, and you own the whole document: "
            "structure, table layout, prose, extra sections, extra tags. "
            "Before publishing, the claims in it are checked against the "
            "verified listing; if any fail, nothing is published and the "
            "problems come back for you to fix and retry.\n\n"
            "Omit `content` to fall back to the fixed template, which is "
            "identical for every model. Prefer writing your own.\n\n"
            "Use dry_run to see the result without publishing."),
        "input_schema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "content": {
                    "type": "string",
                    "description": (
                        "The complete README.md, front matter included. Every "
                        "published file must appear with its real size; "
                        "`base_model` must be the resolved source repo or be "
                        "omitted; cite only arXiv ids the source provides."),
                },
                "dry_run": {"type": "boolean", "default": False},
            },
            "required": ["repo"],
        },
    },
]


# =====================================================
# DISPATCH
# =====================================================
def _need_report():
    result = research.load_latest()
    if not result:
        raise ValueError("No research on file - call research_trending first.")
    return result


def _our_gguf_repo(repo):
    """The quant repo WE publish for a model, whatever the caller named it.

    This exists because of a real and dangerous near-miss. The check used to
    be `if "/" not in repo`, which only rewrote a bare model name. Hand it the
    SOURCE repo id — `ibm-granite/granite-4.2-3b`, the obvious thing to pass
    when that is the model being worked on — and the slash made it look
    already-qualified, so it was used verbatim. The card step then verified
    IBM's repo, found no GGUFs in it, and would have tried to upload a
    README.md over the publisher's own model card.

    Write access is what stopped it, not the code. So: anything that is not
    already one of our `-GGUF` repos is reduced to its model name and mapped
    into our namespace.
    """
    namespace = config.HF_NAMESPACE
    name = repo.strip().rstrip("/").split("/")[-1]
    if not name:
        raise ValueError(f"{repo!r} is not a usable repo id.")
    return f"{namespace}/{name.removesuffix('-GGUF')}-GGUF"


def _source_candidate(verification, job=None):
    """The enriched SOURCE model behind a quant repo, or None.

    This is where a card's provenance comes from — authors, licence, arXiv
    ids, languages, and the source README verbatim. None is a valid answer:
    the source may not be resolvable, or the Hub may be unreachable. The card
    then simply says less, which is the correct outcome. Nothing here is worth
    failing a card over, hence the broad except.
    """
    repo_id = card._source_repo(verification, job)
    if not repo_id:
        return None
    try:
        return hub.enrich(hub.one(repo_id), check_ggufs=False)
    except Exception:
        return None


def _options_from(arguments):
    """RunOptions from tool arguments, falling back to the configured defaults.

    Only the two knobs worth a model's attention are exposed — sequential mode
    and upload concurrency — plus the thread cap. Quantize-worker count is left
    to the CLI, because getting it wrong costs disk and buys nothing.
    """
    options = run_mod.RunOptions.from_config()
    if arguments.get("sequential"):
        options.sequential = True
    if arguments.get("upload_workers"):
        options.upload_workers = int(arguments["upload_workers"])
    if arguments.get("quantize_threads"):
        options.quantize_threads = int(arguments["quantize_threads"])
    return options


def _assessments_for(models):
    """Assessments for named models, whether or not they were ever researched.

    This used to require every model to be in the last research run, and
    raised "not in the last research run: X" otherwise. That error sent the
    agent into a loop: the only remedy it suggests is to research again, and
    researching again cannot help, because the trending list is not where the
    model was going to come from. A user asking for `ibm-granite/granite-4.2-3b`
    — a real model, simply not trending today — produced research at limit=1,
    then 100, then 300, and never an answer.

    `research.assess_one()` has always been able to assess an arbitrary repo;
    it is what `aqx run <any-repo-id>` uses. It was just never reachable from
    the agent. Naming a model IS the judgement the trending filter stands in
    for, so an off-list model is a normal request, not an error.
    """
    result = research.load_latest()
    order = {name: index for index, name in enumerate(models)}
    found, off_list = {}, []

    for name in models:
        # find() raises when a short name is ambiguous, which is a genuine
        # question for the user and must not be swallowed into a Hub lookup.
        assessment = research.find(result, name) if result else None
        if assessment:
            found[name] = assessment
        else:
            off_list.append(name)

    if off_list:
        # One probe for the batch. assess_one() would otherwise measure the
        # machine once per model, which is the same answer at N times the cost.
        sysinfo = sysprobe.probe(measure_disk=False)
        for name in off_list:
            # Lets the Hub's own reason surface for a typo or a gated repo,
            # which is actionable, unlike "not in the last research run".
            found[name] = research.assess_one(name, sysinfo=sysinfo)

    return [found[name] for name in sorted(found, key=order.get)]


def call(name, arguments=None):
    """Run one tool. Returns JSON-serialisable data, or raises ValueError."""
    arguments = arguments or {}

    if name == "probe_system":
        return sysprobe.probe(
            measure_disk=True,
            force_disk=bool(arguments.get("remeasure_disk")))

    if name == "research_trending":
        result = research.research(
            limit=arguments.get("limit"),
            hunt_forks=arguments.get("hunt_forks", True))
        research.save(result)
        # The full result is large (100 models with per-quant size tables) and
        # most of it is never referenced. The model gets the ranked table plus
        # the machine summary, and can pull detail per candidate on demand.
        return {
            "summary": sysprobe.summary(result["system"]),
            "table": report.table(result),
            "counts": {
                "trending": result["trending_count"],
                "base_models": result["kept_count"],
                "ok": sum(1 for a in result["assessments"]
                          if a["verdict"] == "ok"),
                "warn": sum(1 for a in result["assessments"]
                            if a["verdict"] == "warn"),
                "blocked": sum(1 for a in result["assessments"]
                               if a["verdict"] == "blocked"),
            },
            "runnable": [
                {"repo_id": a["repo_id"], "url": a["url"],
                 "params_b": a["params_b"], "hours": a["hours"]["total"],
                 "peak_disk_gb": a["peak_disk_gb"],
                 "upload_gb": a["upload_gb"],
                 "verdict": a["verdict"],
                 "quants": len(a["quants"]),
                 "already_published": a.get("published_count", 0),
                 "warnings": a["warnings"]}
                for a in result["assessments"]
                if a["verdict"] not in ("blocked", "done")],
            "already_done": [
                {"repo_id": a["repo_id"], "target_repo": a["target_repo"],
                 "published": a.get("published_count", 0)}
                for a in result["assessments"] if a["verdict"] == "done"],
        }

    if name == "get_report":
        result = _need_report()
        fmt = arguments.get("format", "table")
        top = arguments.get("top")
        if fmt == "json":
            rows = result["assessments"][:top] if top else result["assessments"]
            return {"generated_at": result["generated_at"], "assessments": rows}
        if fmt == "markdown":
            return {"markdown": report.markdown(result, limit=top)}
        return {"generated_at": result["generated_at"],
                "table": report.table(result, limit=top)}

    if name == "describe_candidate":
        # Same rule as _assessments_for: a model the user names is assessed on
        # demand rather than refused for not being in a trending sweep.
        assessment = _assessments_for([arguments["model"]])[0]
        return {"text": report.detail(assessment), "assessment": assessment}

    if name == "plan_quantization":
        assessments = _assessments_for(arguments["models"])
        blocked = [a for a in assessments if a["verdict"] == "blocked"]
        done = [a for a in assessments if a["verdict"] == "done"]
        jobs = [Job.from_assessment(a) for a in assessments
                if a["verdict"] not in ("blocked", "done")]
        jobs.sort(key=lambda j: j.assessment["hours"]["total"])
        options = _options_from(arguments)
        # Repriced under the requested concurrency: the research figures assume
        # the default, and these arguments are exactly what changes them.
        peaks = {j.base_name: feasibility.peak_disk_for(
            j.assessment, quantize_workers=options.quantize_workers,
            upload_workers=options.upload_workers,
            queue_depth=options.queue_depth, sequential=options.sequential)
            for j in jobs}
        return {
            "concurrency": options.describe(),
            "quants_on_disk_at_once": options.concurrent_quants,
            "blocked": [{"repo_id": a["repo_id"], "why": a["blockers"]}
                        for a in blocked],
            "already_done": [
                {"repo_id": a["repo_id"], "target_repo": a["target_repo"],
                 "published": a["published_count"]} for a in done],
            "order": "smallest first, so a pipeline problem surfaces cheaply",
            "total_hours": round(
                sum(j.assessment["hours"]["total"] for j in jobs), 1),
            "total_upload_gb": round(
                sum(j.assessment["upload_gb"] for j in jobs), 1),
            "peak_disk_gb": max(peaks.values(), default=0),
            "peak_disk_gb_at_default_concurrency": max(
                (j.assessment["peak_disk_gb"] for j in jobs), default=0),
            "jobs": [{
                "repo_id": job.repo_id,
                "target_repo": job.target_repo,
                "quants": job.quants,
                "quant_count": len(job.quants),
                "already_published": job.assessment.get("published_count", 0),
                "source_kind": job.source_kind,
                "imatrix_source": job.imatrix_source,
                "imatrix_ngl": job.imatrix_ngl,
                "fork": job.fork,
                "hours": job.assessment["hours"]["total"],
                "peak_disk_gb": peaks[job.base_name],
                "warnings": job.assessment["warnings"],
            } for job in jobs],
        }

    if name == "start_quantization":
        if not arguments.get("user_approved"):
            raise ValueError(
                "start_quantization requires user_approved=true, and that is "
                "only honest when the user has named these models and said to "
                "go ahead. Show them plan_quantization and ask.")
        assessments = _assessments_for(arguments["models"])
        jobs = [Job.from_assessment(a) for a in assessments
                if a["verdict"] not in ("blocked", "done")]
        if not jobs:
            raise ValueError(
                "nothing to do: every named model is either blocked on this "
                "machine or already fully published.")
        jobs.sort(key=lambda j: j.assessment["hours"]["total"])

        events = []
        outcome = run_mod.run_jobs(jobs, options=_options_from(arguments),
                                   on_event=events.append)

        # Step 5 follows automatically — a run that finishes without a
        # verified listing and a card is not actually done.
        #
        # But it must NEVER be able to fail the run. The quants are already on
        # the Hub by this point; a broken card generator that propagated its
        # exception would return "error" for a six-hour job that succeeded,
        # and the only sane reading of that is "nothing ran" — which is
        # exactly the wrong thing to tell someone about to retry it.
        finished = []
        for job in jobs:
            entry = {"repo": job.target_repo, "card_published": False}
            failures = outcome["jobs"].get(job.base_name, {}).get("failures") or []
            if any(quant == "<model>" for quant, _ in failures):
                # Aborted before publishing anything; the repo does not exist,
                # and a 404 here would obscure the real failure.
                entry["published_nothing"] = True
                finished.append(entry)
                continue
            try:
                verification = card.verify(job)
                entry.update({
                    "url": verification.get("url"),
                    "files": verification.get("count"),
                    "total_gb": verification.get("total_gb"),
                    "missing": verification.get("missing"),
                    "suspect": verification.get("suspect"),
                })
                if not verification.get("error"):
                    # The TEMPLATE card, deliberately. It is a floor, not the
                    # finished article: it guarantees the repo is never left
                    # without a readable card if this session ends here, and
                    # it is the same document for every model. Saying so in
                    # the result matters — a bare card_published=true reads as
                    # "done", and the agent would skip the step that is
                    # actually its own.
                    card.publish(verification, job=job)
                    entry["card_published"] = True
                    entry["card_is_placeholder"] = True
                    entry["next"] = (
                        "A generic template card was published so the repo is "
                        "not left bare. Replace it: get_card_facts, then "
                        "write_model_card with your own content.")
                else:
                    entry["verify_error"] = verification["error"]
            except Exception as e:
                entry["card_error"] = f"{type(e).__name__}: {e}"
            finished.append(entry)
        return {"hours": outcome["hours"], "incomplete": outcome["incomplete"],
                "published": finished,
                "failures": {name: result.get("failures")
                             for name, result in outcome["jobs"].items()
                             if result.get("failures")}}

    if name == "verify_published":
        repo = arguments["repo"]
        if "/" not in repo:
            repo = f"{config.HF_NAMESPACE}/{repo.removesuffix('-GGUF')}-GGUF"
        return card.verify(repo)

    if name in ("get_card_facts", "write_model_card"):
        repo = _our_gguf_repo(arguments["repo"])
        base = repo.split("/")[-1].removesuffix("-GGUF")
        record = config.RUNS_DIR / f"{base}.json"
        job = Job.load(record) if record.exists() else None
        verification = card.verify(job or repo)
        if verification.get("error"):
            raise ValueError(verification["error"])

        # A card describes files. With none there, the run has not published
        # anything yet, and any card written now would be describing files
        # that do not exist — which is exactly what a validator cannot fix by
        # rewording. Say so plainly instead of rejecting draft after draft.
        if not verification["files"]:
            raise ValueError(
                f"{repo} contains no GGUF files, so there is nothing to write "
                "a card about. Quantization has not published anything yet - "
                "run start_quantization and let it finish first.")

        if name == "get_card_facts":
            return card.facts(verification, job=job,
                              candidate=_source_candidate(verification, job))

        content = arguments.get("content")
        try:
            text = card.publish(
                verification, job=job, content=content,
                candidate=(_source_candidate(verification, job)
                           if content else None),
                dry_run=bool(arguments.get("dry_run")))
        except card.CardRejected as rejected:
            # Not an error for the caller to give up on: the card is wrong in
            # ways it can see and fix, so hand back the list and let it retry.
            return {"repo": repo, "published": False,
                    "problems": rejected.problems,
                    "hint": ("Nothing was published. Fix these against "
                             "get_card_facts and call write_model_card again.")}
        return {"repo": repo, "url": verification["url"],
                "published": not arguments.get("dry_run"),
                "problems": [], "card": text}

    raise ValueError(f"unknown tool: {name}")


def call_json(name, arguments=None):
    """Dispatch and serialise, turning failures into a readable tool result.

    An exception must come back as content the model can read and react to.
    A transport-level error would just end the turn, and the user would be
    told nothing useful.
    """
    try:
        return json.dumps(call(name, arguments), indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}, indent=2)
