"""The agent's tools, defined once.

The MCP server, the built-in OpenRouter loop and the Claude Code skill all
expose exactly these, with exactly these schemas. That is the whole trick
behind harness portability: the harness supplies a model and a conversation
loop, and nothing else, so swapping Claude Code for Codex for a raw
OpenRouter key changes who is thinking, never what the agent can do.

The model-work tools are the same shared actions the CLI runs
(`agentquantix.actions`): inspect, plan, start, review, publish, verify. Each
routes the model by metadata to its task pipeline, so there is no separate
text, voice or media tool for a model to pick wrongly.

Two rules the tool boundary enforces, regardless of harness:

  * Nothing here downloads weights, builds anything or writes to the Hub
    except `start_release` and `publish_release`, and both require explicit
    user approval of the exact plan or candidate. There is no tool an
    over-eager model can call that silently starts a six-hour job.

  * Every tool returns JSON-serialisable data with the numbers already
    computed. The model's job is to explain and to ask, not to do arithmetic
    on gigabytes.
"""

from __future__ import annotations

import json

from .. import actions, card, config, hub, report, research, sysprobe, voice
from . import runtime_tools
from ..pipeline.job import Job
from ..runtimes import RuntimeDecisionRequired
from ..tasks import MEDIA, RETRIEVAL, TASKS, TEXT, VOICE

STRING = {"type": "string"}
STRINGS = {"type": "array", "items": STRING}
ROUTING = {
    "task": {"type": "string", "enum": sorted(TASKS),
             "description": "Only to settle an ambiguity the route reported."},
    "engine": {"type": "string",
               "description": "Only to settle an ambiguity the route reported."},
    "family": {"type": "string",
               "description": "Engine catalog family for a custom or renamed repository."},
    "hunt_forks": {"type": "boolean", "default": True,
                   "description": "Search engine forks and open PRs when a source is unresolved."},
}
TEXT_OPTIONS = {
    "sequential": {"type": "boolean", "default": False,
                   "description": "Text: quantize, upload, delete one at a time - one quant on "
                                  "disk instead of three, uploads no longer overlap. Media: price "
                                  "components sequentially."},
    "upload_workers": {"type": "integer",
                       "description": "Text: concurrent uploads; each costs one more quant of peak disk."},
}
BUNDLE_OPTIONS = {
    "target_repo": STRING,
    "quants": {**STRINGS, "description": "Voice/media/retrieval: selected native quant types."},
    "language": STRING,
    "speaker": {"type": "string", "description": "Voice: consented speaker WAV when required."},
}
MEDIA_OPTIONS = {
    "compute_backend": STRING, "contract": {"type": "object"},
    "quality_profile": {"type": "object"}, "conditioning": {"type": "object"},
    "settings": {"type": "object"}, "components": {"type": "array", "items": {"type": "object"}},
}

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
            "THIS machine. Also returns the voice backend catalog. Read-only: "
            "downloads nothing, creates nothing. Takes a minute or two."),
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
        "name": "inspect_model",
        "description": (
            "Read-only. Route ONE model to its task, pipeline and native engine "
            "from Hub metadata and the installed engine catalogs, then return "
            "everything known about it: the text quant list, imatrix strategy, "
            "disk/time breakdown and fork leads; or the voice backend, companions "
            "and native quant set; or the media metadata and loader catalog. "
            "Works for any repo id, trending or not - this is the right tool when "
            "the user names a model; do NOT run research_trending to look for it. "
            "A route with `choices` is a question for the user; a route with a "
            "`blocker` names what is missing."),
        "input_schema": {
            "type": "object",
            "properties": {"model": {"type": "string",
                                     "description": "Any Hub repo id, an audio.cpp package id, "
                                                    "or an unambiguous name from the last report."},
                           **ROUTING},
            "required": ["model"],
        },
    },
    {
        "name": "plan_release",
        "description": (
            "Read-only. The exact plan for the named models, routed by task: "
            "text sweeps (targets, quants, imatrix sources, fork builds, totals, "
            "repriced for the requested concurrency), voice bundles (backend, "
            "companions, quants, quality gate), media plans (revisions, components, "
            "policies, licenses, estimates, evaluator, blockers), and retrieval "
            "bundles (source contract, native quants, parity and quality gates). "
            "This is what you show the user at the approval gate. Always call it "
            "before start_release. Per-model options apply to one model at a time."),
        "input_schema": {
            "type": "object",
            "properties": {"models": STRINGS, **ROUTING, **TEXT_OPTIONS,
                           **BUNDLE_OPTIONS, **MEDIA_OPTIONS},
            "required": ["models"],
        },
    },
    {
        "name": "start_release",
        "description": (
            "Step 4. Build, validate and publish what the user explicitly "
            "approved. Text: quantize and upload the named models smallest-first, "
            "deleting each quant once uploaded. Voice: convert, quantize, run the "
            "real runtime and quality gates; TTS publication also needs a recorded "
            "listening review. Media and retrieval: pass the exact `plan` from "
            "plan_release, unchanged; retrieval runs one model per call and only "
            "publishes passing automated parity/quality candidates. This spends "
            "hours, uses the disk and writes to the Hub; it "
            "requires that the user named these models (or saw this exact plan) and "
            "said to proceed. One pipeline per call. Results carry `run_id`."),
        "input_schema": {
            "type": "object",
            "properties": {
                "models": STRINGS,
                "plan": {"type": "object",
                         "description": "Media/retrieval: the unchanged plan the user approved."},
                "user_approved": {
                    "type": "boolean",
                    "description": "True only when the user has approved these "
                                   "models or this exact plan and said to proceed.",
                },
                **ROUTING, **TEXT_OPTIONS,
                "quantize_threads": {
                    "type": "integer",
                    "description": "Text: threads per llama-quantize. Omit to use every core.",
                },
                **BUNDLE_OPTIONS,
                "publish": {"type": "boolean",
                            "description": "Voice defaults to true (reviewed TTS quants only); "
                                           "media/retrieval default to false (passing candidates only)."},
            },
            "required": ["user_approved"],
        },
    },
    {
        "name": "record_review",
        "description": (
            "Record the USER's explicit human review of exact candidate artifacts "
            "in a run (voice TTS listening, image inspection, video playback and "
            "frames, audio-video watching and listening). Never invent a review: "
            "call this only with a decision the user stated after experiencing the "
            "artifacts. TTS requires listened_audio; video requires watched_video."),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string",
                           "description": "From start_release or list_runs; a source "
                                          "repo id also resolves when unambiguous."},
                "quant": STRING, "accepted": {"type": "boolean"}, "reviewer": STRING,
                "notes": STRING, "watched_video": {"type": "boolean"},
                "listened_audio": {"type": "boolean"},
            },
            "required": ["run_id", "quant", "accepted", "reviewer"],
        },
    },
    {
        "name": "publish_release",
        "description": (
            "Publish a natively validated, human-reviewed voice or media candidate, "
            "or a retrieval quant that passed automated parity and quality gates, "
            "under its saved release policy, then verify remote sizes and hashes. "
            "Text quants publish during start_release and need no separate step. "
            "Requires user_approved=true after the review."),
        "input_schema": {
            "type": "object",
            "properties": {"run_id": STRING, "quant": STRING,
                           "user_approved": {"type": "boolean"}},
            "required": ["run_id", "quant", "user_approved"],
        },
    },
    {
        "name": "list_runs",
        "description": "Recorded text, voice and media runs with their ids, tasks and reviews.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "verify_release",
        "description": (
            "Step 5. Check what actually exists: a published repo (GGUF listing "
            "with real sizes, or a voice/media bundle against its manifest), a "
            "recorded run's built bundles, or a local manifest path. Flags missing "
            "or suspiciously small files and checksum mismatches."),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string",
                           "description": "Repo id, model name, run id or manifest path."},
            },
            "required": ["target"],
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
            "rather than filled in. The `usage` key carries working commands "
            "for this exact repo; copy them instead of writing your own.\n\n"
            "This tool only gathers material. The card is not published until "
            "write_model_card returns published: true - showing it in the "
            "conversation instead leaves the generic placeholder in place."),
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
            "Step 5. Publish the README.md for a text quant repo. (Voice and "
            "media bundle cards are generated from the verified manifest and "
            "published with the bundle.)\n\n"
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
TOOLS.extend(runtime_tools.TOOLS)
EXECUTION_TOOLS = {"start_release", "publish_release"}


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


def _routing(arguments):
    return {key: arguments.get(key) for key in ("task", "engine", "family")}


def _media_args(arguments):
    keys = ("target_repo", "quants", "compute_backend", "contract", "quality_profile",
            "conditioning", "settings", "components", "sequential")
    return {key: arguments[key] for key in keys if arguments.get(key) is not None}


def _voice_args(arguments):
    keys = ("target_repo", "quants", "language", "speaker")
    return {key: arguments[key] for key in keys if arguments.get(key) is not None}


def _start(arguments):
    if arguments.get("user_approved") is not True:
        raise ValueError(
            "start_release requires user_approved=true, and that is only honest "
            "when the user has named these models (or seen this exact plan) and "
            "said to go ahead. Show them plan_release and ask.")
    if arguments.get("plan") is not None:
        approved_plan = arguments["plan"]
        if approved_plan.get("engine") in ("llama.cpp", "clip.cpp") and \
                approved_plan.get("task") in ("embeddings", "reranking",
                                               "zero-shot-image-classification",
                                               "image-feature-extraction"):
            return actions.start_retrieval(
                approved_plan, user_approved=True,
                publish=bool(arguments.get("publish", False)))
        return actions.start_media(arguments["plan"], user_approved=True,
                                   publish=bool(arguments.get("publish", False)))
    models = arguments.get("models") or []
    if not models:
        raise ValueError("name the approved models, or pass the approved media plan")
    routes = [actions.route_named(name, **_routing(arguments)) for name in models]
    for item in routes:
        actions._require_ready(item)
    pipelines = {item.pipeline for item in routes}
    if len(pipelines) > 1:
        raise ValueError("these models resolve to different pipelines; start each separately")
    pipeline = pipelines.pop()
    if pipeline == MEDIA:
        raise ValueError("media releases execute the exact plan the user approved: pass the "
                         "unchanged `plan` from plan_release")
    if pipeline == TEXT:
        options = actions.text_options(arguments.get("sequential"), arguments.get("upload_workers"),
                                       arguments.get("quantize_threads"))
        events = []
        return actions.start_text(models, options, user_approved=True, on_event=events.append)
    if pipeline == RETRIEVAL:
        if len(models) > 1:
            raise ValueError("retrieval releases run one model per call")
        plan = actions.retrieval_plan(
            routes[0].repo_id, task=routes[0].task, engine=routes[0].engine,
            target_repo=arguments.get("target_repo"), quants=arguments.get("quants"))
        return actions.start_retrieval(plan, user_approved=True,
                                       publish=bool(arguments.get("publish", False)))
    if len(models) > 1:
        raise ValueError("voice releases run one model per call")
    plan = actions.voice_plan(routes[0].repo_id, family=actions.voice_family(routes[0], arguments.get("family")),
                              hunt_forks=arguments.get("hunt_forks", True), **_voice_args(arguments))
    if plan.get("status") != "ready":
        raise ValueError(f"voice release is blocked: {plan.get('blocker')}; "
                         f"fork leads: {plan.get('fork_leads', [])}")
    actions.prepare_voice_runtimes(plan, persist=True)
    plan = actions.voice_plan(routes[0].repo_id, family=actions.voice_family(routes[0], arguments.get("family")),
                              hunt_forks=arguments.get("hunt_forks", True), **_voice_args(arguments))
    return actions.run_voice(plan, publish=arguments.get("publish", True),
                             family=arguments.get("family"))


def _call(name, arguments=None):
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
                "ok": sum(1 for a in result["assessments"] if a["verdict"] == "ok"),
                "warn": sum(1 for a in result["assessments"] if a["verdict"] == "warn"),
                "blocked": sum(1 for a in result["assessments"] if a["verdict"] == "blocked"),
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
            "voice": voice.advisory_catalog(),
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

    if name == "inspect_model":
        return actions.inspect(arguments["model"], hunt_forks=arguments.get("hunt_forks", True),
                               **_routing(arguments))

    if name == "plan_release":
        models = arguments["models"]
        text = actions.text_options(arguments.get("sequential"), arguments.get("upload_workers"))
        return actions.plan(models, hunt_forks=arguments.get("hunt_forks", True), text=text,
                            voice_args=_voice_args(arguments), media_args=_media_args(arguments),
                            retrieval_args={key: arguments[key] for key in ("target_repo", "quants")
                                            if arguments.get(key) is not None},
                            **_routing(arguments))

    if name == "start_release":
        return _start(arguments)

    if name == "record_review":
        return actions.review(arguments["run_id"], arguments["quant"],
                              accepted=arguments["accepted"], reviewer=arguments["reviewer"],
                              notes=arguments.get("notes", ""),
                              watched_video=bool(arguments.get("watched_video")),
                              listened_audio=bool(arguments.get("listened_audio")))

    if name == "publish_release":
        return actions.publish(arguments["run_id"], arguments["quant"],
                               user_approved=arguments.get("user_approved"))

    if name == "list_runs":
        return actions.list_runs()

    if name == "verify_release":
        return actions.verify(arguments["target"])

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
                "run start_release and let it finish first.")

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


def _failed(result):
    return bool(result.get("failures") or result.get("incomplete")
                or (result.get("pipeline") == VOICE and actions.voice_failed(result))
                or (result.get("pipeline") == MEDIA and actions.media_failed(result)))


def call(name, arguments=None):
    if name in runtime_tools.NAMES:
        return runtime_tools.call(name, arguments or {})
    if name in EXECUTION_TOOLS:
        from ..runtime_control import execution_session
        with execution_session() as session:
            result = _call(name, arguments)
            if _failed(result):
                session.mark_failed()
            return result
    return _call(name, arguments)


def call_json(name, arguments=None):
    """Dispatch and serialise, turning failures into a readable tool result.

    An exception must come back as content the model can read and react to.
    A transport-level error would just end the turn, and the user would be
    told nothing useful.
    """
    try:
        return json.dumps(call(name, arguments), indent=2, default=str)
    except RuntimeDecisionRequired as e:
        return json.dumps({"error": str(e), "runtime_decision": e.details}, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}, indent=2)
