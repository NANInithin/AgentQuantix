"""Shared inspect/plan/start/review/publish/verify actions.

The CLI and every agent harness call these functions, so routing and the
task-specific gates behave identically wherever AgentQuantix runs. Each action
routes by metadata first and then delegates to the existing pipeline adapter;
none of them relaxes an adapter's own approval, quality or review contract.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import card, config, feasibility, media, report, research, runs, sysprobe, voice
from .pipeline import voice_release
from .pipeline.job import Job
from .tasks import MEDIA, MULTIMODAL, RETRIEVAL, TEXT, VISION, VOICE
from .tasks.router import Route, is_short_name, route as route_model

VOICE_RUN_DEFAULTS = {"language": None, "speaker": None, "pocket_language": "english",
                      "max_roundtrip_wer": 0.35, "max_wer_regression": 0.05}


# =====================================================
# routing
# =====================================================
def route_named(name, *, task=None, engine=None, family=None, fetch=None):
    """Route a repo id, a package id, or a research-report shorthand."""
    if is_short_name(name) and not task and not family:
        result = research.load_latest()
        if result and (assessment := research.find(result, name)):
            return Route(assessment["repo_id"], task="text-generation", pipeline=TEXT,
                         engine="llama.cpp", reasons=["listed in the last research run"])
    return route_model(name, task=task, engine=engine, family=family, fetch=fetch)


def voice_family(item, family=None):
    """The audio.cpp catalog family to pin, if any. Core llama.cpp/whisper.cpp
    backends resolve by repository and must not be narrowed to audio.cpp."""
    return family or (item.family if item.engine == "audio.cpp" else None)


def _require_ready(item):
    if item.choices:
        raise ValueError(f"{item.repo_id}: {item.blocker}: "
                         + json.dumps(item.choices, default=str))
    if not item.ready:
        raise ValueError(f"{item.repo_id}: {item.blocker or 'no pipeline resolved'}")
    return item


# =====================================================
# text
# =====================================================
def assessments_for(models):
    """Assessments for named models, whether or not they were ever researched.

    Naming a model is the judgement the trending filter stands in for, so an
    off-list model is assessed on demand rather than refused. A user asking for
    a real model that is simply not trending once produced research at limit=1,
    100 and 300 and never an answer; this is the fix.
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
        # One probe for the batch rather than one per model.
        sysinfo = sysprobe.probe(measure_disk=False)
        for name in off_list:
            found[name] = research.assess_one(name, sysinfo=sysinfo)
    return [found[name] for name in sorted(found, key=order.get)]


def text_options(sequential=False, upload_workers=None, quantize_threads=None):
    from .pipeline import run as run_mod
    options = run_mod.RunOptions.from_config()
    if sequential:
        options.sequential = True
    if upload_workers:
        options.upload_workers = int(upload_workers)
    if quantize_threads:
        options.quantize_threads = int(quantize_threads)
    return options


def text_plan(models, options):
    assessments = assessments_for(models)
    blocked = [a for a in assessments if a["verdict"] == "blocked"]
    done = [a for a in assessments if a["verdict"] == "done"]
    jobs = sorted((Job.from_assessment(a) for a in assessments
                   if a["verdict"] not in ("blocked", "done")),
                  key=lambda j: j.assessment["hours"]["total"])
    # Repriced under the requested concurrency: research assumes the default.
    peaks = {j.base_name: feasibility.peak_disk_for(
        j.assessment, quantize_workers=options.quantize_workers,
        upload_workers=options.upload_workers, queue_depth=options.queue_depth,
        sequential=options.sequential) for j in jobs}
    return {
        "task": "text-generation", "engine": "llama.cpp",
        "publication": "each GGUF is uploaded as soon as it is built, then verified and carded",
        "concurrency": options.describe(),
        "quants_on_disk_at_once": options.concurrent_quants,
        "blocked": [{"repo_id": a["repo_id"], "why": a["blockers"]} for a in blocked],
        "already_done": [{"repo_id": a["repo_id"], "target_repo": a["target_repo"],
                          "published": a["published_count"]} for a in done],
        "order": "smallest first, so a pipeline problem surfaces cheaply",
        "total_hours": round(sum(j.assessment["hours"]["total"] for j in jobs), 1),
        "total_upload_gb": round(sum(j.assessment["upload_gb"] for j in jobs), 1),
        "peak_disk_gb": max(peaks.values(), default=0),
        "peak_disk_gb_at_default_concurrency": max(
            (j.assessment["peak_disk_gb"] for j in jobs), default=0),
        "jobs": [{
            "run_id": f"text:{job.base_name}",
            "repo_id": job.repo_id, "target_repo": job.target_repo,
            "quants": job.quants, "quant_count": len(job.quants),
            "already_published": job.assessment.get("published_count", 0),
            "source_kind": job.source_kind, "imatrix_source": job.imatrix_source,
            "imatrix_ngl": job.imatrix_ngl, "fork": job.fork,
            "hours": job.assessment["hours"]["total"],
            "peak_disk_gb": peaks[job.base_name],
            "warnings": job.assessment["warnings"],
        } for job in jobs],
    }


def start_text(models, options, *, user_approved, on_event=None):
    """Run approved text jobs, then verify and publish a floor card."""
    from . import hub, runtime_control
    from .pipeline import run as run_mod
    if user_approved is not True:
        raise ValueError(
            "start_release requires user_approved=true, and that is only honest "
            "when the user has named these models and said to go ahead. Show "
            "them plan_release and ask.")
    assessments = assessments_for(models)
    jobs = sorted((Job.from_assessment(a) for a in assessments
                   if a["verdict"] not in ("blocked", "done")),
                  key=lambda j: j.assessment["hours"]["total"])
    if not jobs:
        raise ValueError("nothing to do: every named model is either blocked on "
                         "this machine or already fully published.")
    originals = [job for job in jobs if not job.fork]
    if originals:
        handle = runtime_control.ensure_runtime(
            "llama.cpp", frozen=originals[0].runtime_snapshot or None)
        if handle:
            for job in originals:
                if job.runtime_snapshot and job.runtime_snapshot["identity"] != handle.identity:
                    raise ValueError("these jobs require different frozen runtimes; resume them separately")
                job.runtime_snapshot = handle.snapshot()
    outcome = run_mod.run_jobs(jobs, options=options, on_event=on_event)
    # Our cached repo listing is stale now that the run published.
    hub.forget_our_repos()

    # Verification and the floor card follow automatically, but must never
    # turn a successful upload into a reported failure.
    finished = []
    for job in jobs:
        entry = {"run_id": f"text:{job.base_name}", "repo": job.target_repo,
                 "card_published": False}
        failures = outcome["jobs"].get(job.base_name, {}).get("failures") or []
        if any(quant == "<model>" for quant, _ in failures):
            entry["published_nothing"] = True
            finished.append(entry)
            continue
        try:
            verification = card.verify(job)
            entry.update({key: verification.get(name) for key, name in (
                ("url", "url"), ("files", "count"), ("total_gb", "total_gb"),
                ("missing", "missing"), ("suspect", "suspect"))})
            if not verification.get("error"):
                card.publish(verification, job=job)
                entry.update(card_published=True, card_is_placeholder=True, next=(
                    "A generic template card was published so the repo is not "
                    "left bare. Replace it: get_card_facts, then write_model_card "
                    "with your own content."))
            else:
                entry["verify_error"] = verification["error"]
        except Exception as e:
            entry["card_error"] = f"{type(e).__name__}: {e}"
        finished.append(entry)
    return {"pipeline": TEXT, "hours": outcome["hours"], "incomplete": outcome["incomplete"],
            "published": finished,
            "failures": {name: result.get("failures")
                         for name, result in outcome["jobs"].items() if result.get("failures")}}


# =====================================================
# voice
# =====================================================
def _audio_cpp_catalog_blocker(track):
    if track not in (voice.SEP, voice.VC, voice.ALIGN):
        return None
    from . import runtime_control
    handle = runtime_control.selected_runtime("audio.cpp")
    if handle is None:
        return None
    from .pipeline import voice_release
    if handle.revision in (
            voice_release.AUDIOCPP_REVISION,
            voice_release.AUDIOCPP_REVISION_SHA):
        return None
    return (f"audio.cpp runtime {handle.revision} does not match the vendored "
            f"model-spec catalog {voice_release.AUDIOCPP_REVISION}; install with "
            "aqx runtime install audio.cpp --persistent --revision "
            f"{voice_release.AUDIOCPP_REVISION_SHA}")


def voice_plan(repo_id, *, target_repo=None, quants=None, task=None, family=None,
               hunt_forks=True, language=None, speaker=None):
    task_tracks = {
        "text-to-speech": voice.TTS,
        "automatic-speech-recognition": voice.ASR,
        "audio-source-separation": voice.SEP,
        "voice-conversion": voice.VC,
        "forced-alignment": voice.ALIGN,
    }
    allowed, reason, backend = voice.execution_gate(
        repo_id, track=task_tracks.get(task), family=family)
    if not allowed:
        leads = voice_release.find_backend_forks(repo_id) if hunt_forks else []
        return {"model": repo_id, "status": "blocked", "blocker": reason,
                "fork_hunt": "searched" if hunt_forks else "disabled", "fork_leads": leads,
                "next": ("A lead must still prove its converter, model spec, bundle "
                         "contract, and smoke inference before execution."
                         if leads else "No installed backend or current fork/PR lead was found.")}
    if blocker := _audio_cpp_catalog_blocker(backend.track):
        return {"model": repo_id, "status": "blocked",
                "task": task, "track": backend.track, "backend": backend.id,
                "engine": backend.backend, "family": backend.family,
                "blocker": blocker}
    available = voice.available_quants(repo_id, backend)
    requested = [voice.normalize_quant(backend, quant) for quant in (quants or available)]
    if unsupported := [quant for quant in requested if quant not in available]:
        raise ValueError(f"{backend.id} does not support {unsupported}; choose from {list(available)}")
    source = voice_release.source_metadata(repo_id, family=family)
    source_inputs = voice_release.audiocpp_source_plan(repo_id, backend, source.get("source_files"))
    missing_inputs = [item for item in source_inputs if item["status"] == "missing"]
    suffix = "GGUF" if backend.model_format.endswith("gguf") else "GGML"
    source_name = repo_id.split("/")[-1].replace(":", "-")
    tts = backend.track == voice.TTS
    task = {
        voice.TTS: "text-to-speech",
        voice.ASR: "automatic-speech-recognition",
        voice.SEP: "audio-source-separation",
        voice.VC: "voice-conversion",
        voice.ALIGN: "forced-alignment",
    }[backend.track]
    quality_gates = {
        voice.TTS: ("audio integrity, silence/clipping, ASR round-trip WER/CER, "
                    "then human listening review"),
        voice.ASR: "fixed-corpus WER/CER regression versus base precision",
        voice.SEP: ("both vocal and accompaniment SI-SDR improvements >= 6 dB "
                    "versus mixture; other precisions within 1.0 dB of baseline"),
        voice.VC: ("finite non-silent waveform within 20% of input duration; "
                   "ASR round-trip WER <= 0.35 and within 0.10 of baseline"),
        voice.ALIGN: ("word count and monotonic in-range boundaries; other precisions "
                      "within mean 0.05 s and max 0.20 s of baseline"),
    }
    result = {
        "model": repo_id,
        "status": "blocked" if missing_inputs else "ready",
        "task": task,
        "target_repo": target_repo or f"{config.namespace()}/{source_name}-{suffix}",
        "track": backend.track, "backend": backend.id, "engine": backend.backend,
        "family": backend.family, "backend_status": backend.status,
        "runtime": backend.runtime, "converter": backend.converter,
        "model_format": backend.model_format,
        "quants": requested, "available_quants": list(available),
        "quant_source": ("native converter/quantizer capability; package "
                         "references are narrowed to published precisions"),
        "fork_leads": [],
        "required_companions": list(backend.required_companions),
        "required_source_inputs": source_inputs,
        "speaker_reference": backend.speaker_reference,
        "sample_rate": backend.sample_rate,
        "language": language, "speaker": speaker,
        "source_revision": source["revision"], "source_bytes": source["source_bytes"],
        "source_gated": source["gated"],
        "quality_gate": quality_gates[backend.track],
        "review": "listening" if tts else "none",
        "publication": ("one atomic bundle commit of accepted quants after automated "
                        "gates and a recorded listening review" if tts else
                        "one atomic bundle commit after the WER/CER regression gate"
                        if backend.track == voice.ASR else
                        "atomic bundle of passing precisions after native task gates"),
    }
    if backend.track in (voice.SEP, voice.VC, voice.ALIGN):
        result["source_license"] = source.get("license")
    if missing_inputs:
        result["blocker"] = ("source repository is missing backend-declared tensor inputs: "
                             + ", ".join(item["required_path"] for item in missing_inputs))
    return result


def prepare_voice_runtimes(plan, *, persist=False):
    from . import runtime_control
    return runtime_control.prepare_voice_runtimes(plan["model"], plan["engine"], plan["track"],
                                                  persist=persist)


def run_voice(plan, *, publish=True, human_review=None, family=None, **settings):
    """Execute an approved, ready voice plan and record its run id."""
    if plan.get("status") != "ready":
        raise ValueError(f"voice release is blocked: {plan.get('blocker')}; "
                         f"fork leads: {plan.get('fork_leads', [])}")
    settings = {**VOICE_RUN_DEFAULTS, **{k: v for k, v in settings.items() if v is not None}}
    run_id = runs.write_voice_record(plan["model"], plan)
    speaker = settings["speaker"] or plan.get("speaker")
    options = voice_release.VoiceReleaseOptions(
        quants=plan["quants"], language=settings["language"] or plan.get("language"),
        speaker=Path(speaker) if speaker else None, publish=publish, max_roundtrip_wer=settings["max_roundtrip_wer"],
        max_wer_regression=settings["max_wer_regression"],
        pocket_language=settings["pocket_language"],
        human_review=Path(human_review) if human_review else None,
        family=family or plan.get("family"))
    release = {
        voice.TTS: voice_release.run_tts_release,
        voice.ASR: voice_release.run_asr_release,
        voice.SEP: voice_release.run_audiocpp_sep_release,
        voice.VC: voice_release.run_audiocpp_vc_release,
        voice.ALIGN: voice_release.run_audiocpp_align_release,
    }[plan["track"]]
    result = release(plan["model"], plan["target_repo"], options)
    return {"pipeline": VOICE, "run_id": run_id, **result}


def voice_failed(result):
    return any(entry.get("error") or entry.get("reason", "").endswith("failed")
               or (entry.get("quality") or {}).get("regression", {}).get("passed") is False
               for entry in result.get("results", {}).values())


# =====================================================
# media
# =====================================================
def media_failed(result):
    return bool(result.get("problems")) or any(
        c.get("rejected") or c.get("status") == "quality-rejected"
        for c in result.get("candidates", []))


def start_media(plan, *, user_approved, publish=False, preserve_failed=False):
    from .pipeline import media_release
    if user_approved is not True:
        raise media.MediaValidationError("start_release requires approval of the exact media plan")
    if plan.get("blockers"):
        raise media.MediaValidationError("media plan is blocked: " + "; ".join(plan["blockers"]))
    result = media_release.run_release(plan, user_approved=True, publish=publish,
                                       preserve_failed=preserve_failed)
    return {"pipeline": MEDIA, **result, "run_id": f"media:{result['run_id']}"}


def retrieval_plan(model, *, task, engine, target_repo=None, quants=None):
    from . import retrieval
    return retrieval.plan_release(model, task=task, engine=engine,
                                  target_repo=target_repo, quants=quants)


def start_retrieval(plan, *, user_approved, publish=False):
    if user_approved is not True:
        raise ValueError("start_release requires approval of the exact retrieval plan")
    if plan.get("blockers") or plan.get("status") != "ready":
        raise ValueError("retrieval plan is blocked: " + "; ".join(plan.get("blockers", [])))
    from .pipeline import retrieval_release
    result = retrieval_release.run_release(plan, user_approved=True, publish=publish)
    return {"pipeline": RETRIEVAL, **result, "run_id": f"retrieval:{result['run_id']}"}


def multimodal_plan(model, *, task="multimodal-understanding", engine="llama.cpp",
                    target_repo=None, quants=None):
    from . import multimodal
    return multimodal.plan_release(model, task=task, engine=engine,
                                   target_repo=target_repo, quants=quants)


def vision_plan(model, *, task, family=None, target_repo=None, quants=None):
    from . import runtime_control
    from .pipeline import vision_release
    handle = runtime_control.selected_runtime("vision.cpp")
    try:
        return vision_release.plan_release(
            model, task=task, family=family, target_repo=target_repo, quants=quants,
            runtime_revision=handle.revision if handle else None)
    except vision_release.VisionValidationError as error:
        return {"pipeline": "vision", "engine": "vision.cpp", "model": model,
                "task": task, "family": family, "status": "blocked",
                "blockers": [str(error)]}


def start_vision(plan, *, user_approved, publish=False):
    if user_approved is not True:
        raise ValueError("start_release requires approval of the exact vision plan")
    from .pipeline import vision_release
    if plan.get("blockers") or plan.get("status") != "ready":
        raise ValueError("vision plan is blocked: " + "; ".join(plan.get("blockers", [])))
    from . import runtime_control
    handle = runtime_control.ensure_runtime(
        "vision.cpp", revision={"revision": vision_release.REVISION,
                                "ref": vision_release.REVISION, "channel": "pinned",
                                "repository": "Acly/vision.cpp"},
        required_targets=("vision-cli",))
    result = vision_release.run_release(plan, user_approved=True, publish=publish,
                                        runtime=handle)
    return {"pipeline": VISION, **result, "run_id": f"vision:{result['run_id']}"}


def vision_failed(result):
    return not any(item.get("status") == "passed"
                   for item in result.get("candidates", []))


def start_multimodal(plan, *, user_approved, publish=False):
    if user_approved is not True:
        raise ValueError("start_release requires approval of the exact multimodal plan")
    if plan.get("blockers") or plan.get("status") != "ready":
        raise ValueError("multimodal plan is blocked: " + "; ".join(plan.get("blockers", [])))
    from .pipeline import multimodal_release
    result = multimodal_release.run_release(plan, user_approved=True, publish=publish)
    return {"pipeline": MULTIMODAL, **result, "run_id": f"multimodal:{result['run_id']}"}


# =====================================================
# shared actions
# =====================================================
def inspect(model, *, task=None, engine=None, family=None, hunt_forks=True):
    """Read-only. Routes the model and returns the matching assessment."""
    item = route_named(model, task=task, engine=engine, family=family)
    result = {"route": item.to_dict()}
    if not item.ready:
        return result
    if item.pipeline == TEXT:
        assessment = assessments_for([item.repo_id])[0]
        result.update(text=report.detail(assessment), assessment=assessment)
    elif item.pipeline == VOICE:
        result["voice"] = voice_plan(item.repo_id, task=item.task,
                                     family=voice_family(item, family),
                                     hunt_forks=hunt_forks)
    elif item.pipeline == RETRIEVAL:
        result["retrieval"] = retrieval_plan(item.repo_id, task=item.task, engine=item.engine)
    elif item.pipeline == VISION:
        result["vision"] = vision_plan(item.repo_id, task=item.task, family=item.family)
    elif item.pipeline == MULTIMODAL:
        result["multimodal"] = multimodal_plan(item.repo_id, task=item.task, engine=item.engine)
    else:
        metadata = item.metadata if item.metadata.get("revision") else media.inspect_repository(item.repo_id)
        result.update(metadata=metadata, likely_media=media.looks_like_media(item.repo_id, metadata),
                      capabilities=[c.__dict__ for c in media.capabilities()])
    return result


def plan(models, *, task=None, engine=None, family=None, hunt_forks=True,
         text=None, voice_args=None, media_args=None, retrieval_args=None,
         multimodal_args=None, vision_args=None):
    """Read-only plans for named models, grouped by their resolved pipeline."""
    if isinstance(models, str):
        models = [models]
    if not models:
        raise ValueError("name at least one model")
    routes = [route_named(name, task=task, engine=engine, family=family) for name in models]
    if sum(route.pipeline == RETRIEVAL for route in routes) > 1:
        raise ValueError("retrieval releases run one model per plan")
    if sum(route.pipeline == MULTIMODAL for route in routes) > 1:
        raise ValueError("multimodal releases run one model per plan")
    if sum(route.pipeline == VISION for route in routes) > 1:
        raise ValueError("vision releases run one model per plan")
    per_model = ("target_repo", "quants", "speaker", "components", "contract",
                 "quality_profile", "conditioning", "settings")
    single = {k: v for k, v in {**(voice_args or {}), **(media_args or {}),
                                **(retrieval_args or {}), **(multimodal_args or {}),
                                **(vision_args or {})}.items()
              if k in per_model and v not in (None, [], {})}
    if single and len(models) > 1:
        raise ValueError("target, quant, speaker and media options apply to one model at a time")
    output = {"routes": [r.to_dict() for r in routes], "text": None, "releases": [],
              "blocked": [{"model": r.repo_id, "why": r.blocker, "choices": r.choices}
                          for r in routes if not r.ready]}
    ready = [(name, r) for name, r in zip(models, routes) if r.ready]
    text_models = [name for name, r in ready if r.pipeline == TEXT]
    if text_models:
        if voice_args and any(voice_args.get(k) for k in ("target_repo", "quants")):
            raise ValueError("text sweeps are resolved by the assessment; --target/--quant apply to voice and media")
        output["text"] = text_plan(text_models, text or text_options())
    for name, r in ready:
        if r.pipeline == VOICE:
            entry = voice_plan(r.repo_id, task=r.task,
                               family=voice_family(r, family), hunt_forks=hunt_forks,
                               **{k: v for k, v in (voice_args or {}).items()
                                  if k in ("target_repo", "quants", "language", "speaker")})
        elif r.pipeline == MEDIA:
            args = {k: v for k, v in (media_args or {}).items() if v is not None}
            args.setdefault("task", r.task if r.task in media.MEDIA_TASKS else task)
            args.setdefault("family", family or r.family)
            entry = media.plan_release(r.repo_id, hunt_forks=hunt_forks, **args)
        elif r.pipeline == RETRIEVAL:
            args = {k: v for k, v in (retrieval_args or {}).items() if v is not None}
            entry = retrieval_plan(r.repo_id, task=r.task, engine=r.engine, **args)
        elif r.pipeline == MULTIMODAL:
            args = {k: v for k, v in (multimodal_args or {}).items() if v is not None}
            entry = multimodal_plan(r.repo_id, task=r.task, engine=r.engine, **args)
        elif r.pipeline == VISION:
            args = {k: v for k, v in (vision_args or {}).items() if v is not None}
            entry = vision_plan(r.repo_id, task=r.task, family=r.family, **args)
        else:
            continue
        output["releases"].append({"pipeline": r.pipeline, "model": r.repo_id, "plan": entry})
    return output


def review(run_id, quant, *, accepted, reviewer, notes="", watched_video=False, listened_audio=False):
    """Record a human decision about exact candidate artifacts."""
    from .pipeline import media_release
    if not str(reviewer or "").strip():
        try:
            record = runs.resolve(run_id)
        except ValueError:
            record = None
        if record and record.pipeline == RETRIEVAL:
            raise ValueError("retrieval releases are gated by automated parity and retrieval metrics; there is no human review")
        if record and record.pipeline == MULTIMODAL:
            raise ValueError("multimodal releases are gated by automated task metrics; there is no human review")
        raise ValueError("a human reviewer name is required")
    record = runs.resolve(run_id)
    if record.pipeline == TEXT:
        raise ValueError("text-generation runs have no human review gate")
    if record.pipeline == RETRIEVAL:
        raise ValueError("retrieval releases are gated by automated parity and retrieval metrics; there is no human review")
    if record.pipeline == MULTIMODAL:
        raise ValueError("multimodal releases are gated by automated task metrics; there is no human review")
    if record.pipeline == VISION:
        raise ValueError("vision releases are gated by automated task metrics; there is no human review")
    if record.pipeline == MEDIA:
        review_data = media_release.record_review(
            record.id.partition(":")[2], quant, reviewer, accepted, notes=notes,
            watched_video=watched_video, listened_audio=listened_audio)
        return {"run_id": record.id, "recorded": True, **review_data}
    if record.task == "automatic-speech-recognition":
        raise ValueError("ASR releases are gated by the WER/CER regression, not a listening review")
    matched = next((q for q in voice.TTS_REVIEW_QUANTS if q.casefold() == quant.casefold()), None)
    if matched is None:
        raise ValueError(f"{quant} is not a reviewable TTS quant")
    if not listened_audio:
        raise ValueError("a TTS review must attest that the reviewer listened to the "
                         "generated fixture audio (listened_audio / --listened-audio)")
    path = Path(record.location) / "reviews" / f"{matched}.json"
    voice.write_human_review(path, quant=matched, accepted=bool(accepted),
                             reviewer=reviewer, notes=notes or "")
    return {"run_id": record.id, "path": str(path), "recorded": True,
            "quant": matched, "accepted": bool(accepted)}


def publish(run_id, quant, *, user_approved):
    """Publish an already validated, reviewed candidate under its saved policy."""
    from .pipeline import media_release
    if user_approved is not True:
        raise ValueError("publish_release requires user_approved=true after the user "
                         "has seen the validated, reviewed candidate")
    record = runs.resolve(run_id)
    if record.pipeline == TEXT:
        raise ValueError("text quants are uploaded while the run executes; resume it with "
                         f"`aqx run --resume {record.id}`")
    if record.pipeline == RETRIEVAL:
        from .pipeline import retrieval_release
        return {"run_id": record.id,
                **retrieval_release.publish_run(record.id.partition(":")[2], quant)}
    if record.pipeline == MULTIMODAL:
        from .pipeline import multimodal_release
        return {"run_id": record.id,
                **multimodal_release.publish_run(record.id.partition(":")[2], quant)}
    if record.pipeline == VISION:
        from .pipeline import vision_release
        return {"run_id": record.id,
                **vision_release.publish_run(record.id.partition(":")[2], quant,
                                             root=record.location)}
    if record.pipeline == MEDIA:
        return {"run_id": record.id, **media_release.publish_run(record.id.partition(":")[2], quant)}
    saved = record.details.get("plan") or {}
    if not saved:
        raise ValueError(f"{record.id} has no saved voice plan; run `aqx run {record.repo_id}` "
                         "to rebuild it from its cached artifacts")
    reviews = Path(record.location) / "reviews"
    accepted = sorted(p.stem for p in reviews.glob("*.json")
                      if (json.loads(p.read_text(encoding="utf-8")) or {}).get("accepted")) \
        if reviews.is_dir() else []
    selected = next((q for q in accepted if q.casefold() == quant.casefold()), None)
    if selected is None:
        raise ValueError(f"{quant} has no accepted listening review in {record.id}")
    # Every accepted quant is republished together so bundle.json stays complete.
    current = voice_plan(saved["model"], target_repo=saved.get("target_repo"),
                         quants=accepted, task=saved.get("task"),
                         family=saved.get("family"), hunt_forks=False,
                         language=saved.get("language"), speaker=saved.get("speaker"))
    return run_voice(current, publish=True, family=saved.get("family"))


def _verify_local(path, root=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("pipeline") == "retrieval" and data.get("engine") in ("llama.cpp", "clip.cpp"):
        from .pipeline.retrieval_release import verify_manifest
        problems = verify_manifest(data, root or Path(path).parent)
        return {"target": str(path), "kind": "retrieval-bundle",
                "valid": not problems, "problems": problems}
    if data.get("pipeline") == "multimodal":
        from .pipeline.multimodal_release import verify_manifest
        problems = verify_manifest(data, root or Path(path).parent)
        return {"target": str(path), "kind": "multimodal-bundle",
                "valid": not problems, "problems": problems}
    if data.get("pipeline") == "vision":
        from .pipeline import vision_release
        problems = vision_release.verify_manifest(data, root or Path(path).parent)
        return {"target": str(path), "kind": "vision-bundle",
                "valid": not problems, "problems": problems}
    if isinstance(data, dict) and data.get("engine") == "stable-diffusion.cpp":
        from .pipeline import media_release
        problems = media_release.load_bundle(path).problems(Path(root or Path(path).parent))
        return {"target": str(path), "kind": "media-bundle", "valid": not problems, "problems": problems}
    manifest = voice.load_manifest(path)
    problems = voice.verify_manifest_files(manifest, root or Path(path).parent)
    return {"target": str(path), "kind": "voice-bundle", "valid": not problems, "problems": problems}


def our_repo(name):
    if "/" in name:
        return name
    return f"{config.namespace()}/{name.removesuffix('-GGUF')}-GGUF"


def verify(target, *, root=None):
    """Verify a local manifest, a recorded run, or a published repository."""
    path = Path(target)
    if path.suffix == ".json" and path.is_file():
        return _verify_local(path, root)
    try:
        record = runs.resolve(target)
    except runs.AmbiguousRun:
        raise
    except ValueError:
        record = None
    if record and record.pipeline == TEXT:
        return {"run_id": record.id, **card.verify(Job.load(record.location))}
    if record and record.pipeline in (VOICE, MEDIA):
        pattern = "models/**/publication/*/bundle.json" if record.pipeline == VOICE else "candidates/*/manifest.json"
        bundles = [_verify_local(p) for p in sorted(Path(record.location).glob(pattern))]
        return {"run_id": record.id, "kind": f"{record.pipeline}-run", "bundles": bundles,
                "valid": bool(bundles) and all(b["valid"] for b in bundles),
                "problems": [] if bundles else ["no built bundle manifests in this run yet"]}
    if record and record.pipeline in (RETRIEVAL, MULTIMODAL):
        manifests = sorted(Path(record.location).glob("publication/manifest.json"))
        bundles = [_verify_local(p) for p in manifests]
        return {"run_id": record.id, "kind": f"{record.pipeline}-run", "bundles": bundles,
                "valid": bool(bundles) and all(item["valid"] for item in bundles),
                "problems": [] if bundles else ["no published retrieval manifest in this run"]}
    if record and record.pipeline == VISION:
        manifest = Path(record.location) / "result.json"
        if not manifest.is_file():
            return {"run_id": record.id, "kind": "vision-run", "valid": False,
                    "problems": ["no vision result manifest in this run"]}
        return _verify_local(manifest, root or Path(record.location))
    return verify_remote(our_repo(target))


def verify_remote(repo_id):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError
    api = HfApi(token=config.TOKEN)
    try:
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except RepositoryNotFoundError:
        return {"repo_id": repo_id, "error": f"{repo_id} does not exist"}
    remote = None
    if "bundle.json" in files or "manifest.json" in files:
        remote = voice_release._remote_files(api, repo_id)
    if "bundle.json" in files:
        manifest = voice.load_manifest(hf_hub_download(repo_id, "bundle.json", token=config.TOKEN))
        problems = voice.remote_bundle_problems(manifest, remote)
        return {"repo_id": repo_id, "kind": "voice-bundle", "valid": not problems, "problems": problems}
    if "manifest.json" in files:
        data = json.loads(Path(hf_hub_download(repo_id, "manifest.json", token=config.TOKEN))
                          .read_text(encoding="utf-8"))
        if data.get("engine") == "stable-diffusion.cpp":
            problems = []
            for component in data.get("components", []):
                actual = remote.get(component["output_path"])
                if actual is None:
                    problems.append(f"remote member missing: {component['output_path']}")
                elif actual.get("bytes") not in (None, component["bytes"]):
                    problems.append(f"remote size mismatch: {component['output_path']}")
                elif actual.get("sha256") and actual["sha256"] != component["sha256"]:
                    problems.append(f"remote checksum mismatch: {component['output_path']}")
            return {"repo_id": repo_id, "kind": "media-bundle", "valid": not problems, "problems": problems}
        if data.get("pipeline") == "retrieval" and data.get("engine") in ("llama.cpp", "clip.cpp"):
            from .pipeline.retrieval_release import (_hash, _remote_member_mismatches)
            manifest_path = Path(hf_hub_download(repo_id, "manifest.json", token=config.TOKEN))
            expected_manifest = {"path": "manifest.json",
                                 "bytes": manifest_path.stat().st_size,
                                 "sha256": _hash(manifest_path)}
            expected_files = [*data.get("members", []), *data.get("published_files", []),
                              expected_manifest]
            problems = _remote_member_mismatches(
                expected_files, remote,
                download=lambda name: hf_hub_download(repo_id, name, token=config.TOKEN))
            return {"repo_id": repo_id, "kind": "retrieval-bundle",
                    "valid": not problems, "problems": problems}
        if data.get("pipeline") == "multimodal":
            from .pipeline.native_common import _hash
            manifest_path = Path(hf_hub_download(repo_id, "manifest.json", token=config.TOKEN))
            expected = [*data.get("members", []), *data.get("published_files", []),
                        {"path": "manifest.json", "bytes": manifest_path.stat().st_size,
                         "sha256": _hash(manifest_path)}]
            problems = []
            for member in expected:
                actual = remote.get(member["path"])
                if actual is None or actual.get("bytes") != member["bytes"] or \
                        actual.get("sha256") != member["sha256"]:
                    problems.append(f"remote verification mismatch: {member['path']}")
            return {"repo_id": repo_id, "kind": "multimodal-bundle",
                    "valid": not problems, "problems": problems}
    return card.verify(repo_id)


def list_runs():
    return {"runs": [item.to_dict() for item in runs.discover()]}
