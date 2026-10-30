"""aqx — the AgentQuantix command line.

Every capability of the agent is a subcommand here, and the harness adapters
(MCP server, the built-in OpenRouter loop, the Claude Code skill) all drive
these same functions. That is deliberate: there is exactly one implementation
of "research the trending list" or "run this model", so the agent cannot
behave differently depending on which harness is holding the reins, and the
whole thing still works with no harness at all.

Model-work approval gates:
  * you decide when to trigger  -> `aqx research`
  * you approve the models      -> the prompt in `aqx run`, or `--yes`

Runtime storage and update decisions are separate and are not implied by --yes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from . import (actions, card, config, feasibility, media, report, research, runs,
               sysprobe, transfer)
from .pipeline import build as build_mod, run as run_mod, source as source_mod
from .pipeline.job import Job
from .runtimes import RuntimeDecisionRequired, RuntimeErrorBase
from .tasks import MEDIA, MULTIMODAL, RETRIEVAL, TEXT, VISION, VOICE, TASKS

_ACTIVE_RESUME_IDS = []


def _print(*args):
    """Print without dying on a console that cannot encode the character.

    Windows terminals default to cp1252; a model name with a non-Latin
    character in it would otherwise abort the whole report with a
    UnicodeEncodeError three quarters of the way down.
    """
    text = " ".join(str(a) for a in args)
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "ascii",
                          errors="replace").decode(sys.stdout.encoding or "ascii"),
              flush=True)


# =====================================================
# probe
# =====================================================
def cmd_probe(args):
    info = sysprobe.probe(measure_disk=not args.no_disk,
                          force_disk=args.remeasure_disk)
    if args.json:
        _print(json.dumps(info, indent=2, default=str))
    else:
        _print(sysprobe.summary(info))
        if info.get("disk_gbs_cached") and not args.remeasure_disk:
            _print("\n(disk rate is cached from an earlier measurement; "
                   "--remeasure-disk re-runs it, ideally while nothing else "
                   "is hammering the drive)")
    return 0


# =====================================================
# research
# =====================================================
def cmd_research(args):
    result = research.research(
        limit=args.limit,
        probe_disk=not args.no_disk,
        hunt_forks=not args.no_fork_hunt,
        include_tasks=not args.text_only,
        on_progress=None if args.json else (lambda m: _print(f"  .. {m}")))

    path = research.save(result)
    if args.markdown:
        markdown_path = path.with_suffix(".md")
        markdown_path.write_text(
            report.markdown(result, limit=args.top,
                            include_blocked=not args.hide_blocked),
            encoding="utf-8")

    if args.json:
        _print(json.dumps(result, indent=2, default=str))
        return 0

    _print("")
    _print(sysprobe.summary(result["system"]))
    _print("")
    _print(report.table(result, limit=args.top,
                        include_blocked=not args.hide_blocked))
    _print(f"\nsaved: {path}")
    if args.markdown:
        _print(f"       {path.with_suffix('.md')}")
    _print("\nNext: `aqx show <model>` for detail, `aqx run <model>` to start "
           "(it asks before doing anything).")
    return 0


def _json(value):
    _print(json.dumps(value, indent=2, default=str))


def _announce_resume(run_id):
    if run_id and run_id not in _ACTIVE_RESUME_IDS:
        _ACTIVE_RESUME_IDS.append(run_id)
        _print(f"Resume command if interrupted: `aqx run --resume {run_id}`")


def _completed_resume(run_id):
    if run_id:
        _print(f"Resume with: `aqx run --resume {run_id}`")


def cmd_show(args):
    """Research detail, else a recorded run, else a live read-only inspection."""
    result = research.load_latest()
    failed = False
    for name in args.model:
        assessment = research.find(result, name) if result else None
        if assessment:
            _print("")
            _print(report.detail(assessment))
            continue
        try:
            record = runs.resolve(name)
        except runs.AmbiguousRun:
            raise
        except ValueError:
            record = None
        if record:
            _json(record.to_dict())
            continue
        if "/" not in name and ":" not in name:
            _print(f"{name}: not in the last research run or the run records. "
                   "Name it as a full repo id (org/model).")
            failed = True
            continue
        inspected = actions.inspect(name)
        _print("")
        _print(inspected["text"] if "text" in inspected else
               json.dumps(inspected, indent=2, default=str))
        failed |= not inspected["route"]["ready"]
    return 1 if failed else 0


def cmd_inspect(args):
    result = actions.inspect(args.model, task=args.task, engine=args.engine,
                             family=args.family, hunt_forks=not args.no_fork_hunt)
    if "text" in result and not args.json:
        _print(result["text"])
    else:
        _json(result)
    return 0 if result["route"]["ready"] else 1


def cmd_runs(args):
    records = runs.discover(args.pipeline)
    if args.json:
        _json([item.to_dict() for item in records])
        return 0
    if not records:
        _print("No recorded runs.")
        return 0
    for item in records:
        reviews = item.details.get("reviews")
        _print(f"  {item.id:<40} {item.task or item.pipeline:<28} {item.repo_id}"
               + (f"  reviews: {', '.join(reviews)}" if reviews else ""))
    return 0


# =====================================================
# run
# =====================================================
def _select_interactively(result, preselected, *, allow_runtime_setup=False):
    """The approval gate. Returns the assessments the user said yes to."""
    runnable = [a for a in (result or {}).get("assessments", [])
                if a["verdict"] not in ("blocked", "done")]

    if preselected:
        chosen = []
        for name in preselected:
            assessment = research.find(result, name) if result else None
            if not assessment and "/" in name:
                # A full repo id was named. Assess it on demand rather than
                # refusing work because it did not happen to be trending on
                # the day the last report was written.
                _print(f"{name}: not in the last research run - "
                       "assessing it now...")
                try:
                    assessment = research.assess_one(name)
                except ValueError as e:
                    _print(f"  {e}")
                    continue
            if not assessment:
                _print(f"{name}: not in the last research run. Name it as a "
                       "full repo id (org/model) to assess it directly.")
                continue
            if assessment["verdict"] == "blocked" and assessment.get("arch_ok") is False and allow_runtime_setup:
                from .runtime_control import ensure_runtime
                previous = Job.from_assessment(assessment)
                if ensure_runtime("llama.cpp", frozen=previous.runtime_snapshot or None):
                    assessment = research.assess_one(assessment["repo_id"])
            if assessment["verdict"] == "blocked":
                _print(f"{assessment['repo_id']}: BLOCKED - "
                       f"{assessment['blockers'][0]}")
                continue
            if assessment["verdict"] == "done":
                _print(f"{assessment['repo_id']}: already fully published in "
                       f"{assessment['target_repo']} "
                       f"({assessment['published_count']} quants) - nothing "
                       "to do.")
                continue
            chosen.append(assessment)
        return chosen

    if not runnable:
        _print("Nothing in the last research run is runnable on this machine.")
        return []

    _print("")
    for index, assessment in enumerate(runnable, start=1):
        note = (assessment["warnings"] or [""])[0]
        _print(f"  {index:>2}. {assessment['repo_id']:<44} "
               f"{assessment['hours']['total']:>5.1f}h  "
               f"{assessment['peak_disk_gb']:>5.0f}G peak  "
               f"{len(assessment['quants']):>2} quants  {note[:38]}")
    _print("")
    raw = input("Quantize which? (numbers/names, comma separated, blank = none) ")
    if not raw.strip():
        return []

    chosen = []
    for token in (t.strip() for t in raw.split(",") if t.strip()):
        if token.isdigit() and 1 <= int(token) <= len(runnable):
            chosen.append(runnable[int(token) - 1])
        else:
            try:
                if assessment := research.find(result, token):
                    chosen.append(assessment)
                else:
                    _print(f"  '{token}' matched nothing - ignored.")
            except ValueError as e:
                _print(f"  {e} - ignored.")
    return chosen


# Options that only one pipeline understands. Each defaults to None so a value
# the user actually typed can be told apart from the pipeline's own default.
TEXT_OPTIONS = {"upload_workers", "quantize_workers", "queue_depth", "quantize_threads",
                "xet", "keep_fork"}
VOICE_OPTIONS = {"language", "pocket_language", "speaker", "review",
                 "max_roundtrip_wer", "max_wer_regression"}
MEDIA_OPTIONS = {"component", "contract", "profile", "init_image", "end_image", "ref_image",
                 "ref_audio", "width", "height", "steps", "seed", "frames", "fps", "sampler",
                 "scheduler", "cfg_scale", "compute_backend", "preserve_failed", "plan"}
APPLICABLE = {
    TEXT: TEXT_OPTIONS | {"sequential"},
    VOICE: VOICE_OPTIONS | {"target", "quant", "publish", "output"},
    MEDIA: MEDIA_OPTIONS | {"target", "quant", "publish", "output", "sequential"},
    RETRIEVAL: {"target", "quant", "publish", "output"},
    MULTIMODAL: {"target", "quant", "publish", "output"},
    VISION: {"target", "quant", "publish", "output"},
}


def _check_applicable(args, pipeline):
    for dest in sorted(set().union(*APPLICABLE.values()) - APPLICABLE[pipeline]):
        if getattr(args, dest, None) not in (None, [], False):
            raise ValueError(f"--{dest.replace('_', '-')} does not apply to "
                             f"{pipeline} releases")


def _text_run_options(args):
    def pick(value, default):
        return default if value is None else value
    return run_mod.RunOptions(
        quantize_workers=pick(getattr(args, "quantize_workers", None), config.QUANTIZE_WORKERS),
        upload_workers=pick(getattr(args, "upload_workers", None), config.UPLOAD_WORKERS),
        queue_depth=pick(getattr(args, "queue_depth", None), config.QUEUE_DEPTH),
        quantize_threads=pick(getattr(args, "quantize_threads", None), config.QUANTIZE_THREADS),
        sequential=bool(args.sequential) or config.SEQUENTIAL)


def _confirm(args, question):
    if args.yes:
        return True
    if not sys.stdin.isatty():
        raise ValueError("explicit --yes is required for noninteractive execution")
    return input(question).strip().casefold() in ("y", "yes")


def cmd_run(args):
    # Set before anything reads the policy — the feasibility repricing below
    # asks whether uploads will use xet, and the answer has to be the one this
    # run will actually act on.
    if args.xet:
        os.environ["AQX_XET"] = args.xet
    if args.resume:
        if args.model:
            raise ValueError("--resume takes a run id instead of model names")
        return _resume(args)
    if args.plan:
        if len(args.model) > 1:
            raise ValueError("a saved plan describes exactly one model")
        saved = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        task_spec = TASKS.get(saved.get("task"))
        if task_spec and task_spec.pipeline == RETRIEVAL:
            _check_applicable(args, RETRIEVAL)
            return _run_retrieval(args, saved=saved)
        if task_spec and task_spec.pipeline == MULTIMODAL:
            _check_applicable(args, MULTIMODAL)
            return _run_multimodal(args, saved=saved)
        if task_spec and task_spec.pipeline == VISION:
            _check_applicable(args, VISION)
            return _run_vision(args, saved=saved)
        _check_applicable(args, MEDIA)
        return _run_media(args, args.model[0] if args.model else None, saved=saved)
    if not args.model:
        _check_applicable(args, TEXT)
        return _run_text(args, [])

    routes = [actions.route_named(name, task=args.task, engine=args.engine, family=args.family)
              for name in args.model]
    for item in routes:
        if not item.ready:
            _print(f"{item.repo_id}: BLOCKED - {item.blocker}")
            for choice in item.choices:
                _print(f"    --task {choice['task'] or '?'}  ({choice['pipeline']}: {choice['evidence']})")
    ready = [(name, item) for name, item in zip(args.model, routes) if item.ready]
    if not ready:
        return 2
    pipelines = sorted({item.pipeline for _, item in ready})
    if len(pipelines) > 1:
        raise ValueError(f"these models resolve to different pipelines ({', '.join(pipelines)}); "
                         "run each pipeline separately")
    pipeline = pipelines[0]
    _check_applicable(args, pipeline)
    if pipeline == TEXT:
        code = _run_text(args, [name for name, _ in ready])
    elif len(ready) > 1:
        raise ValueError(f"{pipeline} releases run one model per invocation")
    elif pipeline == VOICE:
        code = _run_voice(
            args, ready[0][1].repo_id,
            family=actions.voice_family(ready[0][1], args.family),
            task=ready[0][1].task)
    elif pipeline == RETRIEVAL:
        code = _run_retrieval(args, ready[0][1])
    elif pipeline == MULTIMODAL:
        code = _run_multimodal(args, ready[0][1])
    elif pipeline == VISION:
        code = _run_vision(args, ready[0][1])
    else:
        code = _run_media(args, ready[0][1].repo_id, route=ready[0][1])
    return code or (1 if len(ready) < len(routes) else 0)


def _run_text(args, names):
    result = research.load_latest()
    if not result and not names:
        _print("No research on file. Run `aqx research` first, or name a "
               "model directly: `aqx run org/model`.")
        return 1

    chosen = _select_interactively(result, names, allow_runtime_setup=not args.dry_run)
    if not chosen:
        _print("Nothing approved - stopping.")
        return 0

    options = _text_run_options(args)
    jobs = [Job.from_assessment(a) for a in chosen]
    return _execute_text(args, jobs, options, result)


def _execute_text(args, jobs, options, result=None, resumed=False):
    # The research run priced these under whatever xet policy was in force at
    # the time. --xet off (or a backend that has since gone missing) can make a
    # BF16 undownloadable, and that is a failure which would otherwise surface
    # AFTER the approval prompt, minutes into a run.
    runnable = []
    for job in jobs:
        gib = job.assessment["bf16_gb"]
        if transfer.exceeds_http_limit(size_gib=gib):
            allowed, why = transfer.for_download(size_gib=gib)
            if not allowed:
                _print(f"  {job.repo_id}: BF16 is {gib:.0f} GiB and {why} - "
                       "cannot be downloaded under the current xet policy, "
                       "dropping it.")
                continue
            # The cap is not download-only. A BF16 over it cannot be PUT on
            # the Hub either, and that failure would land hours later, after
            # the download and the conversion, on a file the run cannot skip.
            can_upload, upload_why = transfer.for_upload(size_gib=gib)
            if not can_upload:
                _print(f"  {job.repo_id}: BF16 is {gib:.0f} GiB and "
                       f"{upload_why} - it could be built but never "
                       "published, dropping it.")
                continue
        runnable.append(job)
    if not runnable:
        _print("Nothing left to run under the current xet policy "
               f"({transfer.summary()}).")
        return 1
    jobs = runnable

    # Conversion needs torch and transformers, which are an optional extra.
    # Reported HERE, before the approval prompt, because the alternative is
    # finding out after several gigabytes of weights have been downloaded.
    # Models whose publisher ships a BF16 GGUF are unaffected and stay.
    if missing := source_mod.converter_missing():
        convertible = [j for j in jobs if j.source_kind == "convert"]
        if convertible:
            _print("")
            for job in convertible:
                _print(f"  {job.repo_id}: needs conversion from safetensors, "
                       f"which requires {', '.join(missing)} - skipping.")
            _print("")
            _print(source_mod.converter_hint(missing))
            jobs = [j for j in jobs if j.source_kind != "convert"]
        if not jobs:
            return 1

    # Smallest first, so a pipeline problem surfaces on the cheapest model
    # rather than eight hours into the largest one.
    jobs.sort(key=lambda j: j.assessment["hours"]["total"])

    if not args.dry_run and any(not job.fork for job in jobs):
        from . import runtime_control
        originals = [job for job in jobs if not job.fork]
        frozen = originals[0].runtime_snapshot or None
        handle = runtime_control.ensure_runtime("llama.cpp", frozen=frozen)
        if handle:
            for job in originals:
                if job.runtime_snapshot and job.runtime_snapshot["identity"] != handle.identity:
                    raise ValueError("these jobs require different frozen runtimes; resume them separately")
                job.runtime_snapshot = handle.snapshot()
            # Reassess only new jobs against the selected installation before
            # displaying the final quant list and asking for model approval.
            # A resumed job keeps the plan it was approved with.
            if not frozen and not resumed:
                refreshed = []
                for job in jobs:
                    if job.fork:
                        refreshed.append(job)
                        continue
                    assessment = research.assess_one(job.repo_id)
                    if assessment["verdict"] == "blocked":
                        _print(f"{job.repo_id}: blocked after runtime selection: {assessment['blockers']}")
                        continue
                    selected = Job.from_assessment(assessment)
                    selected.runtime_snapshot = handle.snapshot()
                    refreshed.append(selected)
                jobs = refreshed
                if not jobs:
                    return 1

    total_hours = sum(j.assessment["hours"]["total"] for j in jobs)
    upload = sum(j.assessment["upload_gb"] for j in jobs)

    # Peak disk is repriced here rather than taken from the research run: the
    # report was computed at the default concurrency, and these flags are
    # precisely the ones that change it. A model that looked too big for the
    # disk can fit under --sequential, and the user needs to see that.
    default_peak = max(j.assessment["peak_disk_gb"] for j in jobs)
    peaks = {j.base_name: feasibility.peak_disk_for(
        j.assessment, quantize_workers=options.quantize_workers,
        upload_workers=options.upload_workers, queue_depth=options.queue_depth,
        sequential=options.sequential) for j in jobs}
    peak = max(peaks.values())

    _print("")
    _print(f"{len(jobs)} model(s), ~{total_hours:.1f} h total, "
           f"{peak:.0f} GB peak disk (one model at a time), "
           f"{upload:.0f} GB to upload")
    _print(f"concurrency: {options.describe()}")
    _print(f"transfer:    {transfer.summary()}")
    if abs(peak - default_peak) > 0.5:
        direction = "down from" if peak < default_peak else "up from"
        _print(f"             peak disk {direction} {default_peak:.0f} GB at "
               "the default concurrency")
    if options.sequential:
        _print("             uploads no longer overlap quantization, so expect "
               "longer than the estimate above")
    _print("")
    for job in jobs:
        fork = f"  [fork: {job.fork['repo']}@{job.fork['ref']}]" if job.fork else ""
        _print(f"  {job.repo_id:<44} -> {job.target_repo}"
               f"  {len(job.quants)} quants, "
               f"imatrix on {job.imatrix_source}, "
               f"{peaks[job.base_name]:.0f} GB peak{fork}")

    # No research on file when models were named directly - `result` is None,
    # and the disk warning simply has nothing to compare against. The
    # assessment itself still blocks on live free disk, so nothing is lost.
    free = ((result or {}).get("system") or {}).get("disk_free_gb") or 0
    if free and peak > free * 0.9:
        _print(f"\n  WARNING: peak needs {peak:.0f} GB but only {free:.0f} GB "
               "was free at research time."
               + ("" if options.sequential else
                  "  --sequential would cut the peak to "
                  f"{max(feasibility.peak_disk_for(j.assessment, sequential=True) for j in jobs):.0f} GB."))

    if args.dry_run:
        _print("\n--dry-run: stopping before any download, build or upload.")
        for job in jobs:
            _print(f"  job record would be written to {job.record_file}")
        return 0

    if not args.yes:
        if input("\nStart? [y/N] ").strip().lower() not in ("y", "yes"):
            _print("Cancelled.")
            return 0

    for job in jobs:
        _announce_resume(f"text:{job.base_name}")
    outcome = run_mod.run_jobs(jobs, options=options,
                               clean_forks=not args.keep_fork)
    failed = bool(outcome.get("incomplete")) or any(
        entry.get("failures") for entry in outcome["jobs"].values())

    # Step 5 runs automatically: verify what landed, then write the card.
    # Guarded, because the quants are already published by this point — a
    # failure here is worth reporting but must not present a successful run as
    # a failed one, nor abandon the remaining models' verification.
    for job in jobs:
        result = outcome["jobs"].get(job.base_name, {})
        # A model that aborted before the pipeline got going never created its
        # repo, so verifying it produces a RepositoryNotFoundError wall of text
        # that buries the actual error above it.
        if any(quant == "<model>" for quant, _ in (result.get("failures") or [])):
            _print(f"\n{job.target_repo}: nothing was published, so there is "
                   "nothing to verify.")
            continue
        _print(f"\nverifying {job.target_repo}...")
        try:
            verification = card.verify(job)
            _print(_verification_text(verification))
            if verification.get("error"):
                failed = True
                continue
            if verification.get("missing") or verification.get("suspect"):
                failed = True
            card.publish(verification, job=job)
            _print(f"model card published to {verification['url']}")
        except Exception as e:
            failed = True
            _print(f"  the quants are uploaded, but the model card step failed: "
                   f"{type(e).__name__}: {e}")
            _print(f"  retry it on its own with: aqx card {job.base_name}")
        _completed_resume(f"text:{job.base_name}")
    return 1 if failed else 0


# =====================================================
# verify / card
# =====================================================
def _verification_text(verification):
    if verification.get("error"):
        return f"  ERROR: {verification['error']}"
    lines = [f"  {verification['count']} GGUF files, "
             f"{verification['total_gb']:.1f} GB total, "
             f"last modified {verification['last_modified']}"]
    for entry in verification["files"]:
        flag = "  <-- SUSPECT, far too small" if entry["suspect"] else ""
        lines.append(f"    {entry['quant']:<10} {entry['gb']:>7.2f} GB  "
                     f"{entry['name']}{flag}")
    if verification["missing"]:
        lines.append(f"  MISSING ({len(verification['missing'])}): "
                     f"{', '.join(verification['missing'])}")
    return "\n".join(lines)


def cmd_verify(args):
    failed = False
    for target in args.target:
        verification = actions.verify(target, root=args.root)
        if "kind" in verification:
            _json(verification)
            failed |= verification.get("valid") is False
            continue
        _print(f"\n{verification.get('run_id') or verification.get('repo_id', target)}")
        _print(_verification_text(verification))
        failed |= bool(verification.get("error"))
    return 1 if failed else 0


def _run_record(target):
    try:
        return runs.resolve(target)
    except runs.AmbiguousRun:
        raise
    except ValueError:
        return None


def cmd_card(args):
    record = _run_record(args.repo)
    if record and record.pipeline in (VOICE, MEDIA):
        if not args.dry_run:
            raise ValueError(f"{record.pipeline} cards are generated from the verified bundle "
                             f"manifest and published with it: aqx publish {record.id} --quant QUANT")
        if record.pipeline == MEDIA:
            from .pipeline import media_release
            if not args.quant:
                raise ValueError("--quant selects the candidate whose card is rendered")
            manifest = Path(record.location) / "candidates" / media.relative_path(args.quant) / "manifest.json"
            _print(media_release.render_card(media_release.load_bundle(manifest)))
            return 0
        cards = sorted(Path(record.location).glob("models/**/publication/*/README.md"))
        if not cards:
            raise ValueError(f"{record.id} has not staged a bundle card yet")
        for path in cards:
            _print(path.read_text(encoding="utf-8"))
        return 0

    repo_id = actions.our_repo(args.repo)
    job = None
    if record and record.pipeline == TEXT:
        job = Job.load(record.location)
        repo_id = job.target_repo
    else:
        path = config.RUNS_DIR / f"{repo_id.split('/')[-1].removesuffix('-GGUF')}.json"
        if path.exists():
            job = Job.load(path)

    verification = card.verify(job or repo_id)
    if verification.get("error"):
        _print(f"{repo_id}: {verification['error']}")
        return 1

    text = card.publish(verification, job=job, dry_run=args.dry_run)
    if args.dry_run:
        _print(text)
    else:
        _print(f"model card published to {verification['url']}")
    return 0


# =====================================================
# review / publish
# =====================================================
def cmd_review(args):
    _json(actions.review(args.run_id, args.quant, accepted=args.accept, reviewer=args.reviewer,
                         notes=args.notes or "", watched_video=args.watched_video,
                         listened_audio=args.listened_audio))
    return 0


def cmd_publish(args):
    record = runs.resolve(args.run_id)
    _print(f"{record.id}: publish {args.quant} of {record.repo_id} -> "
           f"{record.target or 'the target recorded in its plan'}")
    if not _confirm(args, "Publish this validated, reviewed candidate? [y/N] "):
        _print("Cancelled.")
        return 0
    result = actions.publish(record.id, args.quant, user_approved=True)
    _json(result)
    if record.pipeline == VOICE:
        return 0 if any(entry.get("published") for entry in result.get("results", {}).values()) else 1
    return 0 if result.get("published") else 1


# =====================================================
# voice / media execution
# =====================================================
VOICE_NON_ERRORS = {"human listening review required",
                    "accepted and waiting for atomic bundle publication"}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def _voice_args(args, family=None, task=None):
    return {"target_repo": args.target, "quants": args.quant,
            "family": family, "task": task or args.task,
            "hunt_forks": not args.no_fork_hunt, "language": args.language,
            "speaker": args.speaker}


def _run_voice(args, repo, family=None, saved=None, task=None):
    voice_args = _voice_args(args, family, task)
    for key in ("target_repo", "quants", "language", "speaker", "family", "task"):
        if voice_args.get(key) is None and saved:
            voice_args[key] = saved.get(key)
    plan = actions.voice_plan(repo, **voice_args)
    if plan.get("status") == "ready" and not args.dry_run:
        actions.prepare_voice_runtimes(plan)
        plan = actions.voice_plan(repo, **voice_args)
    if args.output:
        media.atomic_json(args.output, plan)
    _json(plan)
    if plan.get("status") != "ready":
        return 2
    if args.dry_run:
        return 0
    if not _confirm(args, "Run this voice release? This downloads, builds and may publish. [y/N] "):
        _print("Nothing started.")
        return 0
    from .pipeline import voice_release
    run_id = f"voice:{voice_release.work_dir(repo).name}"
    _announce_resume(run_id)
    result = actions.run_voice(
        plan, publish=args.publish is not False, human_review=args.review,
        family=voice_args["family"], language=args.language, speaker=args.speaker,
        pocket_language=args.pocket_language, max_roundtrip_wer=args.max_roundtrip_wer,
        max_wer_regression=args.max_wer_regression)
    _json(result)
    _completed_resume(run_id)
    return 0 if all(item.get("published") or item.get("reason") in VOICE_NON_ERRORS
                    for item in result["results"].values()) else 1


def _components(values):
    members = []
    for value in values or []:
        if "=" not in value:
            raise media.MediaValidationError("use --component role=path or role=org/repo::path")
        role, source = value.split("=", 1)
        member = {"role": role, "precision": "source",
                  "quantizable": role not in ("vae", "audio_vae", "llm_vision", "config",
                                              "tokenizer", "scheduler")}
        if "::" in source:
            member["source_repo"], source = source.split("::", 1)
        member["source_path"] = source
        members.append(member)
    return members or None


def _media_args(args, route=None):
    task = args.task if args.task in media.MEDIA_TASKS else (
        route.task if route and route.task in media.MEDIA_TASKS else None)
    return {"target_repo": args.target, "quants": args.quant, "task": task,
            "family": args.family or (route.family if route else None),
            "components": _components(args.component),
            "compute_backend": args.compute_backend or "auto",
            "sequential": bool(args.sequential),
            "contract": _read_json(args.contract), "quality_profile": _read_json(args.profile),
            "conditioning": {key: value for key, value in {
                "init_image": args.init_image, "end_image": args.end_image,
                "ref_image": args.ref_image, "ref_audio": args.ref_audio}.items() if value},
            "settings": {key: getattr(args, key) for key in (
                "width", "height", "steps", "seed", "frames", "fps", "sampler", "scheduler",
                "cfg_scale") if getattr(args, key) is not None}}


def _run_retrieval(args, route=None, saved=None):
    if saved is None:
        if route is None:
            raise ValueError("retrieval execution requires a resolved route")
        plan = actions.retrieval_plan(
            route.repo_id, task=route.task, engine=route.engine,
            target_repo=args.target, quants=args.quant)
    else:
        plan = saved
    if args.output:
        media.atomic_json(args.output, plan)
    _json(plan)
    if args.dry_run:
        return int(bool(plan.get("blockers")))
    if plan.get("status") != "ready" or plan.get("blockers"):
        raise ValueError("retrieval plan is blocked: " + "; ".join(plan.get("blockers", [])))
    if not _confirm(args, "Run this retrieval release? This downloads, builds and validates. [y/N] "):
        _print("Nothing started.")
        return 0
    from .pipeline import retrieval_release
    run_id = f"retrieval:{retrieval_release._slug(plan['model'])}"
    _announce_resume(run_id)
    result = actions.start_retrieval(plan, user_approved=True, publish=bool(args.publish))
    _json(result)
    _completed_resume(result.get("run_id", run_id))
    return int(any(item.get("status") != "passed" for item in result.get("candidates", [])))


def _run_multimodal(args, route=None, saved=None):
    if saved is None:
        if route is None:
            raise ValueError("multimodal execution requires a resolved route")
        plan = actions.multimodal_plan(
            route.repo_id, task=route.task, engine=route.engine,
            target_repo=args.target, quants=args.quant)
    else:
        plan = saved
    if args.output:
        media.atomic_json(args.output, plan)
    _json(plan)
    if args.dry_run:
        return int(bool(plan.get("blockers")))
    if plan.get("status") != "ready" or plan.get("blockers"):
        raise ValueError("multimodal plan is blocked: " + "; ".join(plan.get("blockers", [])))
    if not _confirm(args, "Run this multimodal release? This downloads, builds and validates. [y/N] "):
        _print("Nothing started.")
        return 0
    from .pipeline import multimodal_release
    run_id = f"multimodal:{multimodal_release._slug(plan['model'])}"
    _announce_resume(run_id)
    result = actions.start_multimodal(plan, user_approved=True, publish=bool(args.publish))
    _json(result)
    _completed_resume(result.get("run_id", run_id))
    return int(any(item.get("status") != "passed" for item in result.get("candidates", [])))


def _run_vision(args, route=None, saved=None, runtime_snapshot=None):
    if saved is None:
        if route is None:
            raise ValueError("vision execution requires a resolved route")
        plan = actions.vision_plan(route.repo_id, task=route.task, family=route.family,
                                   target_repo=args.target, quants=args.quant)
    else:
        plan = saved
    if args.output:
        media.atomic_json(args.output, plan)
    _json(plan)
    if args.dry_run:
        return int(bool(plan.get("blockers")))
    if plan.get("status") != "ready" or plan.get("blockers"):
        raise ValueError("vision plan is blocked: " + "; ".join(plan.get("blockers", [])))
    if not _confirm(args, "Run this vision release? This downloads, builds and validates. [y/N] "):
        _print("Nothing started.")
        return 0
    run_id = actions.vision_run_id(plan)
    _announce_resume(run_id)
    result = actions.start_vision(
        plan, user_approved=True, publish=bool(args.publish),
        runtime_snapshot=runtime_snapshot)
    _json(result)
    _completed_resume(result.get("run_id", run_id))
    return int(not any(item.get("status") == "passed"
                       for item in result.get("candidates", [])))


def _run_media(args, repo, route=None, saved=None):
    def make_plan():
        return media.plan_release(repo, hunt_forks=not args.no_fork_hunt, **_media_args(args, route))

    plan = saved if saved is not None else make_plan()
    if saved is not None:
        media.validate_plan(plan, check_runtime=False)
    if not args.dry_run and not plan["blockers"]:
        from . import runtime_control
        revision = ({"revision": plan["engine_revision"], "repository": "leejet/stable-diffusion.cpp"}
                    if saved and not saved.get("runtime_snapshot") else None)
        handle = runtime_control.ensure_runtime(
            "stable-diffusion.cpp", compute_backend=plan["compute_backend"],
            frozen=saved.get("runtime_snapshot") if saved else None, revision=revision)
        if handle and saved is None:
            plan = make_plan()
    if args.output:
        media.atomic_json(args.output, plan)
    _json(plan)
    if args.dry_run:
        return 1 if plan["blockers"] else 0
    if plan["blockers"]:
        raise media.MediaValidationError("media plan is blocked: " + "; ".join(plan["blockers"]))
    if repo and repo != plan["repo_id"]:
        raise media.MediaValidationError("repository differs from the saved approved plan")
    if not _confirm(args, "Build and validate this exact media plan? [y/N] "):
        return 0
    publish = bool(args.publish)
    run_id = f"media:{plan['plan_id']}"
    _announce_resume(run_id)
    result = actions.start_media(plan, user_approved=True, publish=publish,
                                 preserve_failed=bool(args.preserve_failed))
    _json(result)
    _completed_resume(result.get("run_id", run_id))
    if actions.media_failed(result):
        return 1
    if publish and any(not c.get("published") for c in result.get("candidates", [])
                       if c.get("quant") != "f16"):
        return 1
    return 0


def _resume(args):
    record = runs.resolve(args.resume)
    _print(f"resuming {record.id} ({record.task or record.pipeline}): {record.repo_id}")
    _check_applicable(args, record.pipeline)
    if record.pipeline == TEXT:
        return _execute_text(args, [Job.load(record.location)], _text_run_options(args),
                             resumed=True)
    if record.pipeline == VOICE:
        saved = record.details.get("plan") or {}
        return _run_voice(args, record.repo_id,
                          family=args.family or saved.get("family"),
                          saved=saved, task=saved.get("task"))
    if record.pipeline == RETRIEVAL:
        return _run_retrieval(args, saved=record.details["plan"])
    if record.pipeline == MULTIMODAL:
        return _run_multimodal(args, saved=record.details["plan"])
    if record.pipeline == VISION:
        return _run_vision(args, saved=record.details["plan"],
                            runtime_snapshot=record.details.get("runtime_snapshot"))
    plan = json.loads((Path(record.location) / "plan.json").read_text(encoding="utf-8"))
    return _run_media(args, None, saved=plan)


def cmd_plan(args):
    routes = [actions.route_named(name, task=args.task, engine=args.engine, family=args.family)
              for name in args.model]
    for pipeline in {item.pipeline for item in routes if item.ready}:
        _check_applicable(args, pipeline)
    media_args = _media_args(args, next((r for r in routes if r.pipeline == MEDIA), None))
    result = actions.plan(args.model, task=args.task, engine=args.engine, family=args.family,
                          hunt_forks=not args.no_fork_hunt, text=_text_run_options(args),
                          voice_args=_voice_args(args), media_args=media_args,
                          retrieval_args={"target_repo": args.target, "quants": args.quant},
                          multimodal_args={"target_repo": args.target, "quants": args.quant},
                          vision_args={"target_repo": args.target, "quants": args.quant})
    if args.output:
        if len(result["releases"]) != 1:
            raise ValueError("--output saves exactly one release plan")
        media.atomic_json(args.output, result["releases"][0]["plan"])
    _json(result)
    blocked = bool(result["blocked"]) or bool((result["text"] or {}).get("blocked")) or any(
        release["plan"].get("blockers") or release["plan"].get("status") == "blocked"
        for release in result["releases"])
    return 1 if blocked else 0


# =====================================================
# doctor
# =====================================================
def cmd_bootstrap(args):
    from . import bootstrap
    return bootstrap.run(build=not args.check_only)


def cmd_doctor(args):
    info = sysprobe.probe(measure_disk=False)
    problems, notes = [], []
    from .runtimes import RuntimeManager, selected_source
    manager = RuntimeManager()

    def installed_binary(engine, fallback, name):
        installed = manager.inspect(engine, verify=False)
        if installed and name in installed.binaries:
            return installed.binaries[name]
        return build_mod.find_binary(selected_source(engine, fallback), name)

    if not info["hf_token"]:
        problems.append("No Hugging Face token. Set HF_TOKEN, or run "
                        "`hf auth login`. It needs write permission.")
    if not info["llama"]["present"]:
        problems.append(f"No llama.cpp checkout at {config.UPSTREAM_LLAMA}. "
                        "Clone it there, or set INF_ROOT.")
    elif not info["llama"]["binaries"]:
        notes.append("llama.cpp is present but not built - the first run will "
                     "compile llama-quantize and llama-imatrix.")
    elif not installed_binary("llama.cpp", config.UPSTREAM_LLAMA, "llama-tts"):
        notes.append("llama.cpp is built without llama-tts - the first TTS "
                     "release will build that target.")
    if not selected_source("whisper.cpp", config.UPSTREAM_WHISPER).exists():
        notes.append(f"whisper.cpp is not cached at {config.UPSTREAM_WHISPER} - "
                     "the first ASR or TTS round-trip run will clone and build it.")
    elif not installed_binary("whisper.cpp", config.UPSTREAM_WHISPER, "whisper-cli"):
        notes.append("whisper.cpp is present but whisper-cli is not built - "
                     "the first voice run will build it independently.")
    if not selected_source("audio.cpp", config.UPSTREAM_AUDIOCPP).exists():
        notes.append(f"audio.cpp is not cached at {config.UPSTREAM_AUDIOCPP} - "
                     "the first catalog-backed audio.cpp release will clone "
                     "and build the full runtime.")
    elif (not installed_binary("audio.cpp", config.UPSTREAM_AUDIOCPP, "audiocpp_cli")
          or not installed_binary("audio.cpp", config.UPSTREAM_AUDIOCPP, "audiocpp_gguf")):
        notes.append("audio.cpp is present but its CLI/converter is not built - "
                     "the first audio.cpp release will build both tools.")
    if not info["gpus"]:
        notes.append("No CUDA GPU detected - the imatrix pass will run on CPU "
                     "and take considerably longer.")
    if not info["msvc"] and sys.platform == "win32":
        notes.append("Visual Studio build tools not found - building a fork "
                     "for an unsupported architecture would fail.")
    if not info["hub_backends"]["can_exceed_http_limit"]:
        notes.append("Neither hf_xet nor hf_transfer is usable - models with a "
                     f"BF16 over {transfer.HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB "
                     "cannot be downloaded.")
    if transfer.mode() != "auto":
        notes.append(f"xet is pinned by AQX_XET={transfer.mode()} - auto "
                     "would enable it only for downloads that require it "
                     "and keep uploads on the faster plain-HTTP path.")
    if info["disk_free_gb"] < 100:
        notes.append(f"Only {info['disk_free_gb']:.0f} GB free at "
                     f"{info['work_root']} - that limits which models fit.")

    for module in ("huggingface_hub", "datasets", "gguf"):
        try:
            __import__(module)
        except ImportError:
            problems.append(f"Python package '{module}' is not installed in "
                            f"{sys.executable}.")

    history = feasibility.load_history()
    counts = {k: len(v) for k, v in history.items() if v}
    notes.append("timing history: "
                 + (", ".join(f"{k}={n}" for k, n in counts.items())
                    if counts else "empty - the first estimates will be "
                                   "assumptions, not measurements"))

    _print(sysprobe.summary(info))
    _print("")
    for problem in problems:
        _print(f"  PROBLEM  {problem}")
    for note in notes:
        _print(f"  note     {note}")
    if not problems:
        _print("\nReady.")
    return 1 if problems else 0


# =====================================================
# harness entry points
# =====================================================
def cmd_mcp(args):
    from .mcp_server import main as mcp_main
    return mcp_main()


def cmd_agent(args):
    from .agent.loop import main as agent_main
    return agent_main(model=args.model, base_url=args.base_url,
                      prompt=" ".join(args.prompt) if args.prompt else None,
                      max_steps=args.max_steps)


# =====================================================
# parser
# =====================================================
def _add_routing(parser):
    from .runtimes import RUNTIMES
    group = parser.add_argument_group(
        "routing", "Resolved from Hub metadata and engine catalogs; set these only "
                   "to settle a reported ambiguity.")
    group.add_argument("--task", choices=sorted(TASKS))
    group.add_argument("--engine", choices=list(RUNTIMES))
    group.add_argument("--family", help="engine catalog family for a custom or renamed repository")
    group.add_argument("--no-fork-hunt", action="store_true",
                       help="do not search engine forks and open PRs for unresolved sources")


def _add_release_options(parser, *, execution):
    text = parser.add_argument_group(
        "text-generation (llama.cpp)",
        "Files on disk at once = quantize workers + upload workers + queue depth "
        "(3 by default), or exactly 1 with --sequential.")
    text.add_argument(
        "--sequential", action="store_true", default=None,
        help="text: quantize -> upload -> delete, one at a time (smallest disk, slower). "
             "media: price and run components sequentially.")
    text.add_argument("--upload-workers", type=int, metavar="N",
                      help=f"concurrent uploads (default {config.UPLOAD_WORKERS}); "
                           "each costs one more quant of peak disk")
    text.add_argument("--quantize-workers", type=int, metavar="N",
                      help=f"concurrent llama-quantize processes (default {config.QUANTIZE_WORKERS})")
    text.add_argument("--queue-depth", type=int, metavar="N",
                      help=f"finished quants waiting for an uploader (default {config.QUEUE_DEPTH})")
    text.add_argument("--quantize-threads", type=int, metavar="N",
                      help="threads per llama-quantize; omitted, it uses every core")
    if execution:
        text.add_argument(
            "--xet", choices=transfer.MODES, default=None,
            help="xet transfer policy. 'auto' (the default) enables it only for "
                 f"downloads over {transfer.HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB and keeps "
                 "uploads on the faster plain-HTTP path.")
        text.add_argument("--keep-fork", action="store_true", default=None,
                          help="keep any fork checkout after the run")

    bundle = parser.add_argument_group("voice, media, retrieval and multimodal bundles")
    bundle.add_argument("--target", help="target repository (defaults to your namespace)")
    bundle.add_argument("--quant", action="append",
                        help="quant/precision policy; repeat for more (defaults to every native type)")
    bundle.add_argument("--output", help="save the exact reviewable plan as JSON")
    if execution:
        publication = bundle.add_mutually_exclusive_group()
        publication.add_argument("--publish", dest="publish", action="store_true", default=None,
                                 help="publish validated candidates (TTS still requires review)")
        publication.add_argument("--no-publish", dest="publish", action="store_false",
                                 help="build and score locally without writing to the Hub (default)")

    voice_group = parser.add_argument_group("text-to-speech / speech recognition")
    voice_group.add_argument("--language")
    voice_group.add_argument("--speaker", help="consented speaker-reference WAV when required")
    if execution:
        voice_group.add_argument("--pocket-language")
        voice_group.add_argument("--review", help="listening-review JSON (recorded reviews are found automatically)")
        voice_group.add_argument("--max-roundtrip-wer", type=float)
        voice_group.add_argument("--max-wer-regression", type=float)

    media_group = parser.add_argument_group("image / video (stable-diffusion.cpp)")
    media_group.add_argument("--component", action="append",
                             help="resolve an ambiguous member: role=path or role=org/repo::path")
    media_group.add_argument("--contract", help="reviewed component source contract JSON")
    media_group.add_argument("--profile", help="calibrated family/task quality profile JSON")
    media_group.add_argument("--init-image", nargs="?", const="fixtures/reference.png")
    media_group.add_argument("--end-image", nargs="?", const="fixtures/last.png")
    media_group.add_argument("--ref-image", nargs="?", const="fixtures/reference.png")
    media_group.add_argument("--ref-audio", nargs="?", const="fixtures/reference.wav")
    for setting in ("width", "height", "steps", "seed", "frames", "fps"):
        media_group.add_argument("--" + setting, type=int)
    media_group.add_argument("--sampler")
    media_group.add_argument("--scheduler")
    media_group.add_argument("--cfg-scale", type=float)
    media_group.add_argument("--compute-backend", help="auto (default), cpu, cuda, vulkan or metal")
    if execution:
        media_group.add_argument("--plan", help="execute this saved media plan unchanged")
        media_group.add_argument("--preserve-failed", action="store_true", default=None)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="aqx",
        description="AgentQuantix - trending Hugging Face models to GGUF quants.")
    from . import __version__
    parser.add_argument("--version", action="version", version=f"AgentQuantix {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="what this machine can do")
    probe.add_argument("--json", action="store_true")
    probe.add_argument("--no-disk", action="store_true",
                       help="skip the disk throughput measurement")
    probe.add_argument("--remeasure-disk", action="store_true",
                       help="re-run the disk measurement instead of using the cache")
    probe.set_defaults(func=cmd_probe)

    res = sub.add_parser("research",
                         help="steps 1-3: trending -> feasible -> ranked report")
    res.add_argument("--limit", type=int, default=None,
                     help=f"how many trending models to consider "
                          f"(default {config.TRENDING_LIMIT})")
    res.add_argument("--top", type=int, default=None,
                     help="only show the top N in the table")
    res.add_argument("--hide-blocked", action="store_true")
    res.add_argument(
        "--text-only", action="store_true",
        help="only rank text-generation candidates (previous behaviour)")
    res.add_argument("--no-disk", action="store_true")
    res.add_argument("--no-fork-hunt", action="store_true",
                     help="do not search GitHub for forks adding an "
                          "unsupported architecture")
    res.add_argument("--markdown", action="store_true",
                     help="also write the report as markdown")
    res.add_argument("--json", action="store_true")
    res.set_defaults(func=cmd_research)

    show = sub.add_parser("show", help="detail for a researched candidate, a recorded run, or any repo")
    show.add_argument("model", nargs="+")
    show.set_defaults(func=cmd_show)

    inspect = sub.add_parser(
        "inspect", help="read-only: resolve a model's task and engine and show its assessment")
    inspect.add_argument("model")
    _add_routing(inspect)
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(func=cmd_inspect)

    plan = sub.add_parser(
        "plan", help="read-only: the exact release plan for one or more models")
    plan.add_argument("model", nargs="+")
    _add_routing(plan)
    _add_release_options(plan, execution=False)
    plan.set_defaults(func=cmd_plan)

    run_cmd = sub.add_parser(
        "run", help="plan, ask, then build, validate and publish (any supported task)")
    run_cmd.add_argument("model", nargs="*",
                         help="models to run, by name from the last research "
                              "run or as any full repo id (org/model); omit "
                              "to choose interactively")
    run_cmd.add_argument("--resume", metavar="RUN_ID",
                         help="resume a recorded run with its frozen plan and runtime")
    run_cmd.add_argument("--yes", "-y", action="store_true",
                         help="skip the final model confirmation (never approves runtime updates)")
    run_cmd.add_argument("--dry-run", action="store_true",
                         help="show the plan and stop")
    _add_routing(run_cmd)
    _add_release_options(run_cmd, execution=True)
    run_cmd.set_defaults(func=cmd_run)

    review = sub.add_parser("review", help="record a human review of exact candidate artifacts")
    review.add_argument("run_id")
    review.add_argument("--quant", required=True)
    decision = review.add_mutually_exclusive_group(required=True)
    decision.add_argument("--accept", dest="accept", action="store_true", default=None)
    decision.add_argument("--reject", dest="accept", action="store_false")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes", default="")
    review.add_argument("--watched-video", action="store_true",
                        help="confirm video playback and extracted-frame inspection")
    review.add_argument("--listened-audio", action="store_true",
                        help="confirm listening to the generated audio")
    review.set_defaults(func=cmd_review)

    publish = sub.add_parser("publish", help="publish a validated, reviewed candidate")
    publish.add_argument("run_id")
    publish.add_argument("--quant", required=True)
    publish.add_argument("--yes", "-y", action="store_true")
    publish.set_defaults(func=cmd_publish)

    runs_cmd = sub.add_parser("runs", help="list recorded runs and their ids")
    runs_cmd.add_argument("--pipeline", choices=[TEXT, VOICE, MEDIA, RETRIEVAL, MULTIMODAL, VISION])
    runs_cmd.add_argument("--json", action="store_true")
    runs_cmd.set_defaults(func=cmd_runs)

    verify = sub.add_parser(
        "verify", help="check a published repo, a recorded run, or a local bundle manifest")
    verify.add_argument("target", nargs="+")
    verify.add_argument("--root", help="directory holding a local manifest's files")
    verify.set_defaults(func=cmd_verify)

    card_cmd = sub.add_parser("card", help="write or preview the model card")
    card_cmd.add_argument("repo")
    card_cmd.add_argument("--quant", help="bundle candidate to render (media runs)")
    card_cmd.add_argument("--dry-run", action="store_true",
                          help="print the card instead of publishing it")
    card_cmd.set_defaults(func=cmd_card)

    from . import runtime_cli
    runtime_cli.add_parser(sub)

    boot = sub.add_parser(
        "bootstrap",
        help="set up a fresh machine: check prerequisites, clone and build "
             "llama.cpp")
    boot.add_argument("--check-only", action="store_true",
                      help="report prerequisites and stop, without cloning "
                           "or building anything")
    boot.set_defaults(func=cmd_bootstrap)

    doctor = sub.add_parser("doctor", help="is this machine ready to run")
    doctor.set_defaults(func=cmd_doctor)

    mcp = sub.add_parser("mcp", help="run as an MCP server (stdio)")
    mcp.set_defaults(func=cmd_mcp)

    agent = sub.add_parser(
        "agent", help="drive the agent with an OpenRouter / OpenAI-compatible key")
    agent.add_argument("--model", default=os.getenv("AQX_MODEL"),
                       help="model id, e.g. anthropic/claude-opus-5. Must "
                            "support tool calling - everything this agent "
                            "does is tools.")
    agent.add_argument("--base-url", default=os.getenv("AQX_BASE_URL"),
                       help="API base URL (default: OpenRouter)")
    agent.add_argument("--max-steps", type=int, default=40)
    agent.add_argument("prompt", nargs="*",
                       help="what to do; omit for the standard trigger")
    agent.set_defaults(func=cmd_agent)

    return parser


def main(argv=None):
    # Carry a source checkout's history over to the user directory the first
    # time an installed build runs. Done here rather than at import so it is
    # one explicit action per invocation, not a side effect of importing config.
    if migrated := config.migrate_legacy_state():
        _print(f"Moved {', '.join(migrated)} to {config.AQX_HOME} "
               "(state now survives upgrades).")

    _ACTIVE_RESUME_IDS.clear()
    args = build_parser().parse_args(argv)
    try:
        from .runtime_control import execution_session
        with execution_session(interactive=sys.stdin.isatty()) as session:
            result = args.func(args)
            if result:
                session.mark_failed()
            return result
    except RuntimeDecisionRequired as error:
        _print(json.dumps({"runtime_decision": error.details}, indent=2))
        return 2
    except RuntimeErrorBase as error:
        _print(f"runtime: {error}")
        return 1
    except (ValueError, OSError) as error:
        _print(f"aqx: {error}")
        return 1
    except KeyboardInterrupt:
        if _ACTIVE_RESUME_IDS:
            commands = ", ".join(f"`aqx run --resume {run_id}`"
                                 for run_id in _ACTIVE_RESUME_IDS)
            _print(f"\nInterrupted. Resume with: {commands}")
        else:
            _print("\nInterrupted. Nothing is lost - use `aqx runs` to find a resume id.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
