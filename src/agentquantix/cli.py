"""aqx — the AgentQuantix command line.

Every capability of the agent is a subcommand here, and the harness adapters
(MCP server, the built-in OpenRouter loop, the Claude Code skill) all drive
these same functions. That is deliberate: there is exactly one implementation
of "research the trending list" or "run this model", so the agent cannot
behave differently depending on which harness is holding the reins, and the
whole thing still works with no harness at all.

The two human-in-the-loop moments are the only interactive parts:
  * you decide when to trigger  -> `aqx research`
  * you approve the models      -> the prompt in `aqx run`, or `--yes`
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import (card, config, feasibility, report, research, sysprobe,
               transfer)
from .pipeline import run as run_mod, source as source_mod
from .pipeline.job import Job


def _print(*args):
    """Print without dying on a console that cannot encode the character.

    Windows terminals default to cp1252; a model name with a non-Latin
    character in it would otherwise abort the whole report with a
    UnicodeEncodeError three quarters of the way down.
    """
    text = " ".join(str(a) for a in args)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "ascii",
                          errors="replace").decode(sys.stdout.encoding or "ascii"))


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
        on_progress=None if args.json else (lambda m: _print(f"  .. {m}")))

    path = research.save(result)
    if args.markdown:
        markdown_path = path.with_suffix(".md")
        markdown_path.write_text(report.markdown(result), encoding="utf-8")

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


def cmd_show(args):
    result = research.load_latest()
    if not result:
        _print("No research on file. Run `aqx research` first.")
        return 1
    for name in args.model:
        assessment = research.find(result, name)
        if not assessment:
            _print(f"{name}: not in the last research run.")
            continue
        _print("")
        _print(report.detail(assessment))
    return 0


# =====================================================
# run
# =====================================================
def _select_interactively(result, preselected):
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


def cmd_run(args):
    # Set before anything reads the policy — the feasibility repricing below
    # asks whether uploads will use xet, and the answer has to be the one this
    # run will actually act on.
    if args.xet:
        os.environ["AQX_XET"] = args.xet

    result = research.load_latest()
    if not result and not args.model:
        _print("No research on file. Run `aqx research` first, or name a "
               "model directly: `aqx run org/model`.")
        return 1

    chosen = _select_interactively(result, args.model)
    if not chosen:
        _print("Nothing approved - stopping.")
        return 0

    options = run_mod.RunOptions(
        quantize_workers=args.quantize_workers,
        upload_workers=args.upload_workers,
        queue_depth=args.queue_depth,
        quantize_threads=args.quantize_threads,
        sequential=args.sequential)

    jobs = [Job.from_assessment(a) for a in chosen]

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

    free = (result.get("system") or {}).get("disk_free_gb") or 0
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

    outcome = run_mod.run_jobs(jobs, options=options,
                               clean_forks=not args.keep_fork)

    # Step 5 runs automatically: verify what landed, then write the card.
    # Guarded, because the quants are already published by this point — a
    # failure here is worth reporting but must not present a successful run as
    # a failed one, nor abandon the remaining models' verification.
    for job in jobs:
        result = outcome["jobs"].get(job.base_name, {})
        if result.get("skipped"):
            continue
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
                continue
            card.publish(verification, job=job)
            _print(f"model card published to {verification['url']}")
        except Exception as e:
            _print(f"  the quants are uploaded, but the model card step failed: "
                   f"{type(e).__name__}: {e}")
            _print(f"  retry it on its own with: aqx card {job.base_name}")
    return 0


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


def _resolve_repo(name):
    """Accept a full repo id, our target-repo shorthand, or a base model name."""
    if "/" in name:
        return name
    if name.endswith("-GGUF"):
        return f"{config.HF_NAMESPACE}/{name}"
    return f"{config.HF_NAMESPACE}/{name}-GGUF"


def cmd_verify(args):
    for name in args.repo:
        verification = card.verify(_resolve_repo(name))
        _print(f"\n{verification['repo_id']}")
        _print(_verification_text(verification))
    return 0


def cmd_card(args):
    repo_id = _resolve_repo(args.repo)
    job = None
    record = config.RUNS_DIR / f"{repo_id.split('/')[-1].removesuffix('-GGUF')}.json"
    if record.exists():
        job = Job.load(record)

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
# doctor
# =====================================================
def cmd_bootstrap(args):
    from . import bootstrap
    return bootstrap.run(build=not args.check_only)


def cmd_doctor(args):
    info = sysprobe.probe(measure_disk=False)
    problems, notes = [], []

    if not info["hf_token"]:
        problems.append("No Hugging Face token. Set HF_TOKEN, or run "
                        "`hf auth login`. It needs write permission.")
    if not info["llama"]["present"]:
        problems.append(f"No llama.cpp checkout at {config.UPSTREAM_LLAMA}. "
                        "Clone it there, or set INF_ROOT.")
    elif not info["llama"]["binaries"]:
        notes.append("llama.cpp is present but not built - the first run will "
                     "compile llama-quantize and llama-imatrix.")
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
def build_parser():
    parser = argparse.ArgumentParser(
        prog="aqx",
        description="AgentQuantix - trending Hugging Face models to GGUF quants.")
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
    res.add_argument("--no-disk", action="store_true")
    res.add_argument("--no-fork-hunt", action="store_true",
                     help="do not search GitHub for forks adding an "
                          "unsupported architecture")
    res.add_argument("--markdown", action="store_true",
                     help="also write the report as markdown")
    res.add_argument("--json", action="store_true")
    res.set_defaults(func=cmd_research)

    show = sub.add_parser("show", help="full detail for one candidate")
    show.add_argument("model", nargs="+")
    show.set_defaults(func=cmd_show)

    run_cmd = sub.add_parser("run", help="step 4: quantize and upload (asks first)")
    run_cmd.add_argument("model", nargs="*",
                         help="models to run, by name from the last research "
                              "run or as any full repo id (org/model); omit "
                              "to choose interactively")
    run_cmd.add_argument("--yes", "-y", action="store_true",
                         help="skip the final confirmation")
    run_cmd.add_argument("--dry-run", action="store_true",
                         help="show the plan and stop")
    run_cmd.add_argument("--keep-fork", action="store_true",
                         help="keep any fork checkout after the run")

    concurrency = run_cmd.add_argument_group(
        "concurrency",
        "How much to overlap, and therefore how much disk the sweep needs. "
        "Files on disk at once = quantize workers + upload workers + queue "
        "depth (3 by default), or exactly 1 with --sequential.")
    concurrency.add_argument(
        "--sequential", action="store_true",
        help="quantize -> upload -> delete, one at a time, no background "
             "uploader (as scripts/Ornith-1.5-9B.py does it). Smallest "
             "possible disk footprint; slower, because uploads stop hiding "
             "behind the next quantize.")
    concurrency.add_argument(
        "--upload-workers", type=int, default=config.UPLOAD_WORKERS,
        metavar="N",
        help=f"concurrent uploads (default {config.UPLOAD_WORKERS}). Raise "
             "this when a run is upload-bound; each stream costs one more "
             "quant held on disk.")
    concurrency.add_argument(
        "--quantize-workers", type=int, default=config.QUANTIZE_WORKERS,
        metavar="N",
        help=f"concurrent llama-quantize processes (default "
             f"{config.QUANTIZE_WORKERS}). Rarely worth raising - "
             "llama-quantize already threads across every core.")
    concurrency.add_argument(
        "--queue-depth", type=int, default=config.QUEUE_DEPTH, metavar="N",
        help=f"finished quants allowed to wait for an uploader (default "
             f"{config.QUEUE_DEPTH}). Costs one quant of disk each.")
    concurrency.add_argument(
        "--quantize-threads", type=int, default=config.QUANTIZE_THREADS,
        metavar="N",
        help="threads per llama-quantize process (its trailing [nthreads] "
             "argument). Omitted, it uses every core; set it to keep the "
             "machine usable while a sweep runs.")
    run_cmd.add_argument(
        "--xet", choices=transfer.MODES, default=None,
        help="xet transfer policy. 'auto' (the default) enables it only for "
             f"downloads over {transfer.HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB, "
             "where huggingface_hub requires it, and keeps uploads on the "
             "much faster plain-HTTP path. 'on'/'off' pin it for every "
             "transfer.")
    run_cmd.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="step 5: check what is on the Hub")
    verify.add_argument("repo", nargs="+")
    verify.set_defaults(func=cmd_verify)

    card_cmd = sub.add_parser("card", help="step 5: write the model card")
    card_cmd.add_argument("repo")
    card_cmd.add_argument("--dry-run", action="store_true",
                          help="print the card instead of publishing it")
    card_cmd.set_defaults(func=cmd_card)

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

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _print("\nInterrupted. Nothing is lost - re-run to resume.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
