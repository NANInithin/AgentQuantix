"""The quantization run: BF16 in, a published repo full of GGUFs out.

This is the pipeline the existing hand-written scripts implement, generalised
to any model. Four properties are load-bearing and none of them are optional:

  * BOUNDED DISK. Each quant is deleted the instant it is safely on the Hub,
    and the upload queue is bounded, so the number of quant files on disk at
    once is fixed by the run options rather than by how many the sweep
    produces. At the defaults that is three (one being cut, one queued, one
    uploading); at --sequential it is one. Either way a 29-quant run on a 27B
    model peaks near its BF16 size, not near the 500 GB the outputs sum to.

  * MAXIMAL OVERLAP, BY DEFAULT. Uploading a multi-GB quant takes minutes
    during which the CPU is idle; cutting the next one takes minutes during
    which the network is idle. A background uploader turns (quantize + upload)
    per quant into max(quantize, upload) across the run, which is usually
    hours. RunOptions can trade that back for disk when a model needs it.

  * RESUMABLE. status.json plus the Hub file listing mean an interrupted run
    redoes nothing. Re-running the same job is always safe and never uploads
    a file twice.

  * NEVER ABORT THE BATCH. One quant type that this build cannot produce must
    not cost the other twenty-eight, and one model dying must not cost the
    rest of the night. Failures are collected and reported, not raised.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time

from huggingface_hub import HfApi

from .. import config, feasibility, transfer
from . import build as build_mod, imatrix as imatrix_mod, source as source_mod
from .job import Job, Status

GB = 1024 ** 3


@dataclass
class RunOptions:
    """How hard to push, and how much disk to spend doing it.

    The defaults reproduce the arrangement the hand-written scripts use. The
    two interesting departures:

      * `sequential=True` is scripts/Ornith-1.5-9B.py exactly — quantize,
        upload, delete, repeat, with no background uploader. One quant on disk
        at a time. Slower, because the upload no longer hides behind the next
        quantize, but it is the only way to run a model whose quants will not
        fit three-at-a-time.

      * `upload_workers > 1` is the lever for an upload-bound run, which most
        are. Each extra stream costs one more quant held on disk.
    """

    quantize_workers: int = 1
    upload_workers: int = 1
    queue_depth: int = 1
    quantize_threads: int | None = None    # llama-quantize's [nthreads]
    sequential: bool = False

    @classmethod
    def from_config(cls):
        return cls(quantize_workers=config.QUANTIZE_WORKERS,
                   upload_workers=config.UPLOAD_WORKERS,
                   queue_depth=config.QUEUE_DEPTH,
                   quantize_threads=config.QUANTIZE_THREADS,
                   sequential=config.SEQUENTIAL)

    @property
    def concurrent_quants(self):
        """Quant files that can be on disk at once. Drives the peak estimate."""
        return feasibility.concurrent_quants(
            quantize_workers=self.quantize_workers,
            upload_workers=self.upload_workers,
            queue_depth=self.queue_depth,
            sequential=self.sequential)

    def describe(self):
        if self.sequential:
            base = "sequential (quantize -> upload -> delete, one at a time)"
        else:
            base = (f"{self.quantize_workers} quantize / {self.upload_workers} "
                    f"upload worker(s), queue depth {self.queue_depth}")
        threads = ("" if self.quantize_threads is None
                   else f", {self.quantize_threads} threads per quantize")
        return f"{base}{threads}; up to {self.concurrent_quants} quant(s) on disk"


def _hub_files(api, repo_id):
    try:
        return set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception:
        return set()          # repo does not exist yet (first run)


def process(job: Job, llama_dir, llama_quantize, llama_imatrix,
            options: RunOptions | None = None, on_event=None):
    """Run one job to completion. Returns a result dict; never raises for a
    quant-level problem."""
    options = options or RunOptions.from_config()

    def emit(kind, **fields):
        if on_event:
            on_event({"job": job.base_name, "kind": kind, **fields})

    for directory in (job.work_dir, job.models_dir, job.status_file.parent):
        directory.mkdir(parents=True, exist_ok=True)

    # Guards status.json and hub_files, both touched by the upload thread.
    state_lock = threading.Lock()
    status = Status(job.status_file, lock=state_lock)

    api = HfApi(token=config.TOKEN)
    hub_files = _hub_files(api, job.target_repo)

    wanted = [f"{job.base_name}-{q}.gguf" for q in job.quants]
    if job.bf16_path.name in hub_files and all(f in hub_files for f in wanted):
        print(f"[{job.base_name}] all quants already uploaded - skipping.")
        emit("skipped", reason="already complete")
        return {"job": job.base_name, "failures": [], "uploaded": [],
                "skipped": True}

    started = time.time()
    failures = []

    # ---------- BF16 source ----------
    # This is the only transfer that may REQUIRE xet (it is the only file that
    # can exceed the plain-HTTP download cap), and it happens while this is
    # still the only thread doing any transferring.
    source_mod.ensure_bf16(job, llama_dir, hub_files)
    emit("bf16-ready", gb=round(job.bf16_path.stat().st_size / GB, 2))

    # Everything from here on is uploads, and xet is an order of magnitude
    # slower at those. Pinned ONCE, before any uploader thread exists: the
    # setting is process-global, so a per-upload context manager would have one
    # worker's restore landing mid-transfer in another.
    transfer.pin(*transfer.for_upload())

    # ---------- repo + upload helper ----------
    # Created and the BF16 queued BEFORE the imatrix step: the BF16 is already
    # downloaded and valid, and it must not be held hostage to a later failure.
    api.create_repo(repo_id=job.target_repo, repo_type="model", exist_ok=True)
    with state_lock:
        hub_files |= _hub_files(api, job.target_repo)

    def already_up(name):
        with state_lock:
            if name in hub_files:
                return True
        return name in status.load()["uploaded"]

    def upload(path, keep=False):
        """Upload once, with retries. Returns True if the file is on the Hub.

        Retries matter for an unattended run: a single transient network error
        on a multi-GB upload would otherwise abort the whole batch hours in.
        The local file is only deleted once the upload has actually succeeded.
        """
        if not already_up(path.name):
            gigabytes = path.stat().st_size / GB
            for attempt in range(1, 5):
                try:
                    print(f"[{job.base_name}] uploading {path.name} "
                          f"({gigabytes:.2f} GB)...", flush=True)
                    began = time.time()
                    api.upload_file(path_or_fileobj=str(path),
                                    path_in_repo=path.name,
                                    repo_id=job.target_repo, repo_type="model")
                    elapsed = time.time() - began
                    if elapsed > 1 and gigabytes > 0.05:
                        # The transfer mode is recorded with the rate: xet and
                        # plain HTTP differ by roughly 10x, so a median that
                        # mixes them describes neither.
                        feasibility.record(
                            "uploads", mbps=round(gigabytes * 1024 / elapsed, 2),
                            gb=round(gigabytes, 2), file=path.name,
                            xet=transfer.enabled())
                    break
                except Exception as e:
                    if attempt == 4:
                        print(f"[{job.base_name}] UPLOAD FAILED {path.name}: "
                              f"{type(e).__name__}: {e} - keeping the local "
                              "file so a later run can retry.")
                        return False
                    wait = 60 * attempt
                    print(f"[{job.base_name}] upload of {path.name} failed "
                          f"({type(e).__name__}), retrying in {wait}s...")
                    time.sleep(wait)
            status.mark("uploaded", path.name)
            with state_lock:
                hub_files.add(path.name)
            emit("uploaded", file=path.name)

        if config.DELETE_AFTER_UPLOAD and not keep and path.exists():
            path.unlink()
            print(f"[{job.base_name}] removed local {path.name}")
        return True

    # ---------- uploaders ----------
    # The queue depth bounds the disk cost: at most `queue_depth` finished
    # quants wait behind the `upload_workers` in flight, so the sweep never
    # holds more than options.concurrent_quants files at once.
    #
    # In sequential mode there is no queue and no thread at all — send() just
    # uploads inline, which is scripts/Ornith-1.5-9B.py's behaviour exactly.
    upload_failures = []
    upload_queue: queue.Queue | None = None
    uploaders: list[threading.Thread] = []

    def note_failure(path, why):
        upload_failures.append((path.stem.split("-")[-1], why))

    if not options.sequential:
        upload_queue = queue.Queue(maxsize=max(1, options.queue_depth))

        def upload_worker():
            while (item := upload_queue.get()) is not None:
                path, keep = item
                try:
                    if not upload(path, keep):
                        note_failure(path, "upload failed")
                except Exception as e:              # never let the thread die
                    note_failure(path, f"upload: {type(e).__name__}: {e}")
                finally:
                    upload_queue.task_done()
            upload_queue.task_done()

        for _ in range(max(1, options.upload_workers)):
            thread = threading.Thread(target=upload_worker, daemon=True)
            thread.start()
            uploaders.append(thread)

    def send(path, keep=False):
        """Hand a finished file to the uploader, or upload it here and now."""
        if upload_queue is None:
            try:
                if not upload(path, keep):
                    note_failure(path, "upload failed")
            except Exception as e:
                note_failure(path, f"upload: {type(e).__name__}: {e}")
        else:
            upload_queue.put((path, keep))

    def finish_uploads():
        if upload_queue is None:
            return
        # One sentinel per worker; each consumes exactly one and stops.
        for _ in uploaders:
            upload_queue.put(None)
        for thread in uploaders:
            thread.join()

    # The local BF16 is KEPT — every quant below is cut from it. Handed over
    # before the imatrix pass so that step starts now rather than after a
    # multi-GB upload (a no-op when it is already published).
    send(job.bf16_path, keep=True)
    if job.mmproj_path.exists():
        send(job.mmproj_path, keep=True)

    # ---------- imatrix ----------
    imatrix_ok, gap_args, imatrix_error = imatrix_mod.build(
        job, llama_quantize, llama_imatrix, hub_files)
    if not imatrix_ok:
        failures.append(("<imatrix>", imatrix_error or "failed"))
    emit("imatrix", ok=imatrix_ok, source=job.imatrix_source)

    # ---------- quants ----------
    def build_one(quant):
        """Cut one quant and hand it to the uploader. Never raises."""
        out = job.quant_path(quant)
        if out.name in hub_files or out.name in status.load()["uploaded"]:
            return

        if not imatrix_ok and quant in config.IMATRIX_REQUIRED:
            print(f"[{job.base_name}] skipping {quant} - needs an imatrix.")
            failures.append((quant, "skipped: no imatrix"))
            return

        if not out.exists():
            extra = []
            if imatrix_ok and (quant in config.IQ_QUANTS
                               or quant in config.IMATRIX_GUIDED):
                extra += ["--imatrix", str(job.imatrix_path)]
                if quant in config.GAP_AFFECTED:
                    extra += gap_args

            # llama-quantize takes its thread count as a trailing positional
            # argument, AFTER the type. Left off, it uses every core, which is
            # its own default and usually what you want.
            trailing = ([] if options.quantize_threads is None
                        else [options.quantize_threads])

            # Retry: a killed llama-quantize leaves a truncated .gguf that the
            # exists() guard above would happily upload next run, so the
            # partial is removed on every failure. The usual cause is transient
            # memory pressure rather than a bad quant type.
            began = time.time()
            for attempt in range(1, 4):
                try:
                    build_mod.run([llama_quantize, *extra, job.bf16_path,
                                   out, quant, *trailing])
                    break
                except Exception as e:
                    out.unlink(missing_ok=True)
                    if attempt == 3:
                        # Do NOT abort the batch. One quant type this build
                        # cannot produce (an aggressive IQ1_* on an unlucky
                        # tensor shape, say) must not cost the other 28.
                        print(f"[{job.base_name}] GIVING UP on {quant}: {e}")
                        failures.append((quant, f"quantize: {e}"))
                        break
                    print(f"{quant} attempt {attempt} failed, retrying...")
                    time.sleep(60 * attempt)

            # Every path above breaks; a missing file here means we gave up on
            # this quant and already recorded why.
            if not out.exists():
                return
            elapsed = time.time() - began
            if elapsed > 1:
                feasibility.record(
                    "quantize", quant=quant, model=job.base_name,
                    minutes=round(elapsed / 60, 2),
                    gb=round(out.stat().st_size / GB, 2))
            status.mark("generated", out.name)
            emit("quantized", quant=quant,
                 gb=round(out.stat().st_size / GB, 2),
                 minutes=round(elapsed / 60, 1))

        # Hand off and start the next quantize immediately. Blocks only when
        # the upload queue is full, which is exactly the disk bound working.
        send(out, keep=False)

    workers = 1 if options.sequential else max(1, options.quantize_workers)
    if workers == 1:
        for quant in job.quants:
            build_one(quant)
    else:
        # A shared work queue rather than a slice per thread: quant types
        # differ enormously in cost (an IQ1_S is several times an equivalent
        # K-quant), so static partitioning would leave threads idle while one
        # ground through the expensive half.
        pending: queue.Queue = queue.Queue()
        for quant in job.quants:
            pending.put(quant)

        def quantize_worker():
            while True:
                try:
                    quant = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    build_one(quant)
                except Exception as e:              # never let the thread die
                    failures.append((quant, f"quantize: {type(e).__name__}: {e}"))

        threads = [threading.Thread(target=quantize_worker, daemon=True)
                   for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    # Wait for the uploaders to drain BEFORE deleting anything — the BF16 and
    # the last quant may still be in flight.
    finish_uploads()
    failures += upload_failures

    # ---------- cleanup ----------
    if config.DELETE_AFTER_UPLOAD:
        for path in (job.bf16_path, job.mmproj_path):
            if path.exists() and path.name in hub_files:
                path.unlink()
                print(f"[{job.base_name}] local {path.name} deleted.")

        # A quant used only as the imatrix input is never touched by the loop
        # above (it is already on the Hub), so it would otherwise sit on disk
        # forever.
        if job.imatrix_source not in ("BF16", None):
            leftover = job.quant_path(job.imatrix_source)
            if leftover.exists() and leftover.name in hub_files:
                leftover.unlink()
                print(f"[{job.base_name}] removed local {leftover.name} "
                      "(imatrix input, already published).")

    hours = (time.time() - started) / 3600
    if failures:
        print(f"\n[{job.base_name}] finished in {hours:.1f} h WITH "
              f"{len(failures)} FAILURE(S):")
        for quant, why in failures:
            print(f"    {quant}: {why}")
    else:
        print(f"\n[{job.base_name}] done in {hours:.1f} h - all quants "
              "generated and uploaded.")
    emit("done", hours=round(hours, 2), failures=len(failures))

    return {"job": job.base_name, "repo": job.target_repo,
            "hours": round(hours, 2), "failures": failures,
            "uploaded": sorted(hub_files), "skipped": False}


def run_jobs(jobs, options: RunOptions | None = None, on_event=None,
             clean_forks=True):
    """Run several jobs, sharing one build per fork. Returns a report dict.

    Ordering is the caller's; the CLI sorts smallest-first so any pipeline
    problem surfaces on the cheapest model rather than eight hours into the
    largest one.
    """
    options = options or RunOptions.from_config()
    if not config.TOKEN:
        raise SystemExit("No Hugging Face token - set HF_TOKEN, or run "
                         "`hf auth login`.")
    print(f"Sweep concurrency: {options.describe()}")
    print()

    started = time.time()
    report, forks_used = {}, {}

    for job in jobs:
        job.save()
        try:
            # ensure_tools is keyed on the fork, so two models needing the same
            # branch share one checkout and one CUDA compile.
            llama_dir, quantize, imatrix = build_mod.ensure_tools(job.fork)
            if job.fork:
                forks_used[str(llama_dir)] = llama_dir

            # A fork's quant table can differ from upstream's — re-check
            # against the build that will actually run, not the one the
            # research step read.
            from ..archsupport import supported_quants
            available = supported_quants(llama_dir)
            if available:
                missing = [q for q in job.quants if q not in available]
                if missing:
                    print(f"[{job.base_name}] not offered by this llama.cpp "
                          f"build, skipping: {missing}")
                    job.quants = [q for q in job.quants if q in available]

            report[job.base_name] = process(
                job, llama_dir, quantize, imatrix, options=options,
                on_event=on_event)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # One model dying (a bad download, an imatrix OOM) must not cost
            # the others — this is built to be left running unattended.
            print(f"\n[{job.base_name}] ABORTED: {type(e).__name__}: {e}")
            print("Continuing with the next model; re-run to resume this one.\n")
            report[job.base_name] = {
                "job": job.base_name, "repo": job.target_repo,
                "failures": [("<model>", f"{type(e).__name__}: {e}")],
                "uploaded": [], "skipped": False}

    # Only remove a fork once every job that used it finished cleanly —
    # otherwise the resume run would pay for the rebuild.
    incomplete = {k: v for k, v in report.items() if v.get("failures")}
    if clean_forks and not incomplete:
        for llama_dir in forks_used.values():
            build_mod.cleanup_fork(llama_dir)
    elif clean_forks and forks_used:
        print("Keeping the fork checkouts - some work did not finish, and a "
              "resume run would otherwise have to rebuild them.")

    total_h = (time.time() - started) / 3600
    print("\n===================================")
    print(f"FINISHED in {total_h:.1f} h")
    for name, result in report.items():
        problems = result.get("failures") or []
        print(f"  {name:<32} {'OK' if not problems else f'{len(problems)} FAILED'}")
        for quant, why in problems:
            print(f"      {quant}: {why}")
    if incomplete:
        print("\nRe-run the same command to retry only what is missing "
              "(status.json + the Hub listing make it resumable).")
    print("===================================\n")

    return {"hours": round(total_h, 2), "jobs": report,
            "incomplete": list(incomplete)}
