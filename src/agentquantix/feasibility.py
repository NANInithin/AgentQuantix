"""Can THIS machine quantize THIS model, how long will it take, and what breaks.

Step 2/3 of the agent, and the only place where a number is allowed to be a
guess. Everything here is an estimate, so three rules apply throughout:

  * Estimates are derived from measured machine properties (disk throughput,
    free RAM, free VRAM, free disk) rather than from constants baked in on
    somebody else's laptop.
  * Every estimate that has been observed for real is corrected by the
    observation. state/timings.json accumulates actual durations and
    transfer rates, and the model reads them back on the next run.
  * The output distinguishes BLOCKED (this cannot run, here is the wall) from
    WARN (this will run, but you should know something) from OK. A wrong OK
    costs a wasted night; a wrong BLOCKED costs a missed release.

The storage discipline is not an estimate but a design constraint, and it is
what makes the peak-disk figure small: every quant is deleted the moment it is
safely on the Hub, and the upload queue is bounded, so peak disk is the BF16
plus only the files in flight — see concurrent_quants() — no matter how many
quants the sweep produces.
"""

from __future__ import annotations

import json

from . import config, transfer

GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Fallback constants, used only until state/timings.json has a real number.
# ---------------------------------------------------------------------------

# Plain-HTTP Hub transfer, one connection. huggingface_hub uploads a single
# file's LFS parts sequentially, so this is a per-file ceiling that no amount
# of concurrency inside one file can lift.
DEFAULT_NET_MBPS = 20.0

# Xet. This was 2.0 for a long time, from a single measurement that was never
# reproduced, and it made the estimator quote 1.2 h for uploads that finished
# in under ten minutes. Xet uploads chunks over many connections and sends
# fewer bytes (dedup + compression), so it is FASTER than plain HTTP, not
# slower - 64 MB/s effective against 11.65 MB/s on the box where the old
# number was believed.
#
# Deliberately NOT set to that 64: it is one machine on one link, and the
# whole bug being fixed here was a local observation frozen into a constant.
# This is a floor that says "no reason to assume worse than plain HTTP";
# _learned_mbps replaces it with a real median as soon as samples exist, and
# it medians xet and non-xet separately so the two never contaminate.
XET_UPLOAD_MBPS = 20.0

# llama-quantize is a streaming job: read the BF16, write the output. It never
# reaches raw disk throughput, hence the efficiency factor.
DISK_EFFICIENCY = 0.55

# IQ types run an iterative codebook search per block instead of a rounding
# step, so they are CPU-bound rather than IO-bound and cost several times more
# per gigabyte than a K-quant.
IQ_SLOWDOWN = 3.0

# convert_hf_to_gguf.py is single-threaded Python over the whole tensor set.
CONVERT_SLOWDOWN = 2.5

# llama-imatrix has to run a real forward pass over the calibration text.
# Anchored on two observed runs: a 13.62 GB Q2_K that fit in fast memory took
# ~15 min (1.1 min/GB), and a 74.92 GB BF16 on the same 22 GB box took 1.5-3 h
# (~1.8-2.4 min/GB) because every chunk paged most of the file back off disk.
IMATRIX_MIN_PER_GB = 1.1
IMATRIX_THRASH_FACTOR = 1.5

# A fork that has to be compiled from scratch with CUDA. Observed ~11 min on
# this 24-core box; it is a fixed cost per fork, not per model.
FORK_BUILD_MIN = 11.0


# =====================================================
# LEARNED RATES
# =====================================================
def load_history():
    """Observed durations and rates from previous runs, or an empty record."""
    try:
        return json.loads(config.TIMING_HISTORY.read_text())
    except Exception:
        return {"uploads": [], "quantize": [], "imatrix": [], "downloads": []}


def _learned_mbps(history, key, default, match=None):
    """Median observed MB/s for a transfer kind.

    Median rather than mean: one stalled upload that eventually retried would
    drag a mean down for months, while the median shrugs it off.

    `match` filters to comparable observations. It matters for uploads, where
    xet and plain HTTP differ by roughly an order of magnitude: averaging the
    two produces a number that describes neither, and samples recorded before
    the mode was tracked are excluded rather than guessed at. Falling back to
    the constant and flagging the estimate as assumed is more honest than
    quoting a contaminated median.
    """
    rates = [s["mbps"] for s in history.get(key, [])
             if s.get("mbps", 0) > 0 and (match is None or match(s))]
    if not rates:
        return default, False
    rates.sort()
    return rates[len(rates) // 2], True


def record(kind, /, **fields):
    """Append one observation to the timing history. Never raises.

    Called from the pipeline as work completes, so estimates improve with use
    instead of staying frozen at the constants above.

    `kind` is POSITIONAL-ONLY, and that matters: **fields is arbitrary
    observation data, so any caller wanting to record a field of its own
    called "kind" would otherwise collide with the parameter and raise
    "record() got multiple values for argument 'kind'" — which is exactly
    what happened to every download the pipeline tried to time.
    """
    try:
        history = load_history()
        history.setdefault(kind, []).append(fields)
        # Keep the file small and recent — 500 samples per kind is far more
        # than a median needs, and old samples describe an older machine.
        history[kind] = history[kind][-500:]
        config.TIMING_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        config.TIMING_HISTORY.write_text(json.dumps(history, indent=2))
    except Exception:
        pass


# =====================================================
# CONCURRENCY -> DISK
# =====================================================
def concurrent_quants(quantize_workers=None, upload_workers=None,
                      queue_depth=None, sequential=False):
    """How many quant files can exist on disk at the same time.

    This is the number that sets peak disk, and it follows directly from how
    the pipeline is wired rather than from a guess:

        one being WRITTEN per quantize worker
      + one in FLIGHT per upload worker
      + however many are QUEUED waiting for an uploader

    With the defaults (1/1/1) that is three, not two — the queue holds one
    while another uploads and a third is being cut. In sequential mode the
    quantize-upload-delete cycle never overlaps, so exactly one exists at a
    time, which is the cheapest possible disk footprint and the reason the
    mode is worth having.
    """
    if sequential:
        return 1
    return (int(quantize_workers or config.QUANTIZE_WORKERS)
            + int(upload_workers or config.UPLOAD_WORKERS)
            + int(queue_depth or config.QUEUE_DEPTH))


def peak_disk_for(assessment, **options):
    """Recompute an assessment's peak disk under different run options.

    The research step prices every candidate at the default concurrency,
    because it has no idea what flags a run will eventually use. This lets
    `aqx run --sequential` re-answer the question honestly at the moment the
    flags are known — which is exactly when a model that was blocked on disk
    may turn out to fit after all.
    """
    sizes = sorted((assessment.get("quant_sizes_gb") or {}).values(),
                   reverse=True)
    imatrix = assessment.get("imatrix") or {}
    imatrix_disk = (0.0 if imatrix.get("source") == "BF16"
                    else imatrix.get("source_gb", 0.0))
    sweep = (assessment["bf16_gb"]
             + sum(sizes[:concurrent_quants(**options)])
             + imatrix_disk)
    return round(max(assessment.get("peak_convert_gb", 0.0), sweep), 1)


# =====================================================
# SIZES
# =====================================================
def quant_list(candidate):
    """The quant sweep this model should get.

    The full set, exactly as the existing scripts produce it, plus MXFP4_MOE
    when the config declares experts — it is MoE-only and quantizing a dense
    model with it is meaningless.
    """
    quants = list(config.DEFAULT_QUANTS)
    if candidate.n_experts and candidate.n_experts > 1:
        quants += [q for q in config.MOE_QUANTS if q not in quants]
    return quants


def quant_size_gb(params, quant):
    """Estimated output size for one quant type.

    bits-per-weight times parameters, plus ~2% for GGUF metadata, the token
    embeddings that stay at higher precision, and alignment padding.
    """
    bpw = config.BPW.get(quant, 4.5)
    return params * bpw / 8 / GB * 1.02


def bf16_size_gb(candidate):
    """Size of the BF16 GGUF every quant is cut from.

    Prefer the publisher's real file size when they ship one — the estimate
    below is good to a few percent, but a measured byte count is exact, and
    for the >50 GB cap check the difference decides whether the run is
    possible at all.
    """
    if candidate.official_bf16_bytes:
        return candidate.official_bf16_bytes / GB
    if candidate.params:
        return candidate.params * 2 / GB * 1.02
    return candidate.source_bytes / GB or 0.0


# =====================================================
# IMATRIX STRATEGY
# =====================================================
def imatrix_plan(candidate, sysinfo, bf16_gb):
    """What llama-imatrix should run its forward pass over, and with what -ngl.

    The BF16 is always the best answer for quality: activation statistics
    should come from the real weights, not from an approximation of them. It
    is only abandoned when it does not fit in fast memory (available RAM +
    free VRAM), because then every chunk pages most of the file back off disk
    and a 15-minute job becomes a three-hour one.

    The fallback ladder is Q8_0 (near-lossless, so the statistics are still
    essentially the real ones) then Q4_K_M then Q2_K, taking the largest that
    fits. This generalises the hand-written choice in the K2-Horizon script,
    which picked Q2_K for a 74.92 GB BF16 on a 22 GB box.
    """
    fast = max(sysinfo.get("fast_memory_gb") or 0, 1.0)
    # 0.85 leaves room for the KV cache, the compute buffers and the OS.
    budget = fast * 0.85

    source, source_gb, extra_download_gb = "BF16", bf16_gb, 0.0
    if bf16_gb > budget:
        for fallback in ("Q8_0", "Q4_K_M", "Q2_K"):
            size = quant_size_gb(candidate.params or 0, fallback)
            if size <= budget:
                source, source_gb = fallback, size
                # The fallback is cut from the BF16 locally, so it costs disk
                # but not bandwidth — unless it is already published, in which
                # case the pipeline fetches it back instead of re-cutting it.
                extra_download_gb = 0.0
                break
        else:
            # Nothing fits. Run on the smallest anyway and accept the paging;
            # an imatrix from a thrashing Q2_K still beats no IQ quants at all.
            source = "Q2_K"
            source_gb = quant_size_gb(candidate.params or 0, "Q2_K")

    # How many layers fit on the GPU. Embeddings and the output head are large
    # and are not offloaded per-layer, so they come off the top before the
    # remainder is divided by the layer count.
    ngl = 0
    layers = candidate.n_layers or 0
    vram = sysinfo.get("vram_free_gb") or 0
    if layers and vram > 1.5:
        bytes_per_param = source_gb * GB / max(candidate.params or 1, 1)
        embed_params = ((candidate.vocab_size or 0)
                        * (candidate.hidden_size or 0)
                        * (1 if candidate.tied_embeddings else 2))
        embed_gb = embed_params * bytes_per_param / GB
        per_layer_gb = max((source_gb - embed_gb) / layers, 1e-6)
        # 1.5 GB held back for the KV cache and compute buffers.
        ngl = int(max(0, (vram - 1.5) // per_layer_gb))
        ngl = min(ngl, layers if source_gb > vram else 99)

    return {"source": source, "source_gb": round(source_gb, 2),
            "ngl": int(ngl), "extra_download_gb": extra_download_gb,
            "fits_fast_memory": source_gb <= budget}


# =====================================================
# THE ESTIMATE
# =====================================================
def assess(candidate, sysinfo, arch_ok, arch_detail, fork_leads=None,
           history=None):
    """Full feasibility + time + disk verdict for one candidate.

    Returns a plain dict; the report renderer and the MCP tools both consume
    it, and it is what gets written into the run record when a model is
    approved, so a run is reproducible from it alone.
    """
    history = history if history is not None else load_history()
    fork_leads = fork_leads or []
    params = candidate.params or 0
    all_quants = quant_list(candidate)

    # Work already on the Hub under our own namespace is not work. Estimating
    # a full sweep for a repo that is 26/30 finished would misprice it by a
    # factor of seven and bury a nearly-free job at the bottom of the ranking.
    published = set(candidate.published_quants or set())
    quants = [q for q in all_quants if q not in published]
    bf16_published = any(q in published for q in ("BF16", "F16"))
    # Counted against the sweep only. The published set also contains the BF16
    # source and any mmproj, which are not sweep members — including them gives
    # nonsense like "30 of 29 quants already published".
    published_in_sweep = published & set(all_quants)

    bf16_gb = bf16_size_gb(candidate)
    sizes = {q: quant_size_gb(params, q) for q in quants}
    # The BF16 is only an upload cost if it is not already up there.
    total_upload_gb = sum(sizes.values()) + (0.0 if bf16_published else bf16_gb)

    # ---- source strategy ------------------------------------------------
    # A publisher-supplied BF16 GGUF costs one copy of the weights; converting
    # safetensors costs the download PLUS the conversion output PLUS the
    # conversion runtime, and risks a converter that does not know the arch.
    if bf16_published:
        # Our own BF16 is already on the Hub. Refetching it costs one download
        # and skips the conversion entirely — and it is the exact file the
        # existing quants were cut from, so the new ones match them.
        source_kind = "refetch-ours"
        download_gb = bf16_gb
        convert_gb = 0.0
    elif candidate.official_bf16_file:
        source_kind = "official-gguf"
        download_gb = bf16_gb
        convert_gb = 0.0
    else:
        source_kind = "convert"
        download_gb = candidate.source_bytes / GB or bf16_gb
        convert_gb = bf16_gb

    imatrix = imatrix_plan(candidate, sysinfo, bf16_gb)

    # ---- disk -----------------------------------------------------------
    # Two moments compete for the peak. During conversion, the safetensors and
    # the BF16 coexist. During the sweep, the BF16, any imatrix source quant,
    # and however many quants the pipeline's concurrency allows on disk at once
    # all coexist — see concurrent_quants() for where that number comes from.
    # Priced at the default concurrency here; `aqx run --sequential` recomputes
    # it with peak_disk_for() once the actual flags are known.
    concurrency = concurrent_quants()
    largest = sum(sorted(sizes.values(), reverse=True)[:concurrency])
    imatrix_disk = 0.0 if imatrix["source"] == "BF16" else imatrix["source_gb"]
    peak_convert = (download_gb + bf16_gb) if source_kind == "convert" else bf16_gb
    peak_sweep = bf16_gb + largest + imatrix_disk
    peak_disk_gb = max(peak_convert, peak_sweep)

    free_disk = sysinfo.get("disk_free_gb") or 0

    # ---- rates ----------------------------------------------------------
    disk_gbs = sysinfo.get("disk_gbs") or 1.0
    effective_gbs = max(disk_gbs * DISK_EFFICIENCY, 0.05)

    backends = sysinfo.get("hub_backends") or {}
    # What the RUN will actually do, not what the environment happens to say
    # right now: the pipeline pins the upload policy itself, so the estimate
    # has to ask the policy the same question the pipeline will, or it prices
    # a transfer mode the run is not going to use.
    upload_uses_xet, _ = transfer.for_upload()
    up_default = XET_UPLOAD_MBPS if upload_uses_xet else DEFAULT_NET_MBPS
    up_mbps, up_learned = _learned_mbps(
        history, "uploads", up_default,
        match=lambda sample: sample.get("xet") is upload_uses_xet)
    down_mbps, down_learned = _learned_mbps(history, "downloads", DEFAULT_NET_MBPS)

    def transfer_hours(gigabytes, mbps):
        return gigabytes * 1024 / max(mbps, 0.1) / 3600

    # ---- time -----------------------------------------------------------
    download_h = transfer_hours(download_gb, down_mbps)
    convert_h = (convert_gb + download_gb) / effective_gbs / 3600 * CONVERT_SLOWDOWN \
        if source_kind == "convert" else 0.0

    thrash = 1.0
    if not imatrix["fits_fast_memory"]:
        fast = max(sysinfo.get("fast_memory_gb") or 1, 1)
        thrash = 1 + IMATRIX_THRASH_FACTOR * max(0.0, imatrix["source_gb"] / fast - 1)
    imatrix_h = imatrix["source_gb"] * IMATRIX_MIN_PER_GB * thrash / 60
    if imatrix["source"] != "BF16":
        # The fallback source has to be cut from the BF16 first.
        imatrix_h += (bf16_gb + imatrix["source_gb"]) / effective_gbs / 3600

    quantize_h = 0.0
    for quant, size in sizes.items():
        factor = IQ_SLOWDOWN if quant.startswith("IQ") else 1.0
        quantize_h += (bf16_gb + size) / effective_gbs / 3600 * factor

    upload_h = transfer_hours(total_upload_gb, up_mbps)

    # The uploader runs on its own thread one quant behind the quantizer, so
    # the sweep costs max(quantize, upload) rather than their sum. This is the
    # single biggest saving in the pipeline and usually makes the run
    # upload-bound, not compute-bound.
    sweep_h = max(quantize_h, upload_h)
    fork_h = (FORK_BUILD_MIN / 60) if (not arch_ok and fork_leads) else 0.0
    total_h = download_h + convert_h + imatrix_h + sweep_h + fork_h

    # ---- verdict --------------------------------------------------------
    blockers, warnings = [], []

    if not sysinfo.get("hf_token"):
        blockers.append("no Hugging Face token - set HF_TOKEN or run "
                        "`hf auth login`")
    if not params:
        blockers.append("parameter count unknown - cannot size anything")
    if peak_disk_gb > free_disk * 0.9:
        blockers.append(
            f"needs {peak_disk_gb:.0f} GB peak disk, only {free_disk:.0f} GB free")
    if transfer.exceeds_http_limit(size_gib=bf16_gb):
        # Note this asks whether xet COULD be used, not whether it is on right
        # now — the run turns it on for exactly this download. Only a genuinely
        # missing backend, or an explicit AQX_XET=off, is a blocker.
        can_download, why = transfer.for_download(size_gib=bf16_gb)
        if not can_download:
            blockers.append(
                f"BF16 is {bf16_gb:.0f} GiB, over huggingface_hub's "
                f"{transfer.HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB plain-HTTP "
                f"download limit, and {why}")
        else:
            warnings.append(
                f"BF16 is {bf16_gb:.0f} GiB - xet will be enabled for that one "
                "download and disabled again for the uploads")
    if not arch_ok and not fork_leads:
        blockers.append(f"{arch_detail}, and no fork or PR found that adds it")
    if not (sysinfo.get("llama") or {}).get("binaries") and arch_ok:
        warnings.append("llama.cpp is not built here - the run compiles it first")

    if not arch_ok and fork_leads:
        best = fork_leads[0]
        warnings.append(
            f"{arch_detail}; would build {best['repo']}@{best['ref']} "
            f"({best['confidence']} confidence, +{FORK_BUILD_MIN:.0f} min)")
    if not imatrix["fits_fast_memory"]:
        warnings.append(
            f"BF16 ({bf16_gb:.0f} GB) exceeds fast memory "
            f"({sysinfo.get('fast_memory_gb')} GB) - imatrix runs on "
            f"{imatrix['source']} instead, costing some guidance quality")
    if candidate.gated:
        warnings.append("gated repo - the token must already have access")
    if candidate.community_ggufs:
        warnings.append(
            f"{len(candidate.community_ggufs)} community GGUF repo(s) exist "
            f"already, e.g. {candidate.community_ggufs[0]}")
    if candidate.is_multimodal:
        warnings.append(
            "multimodal - the vision tower needs a separate --mmproj export, "
            "and the quants only cover the language tower")
    if upload_uses_xet:
        warnings.append(
            "AQX_XET is pinned on, so uploads run over xet at roughly a tenth "
            "of plain-HTTP speed; AQX_XET=auto would use it only where it is "
            "actually required")

    if published:
        warnings.insert(0, (
            f"{len(published_in_sweep)} of {len(all_quants)} quants already "
            f"published in "
            f"{candidate.target_repo}"
            + (" - nothing left to build" if not quants
               else f" - only the remaining {len(quants)} would be built")))

    # "done" is its own verdict, not a blocker: nothing is wrong, there is
    # simply no work. Keeping it distinct means the report can show it without
    # it looking like a failure, and the ranking can push it out of the way.
    if published and not quants:
        verdict = "done"
        # A finished repo has no work, so it has no cost. Leaving the computed
        # figures in place would show a DONE row asking for 24 minutes and
        # 26 GB of disk for a run that will never happen — and would let it
        # carry blockers that stopped mattering the moment the work finished.
        blockers = []
        download_h = convert_h = imatrix_h = quantize_h = upload_h = 0.0
        sweep_h = fork_h = total_h = 0.0
        download_gb = total_upload_gb = peak_disk_gb = 0.0
    else:
        verdict = "blocked" if blockers else ("warn" if warnings else "ok")

    return {
        "repo_id": candidate.repo_id,
        "url": candidate.url,
        "target_repo": candidate.target_repo,
        "rank": candidate.rank,
        "params": params,
        "params_b": round(params / 1e9, 2) if params else None,
        "architecture": (candidate.architectures or [None])[0],
        "model_type": candidate.model_type,
        "n_experts": candidate.n_experts,
        "n_layers": candidate.n_layers,
        # Carried explicitly rather than left to be re-derived from the warning
        # text: the job builder needs it to decide whether to export an mmproj,
        # and string-matching a human-readable warning for that is a bug
        # waiting for someone to reword the warning.
        "is_multimodal": bool(candidate.is_multimodal),
        "gated": bool(candidate.gated),
        "verdict": verdict,
        "blockers": blockers,
        "warnings": warnings,
        "arch_ok": arch_ok,
        "arch_detail": arch_detail,
        "fork_leads": fork_leads,
        "source_kind": source_kind,
        "official_gguf": candidate.official_gguf,
        "official_bf16_file": candidate.official_bf16_file,
        "community_ggufs": candidate.community_ggufs,
        # `quants` is the work REMAINING — it is what a run would build and
        # what every estimate above is priced on. `all_quants` is the intended
        # sweep, which is what a finished repo gets verified against.
        "quants": quants,
        "all_quants": all_quants,
        "published_quants": sorted(published),
        "published_count": len(published_in_sweep),
        "bf16_published": bf16_published,
        "our_repo_exists": bool(candidate.published_files),
        "quant_sizes_gb": {q: round(s, 2) for q, s in sizes.items()},
        "bf16_gb": round(bf16_gb, 2),
        "download_gb": round(download_gb, 2),
        "upload_gb": round(total_upload_gb, 2),
        "peak_disk_gb": round(peak_disk_gb, 1),
        # Kept so peak_disk_for() can re-answer under different run flags: the
        # conversion moment does not shrink when the sweep is made sequential.
        "peak_convert_gb": round(peak_convert, 1),
        "concurrent_quants": concurrency,
        "free_disk_gb": round(free_disk, 1),
        "imatrix": imatrix,
        "hours": {
            "download": round(download_h, 2),
            "convert": round(convert_h, 2),
            "imatrix": round(imatrix_h, 2),
            "quantize": round(quantize_h, 2),
            "upload": round(upload_h, 2),
            "sweep_overlapped": round(sweep_h, 2),
            "fork_build": round(fork_h, 2),
            "total": round(total_h, 1),
        },
        "rates": {
            "disk_gbs": disk_gbs,
            "upload_mbps": round(up_mbps, 1),
            "upload_learned": up_learned,
            "download_mbps": round(down_mbps, 1),
            "download_learned": down_learned,
        },
    }


def rank_key(assessment):
    """Sort order for the report: runnable first, then cheapest, then trendiest.

    Cheap-and-runnable at the top is deliberate. The models most likely to be
    approved are the ones that finish tonight, and a 40-hour MoE at position 3
    buries them. A part-finished repo therefore rises on its own, without a
    special case: only its remaining quants are priced, so it is cheap.

    Finished models sort below everything runnable but above the blocked ones —
    present, so you can see the work is done, but out of the way.
    """
    verdict_order = {"ok": 0, "warn": 1, "done": 2, "blocked": 3}
    return (verdict_order.get(assessment["verdict"], 3),
            assessment["hours"]["total"],
            assessment["rank"])
