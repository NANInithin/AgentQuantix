"""Paths, environment and the quant tables.

AgentQuantix lives in AGI/AgentQuantix but WORKS in INF: llama.cpp, the .venv
and the temp scratch tree all stay where the existing pipeline put them, so a
run started by the agent is indistinguishable from one started by hand.

Everything here is deliberately a module-level constant read from the
environment once. The pipeline is long-running and unattended; a setting that
could change halfway through a six-hour run is a bug waiting to happen.
"""

from pathlib import Path
import os

# =====================================================
# PATHS
# =====================================================
# Where the agent's own files live (code, reports, run state, tuning history).
AQX_HOME = Path(os.getenv("AQX_HOME") or Path(__file__).resolve().parents[2])

# Where the work happens: the llama.cpp checkout and the temp scratch tree.
#
# Resolution order matters on a machine that already has a setup, and on one
# that has nothing:
#
#   1. AQX_WORK_ROOT / INF_ROOT, if set. Explicit always wins.
#   2. An existing ~/Documents/INF containing llama.cpp — the layout the
#      hand-written scripts use. Adopted automatically so the agent and those
#      scripts share one checkout and one scratch tree rather than each
#      cloning and building their own multi-gigabyte copy.
#   3. ~/.agentquantix — a fresh machine with no prior setup. Not under
#      Documents, because on Windows that is often OneDrive-synced and syncing
#      a llama.cpp build tree is a slow disaster.
def _default_work_root():
    legacy = Path.home() / "Documents" / "INF"
    if (legacy / "llama.cpp").exists():
        return legacy
    return Path.home() / ".agentquantix"


WORK_ROOT = Path(os.getenv("AQX_WORK_ROOT") or os.getenv("INF_ROOT")
                 or _default_work_root())

# Kept as an alias: the name INF_ROOT appears in the existing scripts and in
# every note written while this was Windows-only.
INF_ROOT = WORK_ROOT

UPSTREAM_LLAMA = INF_ROOT / "llama.cpp"
TEMP_DIR = INF_ROOT / "temp"

# Agent state. Kept out of INF so a manual temp wipe in the work tree never
# destroys the run history the estimator learns from.
STATE_DIR = AQX_HOME / "state"
REPORTS_DIR = AQX_HOME / "reports"
RUNS_DIR = STATE_DIR / "runs"
TIMING_HISTORY = STATE_DIR / "timings.json"

# Forks for architectures upstream llama.cpp does not support yet get cloned
# here, one directory per fork, and are reused across models needing the same
# one — the CUDA compile is by far the most expensive step in a fork build.
FORKS_DIR = TEMP_DIR / "aqx-forks"

# =====================================================
# ENVIRONMENT
# =====================================================
# The Hugging Face write token. MLE2 is the name the existing scripts use and
# is already exported in the user's shell; HF_TOKEN is the conventional
# fallback for anyone else running this.
TOKEN = os.getenv("MLE2") or os.getenv("HF_TOKEN")

def _default_namespace():
    """The Hub namespace to publish under.

    Asks the token who it belongs to rather than hardcoding a username: a
    hardcoded default would send someone else's first run at a repo they cannot
    write to, with a 403 as the only explanation. Cached at import because it
    is one network call and the answer cannot change mid-run.

    Falls back to the literal string below only when there is no token yet, so
    that `aqx research` still renders a plausible target repo before a token is
    configured.
    """
    if explicit := os.getenv("AQX_NAMESPACE"):
        return explicit
    if not TOKEN:
        return "your-username"
    try:
        from huggingface_hub import HfApi
        return HfApi(token=TOKEN).whoami()["name"]
    except Exception:
        return "your-username"


HF_NAMESPACE = _default_namespace()

# Delete each local .gguf the moment it is safely on the Hub. This is the
# single most important storage rule in the pipeline: with it, peak disk is the
# BF16 plus only the files in flight (see QUANTIZE_WORKERS below and
# feasibility.concurrent_quants) rather than growing with the sweep.
DELETE_AFTER_UPLOAD = os.getenv("KEEP_LOCAL_GGUF") != "1"

# huggingface_hub's constants.MAX_HTTP_DOWNLOAD_SIZE, in the GiB units this
# codebase measures files in. It is 50 DECIMAL GB upstream, which is 46.57
# GiB - comparing a GiB figure against a bare 50 quietly admits files that
# then fail to download. transfer.py owns the authoritative value.
HTTP_DOWNLOAD_LIMIT_GIB = 50 * 1000 * 1000 * 1000 / 1024 ** 3

# =====================================================
# CONCURRENCY
# =====================================================
# How the sweep is parallelised. These three numbers decide both how fast a run
# goes and how much disk it needs at peak — every extra file in flight is one
# more quant on disk — so they are the main lever for trading time against
# space. `aqx run` exposes all of them as flags.
#
# The defaults overlap one quantize with one upload and allow one finished
# quant to wait in between, which is the arrangement the hand-written scripts
# converged on: it hides the upload time behind the quantize time without
# letting more than three quant files exist at once.

# Concurrent llama-quantize processes. 1 is almost always right: llama-quantize
# already threads internally across every core, so a second process mostly buys
# contention. Worth raising only on a machine with far more cores than one
# process can saturate AND disk to spare.
QUANTIZE_WORKERS = int(os.getenv("AQX_QUANTIZE_WORKERS", "1"))

# Concurrent uploads. Raising this is the useful knob when a run is
# upload-bound, which most are — the Hub will happily take several streams, and
# each one costs one more quant held on disk until it lands.
UPLOAD_WORKERS = int(os.getenv("AQX_UPLOAD_WORKERS", "1"))

# Finished quants allowed to wait for an uploader. Pure disk-for-smoothness:
# a deeper queue means the quantizer never blocks, at one quant of disk each.
QUEUE_DEPTH = int(os.getenv("AQX_QUEUE_DEPTH", "1"))

# Threads passed to llama-quantize itself (its trailing [nthreads] argument).
# None lets it use every core, which is its own default. Set it to leave the
# machine usable while a sweep runs.
QUANTIZE_THREADS = (int(os.getenv("AQX_QUANTIZE_THREADS"))
                    if os.getenv("AQX_QUANTIZE_THREADS") else None)

# Quantize, upload, delete, repeat — never overlapping, exactly as
# scripts/Ornith-1.5-9B.py does it. One quant on disk at a time, which is the
# smallest footprint achievable, at the cost of the upload no longer hiding
# behind the next quantize.
SEQUENTIAL = os.getenv("AQX_SEQUENTIAL") == "1"

# =====================================================
# CALIBRATION
# =====================================================
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"

# Counts dataset ROWS, not written lines — wikitext has many blank/header
# rows, so 500 rows yields noticeably fewer lines of actual text.
CALIBRATION_MAX_LINES = int(os.getenv("AQX_CALIBRATION_ROWS", "500"))

# Keyed by row count so changing the number above builds a NEW file instead of
# silently reusing the old one. Shared by every model — the calibration text is
# model-independent — and deliberately placed in TEMP_DIR next to the file the
# existing K2-Horizon scripts already built, so the two reuse each other's work.
CALIBRATION_FILE = TEMP_DIR / f"calibration-{CALIBRATION_MAX_LINES}.txt"

# =====================================================
# QUANTS
# =====================================================
# Full coverage of llama.cpp's QUANT_OPTIONS table (tools/quantize/quantize.cpp),
# minus: the Q3_K/Q4_K/Q5_K aliases (duplicates of the _M variants) and
# F16/F32/COPY (not quantizations). MXFP4_MOE is MoE-only and is added per
# model rather than living in the default set.
STANDARD_QUANTS = [
    "Q1_0",                                   # 1.125 bpw
    "Q2_0",                                   # 2.25 bpw, group 64
    "Q2_K_S", "Q2_K",
    "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0",
]
IQ_QUANTS = [
    "IQ1_S", "IQ1_M",
    "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_XS", "IQ4_NL",
]

# TQ1_0 / TQ2_0 are ternarizations: they only produce a usable model from
# ternary-TRAINED weights (BitNet-style). Running them over ordinary bf16
# weights succeeds but yields garbage, so they are opt-in, not part of the set.
TERNARY_QUANTS = ["TQ1_0", "TQ2_0"] if os.getenv("BUILD_TERNARY") == "1" else []

# MoE-only. Added automatically when the config declares experts.
MOE_QUANTS = ["MXFP4_MOE"]

DEFAULT_QUANTS = STANDARD_QUANTS + IQ_QUANTS + TERNARY_QUANTS

# Everything below ~6 bit gains measurable quality from imatrix guidance.
# Q4_0/Q4_1/Q5_0/Q5_1 are legacy round-to-nearest and ignore the imatrix;
# Q6_K/Q8_0 are near-lossless. Those six are left unguided.
IMATRIX_GUIDED = {
    "Q1_0", "Q2_0", "Q2_K_S", "Q2_K",
    "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_K_S", "Q4_K_M",
    "Q5_K_S", "Q5_K_M",
}

# Quants that are impossible without imatrix data, so they are skipped rather
# than attempted when the imatrix step fails. llama.cpp's
# tensor_requires_imatrix() hard-aborts on IQ1_*, IQ2_XXS/XS/S and IQ3_XXS, and
# on the Q2_K tensors inside a Q2_K_S file. The remaining IQ mixtures can still
# assign one of those types to some tensor, so the whole IQ set is treated as
# requiring one.
IMATRIX_REQUIRED = set(IQ_QUANTS) | {"Q2_K_S"}

# Mixtures that can assign an imatrix-REQUIRING type to a tensor the imatrix
# has no rows for (see pipeline/imatrix.py gap_layers). Q4_K and above never do.
GAP_AFFECTED = set(IQ_QUANTS) | {
    "Q1_0", "Q2_0", "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
}
GAP_FALLBACK_TYPE = "q4_K"

# Effective bits-per-weight for each type, used ONLY to estimate output sizes
# and therefore upload time before anything is downloaded. These are the
# mixture averages llama.cpp reports on 7B-class models; real files land within
# a few percent, and the estimate is labelled as such everywhere it surfaces.
BPW = {
    "Q1_0": 1.13, "Q2_0": 2.25, "Q2_K_S": 2.60, "Q2_K": 2.96,
    "Q3_K_S": 3.44, "Q3_K_M": 3.74, "Q3_K_L": 4.03,
    "Q4_0": 4.55, "Q4_1": 5.03, "Q4_K_S": 4.58, "Q4_K_M": 4.85,
    "Q5_0": 5.54, "Q5_1": 6.01, "Q5_K_S": 5.52, "Q5_K_M": 5.69,
    "Q6_K": 6.56, "Q8_0": 8.50,
    "IQ1_S": 1.56, "IQ1_M": 1.75,
    "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.50, "IQ2_M": 2.70,
    "IQ3_XXS": 3.06, "IQ3_XS": 3.30, "IQ3_S": 3.44, "IQ3_M": 3.66,
    "IQ4_XS": 4.25, "IQ4_NL": 4.50,
    "MXFP4_MOE": 4.25,
    "TQ1_0": 1.69, "TQ2_0": 2.06,
    "BF16": 16.0, "F16": 16.0,
}

# =====================================================
# CANDIDATE FILTERS  (step 1)
# =====================================================
# Only these produce a text-generation GGUF llama.cpp can run. Everything else
# on the trending list — diffusion, TTS, embeddings, rerankers, ASR — is
# dropped before scoring.
ALLOWED_PIPELINE_TAGS = {
    "text-generation",
    "image-text-to-text",
    "any-to-any",
    "text2text-generation",
}

# Tags that mark a repo as somebody else's weights re-cut: a finetune, a merge,
# an adapter, or a quantization. "Base models" means the original release.
#
# Deliberately NOT here: "fp8" and "compressed-tensors". Large MoEs are now
# routinely TRAINED and released in FP8 by their authors — GLM-5.3 and
# DeepSeek-V4 both carry the fp8 tag on the original release — so treating the
# tag as a derivative marker throws away exactly the models most worth
# quantizing. Somebody else's FP8 requantization is caught by the
# base_model:quantized: relation instead, which is unambiguous.
DERIVATIVE_TAGS = {
    "lora", "peft", "adapter", "merge", "mergekit",
    "gguf", "awq", "gptq", "exl2", "exl3", "mlx", "bitsandbytes",
    "autoround", "quantized",
}

# Substrings in a repo id that give away a derivative even when the tags do
# not. Matched case-insensitively against the model name only, never the org —
# an org called "gguf-org" should not disqualify its original releases.
DERIVATIVE_NAME_HINTS = (
    "-gguf", "-awq", "-gptq", "-exl2", "-exl3", "-mlx", "-4bit", "-8bit",
    "-int4", "-int8", "-fp8", "-w4a16", "-w8a8", "-abliterated", "-uncensored",
    "-merge", "-lora", "-bnb-",
)

TRENDING_LIMIT = int(os.getenv("AQX_TRENDING_LIMIT", "100"))
