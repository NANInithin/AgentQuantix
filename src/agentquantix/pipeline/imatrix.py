"""The importance matrix: calibration text, the forward pass, and its blind spots.

An imatrix records how much each weight actually matters, measured by running
real text through the model. Quantizing below ~6 bits with one is markedly
better than without, and the IQ types cannot be produced without one at all.

Two subtleties this module exists to handle:

  * WHAT to run it on. Ideally the BF16 itself. When the BF16 dwarfs available
    memory, the forward pass pages most of the file off disk for every chunk
    and a 15-minute job becomes a three-hour one — so a smaller quant is cut
    first and used as the source. The statistics then come from an
    approximation of the weights, which is a real quality cost, taken
    knowingly.

  * WHERE it has no data. llama-imatrix only records activations for tensors
    that actually execute. Blocks that never run during normal generation —
    multi-token-prediction / NextN heads, most obviously — end up with no rows,
    and llama-quantize hard-aborts when an imatrix-requiring type lands on one.
    Those blocks are found and pinned to a K-quant instead.
"""

from __future__ import annotations

from pathlib import Path
import re
import time

from huggingface_hub import hf_hub_download

from .. import config, feasibility
from .build import run


def ensure_calibration():
    """The calibration text, built once and shared by every model.

    wikitext-2 is the conventional choice: general English prose, no
    domain skew, and small enough that the imatrix pass is dominated by the
    model's forward cost rather than by the amount of text.
    """
    if config.CALIBRATION_FILE.exists():
        return config.CALIBRATION_FILE

    from datasets import load_dataset

    print(f"Building calibration file from {config.WIKITEXT_REPO} "
          f"({config.WIKITEXT_CONFIG})...")
    config.CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(config.WIKITEXT_REPO, config.WIKITEXT_CONFIG,
                           split="train")

    with config.CALIBRATION_FILE.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(dataset):
            if index >= config.CALIBRATION_MAX_LINES:
                break
            text = example["text"].strip()
            if text:
                handle.write(text + "\n")
    return config.CALIBRATION_FILE


def imatrix_source(job, llama_quantize, hub_files):
    """The GGUF llama-imatrix should run its forward pass over.

    Normally the BF16 itself. When job.imatrix_source names a quant (because
    the BF16 dwarfs memory), that is used instead — fetched back from our own
    repo if it is already published, otherwise cut from the BF16. Note the
    quant is only ever an INPUT here; whether it also gets uploaded is the
    quant loop's business.
    """
    if job.imatrix_source in ("BF16", None):
        return job.bf16_path

    quant = job.imatrix_source
    source = job.quant_path(quant)
    if source.exists():
        return source

    if source.name in hub_files:
        # Already published — downloading it beats re-quantizing from a BF16
        # several times its size, and uses the exact bytes people can see.
        print(f"[{job.base_name}] fetching {quant} back for the imatrix pass...")
        hf_hub_download(repo_id=job.target_repo, filename=source.name,
                        repo_type="model", local_dir=str(job.models_dir),
                        token=config.TOKEN)
        return source

    print(f"[{job.base_name}] cutting {quant} to compute the imatrix on...")
    run([llama_quantize, job.bf16_path, source, quant])
    return source


def gap_layers(bf16_path: Path, imatrix_path: Path):
    """Block indices in the BF16 that the imatrix has no entries for.

    Cheap and self-checking: on a model with no unexecuted blocks it finds
    nothing and costs one pass over two file headers.
    """
    try:
        import gguf
    except ImportError:
        print("gguf not installed - skipping the imatrix coverage check.")
        return []

    block_re = re.compile(r"blk\.(\d+)\.")

    def blocks(path):
        return {int(m.group(1))
                for tensor in gguf.GGUFReader(str(path)).tensors
                if (m := block_re.match(tensor.name))}

    try:
        return sorted(blocks(bf16_path) - blocks(imatrix_path))
    except Exception as e:
        print(f"Could not compare imatrix coverage ({e}) - assuming none missing.")
        return []


def build(job, llama_quantize, llama_imatrix, hub_files):
    """Produce job.imatrix_path. Returns (ok, gap_args).

    A failure here is NOT fatal. llama-quantize does not need an imatrix for
    the standard types, so the sweep continues without one — only the IQ set
    and Q2_K_S get skipped, and the caller records why.
    """
    calibration = ensure_calibration()

    # Stale if the calibration text is newer than the imatrix built from it.
    fresh = (job.imatrix_path.exists()
             and job.imatrix_path.stat().st_mtime >= calibration.stat().st_mtime)

    if not fresh:
        started = time.time()
        try:
            source = imatrix_source(job, llama_quantize, hub_files)
            cmd = [llama_imatrix, "-m", source, "-f", calibration,
                   "-o", job.imatrix_path, "-ngl", job.imatrix_ngl]
            if job.imatrix_chunks:
                cmd += ["--chunks", job.imatrix_chunks]
            run(cmd)
            feasibility.record(
                "imatrix", model=job.base_name, source=job.imatrix_source,
                gb=round(source.stat().st_size / 1024 ** 3, 2),
                minutes=round((time.time() - started) / 60, 1))
        except Exception as e:
            # Remove a partial .dat: llama-quantize would read it and produce
            # a quant guided by half a matrix, which is worse than none.
            job.imatrix_path.unlink(missing_ok=True)
            print(f"[{job.base_name}] IMATRIX FAILED ({e}) - continuing with "
                  "the quants that do not need one.")
            return False, [], str(e)

    gap_args = []
    for layer in gap_layers(job.bf16_path, job.imatrix_path):
        gap_args += ["--tensor-type",
                     rf"blk\.{layer}\.={config.GAP_FALLBACK_TYPE}"]
    if gap_args:
        print(f"[{job.base_name}] imatrix gaps found - forcing those blocks to "
              f"{config.GAP_FALLBACK_TYPE} in low-bit quants.")
    return True, gap_args, None
