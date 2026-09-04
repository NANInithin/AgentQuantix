"""Getting the BF16 GGUF that every quant is cut from.

Three ways in, in order of preference:

  1. Our own repo. A previous run may have uploaded the BF16 and then deleted
     the local copy before finishing. Refetching ours is cheaper than going
     back to the publisher and guarantees the exact bytes people can already
     download.

  2. The publisher's own GGUF repo. When they ship a BF16, downloading it
     costs ONE copy of the weights, against safetensors PLUS a conversion
     output PLUS the conversion runtime — and it sidesteps converter bugs
     entirely, since their file was built with whatever fork actually knows
     the architecture.

  3. Convert the safetensors ourselves. The general case, and the only option
     for a brand-new release. The source weights are deleted the moment the
     conversion succeeds — holding them through the whole sweep would double
     peak disk for no benefit.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import time

from huggingface_hub import hf_hub_download, snapshot_download

from .. import config, feasibility, transfer
from .build import run

GB = 1024 ** 3


def _size_gb(path: Path):
    return path.stat().st_size / GB if path.exists() else 0.0


def _record_download(kind, gigabytes, seconds):
    if gigabytes > 0.05 and seconds > 1:
        feasibility.record("downloads", kind=kind,
                           mbps=round(gigabytes * 1024 / seconds, 2),
                           gb=round(gigabytes, 2))


def ensure_bf16(job, llama_dir: Path, hub_files: set):
    """Put job.bf16_path on disk by whichever route is cheapest. Returns the path."""
    if job.bf16_path.exists():
        return job.bf16_path

    job.models_dir.mkdir(parents=True, exist_ok=True)

    # The BF16 is the one file in the whole run that can exceed
    # huggingface_hub's plain-HTTP download cap, so it is the one transfer that
    # may NEED xet. Every upload afterwards is faster without it, which is why
    # this is decided per download rather than once for the process.
    expected_gib = (job.assessment or {}).get("bf16_gb") or 0
    xet, why = transfer.for_download(size_gib=expected_gib)

    # ---- 1. our own repo ------------------------------------------------
    if job.bf16_path.name in hub_files:
        print(f"[{job.base_name}] BF16 missing locally - refetching ours...")
        started = time.time()
        with transfer.applied(xet, why):
            hf_hub_download(repo_id=job.target_repo,
                            filename=job.bf16_path.name, repo_type="model",
                            local_dir=str(job.models_dir), token=config.TOKEN)
        _record_download("bf16-refetch", _size_gb(job.bf16_path),
                         time.time() - started)
        return job.bf16_path

    # ---- 2. the publisher's GGUF ---------------------------------------
    if job.source_kind == "official-gguf" and job.official_gguf_file:
        print(f"[{job.base_name}] downloading the official BF16 GGUF from "
              f"{job.official_gguf_repo}...")
        started = time.time()
        with transfer.applied(xet, why):
            downloaded = hf_hub_download(
                repo_id=job.official_gguf_repo,
                filename=job.official_gguf_file, repo_type="model",
                local_dir=str(job.models_dir), token=config.TOKEN)
        # Publishers' filenames are frequently ROUNDED and do not match the
        # model repo name (K2-Horizon-3.7B-GGUF ships "K2-Horizon-4B-BF16.gguf").
        # Rename to our convention — metadata-only, instant.
        if Path(downloaded) != job.bf16_path:
            Path(downloaded).replace(job.bf16_path)
        _record_download("official-gguf", _size_gb(job.bf16_path),
                         time.time() - started)
        return job.bf16_path

    # ---- 3. convert the safetensors ------------------------------------
    job.source_dir.mkdir(parents=True, exist_ok=True)
    if not any(job.source_dir.iterdir()):
        print(f"[{job.base_name}] downloading source weights...")
        started = time.time()
        # Safetensors come in shards, each well under the cap, so the size
        # question is asked per shard and answers "leave it alone" — but the
        # policy is still consulted so an explicit AQX_XET=on/off is honoured.
        with transfer.applied(*transfer.for_download(size_gib=0)):
            snapshot_download(repo_id=job.repo_id,
                              local_dir=str(job.source_dir),
                              token=config.TOKEN)
        size = sum(f.stat().st_size for f in job.source_dir.rglob("*")
                   if f.is_file()) / GB
        _record_download("safetensors", size, time.time() - started)

    print(f"[{job.base_name}] converting to BF16 GGUF...")
    run(["python", llama_dir / "convert_hf_to_gguf.py", job.source_dir,
         "--outtype", "bf16", "--outfile", job.bf16_path])

    # The vision tower is a separate export and is NOT part of the quant
    # sweep — mmproj files stay at F16 because quantizing them is not
    # supported and would break the model's image path.
    if job.is_multimodal and not job.mmproj_path.exists():
        print(f"[{job.base_name}] exporting the vision tower (mmproj)...")
        try:
            run(["python", llama_dir / "convert_hf_to_gguf.py", job.source_dir,
                 "--mmproj", "--outtype", "f16", "--outfile", job.mmproj_path])
        except Exception as e:
            # A missing mmproj costs the image path, not the model. The text
            # quants are still perfectly good and worth publishing.
            print(f"[{job.base_name}] mmproj export failed ({e}) - continuing "
                  "with the language tower only.")

    # Source weights are dead the moment the BF16 exists. Deleting them here
    # rather than at the end of the run keeps the safetensors out of the sweep's
    # peak entirely — otherwise every quant in flight would sit on top of a full
    # copy of the original weights.
    try:
        shutil.rmtree(job.source_dir)
        print(f"[{job.base_name}] source weights deleted "
              "(the BF16 GGUF is the only source from here on).")
    except Exception as e:
        print(f"[{job.base_name}] could not delete the source weights: {e}")

    return job.bf16_path
