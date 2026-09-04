"""Step 5: check what actually landed on the Hub, then write the model card.

The verification is not a formality. A run can finish "successfully" with a
quant missing because llama-quantize gave up on it after three tries, or with a
file that uploaded at the wrong size because a retry raced. The card is
generated FROM the verified listing, so it can only ever describe files that
really exist, at the sizes they really are.
"""

from __future__ import annotations

import time

from huggingface_hub import HfApi

from . import config, hub

GB = 1024 ** 3

# Rough guidance, shown next to each quant so the card is useful to someone
# choosing a file rather than just a list of bytes.
QUANT_NOTES = {
    "BF16": "Full precision source. Every quant below is cut from this file.",
    "F16": "Full precision source.",
    "Q8_0": "Effectively lossless. Use when disk and RAM are not the constraint.",
    "Q6_K": "Near-lossless; the last stop before quality becomes measurable.",
    "Q5_K_M": "Very good quality, noticeably smaller than Q6_K.",
    "Q5_K_S": "Slightly smaller than Q5_K_M for a slight quality cost.",
    "Q4_K_M": "The usual default. Best quality-per-byte for most people.",
    "Q4_K_S": "A little smaller than Q4_K_M, a little worse.",
    "Q4_0": "Legacy round-to-nearest. Prefer Q4_K_M unless a runtime needs this.",
    "Q4_1": "Legacy. Prefer Q4_K_M.",
    "Q5_0": "Legacy. Prefer Q5_K_M.",
    "Q5_1": "Legacy. Prefer Q5_K_M.",
    "Q3_K_L": "Small, with real quality loss. Usable when RAM is tight.",
    "Q3_K_M": "Smaller again; noticeable degradation.",
    "Q3_K_S": "Aggressive. Prefer IQ3_M at a similar size.",
    "Q2_K": "Very small, heavily degraded. For experimentation.",
    "Q2_K_S": "Smaller than Q2_K, requires the imatrix.",
    "Q1_0": "Extreme. Included for completeness.",
    "Q2_0": "Extreme, group-64. Included for completeness.",
    "IQ4_XS": "Best sub-4.5bpw option; usually beats Q4_K_S at a smaller size.",
    "IQ4_NL": "Non-linear 4-bit; good on hardware without fast K-quant kernels.",
    "IQ3_M": "Strong at ~3.7bpw, clearly better than Q3_K_M.",
    "IQ3_S": "Slightly smaller than IQ3_M.",
    "IQ3_XS": "Aggressive but coherent.",
    "IQ3_XXS": "Very aggressive; imatrix carries it.",
    "IQ2_M": "The smallest size most people find usable.",
    "IQ2_S": "Below the usual usability line.",
    "IQ2_XS": "Experimental.",
    "IQ2_XXS": "Experimental.",
    "IQ1_M": "Extreme. Expect substantial degradation.",
    "IQ1_S": "Extreme. Expect substantial degradation.",
    "MXFP4_MOE": "MoE-only 4-bit microscaling format for the expert tensors.",
    "TQ1_0": "Ternary. Only meaningful for ternary-trained weights.",
    "TQ2_0": "Ternary. Only meaningful for ternary-trained weights.",
}

# Roughly ordered best-quality-first, so the card's table reads top to bottom
# as "biggest and best" down to "smallest and roughest".
QUALITY_ORDER = [
    "BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_M", "Q4_K_S", "IQ4_NL", "IQ4_XS", "Q4_1", "Q4_0", "MXFP4_MOE",
    "Q3_K_L", "Q3_K_M", "IQ3_M", "IQ3_S", "Q3_K_S", "IQ3_XS", "IQ3_XXS",
    "Q2_K", "IQ2_M", "Q2_K_S", "IQ2_S", "Q2_XS", "IQ2_XS", "IQ2_XXS",
    "Q2_0", "TQ2_0", "IQ1_M", "IQ1_S", "TQ1_0", "Q1_0",
]


def _order(quant):
    return (QUALITY_ORDER.index(quant) if quant in QUALITY_ORDER
            else len(QUALITY_ORDER))


def verify(job_or_repo, expected_quants=None, base_name=None):
    """Compare what is on the Hub against what the run was supposed to produce.

    Accepts a Job or a bare repo id. Returns a dict the caller can print and
    the card generator can consume — including the MISSING list, which is the
    part worth acting on.
    """
    if hasattr(job_or_repo, "target_repo"):
        repo_id = job_or_repo.target_repo
        # The FULL sweep, not just what this run built. A resume run that added
        # four quants to an existing repo should be checked against all thirty,
        # or the check would pass while twenty-six were still missing.
        expected_quants = (expected_quants or job_or_repo.all_quants
                           or job_or_repo.quants)
        base_name = base_name or job_or_repo.base_name
    else:
        repo_id = job_or_repo

    listing = hub.published_files(repo_id)
    if listing.get("error"):
        return {"repo_id": repo_id, "error": listing["error"], "files": []}

    files = []
    for entry in listing["files"]:
        if not entry["name"].lower().endswith(".gguf"):
            continue
        quant = hub.quant_of(entry["name"])
        files.append({
            "name": entry["name"],
            "quant": quant or ("mmproj" if entry["name"].startswith("mmproj")
                               else "?"),
            "bytes": entry["bytes"],
            "gb": round(entry["bytes"] / GB, 2),
            # A GGUF is never a few kilobytes. Anything that small is a failed
            # or truncated upload masquerading as a finished file.
            "suspect": entry["bytes"] < 1_000_000,
        })
    files.sort(key=lambda f: _order(f["quant"]))

    present = {f["quant"] for f in files}
    missing = ([q for q in expected_quants if q not in present]
               if expected_quants else [])

    return {
        "repo_id": repo_id,
        "url": listing["url"],
        "base_name": base_name or repo_id.split("/")[-1].removesuffix("-GGUF"),
        "files": files,
        "count": len(files),
        "total_gb": round(sum(f["bytes"] for f in files) / GB, 2),
        "missing": missing,
        "suspect": [f["name"] for f in files if f["suspect"]],
        "last_modified": listing["last_modified"],
        "error": None,
    }


def render(verification, job=None, assessment=None):
    """The README.md for a quant repo, built from the verified file listing."""
    assessment = assessment or (job.assessment if job else {}) or {}
    base_name = verification["base_name"]

    # The source repo, in descending order of certainty: the job that produced
    # these files, the assessment it came from, or a Hub lookup by name. When
    # none of those is sure, base_model is OMITTED rather than guessed — the
    # Hub builds its model tree from that field, and a wrong value points the
    # tree at a repo that does not exist.
    source_repo = (job.repo_id if job else assessment.get("repo_id")) \
        or hub.resolve_base_model(base_name)

    imatrix_source = (job.imatrix_source if job
                      else (assessment.get("imatrix") or {}).get("source", "BF16"))
    has_imatrix = any(f["quant"] in config.IMATRIX_REQUIRED
                      or f["quant"] in config.IMATRIX_GUIDED
                      for f in verification["files"])

    tags = ["gguf", "llama.cpp", "quantized"]
    if any(f["quant"] in config.IQ_QUANTS for f in verification["files"]):
        tags.append("imatrix")

    front = [
        "---",
        *([f"base_model: {source_repo}"] if source_repo else []),
        "library_name: gguf",
        "pipeline_tag: text-generation",
        "tags:",
        *[f"  - {t}" for t in tags],
        "---",
        "",
    ]

    origin = (f"[{source_repo}](https://huggingface.co/{source_repo})"
              if source_repo else f"`{base_name}`")
    body = [
        f"# {base_name} GGUF",
        "",
        f"GGUF quantizations of {origin}, covering {verification['count']} files "
        f"({verification['total_gb']:.1f} GB total).",
        "",
        "## Files",
        "",
        "| File | Quant | Size | Notes |",
        "|---|---|---:|---|",
    ]
    for entry in verification["files"]:
        note = QUANT_NOTES.get(entry["quant"], "")
        if entry["quant"] == "mmproj":
            note = ("Vision projector. Download alongside any quant below to "
                    "use the image input.")
        body.append(
            f"| [{entry['name']}](https://huggingface.co/{verification['repo_id']}"
            f"/blob/main/{entry['name']}) | `{entry['quant']}` "
            f"| {entry['gb']:.2f} GB | {note} |")

    body += [
        "",
        "## Which one should I download?",
        "",
        "Pick the largest file that leaves a couple of gigabytes of headroom "
        "on the device you will run it on — the model has to fit in RAM (or "
        "VRAM, if you are offloading) alongside the KV cache and the OS.",
        "",
        "- Plenty of memory: **Q6_K** or **Q8_0**.",
        "- The usual choice: **Q4_K_M**.",
        "- Tight on memory: **IQ4_XS**, then **IQ3_M**, then **IQ2_M**.",
        "- The `IQ*` files are imatrix-guided and generally beat a `Q*` file "
        "of similar size, at the cost of slightly slower inference on some "
        "hardware.",
        "",
    ]

    if has_imatrix:
        body += [
            "## Quantization details",
            "",
            f"- Importance matrix computed with `llama-imatrix` over "
            f"{config.CALIBRATION_MAX_LINES} rows of "
            f"[{config.WIKITEXT_REPO}]"
            f"(https://huggingface.co/datasets/{config.WIKITEXT_REPO}) "
            f"(`{config.WIKITEXT_CONFIG}`).",
            f"- The matrix was computed on the **{imatrix_source}** weights."
            + ("" if imatrix_source == "BF16" else
               "  The BF16 exceeded this machine's memory, so a smaller quant "
               "was used as the forward-pass source; the statistics therefore "
               "come from an approximation of the weights rather than the "
               "weights themselves."),
            "- K-quants below 6 bit and the whole `IQ` set are imatrix-guided. "
            "`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1` are legacy round-to-nearest and "
            "ignore it; `Q6_K`/`Q8_0` are near-lossless and do not need it.",
            "- All files are cut from the same BF16 GGUF, so differences "
            "between them are quantization only.",
            "",
        ]

    # `or [None]` rather than a .get() default: the default only applies when
    # the key is ABSENT, and the key is always present — it is an empty list
    # for every architecture upstream already supports. Indexing [0] into that
    # raised IndexError on the common case, which then surfaced as "the run
    # failed" long after the run had actually succeeded.
    fork = (job.fork if job else None) or (assessment.get("fork_leads") or [None])[0]
    if fork:
        body += [
            "## Runtime requirement",
            "",
            f"This architecture is not in upstream llama.cpp yet. These files "
            f"were built with [`{fork['repo']}`](https://github.com/"
            f"{fork['repo']}) at `{fork['ref']}`, and need that build (or "
            "upstream once it merges) to run.",
            "",
        ]

    body += [
        "## Usage",
        "",
        "```bash",
        f"llama-cli -hf {verification['repo_id']}:Q4_K_M -p \"Hello\"",
        "```",
        "",
        "Or download one file and point at it directly:",
        "",
        "```bash",
        f"huggingface-cli download {verification['repo_id']} "
        f"{base_name}-Q4_K_M.gguf --local-dir .",
        f"llama-cli -m {base_name}-Q4_K_M.gguf -p \"Hello\"",
        "```",
        "",
        "---",
        "",
        f"Quantized with [llama.cpp](https://github.com/ggml-org/llama.cpp) by "
        f"AgentQuantix on {time.strftime('%Y-%m-%d')}. "
        "Licensing follows the base model.",
        "",
    ]
    return "\n".join(front + body)


def publish(verification, job=None, assessment=None, dry_run=False):
    """Render the card and push it to the repo as README.md."""
    text = render(verification, job=job, assessment=assessment)
    if dry_run:
        return text

    api = HfApi(token=config.TOKEN)
    api.upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=verification["repo_id"],
        repo_type="model",
    )
    return text
