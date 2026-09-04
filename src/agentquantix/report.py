"""Rendering a research result for a human at an approval gate.

The report has one job: let the user say yes or no to each model in a few
seconds. That means the things a no is usually based on — it will not fit, it
will take three days, somebody already published it, it needs a fork — have to
be visible without scrolling into a detail section.

Two renderings, same data. `table()` is what a terminal or a chat gets;
`markdown()` is the durable record written next to the JSON.
"""

from __future__ import annotations

from . import sysprobe

VERDICT_MARK = {"ok": "OK  ", "warn": "WARN", "done": "DONE", "blocked": "BLOCK"}


def _hours(value):
    if value is None:
        return "?"
    if value < 1:
        return f"{value * 60:.0f}m"
    if value < 48:
        return f"{value:.1f}h"
    return f"{value / 24:.1f}d"


def _size(gigabytes):
    if gigabytes is None:
        return "?"
    return f"{gigabytes:.0f}G" if gigabytes >= 10 else f"{gigabytes:.1f}G"


def table(result, limit=None, include_blocked=True):
    """Fixed-width table, best candidates first."""
    rows = [a for a in result["assessments"]
            if include_blocked or a["verdict"] != "blocked"]
    if limit:
        rows = rows[:limit]

    header = (f"{'':<5} {'MODEL':<44} {'PARAMS':>7} {'BF16':>6} "
              f"{'PEAK':>6} {'UPLOAD':>7} {'EST':>6} {'QUANTS':>7}  NOTE")
    lines = [header, "-" * len(header)]

    for assessment in rows:
        # A blocker is the most important thing to say about a row — except on
        # a finished one, where it is a leftover observation about work that no
        # longer needs doing, and the useful line is "all 29 already published".
        note = ""
        if assessment["blockers"] and assessment["verdict"] != "done":
            note = assessment["blockers"][0]
        elif assessment["warnings"]:
            note = assessment["warnings"][0]
        if len(note) > 60:
            note = note[:57] + "..."

        # "12/30" reads as "twelve left to build of thirty" — the count that
        # decides whether a row is worth hours or minutes.
        todo = len(assessment["quants"])
        total = len(assessment.get("all_quants") or assessment["quants"])
        quant_text = f"{todo}/{total}" if todo != total else str(total)

        params = (f"{assessment['params_b']:.1f}B"
                  if assessment["params_b"] else "?")
        if assessment["n_experts"]:
            params += "*"                       # MoE: stored, not active
        lines.append(
            f"{VERDICT_MARK[assessment['verdict']]:<5} "
            f"{assessment['repo_id'][:44]:<44} "
            f"{params:>7} "
            f"{_size(assessment['bf16_gb']):>6} "
            f"{_size(assessment['peak_disk_gb']):>6} "
            f"{_size(assessment['upload_gb']):>7} "
            f"{_hours(assessment['hours']['total']):>6} "
            f"{quant_text:>7}  {note}")

    counts = {}
    for assessment in result["assessments"]:
        counts[assessment["verdict"]] = counts.get(assessment["verdict"], 0) + 1
    lines.append("-" * len(header))
    lines.append(
        f"{result['trending_count']} trending -> {result['kept_count']} base "
        f"models -> {counts.get('ok', 0)} ok, {counts.get('warn', 0)} with "
        f"warnings, {counts.get('done', 0)} already done, "
        f"{counts.get('blocked', 0)} blocked."
        + ("  (* = MoE, stored params)" if any(
            a["n_experts"] for a in result["assessments"]) else ""))
    return "\n".join(lines)


def detail(assessment):
    """Everything known about one candidate, for a 'tell me more' follow-up."""
    hours = assessment["hours"]
    imatrix = assessment["imatrix"]
    lines = [
        f"{assessment['repo_id']}   (trending #{assessment['rank']})",
        f"  {assessment['url']}",
        f"  publish to      {assessment['target_repo']}",
        f"  architecture    {assessment['architecture']} "
        f"({assessment['model_type']})"
        + (f", {assessment['n_experts']} experts" if assessment["n_experts"] else "")
        + (f", {assessment['n_layers']} layers" if assessment["n_layers"] else ""),
        f"  arch support    {assessment['arch_detail']}",
        f"  source          {assessment['source_kind']}"
        + (f" ({assessment['official_gguf']} / {assessment['official_bf16_file']})"
           if assessment["official_bf16_file"] else ""),
        f"  quants          {len(assessment['quants'])} to build: "
        f"{', '.join(assessment['quants']) or '(none - all published)'}",
        *([f"  already up      {assessment['published_count']} in "
           f"{assessment['target_repo']}: "
           f"{', '.join(assessment['published_quants'])}"]
          if assessment.get("published_count") else []),
        f"  imatrix         on {imatrix['source']} "
        f"({imatrix['source_gb']} GB), -ngl {imatrix['ngl']}"
        + ("" if imatrix["fits_fast_memory"] else "  [does not fit fast memory]"),
        f"  disk            {assessment['bf16_gb']} GB BF16, "
        f"{assessment['peak_disk_gb']} GB peak, "
        f"{assessment['free_disk_gb']} GB free",
        f"  transfer        {assessment['download_gb']} GB down, "
        f"{assessment['upload_gb']} GB up "
        f"@ {assessment['rates']['upload_mbps']} MB/s"
        + ("" if assessment["rates"]["upload_learned"] else " (assumed)"),
        f"  time            {_hours(hours['total'])} total = "
        f"{_hours(hours['download'])} download + "
        f"{_hours(hours['convert'])} convert + "
        f"{_hours(hours['imatrix'])} imatrix + "
        f"{_hours(hours['sweep_overlapped'])} sweep"
        + (f" + {_hours(hours['fork_build'])} fork build"
           if hours["fork_build"] else ""),
        f"                  (sweep = max of {_hours(hours['quantize'])} "
        f"quantize and {_hours(hours['upload'])} upload, run in parallel)",
    ]
    for lead in assessment["fork_leads"]:
        lines.append(f"  fork lead       {lead['repo']}@{lead['ref']} "
                     f"[{lead['confidence']}] {lead['why']}")
    for blocker in assessment["blockers"]:
        lines.append(f"  BLOCKED         {blocker}")
    for warning in assessment["warnings"]:
        lines.append(f"  warning         {warning}")
    return "\n".join(lines)


def markdown(result, limit=None):
    """The durable report written to reports/<stamp>.md."""
    system = result["system"]
    rows = result["assessments"][:limit] if limit else result["assessments"]

    out = [
        "# AgentQuantix research report",
        "",
        f"Generated {result['generated_at']} in {result['elapsed_s']}s.",
        "",
        "## This machine",
        "",
        "```",
        sysprobe.summary(system),
        "```",
        "",
        "## Candidates",
        "",
        f"{result['trending_count']} trending models, {result['kept_count']} of "
        "them original base models with a text-generation path. Sorted by "
        "verdict, then by estimated wall-clock time — the ones that finish "
        "soonest are the ones most likely to be worth starting tonight.",
        "",
        "| | Model | Params | BF16 | Peak disk | Upload | Est. time | Quants | Note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for assessment in rows:
        note = (assessment["blockers"] or assessment["warnings"] or [""])[0]
        params = (f"{assessment['params_b']:.1f}B" if assessment["params_b"] else "?")
        if assessment["n_experts"]:
            params += f" ({assessment['n_experts']}E)"
        out.append(
            f"| {VERDICT_MARK[assessment['verdict']].strip()} "
            f"| [{assessment['repo_id']}]({assessment['url']}) "
            f"| {params} "
            f"| {assessment['bf16_gb']:.1f} GB "
            f"| {assessment['peak_disk_gb']:.0f} GB "
            f"| {assessment['upload_gb']:.0f} GB "
            f"| {_hours(assessment['hours']['total'])} "
            f"| {len(assessment['quants'])}"
            f"{'/' + str(len(assessment.get('all_quants') or [])) if assessment.get('published_count') else ''} "
            f"| {note} |")

    out += ["", "## Detail", ""]
    for assessment in rows:
        out += ["```", detail(assessment), "```", ""]

    if result["dropped"]:
        out += ["## Dropped before scoring", "",
                "| Rank | Model | Reason |", "|---:|---|---|"]
        for item in sorted(result["dropped"], key=lambda d: d["rank"]):
            out.append(f"| {item['rank']} | {item['repo_id']} | {item['reason']} |")
        out.append("")

    if result["errors"]:
        out += ["## Could not read", ""]
        for item in result["errors"]:
            out.append(f"- `{item['repo_id']}` — {item['error']}")
        out.append("")

    out += [
        "## How the estimates are made",
        "",
        "- **Peak disk** is the larger of the conversion moment (safetensors +"
        " BF16 GGUF) and the sweep moment (BF16 + the two largest quants +"
        " any imatrix source quant). Every quant is deleted the moment it is"
        " safely on the Hub and the uploader runs one quant behind the"
        " quantizer, so the sweep never holds more than two at once.",
        "- **Est. time** overlaps quantization with upload, because the"
        " pipeline does: the sweep costs `max(quantize, upload)`, not their"
        " sum. Most runs are upload-bound.",
        "- **Transfer rates** are the median of what previous runs actually"
        " achieved, from `state/timings.json`; where no observation exists yet"
        " the report says *(assumed)*.",
        "- **imatrix source** is the BF16 whenever it fits in available RAM +"
        " free VRAM. When it does not, the largest of Q8_0 / Q4_K_M / Q2_K"
        " that does is used instead — the same trade the K2-Horizon MoVA run"
        " made by hand.",
        "",
    ]
    return "\n".join(out)
