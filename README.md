# AgentQuantix

[![CI](https://github.com/NANInithin/AgentQuantix/actions/workflows/ci.yml/badge.svg)](https://github.com/NANInithin/AgentQuantix/actions/workflows/ci.yml)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](#)

Finds newly trending Hugging Face models, works out which of them **your
machine** can turn into llama.cpp GGUF quants, and — once you approve specific
models — quantizes them, uploads them, and writes their model cards.

Two human gates, and nothing else:

1. **You decide when to start.** It never researches on its own.
2. **You decide which models get quantized.** It never starts a run for a model
   you did not name.

Everything between and around those is automatic, including the final
verification and the model card.

```
$ aqx research

      MODEL                          PARAMS   BF16   PEAK  UPLOAD    EST  QUANTS  NOTE
WARN  IFM/K2-Horizon-MoVA-36B-A4B     37.4B*    70G   138G     55G   3.0h    3/30  27 of 30 already published
WARN  Qwen/Qwen3.8-27B                27.8B    53G   132G    422G  11.5h      29  BF16 is 53 GiB - xet enabled for that download
DONE  IFM/K2-Horizon-7B                9.0B    17G   0.0G    0.0G     0m    0/29  29 of 29 already published
BLOCK zai-org/GLM-5.3                753.3B*  1431G  3581G  11836G  21.0d      30  needs 3581 GB peak disk, only 441 GB free
```

**Why this exists.** Quantizing a model by hand means writing a script per
model, guessing whether it will fit, watching disk fill up, and discovering
four hours in that the architecture needs a llama.cpp fork. This does the
arithmetic first, tells you what it will cost before you commit, and then runs
unattended without filling the drive.

---

## Quick start

**[USAGE.md](USAGE.md) is the step-by-step walkthrough** — install, the normal
cycle, reading the report, what to do when a run stops, and the command
reference. Start there. This file is the reference behind it: what each step
does internally, the environment, and the layout.

```bash
uv tool install ./AgentQuantix[all]   # or: pipx install ./AgentQuantix[all]

aqx bootstrap     # fresh machine: prerequisites, clone + build llama.cpp
aqx doctor        # is this machine ready
aqx research      # steps 1-3, read-only
aqx show <model>  # full detail on one candidate
aqx run <model>   # steps 4-5, asks before starting
```

On a bare VM with no Python at all, `install.sh` / `install.ps1` install uv
first and then the above. Or drive it with a model instead of by hand — see
**Harnesses** below.

Runs on Windows and Linux. CUDA is detected, not assumed: no nvcc means a CPU
build and a slower imatrix pass, not a failure.

---

## What each step does

**1-3. `aqx research`** — reads the top 100 trending models from the Hub REST
API (`sort=trendingScore`, exact and deterministic), drops everything that is
not an original text-capable base model, then for each survivor reads the real
parameter count from the safetensors header, the architecture from
`config.json`, and checks it against `convert_hf_to_gguf.py
--print-supported-models`. Measures this machine — CPU, RAM, VRAM, free disk,
disk throughput — and estimates disk, transfer and wall-clock time for a full
quant sweep. Downloads nothing, creates nothing, safe to run any time.

It also checks `<namespace>/<name>-GGUF` for every candidate, so the report
knows what you have already published. A finished repo is marked `DONE` and
costed at zero; a part-finished one is priced on its **remaining** quants only,
which makes an interrupted run the cheapest work on the list rather than a
full-price duplicate.

Verdicts are `OK` / `WARN` / `DONE` / `BLOCK`, sorted runnable-and-cheapest
first, because the models most likely to be approved are the ones that finish
tonight.

Nothing in this step involves a language model. It is deterministic Python
against the Hub REST API, `config.json`, and llama.cpp's own architecture and
quant tables — a model only enters when you drive the agent through a harness,
and even then it calls this same code.

**4. `aqx run <model>`** — the pipeline, unchanged in every way that matters
from the hand-written scripts in `INF/scripts`:

- BF16 source: the publisher's own GGUF when they ship one (one copy of the
  weights instead of safetensors + conversion output + conversion runtime),
  otherwise `convert_hf_to_gguf.py`, with the source weights deleted the moment
  the conversion succeeds.
- imatrix from 500 rows of wikitext-2, computed on the BF16 when it fits in
  available RAM + free VRAM, and on the largest of Q8_0 / Q4_K_M / Q2_K that
  does when it does not.
- Blocks the imatrix has no rows for (NextN / multi-token-prediction heads)
  are found and pinned to `q4_K` so low-bit types do not hard-abort on them.
- The full 29-type sweep, imatrix-guided where it helps, `MXFP4_MOE` added for
  MoE models, ternary opt-in via `BUILD_TERNARY=1`.
- **Quantization and upload overlap** on a bounded queue, so the sweep costs
  `max(quantize, upload)` rather than their sum.
- **Every quant is deleted the moment it is safely on the Hub**, so peak disk
  is BF16 plus however many files the concurrency allows in flight —
  `quantize workers + upload workers + queue depth`, three at the defaults, or
  one under `--sequential`. A 29-quant sweep of a 27B model peaks around
  132 GB, not the 422 GB its outputs sum to.
- Resumable from `status.json` plus the Hub listing. Nothing is ever uploaded
  twice, and an interrupted run redoes nothing.
- One failed quant never aborts the batch; one failed model never aborts the
  night.

**5. verification + card** — runs automatically at the end. Lists what is
actually on the Hub with real sizes, flags anything missing from the intended
sweep or suspiciously small, and generates the README from that verified
listing so the card can only describe files that exist.

### Unsupported architectures

When upstream llama.cpp cannot convert a model, the agent searches GitHub for
the publisher's own llama.cpp fork on a branch named for the architecture, or
an open upstream PR adding it — then clones and builds it in
`INF/temp/aqx-forks/`, reusing one build across every model that needs it.
That is the K2-Horizon case, automated. Set `GITHUB_TOKEN` to raise the
unauthenticated search rate limit; without it a busy trending list can exhaust
it and quietly find fewer leads.

---

## Harnesses

All logic lives in the CLI and one tool registry
(`src/agentquantix/agent/tools.py`). Every harness is a thin adapter over the
same eight tools and the same system prompt, so the agent behaves identically
wherever it runs — and works with no harness at all.

| | Setup |
|---|---|
| **Claude Code** | `python scripts/install.py` copies the skill and merges `.mcp.json` into INF. Then `/quantix`. |
| **OpenRouter / any API key** | `aqx agent --model anthropic/claude-opus-5`. Reads `OPENROUTER_API_KEY`, `AQX_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY` or `TOGETHER_API_KEY`. Any OpenAI-compatible endpoint via `--base-url`. |
| **Codex** | append `adapters/codex.config.toml` to `~/.codex/config.toml` |
| **OpenCode** | merge `adapters/opencode.json` into your opencode config |
| **Kimi CLI / anything MCP** | point it at `python -m agentquantix.mcp_server` with `PYTHONPATH` set to `src` |

The MCP server is written directly against the JSON-RPC wire format and the
OpenRouter loop against `urllib` — neither needs the `mcp` or `openai` package,
because a portability layer that needs a pip install in someone else's
environment is not portable.

The approval gate is enforced three times over: the prompt tells the model not
to start a run unasked, the tool refuses without `user_approved=true`, and the
built-in loop asks you directly before letting the call through.

---

## Environment

| Variable | Meaning |
|---|---|
| `MLE2` / `HF_TOKEN` | Hugging Face token with write permission. Required. |
| `INF_ROOT` | Where llama.cpp and the scratch tree live. Default `~/Documents/INF`. |
| `AQX_NAMESPACE` | Hub namespace to publish under. Defaults to whoever the token belongs to. |
| `AQX_XET` | `auto` (default), `on` or `off`. Auto enables xet only for downloads over 46.6 GiB, where huggingface_hub requires it, and keeps uploads on the ~10x faster plain-HTTP path. You do not need to set `HF_HUB_DISABLE_XET` yourself. |
| `KEEP_LOCAL_GGUF=1` | Do not delete quants after upload. Costs a lot of disk. |
| `AQX_SEQUENTIAL=1` | Quantize → upload → delete, one at a time. One quant on disk. Same as `--sequential`. |
| `AQX_UPLOAD_WORKERS` | Concurrent uploads (default 1). Each costs one more quant of peak disk. |
| `AQX_QUEUE_DEPTH` | Finished quants allowed to wait for an uploader (default 1). |
| `AQX_QUANTIZE_WORKERS` | Concurrent `llama-quantize` processes (default 1). |
| `AQX_QUANTIZE_THREADS` | Threads per `llama-quantize` (default: every core). |
| `BUILD_TERNARY=1` | Include TQ1_0 / TQ2_0. Only meaningful for ternary-trained weights. |
| `KEEP_FORK=1` | Keep fork checkouts after a run. |
| `GITHUB_TOKEN` | Raises the fork-hunt search rate limit. |
| `AQX_TRENDING_LIMIT` | How many trending models to consider. Default 100. |

---

## Layout

```
src/agentquantix/
├── config.py         paths, quant tables, bpw estimates, candidate filters
├── sysprobe.py       what machine are we on (step 2)
├── hub.py            trending, filtering, metadata, published-file listing
├── archsupport.py    the llama.cpp arch gate, and the fork hunt
├── feasibility.py    disk/time/RAM estimates and the OK/WARN/BLOCK verdict
├── research.py       steps 1-3 end to end
├── report.py         the ranked table, the detail view, the markdown report
├── card.py           step 5: verification and the model card
├── cli.py            aqx
├── mcp_server.py     MCP over stdio, no dependencies
├── pipeline/
│   ├── job.py        one approved model, frozen at approval + resume state
│   ├── build.py      llama.cpp / fork builds
│   ├── source.py     BF16 acquisition
│   ├── imatrix.py    calibration, the forward pass, coverage gaps
│   └── run.py        the orchestrator: overlap, delete-on-upload, resume
└── agent/
    ├── prompt.py     the system prompt, shared by every harness
    ├── tools.py      the eight tools, defined once
    └── loop.py       the OpenAI-compatible loop for raw API keys
```

`state/` holds run records and `timings.json`, which accumulates what runs
actually achieved so the estimates stop being assumptions. `reports/` holds
research results. Both are machine-local and gitignored.
