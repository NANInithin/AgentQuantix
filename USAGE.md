# Using AgentQuantix

Three ways to drive it. The CLI is the ground truth — the Claude Code skill and
the API-key loop call exactly the same code, so nothing behaves differently
depending on which one you pick.

See [README.md](README.md) for what it does internally, the environment
variables, and the file layout.

---

## Install

**If you already have Python**, install the CLI straight from the repo:

```bash
uv tool install "git+https://github.com/NANInithin/AgentQuantix#egg=agentquantix[all]"
```

or from a local checkout:

```bash
uv tool install "./AgentQuantix[all]"
```

That puts `aqx` on PATH. `pipx install` works identically if you prefer it.
The `[all]` extra pulls `hf_xet` (large downloads) and `ninja` (much faster
builds) — both change what the tool can actually do, so take them.

**On a bare machine** — a fresh VM with nothing — use the bootstrap installer,
which brings its own Python:

```bash
curl -fsSL https://raw.githubusercontent.com/NANInithin/AgentQuantix/main/install.sh | bash
```

```powershell
irm https://raw.githubusercontent.com/NANInithin/AgentQuantix/main/install.ps1 | iex
```

Then:

```bash
export HF_TOKEN=hf_...    # write permission; or just run `hf auth login`
aqx bootstrap             # checks prerequisites, clones + builds llama.cpp
```

`aqx bootstrap` will not install system packages behind your back. It reports
what is missing and prints the exact command for each — those need a package
manager and usually root, and the CUDA choice depends on the machine.
`--check-only` reports and stops.

The **work root** (llama.cpp plus the scratch tree) resolves in this order:
`AQX_WORK_ROOT`, then an existing `~/Documents/INF` containing llama.cpp — so
this machine keeps sharing one checkout with the hand-written scripts — then
`~/.agentquantix` on a fresh machine.

## One-time setup

```bash
aqx doctor
```

Confirms token, llama.cpp build, GPU, disk, transfer policy. It should end with
`Ready.`

One more, **while the machine is idle**:

```powershell
aqx probe --remeasure-disk
```

Disk throughput drives every time estimate, and it is cached per volume. If the
first measurement happened while a quantization job was running it will read
low — roughly half — and every estimate built on it will be pessimistic. Run
this once on a quiet machine and it is right from then on.

---

## The normal cycle

### 1. Trigger the research

```powershell
aqx research
```

Top 100 trending, filtered, sized, checked against this box. Two-ish minutes.
Read-only — downloads nothing, creates nothing, safe to run whenever.

Useful variants:

```powershell
aqx research --limit 30          # faster, top 30 only
aqx research --hide-blocked      # only what's actually runnable
aqx research --markdown          # also writes reports/<stamp>.md
```

### 2. Read the table

```
      MODEL                          PARAMS   BF16   PEAK  UPLOAD    EST  QUANTS  NOTE
WARN  IFM/K2-Horizon-MoVA-36B-A4B     37.4B*    70G   138G     55G   3.0h    3/30  27 of 30 already published...
WARN  Qwen/Qwen3.8-27B                27.8B    53G   132G    422G  11.5h      29  BF16 is 53 GiB - xet for that download...
DONE  IFM/K2-Horizon-7B                9.0B    17G   0.0G    0.0G     0m    0/29  29 of 29 already published...
BLOCK zai-org/GLM-5.3                753.3B*  1431G  3581G  11836G  21.0d      30  needs 3581 GB peak disk...
```

- `OK` / `WARN` / `DONE` / `BLOCK` — WARN means it runs, but read the note.
  DONE means it is already fully published under your namespace, so there is no
  work. BLOCK means it can't run, and the note is the wall.
- **QUANTS** is `remaining/total` when some are already up, or a plain count
  when nothing is. `3/30` means twenty-seven are already on the Hub and a run
  would build three.
- **PEAK** is peak disk *during* the run, not total output. 29 quants of a 27B
  model is 132 GB, not the 422 GB the outputs sum to, because each file is
  deleted the moment it is uploaded.
- **EST** overlaps quantize with upload. Most runs are upload-bound.
- `*` after params = MoE, so that is stored parameters, not active.
- Sorted runnable-and-cheapest first, not by trending rank. **Part-finished
  repos rise on their own** — only their remaining quants are priced, so a
  repo that is 27/30 done costs an hour or two and sorts near the top. Those
  are usually the best work on the list.

Every row checks `<your-hf-username>/<name>-GGUF` on the Hub, so the agent never
proposes work you have already done, and never quotes a full-sweep price for a
repo that is nearly finished.

### 3. Look closer at anything interesting

```powershell
aqx show Qwen3.8-27B
```

Full breakdown: quant list, imatrix source and why, the `-ngl` it will use,
disk math, the time split, fork leads, every warning. Partial names work —
`aqx show qwen3.8-27b` is fine.

### 4. Approve and run

```powershell
aqx run Qwen3.8-27B
```

Prints the plan, asks `Start? [y/N]`, then runs for hours. Or omit the name to
pick from a numbered list:

```powershell
aqx run
```

Always worth doing first:

```powershell
aqx run Qwen3.8-27B --dry-run     # show the plan, touch nothing
```

Multiple models run smallest-first, so a pipeline problem surfaces on the cheap
one:

```powershell
aqx run gpt2 Qwen3.8-27B
```

#### Trading time against disk

How much overlap the sweep uses decides how much disk it needs, because every
file in flight is one more quant sitting on the drive:

```
files on disk at once = quantize workers + upload workers + queue depth
```

Three by default: one being cut, one queued, one uploading. `aqx run` reprices
peak disk for whatever you choose and tells you how it moved.

| Flag | Default | Effect |
|---|---|---|
| `--sequential` | off | Quantize → upload → delete, one at a time, no background uploader — exactly how `scripts/Ornith-1.5-9B.py` works. **One** quant on disk. Slower, because uploads stop hiding behind the next quantize. |
| `--upload-workers N` | 1 | Concurrent uploads. The useful knob when a run is upload-bound, which most are. Each stream costs one more quant of disk. |
| `--queue-depth N` | 1 | Finished quants allowed to wait for an uploader. Pure disk-for-smoothness. |
| `--quantize-workers N` | 1 | Concurrent `llama-quantize` processes. Rarely worth raising — it already threads across every core, so a second process mostly buys contention. |
| `--quantize-threads N` | all cores | Its trailing `[nthreads]` argument. Set it to keep the machine usable during a long sweep. |

Qwen3.8-27B, same 29 quants, three ways:

```powershell
aqx run Qwen3.8-27B --dry-run                              # 132 GB peak
aqx run Qwen3.8-27B --dry-run --sequential                 # 104 GB peak
aqx run Qwen3.8-27B --dry-run --upload-workers 3 --queue-depth 2   # 187 GB peak
```

**When to reach for each:** `--sequential` when a model is blocked on disk, or
when you want to keep the drive free for something else — it is the only way to
run a model whose quants will not fit three at a time. `--upload-workers 2-3`
when the estimate is upload-bound and you have disk to spare. `--quantize-threads`
when you want to use the machine for something else while it grinds.

All five also work as environment variables (`AQX_SEQUENTIAL=1`,
`AQX_UPLOAD_WORKERS`, `AQX_QUEUE_DEPTH`, `AQX_QUANTIZE_WORKERS`,
`AQX_QUANTIZE_THREADS`) if you want a permanent default.

### 5. Nothing — step 5 is automatic

When the run finishes it verifies what actually landed on the Hub and publishes
the model card. You will see the file listing with real sizes and anything
missing.

To re-check later, or check a repo the old hand-written scripts produced:

```powershell
aqx verify K2-Horizon-0.9B
aqx card K2-Horizon-0.9B --dry-run    # print it
aqx card K2-Horizon-0.9B              # publish it
```

---

## xet — handled for you

This used to be a manual lever. It is now automatic, because the correct
setting is not per-run, it is **per transfer**:

- hf_xet is **required** to *download* any file over huggingface_hub's
  plain-HTTP cap — 50 decimal GB, which is **46.6 GiB**. Without it the
  download raises outright.
- hf_xet is roughly **10x slower to upload** on this connection (~2 MB/s
  against ~20 MB/s), and uploads dominate almost every run.

So a 75 GB model wants xet **on** for one download and **off** for the thirty
uploads that follow. `aqx run` does exactly that: it enables xet for the BF16
fetch when the file is over the cap, then pins it off before the first upload
thread starts. Time estimates are priced on that, which is why they are hours
rather than days.

Override only if you have a reason:

```powershell
aqx run <model> --xet auto     # the default
aqx run <model> --xet off      # never use it; models over 46.6 GiB become unrunnable
aqx run <model> --xet on       # always use it; expect ~10x slower uploads
```

`AQX_XET=auto|on|off` does the same permanently. You no longer need to set
`HF_HUB_DISABLE_XET` yourself — the pipeline sets and restores it around each
phase.

---

## If it stops

Ctrl-C, a reboot, a dead network — just re-run the identical command.
`status.json` plus the Hub listing mean it redoes nothing and never uploads
twice. A failed quant type does not abort the batch; a failed model does not
abort the night. The end-of-run summary lists what failed and why.

---

## Driving it with a model instead

**Claude Code** — restart your session in INF, approve the two MCP servers,
then:

```
/quantix
```

Then talk to it: *"what's worth quantizing tonight?"* It researches, presents
the table, asks which ones, and will not start without you naming them.

**Any API key, no harness:**

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
aqx agent --model anthropic/claude-opus-5
```

Also reads `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`,
`TOGETHER_API_KEY`. Any OpenAI-compatible endpoint via `--base-url`. It asks you
directly before letting a quantization start, regardless of what the model
thinks it is allowed to do.

**The model must support tool calling.** Everything this agent does is tools —
a text-only model cannot drive it at all. Omit `--model` to use the default,
which does.

Two failures worth recognising, both reported as a plain explanation rather
than a traceback:

- *"filtered out by your account's privacy settings"* — OpenRouter removed
  every endpoint for that model because your account forbids models that train
  on prompts. Either allow it at
  [openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy), or
  pick a model that does not require the permission.
- *"cannot do tool calling"* — the model has endpoints, but none that accept a
  `tools` payload. Check the model's provider page lists Tools support.

**Codex / OpenCode / Kimi** — copy the matching file from
[`adapters/`](adapters/) into that tool's config.

---

## Two cautions

**Do not start a run while another quantization job is going.** They fight over
RAM, disk and bandwidth, and both slow to a crawl. Check first:

```powershell
Get-Process llama-quantize,llama-imatrix -ErrorAction SilentlyContinue
```

If that prints anything, wait.

**Try a small model first on a new machine.** The pipeline's first real
end-to-end run was `MiniCPM5-1B` on 2026-09-04: 30 files, 18.1 GB published,
every local quant deleted as it uploaded. It works — but a new machine has a
new toolchain, and a 1B model costs minutes rather than hours to find that out.

---

## Command reference

| Command | What it does |
|---|---|
| `aqx doctor` | Is this machine ready. Run after any environment change. |
| `aqx probe` | Measure the machine. `--remeasure-disk` to refresh the cached disk rate. |
| `aqx research` | Steps 1-3. `--limit N`, `--top N`, `--hide-blocked`, `--markdown`, `--json`, `--no-fork-hunt`. |
| `aqx show <model>...` | Full detail on candidates from the last research run. |
| `aqx run [model...]` | Steps 4-5. `--dry-run`, `--yes`, `--keep-fork`. Omit the name to choose interactively. |
| `aqx verify <repo>...` | What is actually on the Hub, with real sizes and anything missing. |
| `aqx card <repo>` | Write and publish the model card. `--dry-run` to print it instead. |
| `aqx agent [prompt]` | Drive it with an API key. `--model`, `--base-url`, `--max-steps`. |
| `aqx mcp` | Run as an MCP server on stdio. Harnesses launch this; you never do by hand. |

Research results land in `reports/` (`latest.json` is what `show` and `run`
read). Run records and the learned timing history land in `state/`.
