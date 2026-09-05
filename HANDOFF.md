# AgentQuantix — session handoff

**Date:** 2026-09-05
**Version:** 0.2.0
**Repo:** https://github.com/NANInithin/AgentQuantix
**HEAD:** `a43ccd8`
**CI:** green (8 jobs — ubuntu/windows × py3.10/3.13, hub 0.36 + 1.x, lint, build)
**Tests:** 136, all offline

---

## 1. The blocking issue — NOT a code problem

Uploads from the Linux VM (`friday`, Vultr Amsterdam, `45.77.138.62`) run at
**12.3 MB/s single-stream** against **538 Mbps measured capacity**.

Root cause, from `ss -tni dst 18.239.81.3` taken during a live upload:

| socket | bytes_sent | bytes_retrans | loss |
|---|---:|---:|---:|
| :52016 | 4,301,965,297 | 1,487,495,568 | **34.6%** |
| :55888 | 1,411,813,159 | 471,011,938 | **33.4%** |
| :59702 | 1,309,017,250 | 443,533,781 | **33.9%** |
| :49320 | 13,849,861 | 4,618,152 | **33.3%** |

**A third of every transmitted byte is retransmitted.** Healthy is <0.1%.

Supporting evidence:

- Ratios cluster at 33–35% across four independent connections. Random
  congestion does not produce a constant fraction — this looks like ECMP
  across three links with one blackholing.
- `rehash:1735` / `rehash:518` / `rehash:420`. Linux rehashes a flow's source
  port after repeated timeouts specifically to land on a different ECMP path.
  The kernel has tried 1,735 times to escape the bad link on one connection.
- Congestion control is **BBR**, which estimates bandwidth from delivery rate
  rather than treating loss as congestion. That is the only reason throughput
  is 12 MB/s and not near-zero. CUBIC would have collapsed.
- RTT to the endpoint is **1.15 ms** — not a bandwidth-delay-product limit.
- `Send-Q` sits full at ~5 MB with `Recv-Q` at 0 — the application is ahead and
  blocked; nothing local is the bottleneck.

### Ruled out

| Hypothesis | Killed by |
|---|---|
| Bandwidth-delay product / RTT | RTT to `18.239.81.3` is 1.15 ms |
| Provider egress cap | Ookla: 538.79 Mbps up |
| CloudFront per-connection cap | User uploads fast from laptop + Codespace |
| `huggingface_hub` / Python slowness | Send-Q full — app is ahead of the kernel |
| AgentQuantix transfer policy | Loss is at the TCP layer, below anything we control |

### Next steps (not code)

```bash
sudo apt install -y mtr-tiny
mtr -rwc 100 18.239.81.3
```

Loss appearing at one hop **and persisting to the end** is the culprit. Loss
at a middle hop that does not propagate is ICMP deprioritisation — ignore it.

Then open a Vultr ticket with the mtr output and the retransmission table
above. This is their transit path.

### Consequences while it is broken

- **Egress cost.** ~1.5× the bytes leave the box. A 250 GB sweep pushes ~375 GB.
  If the plan meters bandwidth, that overage is real money.
- **Any xet-vs-plain-HTTP measurement taken on this VM is void.** See §4.
- Parallel upload workers remain the only useful lever: four connections each
  getting ~12 MB/s still aggregates.

---

## 2. What was shipped this session

| Commit | What |
|---|---|
| `321941a` | Standard HF token vars (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `hf auth login`) instead of the private `MLE2` |
| `d74cf0d` | Linux build fix — Ninja was routed through `cmd.exe` (gated on generator, not on vcvars) |
| `9a16a2e` | huggingface_hub 1.x support; version floor raised 0.30 → 0.36 (0.30 lacks `HF_HUB_DISABLE_XET`) |
| `5b40942` | Read the architecture list without importing torch (source-parsing fallback, verified superset 276 vs 275) |
| `0b3ce0b` | `record(kind, **fields)` collided with a caller passing `kind=` — made positional-only |
| `5eeb200` | Report missing converter deps **before** downloading, not after |
| `7d58310` | `run_verbose()` — surface the converter's real error, not argv + exit code |
| `77ac66a` | `aqx run <any-repo-id>`; custom-code dep hints; `stdin=DEVNULL` |
| `a81ad0b` | All four converter deps checked, not two; `[full]` extra documented |
| `a43ccd8` | The 50 GB cap applies to **uploads** too |

---

## 3. Verified state

### Works on Linux (VM, AMD EPYC-Turin, 8 logical, 30.33 GB RAM, 449 GB disk, no GPU)

Full pipeline, end to end, on `Qwen/Qwen3-0.6B`:

```
FINISHED in 0.7 h — 30 GGUF files, 12.8 GB, model card published
```

install.sh → bootstrap (llama.cpp CPU build, 265 targets) → doctor → research
→ run (download → convert → imatrix → 29 quants → upload) → verify → card.

Disk discipline held: 6.6% of 449 GB used at peak.

### Works on Windows

Same, from the original INF checkout.

### Estimator calibration

`Qwen3-0.6B` was estimated at 0.2 h, took 0.7 h. The whole error was upload
throughput — the estimator had no timing history on that box and used its
default network rate. Uploads/downloads are learned from
`~/.agentquantix/state/timings.json` and median separately by xet mode, so it
self-corrects. **Note:** samples collected on this VM are contaminated by the
packet loss in §1.

---

## 4. Open items

### Deferred by the user

**PyPI publishing.** `publish.yml` exists with OIDC trusted publishing; the
`release:` trigger is commented out. Needs the user to create pending
publishers first:

- TestPyPI and PyPI, Owner `NANInithin`, Repository `AgentQuantix`,
  Workflow `publish.yml`, Environment `testpypi` / `pypi`

Then uncomment the trigger.

### Offered, not built

1. **Learn the xet upload policy instead of hardcoding it.**
   `transfer.for_upload()` carries a "~10x slower" constant measured on the
   user's *home* connection. Xet uploads chunks over many parallel connections,
   which is exactly what a single-stream-limited path lacks. The timing history
   already records `xet: true/false` per sample and medians the two modes
   separately — the policy should read it rather than trust one measurement.
   **Blocked on §1**: cannot be measured on a path dropping 33% of packets.

2. **Feed recorded quantize timings into the estimate.**
   `run.py` records `{quant, model, minutes, gb}` per quant, and
   `feasibility.py` never reads it — `quantize_h` is derived from measured disk
   throughput instead. Real measurements sitting unused next to a modelled
   number. Smaller term than upload; was not the source of the 0.2→0.7 h miss.

3. **Flag to skip the BF16 upload for models over the cap.**
   A >46.57 GiB BF16 must use xet, at ~2 MB/s ≈ 7 h per model, for the least-used
   file in the repo. Quants are unaffected — they are cut locally before
   anything is published.

### Known, unfixed

- **`snapshot_download` pulls whole repos**, including TensorFlow/Rust/ONNX
  duplicates. gpt2 was estimated at 1 GB peak and downloaded 5.63 GB. Peak-disk
  estimates are understated for multi-framework repos.

---

## 5. Models that cannot be converted (do not retry these)

| Model | Why |
|---|---|
| `openai-community/gpt2` | `h.0.attn.bias` is a causal-mask buffer in the 2019 checkpoint; llama.cpp's converter neither maps nor skips it. Not fixable here. |
| `baidu/Unlimited-OCR` | `trust_remote_code` custom code imports PIL, addict, matplotlib, torchvision. Installable, but a per-model dependency chase. |

**Good smoke test:** `Qwen/Qwen3-0.6B` — standard arch, no remote code, no
vision tower, ~1.2 GB BF16, full 29-quant sweep in well under an hour.

---

## 6. Environment notes

- **Tailscale broke DNS on the VM.** It owned `/etc/resolv.conf` and pointed at
  MagicDNS (`100.100.100.100`), which could not fetch the device's DNS config
  (`tailscale status` health check) and so answered tailnet names only.
  Fixed with `sudo tailscale set --accept-dns=false`. Will recur if Tailscale
  is re-enabled with `--accept-dns`.
- **Tool venv is Python 3.14.** torch has wheels; some model-specific deps may
  not. `uv tool install --python 3.12` is the escape hatch.
- **Install the right extra.** `[all]` = hf_xet + ninja. `[full]` adds torch,
  transformers, sentencepiece, protobuf — required to convert safetensors,
  which is the general case. `install.sh` takes `AQX_EXTRA=full`.

---

## 7. Process notes for whoever picks this up

Recurring root cause of the Linux bugs: **code paths that only execute in
circumstances the dev machine never had.** Windows had torch, a pre-existing
BF16, `cmd.exe`, an older huggingface_hub, and ran scripts as files rather than
pipes. Every one of those hid a real bug. Tests now simulate `sys.platform` and
validate call signatures offline so both platforms exercise both branches.

Mistakes made this session, recorded so they are not repeated:

- **Pushed a commit with a failing test and a lint error** (`7d58310`) — the
  push was not gated on the test result. Fixed in `5d4d8f0`.
- **Verified a fix against a cached wheel.** `uv tool install` reused the old
  build. Always `--reinstall`.
- **Asserted the upload size cap from memory** after verifying the download cap
  against the installed package, and wrote a confident comment around the wrong
  claim. Cost the user a failed 51.90 GiB upload. Verify both directions.
- **Gave install instructions that omitted `[full]`** twice, sending the user
  round the loop.
- **Recommended two models that cannot be converted.** Check checkpoint age and
  `trust_remote_code` before suggesting a test model.
- **Bash heredocs in this environment process backslash escapes even when
  quoted** (`<<'PY'`). Use Write/Edit for content containing escapes.
