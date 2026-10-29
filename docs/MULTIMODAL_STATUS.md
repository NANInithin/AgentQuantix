# AgentQuantix 0.5.4 multimodal status

0.5.4 adds explicit llama.cpp libmtmd multimodal understanding. Plan-time
vision/audio claims come from the Hub config; converted projector metadata is
authoritative. A projector that does not cover the claimed modalities blocks
the run. `image-text-to-text` and `any-to-any` remain on text-generation unless
the user selects `--task multimodal-understanding`.

## Support matrix

| Task | Native route | Acceptance |
|---|---|---|
| Vision VQA and document OCR | `llama-server --mmproj --jinja`, `/v1/chat/completions` with `image_url` | Pending Linux CPU result |
| Video order | `llama-mtmd-cli --video`; ffmpeg required | CLI-only; pending Linux CPU result |
| Audio transcription | `llama-server --mmproj --jinja`, `/v1/chat/completions` with `input_audio` | ASR-style only; pending Linux CPU result |
| Projector | F16 `mmproj` GGUF | F16 only; projector quantization is unaccepted |

Audio input is experimental upstream. Audio QA is not accepted. No Windows,
GPU or Metal testing has been performed. Hub publication has not been exercised.

## Quality gates

| Candidate | Vision | Audio | Video |
|---|---|---|---|
| F16 baseline | VQA accuracy `>= 0.8`; OCR CER `<= 0.05`; if a Transformers VQA reference exists, accuracy `>= reference - 0.2` | ASR WER `<= 0.10` | `video_order_ok` must be true when runtime video support is available |
| Text quant vs F16 | VQA accuracy no more than `0.2` below F16; OCR CER no more than `0.05` above F16 | ASR WER no more than `0.05` above F16 | Must remain true if F16 passed video order |

Only metrics for actual projector modalities apply. A candidate with no
applicable gates fails. Video is skipped, not counted as a pass, if
`llama-mtmd-cli --help` lacks `--video` or ffmpeg is absent.

## Fixtures and licence

`src/agentquantix/fixtures/multimodal/` contains the authored invoice image,
the 4-second 4-fps red-circle-then-blue-square video, ordered extracted frames,
and `cases.json`. The VQA cases refer to the existing five CC0 retrieval PNGs;
the ASR case refers to the existing `fixtures/voice/jfk.wav` and transcript.
The invoice/video generator is `scripts/generate_multimodal_fixtures.py`.
Authored fixtures are dedicated to the public domain under CC0 1.0 Universal;
see `fixtures/multimodal/LICENSE.md`.

## Revisions and native acceptance

Native Linux CPU acceptance has not run. At 2026-10-29 14:35:27 UTC, the GitHub
API rate-limit endpoint reported 0 of 60 core requests remaining, resetting at
2026-10-29 15:07:25 UTC. Runtime installation and inference were not attempted.
This is a blocked preflight, not a runtime or model failure. Record the runtime
and source revisions, model licences and per-quant metrics from
`/home/ubuntu/spike/acceptance/{smolvlm2,qwen3asr}-results.json` after the
sequential native runs. The separate 0.5.4b feasibility spike used llama.cpp
pin `8642541`; it does not count as this implementation's acceptance.

### HuggingFaceTB/SmolVLM2-500M-Video-Instruct

Source revision, licence, runtime revision and per-quant candidate results:
pending native acceptance.

| Text quant | Status | Bytes | VQA accuracy | OCR CER/WER | Video order |
|---|---|---:|---:|---:|---|
| F16 | pending | — | — | — | — |
| Q8_0 | pending | — | — | — | — |
| Q6_K | pending | — | — | — | — |
| Q5_K_M | pending | — | — | — | — |
| Q4_K_M | pending | — | — | — | — |

### Qwen/Qwen3-ASR-0.6B

Source revision, licence, runtime revision and per-quant candidate results:
pending native acceptance.

| Text quant | Status | Bytes | ASR WER | ASR CER |
|---|---|---:|---:|---:|
| F16 | pending | — | — | — |
| Q8_0 | pending | — | — | — |
| Q6_K | pending | — | — | — |
| Q5_K_M | pending | — | — | — |
| Q4_K_M | pending | — | — | — |
