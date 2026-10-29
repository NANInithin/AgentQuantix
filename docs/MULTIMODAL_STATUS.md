# AgentQuantix 0.5.4 multimodal status

0.5.4 adds explicit llama.cpp libmtmd multimodal understanding. Plan-time
vision/audio claims come from the Hub config; converted projector metadata is
authoritative. A projector that does not cover the claimed modalities blocks
the run. `image-text-to-text` and `any-to-any` remain on text-generation unless
the user selects `--task multimodal-understanding`.

## Support matrix

| Task | Native route | Acceptance |
|---|---|---|
| Vision VQA and document OCR | `llama-server --mmproj --jinja`, `/v1/chat/completions` with `image_url` | SmolVLM2 F16 evaluated on Linux CPU; rejected by VQA and OCR gates |
| Video order | `llama-mtmd-cli --video`; ffmpeg required | CLI-only; SmolVLM2 F16 passed frame order at 1 fps |
| Audio transcription | `llama-server --mmproj --jinja`, `/v1/chat/completions` with `input_audio` | Qwen3-ASR F16 and Q8_0/Q6_K/Q5_K_M/Q4_K_M passed on Linux CPU |
| Projector | F16 `mmproj` GGUF | F16 only; projector quantization is unaccepted |

Audio input is experimental upstream. Audio QA is not accepted. No Windows,
GPU or Metal testing has been performed. Hub publication has not been exercised.
The recorded results are model-specific Linux CPU evidence, not a general
platform support claim. The default `image-text-to-text` route remains
text-generation unless `--task multimodal-understanding` is selected.

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

Both models were run sequentially on Linux CPU with llama.cpp revision
`8642541b255bfb5dc39cd351a178be09013a55d6`. Runtime update lookup returned
HTTP 403 during inference; the already-installed runtime at that exact revision
was used. Runs used `--no-publish`; no Hub publication was exercised.

The video CLI supports `--video`. Acceptance uses
`--video-fps 1 --video-timestamp-interval 1000`; the CLI defaults of 4 fps and
5000 ms produced an excessive media batch and a memory-slot failure on this
fixture. The reduced sampling completed and retained the video-order gate.

The authoritative result files are
`/home/ubuntu/spike/acceptance/smolvlm2-results.json` and
`/home/ubuntu/spike/acceptance/qwen3asr-results.json`. The separate 0.5.4b
feasibility spike is not used as implementation acceptance.

### HuggingFaceTB/SmolVLM2-500M-Video-Instruct

Source revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; licence
`apache-2.0`; architecture `SmolVLMForConditionalGeneration`; runtime revision
`8642541b255bfb5dc39cd351a178be09013a55d6`. The F16 projector metadata reports
`general.type=mmproj`, vision encoder present, projector type `idefics3`,
image size 512, preprocessing image size 2048, mean/std `[0.5, 0.5, 0.5]`,
and patch size 16. The Transformers VQA reference was available at accuracy
0.8.

| Text quant | Status | Bytes | VQA accuracy | OCR CER/WER | Video order |
|---|---|---:|---:|---:|---|
| F16 | quality-rejected | 820421760 | 0.6 | 0.5681818181818182 / 0.5714285714285714 | true |
| Q8_0 | not evaluated | — | — | — | — |
| Q6_K | not evaluated | — | — | — | — |
| Q5_K_M | not evaluated | — | — | — | — |
| Q4_K_M | not evaluated | — | — | — | — |

F16 failed the absolute VQA (`0.8`) and OCR CER (`0.05`) gates. Its video
answer correctly described a red circle followed by a blue square. The run
stopped before quantization because there was no passing F16 baseline for the
relative quant gates. The stored OCR answer was `Invoice 4821 is due on
2026-11-30.`, which omitted invoice text.

### Qwen/Qwen3-ASR-0.6B

Source revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`; licence
`apache-2.0`; architecture `Qwen3ASRForConditionalGeneration`; runtime revision
`8642541b255bfb5dc39cd351a178be09013a55d6`. The F16 projector metadata reports
`general.type=mmproj`, audio encoder present, projector type `qwen3a`, and
128 mel bins. No audio reference is available in this milestone.

| Text quant | Status | Bytes | ASR WER | ASR CER |
|---|---|---:|---:|---:|
| F16 | passed | 1509343264 | 0.0 | 0.018691588785046728 |
| Q8_0 | passed | 804749344 | 0.0 | 0.018691588785046728 |
| Q6_K | passed | 622729248 | 0.0 | 0.018691588785046728 |
| Q5_K_M | passed | 551373856 | 0.0 | 0.018691588785046728 |
| Q4_K_M | passed | 484215840 | 0.0 | 0.018691588785046728 |

All candidates returned the transcript
`And so, my fellow Americans, ask not what your country can do for you; ask
what you can do for your country.` after generic ASR markup removal. Audio QA
was not evaluated.
