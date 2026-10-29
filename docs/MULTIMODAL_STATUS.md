# AgentQuantix 0.5.4 multimodal status

0.5.4 adds explicit llama.cpp libmtmd multimodal understanding. Plan-time
vision/audio claims come from the Hub config; converted projector metadata is
authoritative. A projector that does not cover the claimed modalities blocks
the run. `image-text-to-text` and `any-to-any` remain on text-generation unless
the user selects `--task multimodal-understanding`.

## Support matrix

| Task | Native route | Acceptance |
|---|---|---|
| Vision VQA and document OCR | `llama-server --mmproj --jinja`, `/v1/chat/completions` with `image_url` | SmolVLM2 F16, Q8_0, Q6_K, Q5_K_M and Q4_K_M passed on Linux CPU |
| Video order | `llama-mtmd-cli --video`; ffmpeg required | CLI-only; SmolVLM2 F16 and all four text quants passed frame order at 1 fps |
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

Both models were rerun sequentially on Linux CPU with llama.cpp revision
`8642541b255bfb5dc39cd351a178be09013a55d6`. The reruns reused the installed
runtime and cached conversions; only evaluation journal entries were
invalidated. Runs used `--no-publish`; no Hub publication was exercised.

The video CLI supports `--video`. Acceptance uses
`--video-fps 1 --video-timestamp-interval 1000`; the CLI defaults of 4 fps and
5000 ms produced an excessive media batch and a memory-slot failure on this
fixture. The reduced sampling completed and retained the video-order gate.

Server requests now put `image_url` or `input_audio` before the text part.
SmolVLM2 VQA requests took 16.113-20.947 seconds and OCR requests took
8.263-10.475 seconds; Qwen3-ASR requests took 1.166-1.765 seconds. These are
not approximately five times the corresponding feasibility-spike timings, so
no server thread override was added. The previous run's CPU-heavy post-run
parser was stopped before this sequential rerun.

The authoritative result files are
`/home/ubuntu/spike/acceptance/smolvlm2-results.json` and
`/home/ubuntu/spike/acceptance/qwen3asr-results.json`. The separate 0.5.4b
feasibility spike is not used as implementation acceptance. The prior result
files are preserved as
`/home/ubuntu/spike/acceptance/smolvlm2-results-pre-order-fix.json` and
`/home/ubuntu/spike/acceptance/qwen3asr-results-pre-order-fix.json`.

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
| F16 | passed | 820421760 | 1.0 | 0.0 / 0.0 | true |
| Q8_0 | passed | 436805760 | 1.0 | 0.0 / 0.0 | true |
| Q6_K | passed | 417759360 | 1.0 | 0.0 / 0.0 | true |
| Q5_K_M | passed | 325560960 | 1.0 | 0.0 / 0.0 | true |
| Q4_K_M | passed | 303250560 | 0.8 | 0.0 / 0.0 | true |

F16 VQA accuracy was 1.0 versus the Transformers reference's 0.8. Its stored
OCR answer was ` INVOICE 4821 Total: 37.50 EUR Due 2026-11-30`; CER and WER were
both 0.0 after whitespace normalization. Every candidate passed video order.
Q4_K_M missed only the black-stripes VQA case and passed at the 0.8 threshold.
The F16 video answer was `First, the video shows a red circle on a white
background, which is a common symbol for Japan. Then, it transitions to a blue
square on the same white background, which is also a symbol for Japan.`

Per-request elapsed seconds are recorded below. VQA times follow the order
red-circle, blue-square, green-triangle, yellow-star, black-stripes; values
are rounded to three decimals.

| Text quant | VQA requests (seconds) | OCR | Video |
|---|---|---:|---:|
| F16 | 18.904, 18.185, 19.180, 20.947, 17.799 | 9.997 | 72.404 |
| Q8_0 | 18.973, 18.934, 16.611, 16.113, 20.367 | 10.475 | 59.562 |
| Q6_K | 17.526, 19.970, 16.637, 17.970, 17.785 | 8.960 | 66.723 |
| Q5_K_M | 18.155, 17.983, 16.939, 17.577, 19.503 | 8.834 | 61.488 |
| Q4_K_M | 20.083, 18.169, 17.046, 17.035, 16.562 | 8.263 | 60.507 |

The native invocation wall time was 14m12.091s. It reused F16 text/projector
conversion and quantized/evaluated the text candidates.

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
what you can do for your country.` after generic ASR markup removal. The raw
answer began `language English<asr_text>`. Audio QA was not evaluated. The
native invocation wall time was 42.256s.

| Text quant | ASR request seconds |
|---|---:|
| F16 | 1.765 |
| Q8_0 | 1.420 |
| Q6_K | 1.211 |
| Q5_K_M | 1.166 |
| Q4_K_M | 1.254 |
