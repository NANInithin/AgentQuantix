# 0.6.0 draft release notes

**DRAFT — not released; version/tag/publication require owner approval**

This draft consolidates work recorded for milestones 0.5.1 through 0.5.9. It
does not change the source package version, which remains 0.4.5. Acceptance
means only the task, family, precision, and Linux x86_64 CPU cell recorded in
the [support matrix](SUPPORT_MATRIX.md).

## Accepted native task/family/precision evidence

| Task | Accepted family and precisions | Status document | Evidence artifact |
|---|---|---|---|
| Automatic speech recognition | Moonshine Streaming Tiny: Q8_0; Qwen3-ASR: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M | [Audio tasks](AUDIO_TASKS_STATUS.md), [multimodal](MULTIMODAL_STATUS.md) | `audio-moonshine-tiny-q8_0-results.json`; `qwen3asr-results.json` |
| Audio source separation | BS-RoFormer: Q8_0 | [Audio tasks](AUDIO_TASKS_STATUS.md) | `audio-bs-roformer-results.json` |
| Voice conversion | MeanVC2: F32, Q4_K | [Audio tasks](AUDIO_TASKS_STATUS.md) | `audio-meanvc2-f32-q4_k-combined-results.json` |
| Forced alignment | Qwen3 Forced Aligner: all nine evaluated precisions | [Audio tasks](AUDIO_TASKS_STATUS.md) | `audio-qwen3-forced-aligner-source-results.json` |
| Embeddings | BAAI/bge-small-en-v1.5: F16, Q8_0, Q6_K, Q5_K_M, Q5_0, Q4_K_M, Q4_0; all-MiniLM-L6-v2: same except Q4_0 | [Retrieval](RETRIEVAL_STATUS.md) | `bge-results.json`; `minilm-results.json` |
| Reranking | BAAI/bge-reranker-base: F16, Q8_0, Q6_K, Q5_K_M, Q5_0, Q4_K_M, Q4_0 | [Retrieval](RETRIEVAL_STATUS.md) | `reranker-results.json` |
| Image feature extraction and zero-shot image classification | openai/clip-vit-base-patch32: F16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0 | [Retrieval](RETRIEVAL_STATUS.md) | `clip-results.json` |
| Multimodal understanding | SmolVLM2-500M-Video-Instruct and Qwen3-ASR-0.6B: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M; audio QA is not accepted | [Multimodal](MULTIMODAL_STATUS.md) | `smolvlm2-results.json`; `qwen3asr-results.json` |
| Mask generation | MobileSAM: F32, F16 | [Vision](VISION_STATUS.md) | `vision-mobilesam-results.json` |
| Image segmentation | BiRefNet: F16; RF-DETR seg-nano: F32, F16 | [Vision](VISION_STATUS.md), [detection](DETECTION_STATUS.md) | `vision-birefnet-results.json`; `detect-rfdetr-seg-nano-results.json` |
| Image classification | YOLO26n-cls: F32, F16, Q8_0; narrow reference-parity sample | [Detection](DETECTION_STATUS.md) | `detect-yolo26n-cls-results.json` |
| Keypoint detection | YOLO26n-pose: F32, F16, Q8_0 | [Detection](DETECTION_STATUS.md) | `detect-yolo26n-pose-results.json` |
| Depth estimation | Vision.cpp Depth Anything V2 Small: F32, F16; DA2 metric Hypersim Small: F32, F16, Q8_0, Q4_K; DA3-Small: F32, F16 | [Vision](VISION_STATUS.md), [depth and SAM](DEPTH_SAM_STATUS.md) | `vision-depth-anything-results.json`; `depth-da2-metric-hypersim-small-results.json`; `depth-da3-small-results.json` |

These results are model- and sample-specific. They do not establish acceptance
for unlisted families or precisions. Text generation, TTS, and stable-diffusion.cpp
media have no accepted family/precision in this matrix.

## Milestone summary

- **0.5.1 — runtime lifecycle:** shared runtime identities, approval decisions,
  immutable build records, adoption, activation/rollback, locks, leases,
  pruning, and frozen-runtime resume. See
  [runtime lifecycle status](RUNTIME_LIFECYCLE_STATUS.md).
- **0.5.2 — shared task router and CLI:** common task routing and shared
  inspect/plan/run/review/publish/verify actions replaced modality-specific
  groups. See [CHANGELOG.md](../CHANGELOG.md) for the breaking changes.
- **0.5.3 — retrieval:** embeddings, reranking, CLIP image features, and
  zero-shot classification, with the accepted Linux CPU cells above. See
  [retrieval status](RETRIEVAL_STATUS.md).
- **0.5.4 — multimodal understanding:** llama.cpp/libmtmd vision, document,
  video-order, and ASR-style audio evaluation with an F16 projector. Audio QA
  remains unaccepted. See [multimodal status](MULTIMODAL_STATUS.md).
- **0.5.5 — extended audio:** source separation, voice conversion, and forced
  alignment were added to the shared voice pipeline. Rejected, blocked, and
  untested audio work is listed below. See
  [audio task status](AUDIO_TASKS_STATUS.md).
- **0.5.6 — vision.cpp:** prompted masks, image segmentation, and relative
  depth were added; inpainting and super-resolution did not pass their naive
  baselines. See [vision status](VISION_STATUS.md).
- **0.5.7 — detection:** family-routed ultralytics-ggml and rf-detr.cpp
  support was added; the accepted classification, pose, and segmentation
  results and rejected detection result are listed above. See
  [detection status](DETECTION_STATUS.md).
- **0.5.8 — depth-anything.cpp:** DA2 metric Hypersim Small and DA3-Small were
  added. DA3 Q8_0 and Q4_K fail camera gates; SAM3 remains blocked. See
  [depth and SAM status](DEPTH_SAM_STATUS.md).
- **0.5.9 — hardening and feature freeze:** package runtime fixtures, shared
  vision-prefixed detection/depth resume records, cache invalidation, the
  [support matrix](SUPPORT_MATRIX.md), and the forecasting feasibility record.

## Breaking CLI changes

In 0.5.2, `aqx voice` and `aqx media` were removed without aliases. Use the
shared `aqx inspect`, `aqx plan`, `aqx run`, `aqx review`, `aqx publish`, and
`aqx verify` commands. TTS review requires explicit `--listened-audio`.
`aqx run --resume RUN_ID` resolves pipeline-prefixed run IDs; detection and
depth use the shared `vision:` prefix.

## Deferred, rejected, and blocked work

| Work | State and reason | Status document |
|---|---|---|
| Time-series forecasting | Blocked and not offered. timeseries.cpp declares no license; the pinned loader rejects Chronos tensor names longer than 64 characters, and TimesFM conversion fails on the local snapshot. No model reached forecasting or reference evaluation. | [Forecasting](FORECASTING_STATUS.md) |
| SAM3 | Blocked. Official weights are gated under the SAM License; an ungated source has no independently verifiable conversion provenance, and sam3cpp is not accepted. | [Depth and SAM](DEPTH_SAM_STATUS.md) |
| Inpainting and super-resolution | Unimplemented. MI-GAN did not beat mean-fill PSNR/SSIM; Real-ESRGAN anime-6B did not beat the bicubic PSNR baseline. | [Vision](VISION_STATUS.md) |
| YOLO detection, segmentation, depth, and World | Not accepted. The 0.5.7 samples missed their task/reference gates: YOLO26n detection recall was 0.8889, instance segmentation failed its mask gates, semantic segmentation had pixel agreement 0.9631 and mIoU 0.7778, YOLO26 depth had Spearman 0.9861 and AbsRel 0.1284, and YOLOv8s-World native precision was 0.9024. OBB was not evaluated. | [Detection](DETECTION_STATUS.md) |
| Object detection | Rejected for this sample. RF-DETR nano F32 reference recall was 0.9478, below the 0.95 gate; lower precisions cannot pass after a failed baseline. | [Detection](DETECTION_STATUS.md) |
| Audio generation, audio-to-audio, classification, VAD, enhancement, diarization, and MIDI | Not accepted. Audio generation lacks an eligible package within the evaluation size preference or a permissive license; the pinned backend exposes no standalone classification task; VAD has no accepted standalone source; enhancement paths worsened SNR; diarization and MIDI candidates have non-commercial restrictions. | [Audio tasks](AUDIO_TASKS_STATUS.md) |
| Text generation and TTS | Experimental in the support matrix with no accepted native family/precision evidence. Required native acceptance is still outstanding. | [Support matrix](SUPPORT_MATRIX.md), [release validation](RELEASE_VALIDATION.md) |
| Stable-diffusion.cpp image/video/audio-video | Experimental with no accepted native family/precision. Real-model inference, quality calibration, MiniMax-H3 human review, and post-download bundle/card verification are outstanding. | [Media implementation](MEDIA_IMPLEMENTATION_STATUS.md), [stable-diffusion.cpp release plan](STABLE_DIFFUSION_CPP_RELEASE_PLAN.md) |

## Release boundary

Windows/MSVC builds and acceptance for the new engines, GPU/Metal, real Hub
publication and re-download, and owner approval have not been established.
This document is a draft only; version, tag, package publication, and release
announcement require owner approval.
