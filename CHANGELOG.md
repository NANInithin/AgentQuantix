# Changelog

## 0.5.9 — unreleased development

- Package runtime fixture data and route catalogs in wheels and sdists; add an
  installed-artifact smoke check for task routing and the detection manifest.
- Publish the [feature-freeze support matrix](docs/SUPPORT_MATRIX.md), covering
  every TaskSpec and distinguishing accepted evidence from catalog-only routes.
- Defer timeseries.cpp forecasting pending a source license and upstream fixes;
  see [forecasting status](docs/FORECASTING_STATUS.md).
- Resolve detection and depth runs through the shared `vision:` run index and
  resume vision-engine releases against their frozen runtime snapshots.

## 0.5.8 — unreleased development

- Add pinned `depth-anything.cpp` support for DA2 metric Hypersim Small and
  DA3-Small. Convert F32, derive F16/Q8_0/Q4_K with the native quantizer,
  evaluate against the PyTorch references on the license-filtered COCO sample,
  fail closed on baseline gates, and publish accepted precisions atomically.
- Keep DA2 relative Small on vision.cpp. Keep SAM3 blocked because official
  weights are gated and no ungated source has verifiable conversion provenance.
- See [depth and SAM status](docs/DEPTH_SAM_STATUS.md) for the native result
  matrix, source and engine licenses, and sample limitations.

## 0.5.7 — unreleased development

- Add family-routed `ultralytics-ggml` and `rf-detr.cpp` detection support with
  SHA-verified sources, pinned converters, D(0.25) parity gates and atomic
  publication of passing precisions.
- Reject RF-DETR nano object detection on this sample: F32 recall 0.9478 is
  below 0.95, so no precision is accepted and no object-detection family is
  accepted in 0.5.7. Keep the family in the catalog for run-time gating.
- Keep YOLO detection, instance segmentation, semantic segmentation, depth,
  YOLO-World and OBB blocked or unoffered according to the 0.5.7 spike gates.
- See [detection task status](docs/DETECTION_STATUS.md) for spike evidence,
  licences, native acceptance results and sample limits.

## 0.5.6 — unreleased development

- Add a vision.cpp pipeline for MobileSAM prompted masks, BiRefNet image
  segmentation and Depth Anything V2 relative depth, with source-aware F32/F16
  conversion, automated quality gates and atomic passing-precision bundles.
- Keep inpainting and super-resolution unimplemented: native outputs did not
  beat the naive baseline under PSNR/SSIM, and no perceptual gate is established.
- See [vision task status](docs/VISION_STATUS.md) for spike evidence and native
  acceptance limits.

## 0.5.5 — unreleased development

- Extend the catalog-driven audio.cpp voice pipeline with source separation,
  voice conversion and forced alignment. Each task has native quality gates,
  precision regression checks and atomic bundles.
- Add deterministic CC0 separation and voice-reference fixtures. Keep audio
  generation, classification and standalone VAD explicitly blocked with
  licensing or upstream-support reasons.
- Record source licensing and disclose prebuilt package provenance and
  measurement limits in task plans and model cards.
- Native Linux CPU acceptance on audio.cpp v0.8.0 accepts BS-RoFormer Q8_0,
  MeanVC2 F32/Q4_K, all nine Qwen3 aligner precisions, and the Moonshine
  Streaming Tiny Q8_0 ASR regression; HTDemucs fails the separation gate. No
  enhancement path improved SNR.
- The Moonshine regression exposed a redundant language option for
  single-language ASR backends. The command now omits that flag when catalog
  metadata declares a single language.
- See [audio task status](docs/AUDIO_TASKS_STATUS.md) for native results and
  untested categories.

## 0.5.4 — unreleased development

- Add the `multimodal-understanding` llama.cpp pipeline for explicit vision and
  audio task routing, native F16 projector conversion, image/document/video
  evaluation, ASR-style audio evaluation, resumable runs and atomic bundles.
- Keep `image-text-to-text` and `any-to-any` on the text-generation pipeline
  unless `--task multimodal-understanding` is selected. List runs with
  `aqx runs --pipeline multimodal`; install reference dependencies with
  `pip install "agentquantix[multimodal]"`.
- Keep projectors at F16; gate text quants against the F16 multimodal task
  baseline. Video is CLI-only and requires ffmpeg; audio input remains
  experimental upstream and the accepted audio path is ASR-style only.

## 0.5.3 — unreleased development

- Add retrieval tasks for embeddings and reranking on llama.cpp, plus image
  feature extraction and zero-shot image classification on clip.cpp. Use
  `--task` to resolve a task explicitly when metadata is ambiguous; list saved
  retrieval runs with `aqx runs --pipeline retrieval`.
- Use `llama-server --embedding` with `/v1/embeddings` for embeddings and
  `llama-server --reranking` with `/v1/rerank` for reranking. clip.cpp uses
  `extract`, `zsl` and `quantize`. Reranking `relevance_score` values are raw,
  unbounded logits, not probabilities.
- Normalize CLIP text prompts to lowercase and remove punctuation before
  encoding. The `llama-embedding` CLI is not used: `--pooling rank` produces
  broken output at the pinned llama.cpp revision, and its async logger can
  truncate stdout nondeterministically. The server embedding endpoint avoids
  that CLI path.
- Add the optional `retrieval` extra for reference evaluation dependencies:
  `pip install "agentquantix[retrieval]"`.

## 0.5.2 — unreleased development

- Add a shared task/capability registry (`agentquantix.tasks`) and metadata
  routing: installed engine catalogs, Hub task metadata, library contracts and
  transformers head classes resolve each repository's task, pipeline and engine.
  Conflicting evidence returns a short `--task` choice. Recognised tasks without
  an adapter (embeddings, reranking, CLIP, multimodal understanding, extended
  audio, vision, detection, depth, SAM, forecasting) are blocked with their
  milestone instead of falling into the text imatrix pipeline.
- Add shared actions used by both the CLI and every agent harness: `inspect`,
  `plan`, `run` (including `--resume RUN_ID` and `--plan FILE`), `review`,
  `publish`, `verify`, `card` and `runs`. Run ids are `text:<name>`,
  `voice:<slug>` and `media:<plan_id>`; existing records are discovered in place
  and older voice workspaces gain a `run.json` on first use.
- **Breaking:** remove the `aqx voice` and `aqx media` command groups with no
  aliases, and replace the modality-specific agent tools with the shared ones.
  TTS reviews now require an explicit `--listened-audio` attestation.

| Removed | Replacement |
|---|---|
| `aqx voice list`, `aqx media backends` | `aqx runtime show <engine>` |
| `aqx media backends --build` | `aqx runtime install stable-diffusion.cpp` |
| `aqx media inspect <repo>` | `aqx inspect <repo>` |
| `aqx voice plan <repo>`, `aqx media plan <repo>` | `aqx plan <repo>` |
| `aqx voice run <repo>`, `aqx media run <repo>` | `aqx run <repo>` (`--plan FILE` for a saved media plan) |
| `aqx voice review <repo> <quant> --accept` | `aqx review <run_id> --quant <quant> --accept --listened-audio --reviewer NAME` |
| `aqx media review <run_id> --accept <quant>` | `aqx review media:<run_id> --quant <quant> --accept --reviewer NAME` |
| `aqx media publish <run_id> --quant <quant>` | `aqx publish media:<run_id> --quant <quant>` |
| `aqx voice verify <bundle.json>`, `aqx media verify <manifest.json>` | `aqx verify <path>` |
| tools `describe_candidate`, `inspect_media_model` | `inspect_model` |
| tools `plan_quantization`, `plan_voice_release`, `plan_media_release` | `plan_release` |
| tools `start_quantization`, `start_voice_release`, `start_media_release` | `start_release` |
| tools `record_voice_review`, `record_media_review` | `record_review` |
| tool `publish_media_release` | `publish_release` |
| tool `verify_published` | `verify_release` |

## 0.5.1 — unreleased development

- Add shared original-runtime installation/update management for llama.cpp,
  audio.cpp, whisper.cpp and stable-diffusion.cpp, with persistent/temporary
  storage choices, immutable revision/build manifests and explicit update approval.
- Add runtime CLI/MCP maintenance, offline installed-version use, local-source
  setup, adoption/migration, side-by-side builds, rollback and lease-safe cleanup.
- Bind interrupted text/voice/media work to recorded runtimes and use selected
  audio/media capability catalogs after updates.
- Preserve the current package version and outstanding 0.5.0 native acceptance
  requirements. See docs/RUNTIME_LIFECYCLE_STATUS.md for implementation boundaries.

## 0.4.5

- Stop text quantization after exhausted upload retries. Reservations include
  retained failures so concurrent producers cannot exceed their file budget.
- Treat fresh remote listings as authoritative on resume. Hub listing outages
  fail closed and preserve local artifacts.
- Write text quantization and BF16 conversion outputs atomically; interrupted
  downloads are retried, and upload status updates use atomic replacement.
- Return nonzero CLI status for incomplete jobs, failed verification, suspicious
  files, or failed model-card publication.
- Add recovery regression tests and an opt-in native acceptance runner for
  llama.cpp text/TTS, audio.cpp TTS/ASR, and whisper.cpp ASR.

## 0.5.0 — unreleased

Development follows `docs/STABLE_DIFFUSION_CPP_RELEASE_PLAN.md`. The separate
media engine, evidence catalog, plans, bundles, CLI/tools, conversion journal,
quality interfaces and review/publication gates are experimental. A catalog
entry proves loader evidence; it does not prove model-level native acceptance.
The final 0.5.0 version must not be cut before the plan's native, cross-platform,
MiniMax-H3 human review, publication and re-download acceptance gates pass.
