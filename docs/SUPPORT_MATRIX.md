# AgentQuantix support matrix

This is the feature-freeze view of every registered `TaskSpec`. “Accepted”
means at least one named family and precision passed native evaluation; it
does not imply every catalog entry, precision, or runtime is accepted.
Catalog presence alone is not acceptance. Platform evidence is Linux x86_64
CPU only. Windows, GPU/Metal, and real Hub publication have not been verified.

| TaskSpec | Engine / family | State | Accepted precisions | Platform | Status document | Evidence artifact |
|---|---|---|---|---|---|---|
| `text-generation` | llama.cpp / architecture-selected GGUF | experimental | None in this acceptance set | Linux x86_64 CPU only | [runtime lifecycle](RUNTIME_LIFECYCLE_STATUS.md) | None |
| `text-to-speech` | llama.cpp (`pocket-tts`, `qwen3-tts`); audio.cpp catalog | experimental | None in this acceptance set | Linux x86_64 CPU only | [audio tasks](AUDIO_TASKS_STATUS.md) | None |
| `automatic-speech-recognition` | whisper.cpp (`whisper`); audio.cpp (`moonshine_asr`, `qwen3_asr`) | accepted | Moonshine Tiny: Q8_0; Qwen3-ASR: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M | Linux x86_64 CPU only | [audio tasks](AUDIO_TASKS_STATUS.md), [multimodal](MULTIMODAL_STATUS.md) | `audio-moonshine-tiny-q8_0-results.json`; `qwen3asr-results.json` |
| `audio-source-separation` | audio.cpp (`bs_roformer`, `htdemucs`, `mel_band_roformer`) | accepted | BS-RoFormer: Q8_0; HTDemucs rejected | Linux x86_64 CPU only | [audio tasks](AUDIO_TASKS_STATUS.md) | `audio-bs-roformer-results.json`; `audio-htdemucs-f16-results.json`; `audio-htdemucs-q8_0-results.json` |
| `voice-conversion` | audio.cpp (`meanvc2`) | accepted | F32, Q4_K | Linux x86_64 CPU only | [audio tasks](AUDIO_TASKS_STATUS.md) | `audio-meanvc2-f32-q4_k-combined-results.json` |
| `forced-alignment` | audio.cpp (`qwen3_forced_aligner`) | accepted | All nine evaluated precisions; see result manifest | Linux x86_64 CPU only | [audio tasks](AUDIO_TASKS_STATUS.md) | `audio-qwen3-forced-aligner-source-results.json` |
| `text-to-image` | stable-diffusion.cpp / task-compatible entries in the [pinned capability catalog](../src/agentquantix/fixtures/stable_diffusion_cpp_capabilities.json) | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `image-to-image` | stable-diffusion.cpp / task-compatible entries in the pinned capability catalog | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `image-editing` | stable-diffusion.cpp / task-compatible entries in the pinned capability catalog | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `text-to-video` | stable-diffusion.cpp / task-compatible entries in the pinned capability catalog | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `image-to-video` | stable-diffusion.cpp / task-compatible entries in the pinned capability catalog | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `first-last-to-video` | stable-diffusion.cpp / no matching registered family | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `reference-to-video` | stable-diffusion.cpp (`minimax_h3`) | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `text-to-audio-video` | stable-diffusion.cpp (`minimax_h3`) | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `image-to-audio-video` | stable-diffusion.cpp (`minimax_h3`) | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `first-last-to-audio-video` | stable-diffusion.cpp (`minimax_h3`) | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `reference-to-audio-video` | stable-diffusion.cpp (`minimax_h3`) | experimental | None accepted | Linux x86_64 CPU only | [media implementation](MEDIA_IMPLEMENTATION_STATUS.md) | None |
| `embeddings` | llama.cpp (`BAAI/bge-small-en-v1.5`, `all-MiniLM-L6-v2`) | accepted | BGE: F16, Q8_0, Q6_K, Q5_K_M, Q5_0, Q4_K_M, Q4_0; MiniLM: same except Q4_0 rejected | Linux x86_64 CPU only | [retrieval](RETRIEVAL_STATUS.md) | `bge-results.json`; `minilm-results.json` |
| `reranking` | llama.cpp (`BAAI/bge-reranker-base`) | accepted | F16, Q8_0, Q6_K, Q5_K_M, Q5_0, Q4_K_M, Q4_0 | Linux x86_64 CPU only | [retrieval](RETRIEVAL_STATUS.md) | `reranker-results.json` |
| `zero-shot-image-classification` | clip.cpp (`openai/clip-vit-base-patch32`) | accepted | F16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0 | Linux x86_64 CPU only | [retrieval](RETRIEVAL_STATUS.md) | `clip-results.json` |
| `image-feature-extraction` | clip.cpp (`openai/clip-vit-base-patch32`) | accepted | F16, Q8_0, Q5_1, Q5_0, Q4_1, Q4_0 | Linux x86_64 CPU only | [retrieval](RETRIEVAL_STATUS.md) | `clip-results.json` |
| `multimodal-understanding` | llama.cpp/libmtmd (`SmolVLM2-500M-Video-Instruct`, `Qwen3-ASR-0.6B`) | accepted | SmolVLM2 and Qwen3-ASR: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M; audio QA not accepted | Linux x86_64 CPU only | [multimodal](MULTIMODAL_STATUS.md) | `smolvlm2-results.json`; `qwen3asr-results.json` |
| `audio-generation` | audio.cpp / no accepted family | blocked | None | Linux x86_64 CPU only | [audio tasks](AUDIO_TASKS_STATUS.md) | None |
| `audio-to-audio` | audio.cpp / no release adapter | unimplemented | None | Linux x86_64 CPU only | [audio tasks](AUDIO_TASKS_STATUS.md) | None |
| `audio-classification` | audio.cpp / no native classification task | blocked | None | Linux x86_64 CPU only | [audio tasks](AUDIO_TASKS_STATUS.md) | None |
| `voice-activity-detection` | audio.cpp/whisper.cpp / no accepted standalone source | blocked | None | Linux x86_64 CPU only | [audio tasks](AUDIO_TASKS_STATUS.md) | None |
| `image-segmentation` | vision.cpp (`BiRefNet`); rf-detr.cpp (`rfdetr-seg-nano`) | accepted | BiRefNet: F16; RF-DETR seg-nano: F32, F16 | Linux x86_64 CPU only | [vision](VISION_STATUS.md), [detection](DETECTION_STATUS.md) | `vision-birefnet-results.json`; `detect-rfdetr-seg-nano-results.json` |
| `super-resolution` | vision.cpp (`Real-ESRGAN anime-6B`) | unimplemented | None; failed naive-baseline comparison | Linux x86_64 CPU only | [vision](VISION_STATUS.md) | None (spike evidence only) |
| `inpainting` | vision.cpp (`MI-GAN`) | unimplemented | None; failed naive-baseline comparison | Linux x86_64 CPU only | [vision](VISION_STATUS.md) | None (spike evidence only) |
| `object-detection` | rf-detr.cpp (`rfdetr-nano`); ultralytics-ggml (no accepted detection family) | rejected | None; every RF-DETR nano precision rejected | Linux x86_64 CPU only | [detection](DETECTION_STATUS.md) | `detect-rfdetr-nano-results.json` |
| `image-classification` | ultralytics-ggml (`yolo26n-cls`) | accepted | F32, F16, Q8_0; narrow parity sample | Linux x86_64 CPU only | [detection](DETECTION_STATUS.md) | `detect-yolo26n-cls-results.json` |
| `keypoint-detection` | ultralytics-ggml (`yolo26n-pose`) | accepted | F32, F16, Q8_0 | Linux x86_64 CPU only | [detection](DETECTION_STATUS.md) | `detect-yolo26n-pose-results.json` |
| `depth-estimation` | vision.cpp (`Depth Anything V2 Small`); depth-anything.cpp (`da2-metric-hypersim-small`, `da3-small`) | accepted | Vision.cpp: F32, F16; DA2 metric: F32, F16, Q8_0, Q4_K; DA3: F32, F16 (Q8_0/Q4_K rejected) | Linux x86_64 CPU only | [vision](VISION_STATUS.md), [depth/SAM](DEPTH_SAM_STATUS.md) | `vision-depth-anything-results.json`; `depth-da2-metric-hypersim-small-results.json`; `depth-da3-small-results.json` |
| `mask-generation` | vision.cpp (`MobileSAM`); sam3cpp (blocked) | accepted | MobileSAM: F32, F16 | Linux x86_64 CPU only | [vision](VISION_STATUS.md), [depth/SAM](DEPTH_SAM_STATUS.md) | `vision-mobilesam-results.json` |
| `time-series-forecasting` | timeseries.cpp (`Chronos T5 tiny`, `TimesFM 1.0 200M`) | blocked | None; neither model reached forecasting | Linux x86_64 CPU only | [forecasting](FORECASTING_STATUS.md) | `/home/ubuntu/spike/0.5.9/REPORT.md`; no acceptance artifact |

Forecasting remains out of scope pending a licence and upstream fixes. Runtime
catalogues can contain experimental or blocked families; registry presence is
not a support claim. Artifact basenames refer to `/home/ubuntu/spike/acceptance/`
unless an absolute path is shown.
