# Usage

The shared entry point is `aqx run org/model`. Routing uses Hub metadata and
engine catalogs; add `--task` or `--family` only to resolve an ambiguity or a
custom repository. The accepted evidence scope is Linux x86_64 CPU and is
limited to the task, family, and precision recorded in the
[support matrix](docs/SUPPORT_MATRIX.md).

The examples use `org/model` where any Hub repository can be substituted.
`--no-publish` keeps an evaluation local. Use the matching task's status
document for its complete evidence and limits.

## Install and common commands

Install the extras needed by your task. For development plus media and
detection helpers:

```sh
python -m pip install -e '.[dev,media,detection]'
```

The `voice`, `retrieval`, `multimodal`, `convert`, and `media-quality` extras
provide additional optional dependencies. Per-task converter environments are
created automatically and kept separate from the main install.

```sh
aqx inspect org/model
aqx plan org/model
aqx run org/model --no-publish
aqx runs
aqx run --resume RUN_ID
```

`aqx run --resume RUN_ID` uses the recorded plan and frozen runtime. Detection
and depth run IDs use the shared `vision:` prefix; use
`aqx runs --pipeline vision` to list vision, detection, and depth records.

## Text

**Task:** `text-generation`

**Target:** `org/model`

**Commands:**

```sh
aqx plan org/model --task text-generation
aqx run org/model --task text-generation --no-publish
aqx runs --pipeline text
aqx run --resume RUN_ID
```

**Quality gate:** No text-generation family or precision has accepted native
evidence in the support matrix; the native recipe acceptance requires
conversion, quantization, inference, and the expected-text check in
[release validation](docs/RELEASE_VALIDATION.md).

## Voice

**Tasks:** `text-to-speech`, `automatic-speech-recognition`,
`audio-source-separation`, `voice-conversion`, `forced-alignment`

**Targets:** `org/model` or `audio.cpp:<family>`

**Commands:**

```sh
aqx plan org/model --task text-to-speech
aqx plan audio.cpp:bs_roformer --task audio-source-separation
aqx run audio.cpp:bs_roformer --task audio-source-separation --no-publish
aqx runs --pipeline voice
aqx run --resume RUN_ID
```

**Quality gate:** Separation requires at least 6 dB improvement for both stems
at baseline and other precisions within 1 dB; voice conversion requires finite,
non-silent output, duration within 20%, and round-trip WER at most 0.35 at
baseline and at most baseline +0.10 otherwise; alignment requires valid
monotonic intervals and mean/max baseline deltas at most 0.05/0.20 seconds.
TTS publication also requires listening review, and no TTS precision is
accepted in the support matrix.

See [audio task status](docs/AUDIO_TASKS_STATUS.md) for accepted families,
precision results, rejected tasks, and gate definitions.

## Media

**Tasks:** `text-to-image`, `image-to-image`, `image-editing`,
`text-to-video`, `image-to-video`, `first-last-to-video`,
`reference-to-video`, `text-to-audio-video`, `image-to-audio-video`,
`first-last-to-audio-video`, `reference-to-audio-video`

**Target:** `org/model`; select a known engine family with `--family FAMILY`.

**Commands:**

```sh
aqx plan org/model --task text-to-image --output media-plan.json
aqx run org/model --plan media-plan.json --no-publish
aqx runs --pipeline media
aqx run --resume RUN_ID
aqx review media:RUN_ID --quant q4_0 --accept --reviewer 'Your name'
aqx publish media:RUN_ID --quant q4_0
```

**Quality gate:** Media has no accepted native family in the support matrix;
publication requires structural checks, a calibrated quality profile and
high-precision baseline gates, plus human review (video playback and, for
audio-video, listening); without a profile the run is structural-only and
cannot publish.

See [media usage](docs/MEDIA_USAGE.md) and
[media implementation status](docs/MEDIA_IMPLEMENTATION_STATUS.md). Use
`--watched-video` for video reviews and `--listened-audio` for audio-video
reviews; media publication remains gated on native acceptance.

## Retrieval

**Tasks:** `embeddings`, `reranking`, `image-feature-extraction`,
`zero-shot-image-classification`

**Target:** `org/model`

**Commands:**

```sh
aqx plan org/model --task embeddings
aqx run org/model --task embeddings --no-publish
aqx plan org/model --task reranking
aqx plan org/model --task image-feature-extraction
aqx plan org/model --task zero-shot-image-classification
aqx runs --pipeline retrieval
aqx run --resume RUN_ID
```

**Quality gate:** Embeddings require F16/reference cosine at least 0.999 and
quant/F16 cosine at least 0.98 plus normalization and retrieval-score checks;
reranking requires F16 max logit error at most 0.1 and exact relevant ordering,
with quant top-1 agreement; CLIP requires F16 image/text cosine at least 0.999
and quant cosine at least 0.95, with classification accuracy no worse than
F16.

See [retrieval status](docs/RETRIEVAL_STATUS.md) for accepted models and exact
per-task thresholds.

## Multimodal

**Task:** `multimodal-understanding`

**Target:** `org/model`

**Commands:**

```sh
aqx plan org/model --task multimodal-understanding
aqx run org/model --task multimodal-understanding --no-publish
aqx runs --pipeline multimodal
aqx run --resume RUN_ID
```

**Quality gate:** The evaluated F16 baseline must pass the image, audio, or
video task gates; lower text precisions are compared against F16 with the
shared projector, and a missing or mismatched projector blocks release.

See [multimodal status](docs/MULTIMODAL_STATUS.md) for the accepted models,
evaluated precisions, and limits.

## Vision

**Tasks:** `mask-generation`, `image-segmentation`, `depth-estimation`

**Targets:** `org/model` or `vision.cpp:<family>`

**Commands:**

```sh
aqx plan vision.cpp:sam --task mask-generation
aqx plan vision.cpp:birefnet --task image-segmentation
aqx plan vision.cpp:depth-anything --task depth-estimation
aqx run vision.cpp:sam --task mask-generation --no-publish
aqx runs --pipeline vision
aqx run --resume RUN_ID
```

**Quality gate:** MobileSAM requires point and box IoU at least 0.90 and
lower-precision IoU within 0.01 with pixel agreement at least 0.99; BiRefNet
requires IoU at least 0.90 and boundary F-score at least 0.80, with the same
lower-precision bounds; vision.cpp depth is relative/ordinal and uses the
three-scene Spearman and scale-shift AbsRel gates in its status document.

See [vision status](docs/VISION_STATUS.md). The vision.cpp depth family is
relative depth only; metric depth is documented under the separate depth
pipeline below.

## Detection

**Tasks:** `object-detection`, `image-segmentation`, `image-classification`,
`keypoint-detection`

**Targets:** `ultralytics-ggml:<family>` or `rf-detr.cpp:<family>`

**Commands:**

```sh
aqx plan ultralytics-ggml:yolo26n-cls --task image-classification
aqx run ultralytics-ggml:yolo26n-cls --task image-classification --no-publish
aqx plan ultralytics-ggml:yolo26n-pose --task keypoint-detection
aqx plan rf-detr.cpp:rfdetr-seg-nano --task image-segmentation
aqx runs --pipeline vision
aqx run --resume RUN_ID
```

**Quality gate:** A family is accepted only when its native outputs pass the
task-specific reference and precision-regression thresholds; object detection
is rejected, and no object-detection family or precision is accepted in the
support matrix.

See [detection status](docs/DETECTION_STATUS.md) for the accepted classification,
pose, and segmentation results and the rejected RF-DETR object-detection
baseline.

## Depth

**Task:** `depth-estimation`

**Targets:** `depth-anything.cpp:<family>` or `vision.cpp:<family>`

**Commands:**

```sh
aqx plan depth-anything.cpp:da2-metric-hypersim-small --task depth-estimation
aqx run depth-anything.cpp:da2-metric-hypersim-small --task depth-estimation --no-publish
aqx plan depth-anything.cpp:da3-small --task depth-estimation
aqx runs --pipeline vision
aqx run --resume RUN_ID
```

**Quality gate:** DA2 F32 and its quant candidates require Spearman at least
0.99, AbsRel at most 0.05, and median absolute error at most 5% of reference
median; DA3 additionally requires rotation at most 1 degree, translation L2
at most 0.001, and intrinsics relative error at most 0.01, so Q8_0 and Q4_K
are rejected for DA3.

See [depth and SAM status](docs/DEPTH_SAM_STATUS.md) for family semantics,
producers, results, and the SAM3 blocker.

## Reviews, publication, and status

The CLI help documents the shared review and publish forms:

```sh
aqx review RUN_ID --quant QUANT --accept --reviewer 'Your name'
aqx publish RUN_ID --quant QUANT
aqx verify TARGET
```

Reviews are bound to the run and candidate artifact. Use `--listened-audio`
for TTS, `--watched-video` for video, and both flags for audio-video. A review
does not itself publish a candidate; publication is a separate command and is
blocked when the release gates are not satisfied.
