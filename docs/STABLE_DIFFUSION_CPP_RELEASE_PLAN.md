# stable-diffusion.cpp release plan

## Release target

**AgentQuantix v0.5.0** adds `stable-diffusion.cpp` as the native GGUF
backend for image, image-editing, video, and joint audio-video models.

This is a backend-wide integration, not a MiniMax-H3 allowlist. AgentQuantix
must expose every model family and quantization route supported by the pinned
`stable-diffusion.cpp` revision when that support can be verified from the
runtime's source, converter, or tests. Upgrading the pinned backend and syncing
its capability manifest must be sufficient to discover newly supported
families without adding model-specific branches to the core pipeline.

MiniMax-H3 is the first complete audio-video acceptance model because it
exercises the hardest parts of the design: two quantizable GGUF components,
two retained VAE companions, optional vision weights, reference conditioning,
video output, and stereo audio output.

References:

- [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp)
- [GGUF conversion and quantization](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/quantization_and_gguf.md)
- [MiniMax-H3 runtime contract](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/minimax_h3.md)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

## Product principles

1. **The backend defines support**

   AgentQuantix must not maintain a hand-written list of a few fashionable
   models. The pinned `stable-diffusion.cpp` revision, its loaders, converter,
   documentation, and executable probes define what is supported.

2. **A component bundle is the publication unit**

   Diffusion releases are usually not one GGUF. A runnable release may require
   a diffusion transformer or UNet, one or more text encoders, a VAE, an audio
   VAE, a vision tower, tokenizer/configuration files, and scheduler metadata.
   AgentQuantix must plan, resume, verify, publish, and document the complete
   bundle atomically.

3. **Runtime support and conversion support are separate facts**

   A model may load from an existing GGUF while its original SafeTensors
   checkpoint cannot be converted by the pinned revision. The capability
   registry must report `load`, `convert`, and `quantize` independently and
   must never infer one from another.

4. **Quantization is component-aware**

   Quantize only components and tensors accepted by the backend. VAEs, audio
   VAEs, normalization tensors, embeddings, and small conditioning layers may
   need to remain FP16, BF16, or FP32. A requested release quant describes a
   bundle policy, not a command to force every tensor to the same type.

5. **Real inference is the release gate**

   A valid GGUF header is necessary but insufficient. Every candidate must be
   exercised through `sd-cli` using the intended generation or editing mode.

6. **Quality is measured against a high-precision baseline**

   Image and video quantization can produce structurally valid but semantically
   broken output. Candidate quants must be compared with the same seed, prompt,
   dimensions, steps, scheduler, and conditioning inputs as the BF16/FP16
   baseline.

7. **No false universality**

   Support means the exact source layout, component roles, conversion path, and
   runtime command have been resolved. A matching model name or architecture
   string alone is not executable evidence.

## Scope

### Included in v0.5.0

- Text-to-image generation.
- Image-to-image generation.
- Supported image editing and reference-image workflows.
- Text-to-video and image-to-video generation.
- Joint audio-video generation where the backend supports it, beginning with
  MiniMax-H3.
- Direct conversion from the checkpoint formats accepted by
  `stable-diffusion.cpp`.
- Installation of already-published GGUF component bundles.
- Every quant type reported by the pinned converter, filtered through each
  component's tensor policy and runtime validation.
- CPU, CUDA, Vulkan, Metal, HIP, and other hardware modes already exposed by
  the backend, selected through the existing AgentQuantix system probe.
- Backend fork and pull-request hunting for unresolved diffusion families.
- Atomic Hugging Face publication with generated model cards and quality data.

### Deferred unless already exposed by the backend contract

- Training, fine-tuning, and LoRA creation.
- Serving APIs or a permanent generation server.
- Arbitrary ComfyUI graphs and third-party custom nodes.
- Quality claims based only on metadata or community filenames.
- Reimplementation of unsupported model architectures inside AgentQuantix.
- Distributing components whose upstream license prohibits redistribution.

## Architecture

### 1. Generalize the runtime capability registry

Introduce a modality-neutral registry instead of extending `VoiceBackend` with
image and video conditionals. A backend capability should include:

```text
RuntimeCapability
  id
  engine
  family
  tasks
  source_matchers
  architecture_matchers
  load_formats
  converter
  quantizer
  runtime
  component_contract
  quant_policies
  conditioning_inputs
  output_contract
  license_policy
  evidence
  backend_revision
```

The existing text and voice registries can remain operational during v0.5.0,
but the new registry must be designed so they can migrate to the same data
model later.

Use `engine` for `stable-diffusion.cpp` and reserve `compute_backend` for CUDA,
Vulkan, Metal, CPU, and similar hardware implementations. This avoids the two
different meanings of “backend” colliding in plans and reports.

### 2. Generate a pinned capability manifest

`stable-diffusion.cpp` does not currently provide the same `model_specs`
catalog used by `audio.cpp`. Add a deterministic sync tool that inspects a
pinned checkout and produces a reviewed manifest containing:

- Model families detected by the loaders.
- Supported task modes and their `sd-cli` arguments.
- Required and optional component roles.
- Accepted source and runtime formats.
- Converter and quantization support.
- Component-specific tensor rules documented upstream.
- Evidence paths, commit SHA, and extraction timestamp.

The checked-in manifest is an offline planning fallback and must always record
the upstream commit from which it was generated. CI must fail when regeneration
from the pinned revision changes the manifest unexpectedly.

Where source inspection is ambiguous, mark the route `experimental` or
`runtime-only`; do not silently turn documentation examples into universal
conversion claims.

### 3. Represent component graphs explicitly

Add a bundle graph with typed members and dependencies:

```text
MediaBundle
  manifest.json
  components[]
    role
    source_repo
    source_revision
    source_path
    output_path
    format
    precision
    sha256
    required
    quantized
    converter_revision
  runtime_recipe
  quality_results
  license_results
```

Initial component roles should cover:

- `diffusion_model`
- `unet`
- `text_encoder`
- `text_encoder_2`
- `llm`
- `llm_vision`
- `vae`
- `audio_vae`
- `clip_l`
- `clip_g`
- `t5xxl`
- `controlnet`
- `upscaler`
- `tokenizer`
- `config`
- `scheduler`

Unknown required roles block publication. Optional LoRAs, ControlNets, and user
inputs are recorded in the runtime recipe but are not copied into a base-model
release unless the user explicitly requests them and their licenses permit it.

### 4. Add a dedicated media pipeline

Create a separate pipeline rather than placing diffusion logic in the text or
voice release modules:

```text
src/agentquantix/media.py
src/agentquantix/pipeline/media_release.py
src/agentquantix/fixtures/media/
src/agentquantix/fixtures/stable_diffusion_cpp_capabilities.json
```

Expose matching agent tools and CLI commands:

```bash
aqx media backends
aqx media inspect MiniMax-AI/MiniMax-H3
aqx media plan MiniMax-AI/MiniMax-H3
aqx media run MiniMax-AI/MiniMax-H3 --sequential
aqx media review <run-id> --accept Q4_K_M --reviewer "name"
```

The natural-language agent may continue accepting `quantize <repo>`, but it
must inspect the model and route it to text, voice, or media before planning.
It must never send a diffusion model through the llama.cpp text converter.

### 5. Build and pin stable-diffusion.cpp independently

- Pin a tested upstream commit, not a moving default branch.
- Clone with required submodules and record all resolved revisions.
- Cache builds by engine revision, platform, compute backend, compiler, and
  relevant CMake flags.
- Build `sd-cli` and any required converter targets.
- Detect available compute backends through the system probe.
- Keep the `stable-diffusion.cpp` ggml revision isolated from llama.cpp and
  audio.cpp; never assume ABI compatibility between their builds.
- Surface build provenance in every plan, manifest, card, and failure report.

Because upstream changes rapidly, an engine upgrade is its own reviewed change
with manifest regeneration and regression tests.

## Planning and source resolution

### 6. Resolve repositories without per-model branches

Planning should inspect:

- Hugging Face `config.json`, model index, component directories, and
  SafeTensors indexes.
- Architecture names, pipeline tags, library name, tags, and model-card data.
- File names only as supporting evidence, never as the sole match.
- The generated backend capability manifest.
- Existing official or community GGUF components.

Resolution produces one of four states:

- `convertible`: source components map completely to a verified converter
  contract.
- `prebuilt`: a complete runnable GGUF bundle already exists.
- `runtime-only`: upstream can run the family, but AgentQuantix lacks a proven
  source-to-GGUF conversion contract.
- `blocked`: the engine does not support the architecture or required
  companions.

Ambiguous component mapping must return a structured question or blocked plan,
not a guessed conversion command.

### 7. Add stable-diffusion.cpp fork hunting

When no installed capability resolves a likely image or video model, search:

- Publisher-owned `stable-diffusion.cpp` forks.
- Open upstream pull requests mentioning the family or architecture.
- Converter scripts and runtime branches linked by the publisher.
- Upstream issues only as non-executable leads.

Cache positive and negative results with the existing evidence model. A fork
or pull request becomes executable only after AgentQuantix verifies its build,
converter, component contract, and smoke inference. Plans must distinguish
`lead`, `buildable`, and `validated` status.

## Conversion and quantization

### 8. Probe converter capabilities at runtime

Use `sd-cli` help and a small converter probe to discover supported types from
the pinned executable. The checked-in fallback must never override a newer
successful probe.

For source formats accepted upstream, use the native conversion flow:

```bash
sd-cli -M convert \
  -m component.safetensors \
  -o component-Q8_0.gguf \
  --type q8_0
```

Preserve exact stdout, stderr, command arguments, duration, input SHA-256,
output SHA-256, and engine revision. Reject empty outputs, duplicate output
paths, missing metadata, and inputs modified during conversion.

### 9. Model quantization as named bundle policies

A bundle label such as `Q4_K_M` may represent:

- Diffusion transformer: `Q4_K_M` or the closest backend-supported policy.
- LLM/text encoder: `Q4_K_M`.
- VAE: retained at FP16.
- Audio VAE: retained at FP32.
- Small or quality-sensitive tensors: retained at a higher precision through
  `--tensor-type-rules`.

Write the actual precision of every component and tensor-rule override to the
manifest and card. Never imply that an entire bundle is uniformly four-bit.

Expose every quant type supported by the selected converter for the selected
component. Do not assume llama.cpp's quant table applies to
`stable-diffusion.cpp`. Candidate policies that load but fail quality gates are
recorded as rejected rather than hidden.

### 10. Support sequential and resumable execution

For large models, especially MiniMax-H3:

- Download and verify one component at a time when `--sequential` is active.
- Convert to a temporary path and atomically rename only after validation.
- Remove source shards only after the converted artifact is hashed and its
  resumable state is committed.
- Reuse immutable companions across quant candidates.
- Resume only when source revision, engine revision, command, file size, and
  SHA-256 all match the state journal.
- Preserve failed artifacts only when requested for diagnosis.

## Feasibility reporting

### 11. Add image/video resource units

Text parameter counts and audio real-time factor are insufficient. A media
plan must estimate and later report:

- Source download and peak conversion disk.
- Bundle size per candidate policy.
- Host RAM and peak VRAM.
- Model-load time.
- First-image or first-frame latency.
- Seconds per image at the fixture resolution and step count.
- Seconds per video frame and total render time.
- Frames per second of generated output versus processing throughput.
- Resolution, frame count, output FPS, diffusion steps, and scheduler.
- CPU/GPU offload configuration and transfer overhead.
- For joint models, audio sample rate, channels, duration, and whether audio
  decoding adds a separate peak.

Every estimate must label its confidence and state whether it comes from a
measured local baseline, an upstream example, or a conservative heuristic.

## Validation and quality gates

### 12. Validate the complete runtime recipe

For every candidate bundle, invoke `sd-cli` with all required components.
Validation fails on:

- Missing or unreadable bundle members.
- Runtime load or generation failure.
- NaN/Inf diagnostics or backend assertions.
- Empty output or invalid image/video/audio container.
- Wrong dimensions, frame count, FPS, duration, or sample rate.
- Output that is blank, uniformly colored, fully transparent, silent, clipped,
  or obviously corrupt.
- A required conditioning input being ignored or rejected.
- A command that only works against files outside the published bundle.

### 13. Use deterministic media fixtures

Commit tiny, clearly licensed fixtures:

- Text prompts covering people-free scenes, readable shapes/colors, spatial
  relations, numbers, and multilingual text.
- A small reference image for image-to-image and editing.
- First and last frames for conditioned video.
- A short reference WAV for audio-video conditioning.
- Expected structural assertions and fixture licenses.

CI runs only low-cost conversion, manifest, bundle-integrity, and smoke tests.
Full-resolution quality benchmarks remain opt-in or scheduled because they are
too expensive for every pull request.

### 14. Add two-stage image quality assessment

Automated gates compare each quant with a high-precision baseline generated
using identical settings:

- Valid dimensions, channels, bit depth, and dynamic range.
- Non-blank and non-corrupt output.
- Perceptual similarity against the baseline.
- Prompt-image alignment using a pinned evaluator.
- Conditioning preservation for image-to-image and editing fixtures.
- Determinism within the tolerance of the selected compute backend.

Thresholds must be calibrated per family and task. Passing automation makes a
candidate reviewable, not automatically publishable. Require human review for
the first release of each family and for candidates near the quality boundary.

### 15. Add video and audio-video quality assessment

Video gates add:

- Decodable container and all expected frames.
- Correct resolution, FPS, duration, and frame ordering.
- Black-frame, frozen-frame, flash, and gross temporal-discontinuity checks.
- Prompt alignment on sampled frames.
- Start/end/reference conditioning checks where applicable.
- Temporal consistency relative to the high-precision baseline.

Joint audio-video gates add:

- Required stereo audio stream exists and decodes.
- Correct sample rate and duration within tolerance.
- Silence and clipping limits.
- Audio duration agrees with video duration.
- Reference audio is accepted when the selected mode requires it.

Human review must include both playback and several extracted frames. A silent
video-only MiniMax-H3 run may be a supported explicit mode, but it cannot pass
an audio-video release recipe that declares `audio_vae` required.

### 16. Keep evaluators reproducible and optional by tier

The mandatory structural tier should rely on small native utilities where
possible. Heavier perceptual, prompt-alignment, and temporal evaluators must be
pinned, cached, named in `quality.json`, and isolated behind an optional
`media-quality` dependency group. The plan must disclose when a metric could
not run; it must never silently replace it with a weaker metric.

## MiniMax-H3 acceptance route

### 17. Implement MiniMax-H3 as a bundle, not a special backend

The first MiniMax-H3 recipe must resolve these roles:

| Role | Expected publication format | Quantization policy |
| --- | --- | --- |
| Diffusion transformer | GGUF | Candidate quant |
| MiniMax-H3 Qwen3-VL-32B text encoder | GGUF | Candidate quant |
| Video VAE | SafeTensors | Retain documented precision |
| Audio VAE | SafeTensors | Retain documented precision |
| Vision tower | GGUF or bundled with the text encoder | Follow source layout |

The text encoder must be the MiniMax-H3 variant documented by upstream, not an
arbitrary Qwen3-VL checkpoint. The resolver must verify its truncation/layout
and final-normalization requirements from component metadata or a validated
manifest entry.

The acceptance matrix must cover:

- Text-to-audio-video.
- Image-to-audio-video.
- First/last-frame-to-audio-video when the selected model variant supports it.
- Reference image and reference audio conditioning for the Ref2VA variant.
- An explicit video-without-audio route obtained by omitting the audio VAE.

Do not claim support for distilled or adapter-based text encoders unless the
pinned backend supports the required projection adapter and a smoke test
passes. A community GGUF filename is not enough.

### 18. Define MiniMax-H3 release gates

A MiniMax-H3 candidate is publishable only when:

- Every required component is present and hashed.
- Both GGUF components load with the pinned `sd-cli`.
- The exact published bundle command completes without external model files.
- Video and stereo audio decode successfully.
- Structural and automated quality gates pass against the high-precision
  baseline.
- A human reviewer accepts the candidate video and audio together.
- The upstream model and every companion permit redistribution.
- Remote post-publication verification confirms every manifest member.

## Publication and documentation

### 19. Publish atomically

Upload all bundle members, `manifest.json`, `quality.json`, fixture metadata,
and the model card in one Hugging Face commit. Resume skips a member only when
the remote size and SHA-256 match. After commit, fetch a fresh repository
listing and verify every required file before reporting success.

Do not publish partial quant directories or cards that reference unavailable
companions. If Hugging Face cannot provide transactional behavior for the
payload size, upload to a staging revision and expose the final revision only
after verification.

### 20. Add media-specific model cards

Each card must state:

- Engine name and exact `stable-diffusion.cpp` revision.
- Supported tasks and conditioning modes.
- Every required component and its actual precision.
- Which files remain SafeTensors and why.
- Recommended `sd-cli` command copied from the verified runtime recipe.
- Required RAM/VRAM, tested compute backend, resolution, steps, and measured
  generation time.
- Quality results per candidate quant and rejected candidates.
- Base-model and component licenses.
- Safety limitations, intended use, and provenance.
- For audio-video models, audio sample rate/channels and reference-audio terms.

The card must not say “the model is Q4” when only some components are Q4.

## CLI and agent behavior

### 21. Preserve explicit approval boundaries

`plan_media_release` is read-only. `start_media_release` requires the user to
have seen and approved the exact plan, including source revisions, bundle
members, quants, resource estimate, license result, and target repository.
Human quality review remains a separate publication gate.

The agent must report:

- Why a particular engine and family were selected.
- Which components will be converted or retained.
- All candidate quant policies.
- Expected disk, RAM/VRAM, and render time.
- Any experimental capability or unresolved companion.
- Fork-hunt results when blocked.

## Testing strategy

### 22. Unit tests

- Capability-manifest parsing and regeneration.
- Repository-to-family matching without name-only false positives.
- Component-graph resolution and missing-member failures.
- Quant policy expansion and tensor-rule preservation.
- License and redistribution decisions per component.
- Resume-journal invalidation when any revision or hash changes.
- CLI and agent routing away from the text and voice pipelines.
- Fork-hunt evidence classification.

### 23. Offline integration tests

- Fake `sd-cli` converter and runtime processes with recorded command lines.
- Single-file image bundle.
- Multi-encoder image bundle.
- Video bundle with retained VAE.
- MiniMax-H3-shaped audio-video bundle.
- Runtime-only and prebuilt-GGUF plans.
- Partial upload, resume, atomic commit, and post-upload verification.
- Structural rejection of blank images, frozen video, missing audio, and
  malformed containers.

### 24. Native acceptance tests

Run on supported Linux and Windows hosts:

- Build the pinned engine from a clean cache.
- Convert a small supported image model and run every advertised quant policy.
- Run at least one image-editing or conditioning route.
- Run a short video fixture.
- Run the complete MiniMax-H3 bundle on capable hardware.
- Re-download the published candidate and execute the card command exactly.

CPU-only CI is permitted to skip expensive generation, but it must still test
capability extraction, bundle planning, conversion command construction, and
artifact validation.

## Delivery sequence

### Milestone A — registry and backend build

- Add the modality-neutral capability and bundle types.
- Pin and cache `stable-diffusion.cpp`.
- Add capability-manifest generation and drift tests.
- Expose backend inspection through CLI and agent tools.

### Milestone B — image conversion and publication

- Implement component resolution, native conversion, and quant policy probing.
- Support text-to-image and image-to-image smoke inference.
- Add image fixtures, automated gates, cards, and atomic publication.
- Validate at least one classic single-file model and one multi-component
  transformer model.

### Milestone C — video support

- Add frame/video container validation and feasibility metrics.
- Support the video families declared by the pinned capability manifest.
- Add temporal quality gates and human review artifacts.

### Milestone D — MiniMax-H3 audio-video release

- Resolve and publish the complete multi-component bundle.
- Validate T2VA, I2VA, FL2VA, and Ref2VA routes supported by the selected
  variant.
- Add joint video/audio gates and listening review.
- Re-download and reproduce the documented command.

### Milestone E — release hardening

- Complete fork hunting and blocked-plan reporting.
- Test resumability and sequential mode on large bundles.
- Run Windows/Linux native acceptance suites.
- Update README, USAGE, package extras, and architecture documentation.
- Cut v0.5.0 only after all acceptance criteria below pass.

## v0.5.0 acceptance criteria

The release is complete only when:

1. AgentQuantix routes supported image and video repositories to
   `stable-diffusion.cpp` without a per-model core-code allowlist.
2. The capability manifest is generated from and pinned to an exact upstream
   revision.
3. Plans distinguish load, conversion, and quantization support.
4. Multi-component bundles resume, verify, and publish atomically.
5. Quant choices come from the selected backend and preserve higher-precision
   components or tensors when required.
6. Every published candidate passes real `sd-cli` inference.
7. Image, video, and audio-video outputs pass their structural and quality
   gates against a high-precision baseline.
8. MiniMax-H3 completes as a verified GGUF-plus-companions bundle, including a
   human-reviewed video with audio.
9. A newly supported upstream family can be added by upgrading the pinned
   backend and regenerating capability data, without adding a model-specific
   branch to the media pipeline.
10. The published card command works after downloading only the files listed
    in the bundle manifest.

## Principal risks

- **Fast-moving upstream:** pin commits and treat upgrades as reviewed manifest
  migrations.
- **No machine-readable upstream catalog:** generate an evidence-bearing
  manifest and fail on drift instead of relying on names.
- **Huge multi-component models:** make sequential conversion, shared
  companions, and resumability mandatory.
- **Quantized output can be valid but wrong:** compare with a deterministic
  high-precision baseline and retain human review.
- **Mixed licenses:** evaluate and record redistribution permission for every
  component, not only the top-level repository.
- **Hardware-specific output differences:** record compute backend and use
  tolerance-based comparisons rather than byte equality across devices.
- **GGUF dialect assumptions:** invoke only the engine that produced or
  validated the artifact and record that engine in the manifest.
- **Unsupported community adapters:** classify them as leads until the exact
  adapter path is supported and tested by the pinned runtime.

