# Media implementation status

Source version: **0.4.5**. The v0.5.0 track is implemented as experimental
development against stable-diffusion.cpp commit
`8cd731e7b4667eb8f40a153315b9a7125d09a58d`.

The original STABLE_DIFFUSION_CPP_RELEASE_PLAN.md is unchanged. Real-model
acceptance and final quality calibration are deferred to the user's manual
testing, as requested. No final v0.5.0 release, model publication, or human
acceptance is implied by this implementation.

## Implemented surfaces

- Modality-neutral runtime capabilities and typed component graphs.
- Deterministic evidence-bearing catalog extraction covering all 56 loader
  families in the pin, with source hashes and CI drift checking.
- Independent native builds keyed by revision, platform, compiler, compute
  backend and CMake flags; resolved submodule/build provenance is recorded.
- Native converter probes, exact case-sensitive quant names, and native GGUF
  header inspection that tolerates engine types absent from the Python gguf
  library. Successful executable probes override the offline fallback.
- Read-only metadata planning, exact source revisions, separate load/convert/
  quantize facts, structured ambiguity blockers, license results and estimates.
- Component source contracts from unambiguous native examples, including
  documented companion source locations. Explicit CLI component selections
  handle ambiguous layouts without adding model-name branches to the pipeline.
- Shared CLI actions and MCP/agent tools (the separate `aqx media` group was
  removed in 0.5.2) with exact-plan approval, source/catalog/fixture integrity
  checks, and metadata routing away from text conversion.
- Atomic conversion, SHA-256 journals, revision/command-sensitive resume,
  sequential source processing, and shared retained companions.
- Deterministic licensed prompts, geometric reference/first/last frames, and
  a synthetic stereo WAV; repeated generation uses identical settings.
- Structural image checks and ffmpeg/ffprobe video/audio checks, plus pinned
  external evaluator interfaces for perceptual, prompt, conditioning, temporal,
  and determinism metrics. Missing metrics are disclosed and prevent publication.
- Family/task quality profiles, high-precision baseline comparisons, explicit
  human review bound to exact artifacts, and joint playback/listening gates.
- MiniMax-H3 component/tensor constraints, conditioned recipe modes, explicit
  video-only recipes, and unsupported conditioning-combination blockers.
- Cached fork/PR/issue leads, complete-bundle atomic commits, strict remote
  size/hash checks, and cards built from actual component precision and recipes.
- An opt-in native acceptance runner and documented manual-testing workflow.

## Validation boundaries

A native Windows CPU engine build and synthetic-tensor converter probe were
completed during development. This verifies buildability and converter behavior;
it does not establish real model-generation quality. Offline tests cover Python
orchestration, fake native subprocesses, journals, gates and publication behavior.

Full Windows/Linux model acceptance, family/task evaluator deployment and
threshold calibration, MiniMax-H3 human review, and executing card commands after
re-downloading published bundles remain manual acceptance work. Measurements
that cannot yet be established are explicitly reported as unavailable or as
conservative heuristics. Ambiguous source layouts and unverifiable architectures
remain blocked rather than being advertised as universally supported.

For local testing, a quality profile is optional: valid generations are marked
`structural-only` and retained for inspection. They cannot be published through
the release pipeline until the required quality, license, and human-review
conditions are satisfied. See MEDIA_USAGE.md for commands.
