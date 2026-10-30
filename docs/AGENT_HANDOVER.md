# Agent handover — 29 October 2026

## Start here

The user approved the [0.5.1–0.6.0 implementation plan](RELEASE_0_6_0_PLAN.md).
0.5.1 (shared runtime lifecycle) and **0.5.2 (shared task router, unified CLI
and agent actions)** and **0.5.3 (retrieval)** are implemented in local source.
The package version remains 0.4.5. Native acceptance for the tested 0.5.3
model/precision matrix is complete on Linux CPU, including llama.cpp at
`8642541b255bfb5dc39cd351a178be09013a55d6` with the llama-server-only runtime
and clip.cpp. Remaining gaps are Windows/MSVC, GPU/Metal, and real Hub
publication; acceptance runs used `--no-publish`. See
[retrieval status](RETRIEVAL_STATUS.md) for the exact matrix. **0.5.4
multimodal understanding is implemented, and sequential Linux CPU native
evaluation is complete for the tested models. SmolVLM2-500M-Video-Instruct
passed F16 and all four configured text quants; Qwen3-ASR-0.6B also passed its
full evaluated matrix. See [multimodal status](MULTIMODAL_STATUS.md).
Remaining gaps include Windows/MSVC, GPU/Metal and real Hub publication. The
0.5.5 implementation adds audio.cpp source separation, voice conversion and
forced alignment to the shared voice pipeline. See
[audio task status](AUDIO_TASKS_STATUS.md) for the native matrix and remaining
limitations. 0.5.6 adds vision.cpp prompted masks, image segmentation and
relative depth; see [vision task status](VISION_STATUS.md). 0.5.7 adds
ultralytics-ggml and rf-detr.cpp family-routed detection; see
[detection task status](DETECTION_STATUS.md). The RF-DETR nano F32 baseline
failed reference recall (0.9478 < 0.95), so no object-detection precision or
family is accepted on this sample in 0.5.7; keep the family cataloged for
run-time gating. The next milestone is **0.5.8**.

0.5.2 summary: `agentquantix/tasks/` (registry + metadata router),
`agentquantix/runs.py` (`text:`/`voice:`/`media:` run ids over existing records),
and `agentquantix/actions.py` (shared inspect/plan/start/review/publish/verify).
0.5.3 adds `retrieval:` run ids. `aqx voice`/`aqx media` are removed without
aliases; agent tools are `inspect_model`, `plan_release`, `start_release`,
`record_review`, `publish_release`, `list_runs`, `verify_release`. New tasks get
a pipeline by setting `TaskSpec.pipeline` in `tasks/registry.py` and adding an
adapter branch in `actions.py`; until then the router blocks them with their
milestone.

The 0.5.1 and 0.5.2 implementation notes below are retained as historical
context; the milestone state above is current.

Read the approved plan and [runtime implementation status](RUNTIME_LIFECYCLE_STATUS.md)
before editing. The package version remains **0.4.5** until a release is
approved.

## User decisions — do not ask again

- Normal execution should be **`aqx run org/model`**, with metadata-based task
  and engine routing. Standard source layouts should not require long commands.
- **Remove `aqx media` and `aqx voice` in 0.5.2**, once all their actions have
  shared replacements. Do not keep aliases or hidden forwarding paths.
- Original runtimes: ask whether to keep them for regular use. Yes means
  persistent storage outside `temp`; no means a temporary original build.
- Compatibility forks and PR builds remain temporary. Classify originals by
  their designated registry upstream, not GitHub's `fork` flag. In particular,
  Asher-1/ultralytics-ggml is the designated original provider of that adapter.
- Before execution, check the required engines for updates. Prefer the latest
  stable release; if none exists, resolve the upstream default branch's commit.
  Ask before updating; model `--yes` does not authorize runtime upgrades.
- If the check is unavailable, disclose it and continue with the installed
  revision. Frozen resumes retain their recorded runtime, even if another
  version becomes active. Updates for future runs must not switch an old job.
- All requested engines belong in the roadmap. Do not substitute a registry
  entry, a valid GGUF header, or a mocked executable for actual model support.

## Completed work

### Runtime lifecycle

- `src/agentquantix/runtimes/spec.py`: engine identities, handles, hashes and
  explicit decision/cancellation/error types.
- `runtimes/manager.py`: upstream discovery; immutable source/build records;
  persistent/temporary decisions; adoption and media-cache migration;
  side-by-side builds; atomic activation; rollback; OS locks; leases; cleanup
  and pruning; complete-local-source offline setup; exact-SHA reconstruction
  of missing temporary installations; retained catalog resources.
- `runtime_builders.py`: actual build/probe contracts for llama.cpp, audio.cpp,
  whisper.cpp, stable-diffusion.cpp, and clip.cpp. Captures toolchain,
  resources, shared libraries, binary hashes and catalogs.
- `runtime_control.py`: interactive policy and per-execution binding; required
  binary scoping; existing media-cache discovery; voice runtime journals;
  explicit failed-update fallback and future-only updates for frozen resumes.
- `runtime_cli.py` and `agent/runtime_tools.py`: inspect/check/install/update/
  adopt/rollback/prune controls. MCP tools return structured requirements and
  never read terminal input. Runtime preparation requires an exact target SHA
  plus runtime-specific approval.

Registry identities exist for all eleven planned engines. The remaining
post-0.5.3 engine builders and task pipelines are not implemented yet.

### Retrieval 0.5.3

- Implemented tasks: embeddings and reranking on llama.cpp; image feature
  extraction and zero-shot image classification on clip.cpp.
- Embeddings and reranking use `llama-server`; clip.cpp uses `extract`, `zsl`,
  and `quantize`. See [retrieval status](RETRIEVAL_STATUS.md) for the support
  matrix and quality gates.
- Focused offline verification on code commit `5d3471a`: 86 passed, 1 skipped;
  Ruff and `git diff --check` passed. These checks are not native acceptance.
- clip.cpp native acceptance passed on Linux x86_64 CPU for
  `openai/clip-vit-base-patch32`. This is a model-specific Linux CPU result,
  not a Windows, GPU, or Metal support claim.

### Multimodal 0.5.4

- Implemented the explicit `multimodal-understanding` task on llama.cpp.
  Plan-time vision/audio claims are checked against the converted F16
  projector metadata; a missing or mismatched projector blocks the run.
- Text quants are evaluated against F16 with a shared F16 projector.
  Server image/audio cases and CLI video cases are journaled; passing text
  quants plus the projector are published as one bundle.
- Qwen3-ASR-0.6B passed F16, Q8_0, Q6_K, Q5_K_M and Q4_K_M on Linux CPU with
  WER 0.0 and CER 0.018691588785046728. This is ASR-style understanding only;
  no audio QA or audio reference was evaluated.
- SmolVLM2-500M-Video-Instruct passed F16, Q8_0, Q6_K, Q5_K_M and Q4_K_M.
  F16 VQA accuracy was 1.0, OCR CER/WER were 0.0, and all five candidates
  passed video order at `--video-fps 1 --video-timestamp-interval 1000`.
- Server request payloads put image/audio before prompt text. The corrected
  reruns passed the native gates; request timings were near or faster than the
  feasibility-spike samples, so no server thread override was needed.
- A native ASR answer exposed an ASR-prefix markup scoring bug. `strip_markup`
  now removes the prefix generically, and the corrected Qwen rerun is
  authoritative. Result JSONs and logs are preserved in
  `/home/ubuntu/spike/acceptance/`.
- Windows/MSVC, GPU/Metal and real Hub publication remain untested. See
  [multimodal status](MULTIMODAL_STATUS.md).

### Expanded audio 0.5.5

- audio.cpp separation (`sep`), voice conversion (`vc`) and forced alignment
  (`align`) are catalog-derived voice tracks. New task specs use native task
  gates and atomic bundles; the existing ASR path remains on the shared
  generic release flow.
- The real lifecycle runtime was pinned to audio.cpp v0.8.0,
  `4af143229384fb6da3f373dc87de145ae954609b`. Linux CPU acceptance passed
  BS-RoFormer Q8_0, MeanVC2 F32/Q4_K and Qwen3 Forced Aligner's full nine-type
  source-conversion matrix. HTDemucs F16/Q8_0 failed the vocal SI-SDR gate.
  The Qwen run initially found an input-namespace bug; commit `8b83640` fixes
  it using the model-spec GGUF prefix contract, and the rerun passed. Its
  package fallback was not needed.
- The pinned Moonshine Streaming Tiny Q8_0 ASR regression passed at WER 0.0.
  Its first run found that the generic command sent `--language en` to a
  single-language runtime that rejects the option. The catalog-driven command
  now omits a redundant language flag only for single-language backends;
  multilingual Qwen3-ASR still receives its language. The failed and passing
  run logs and result JSON are preserved under
  `/home/ubuntu/spike/acceptance/`.
- Separation uses the deterministic speech-plus-chord mixture and true stems;
  VC uses a deterministic pitch-shifted JFK reference; alignment reuses JFK
  and its transcript. See `fixtures/voice/` and
  `scripts/generate_audio_task_fixtures.py`.
- Audio task acceptance data, negative results, licenses, package/source
  provenance and per-precision metrics are in
  [AUDIO_TASKS_STATUS.md](AUDIO_TASKS_STATUS.md). Native acceptance and
  cross-platform/Hub publication limitations are recorded there.

### Vision 0.5.6

- The pinned vision.cpp v0.3.0 lifecycle runtime is
  `b766b0acc30f6706f1bfe46769635e361172ce96`; Linux CPU acceptance passed
  MobileSAM F32/F16, BiRefNet F16 and Depth Anything V2 Small F32/F16. MobileSAM
  F16 achieved point IoU 0.985356, box IoU 0.993852 and pixel agreement
  0.999980. BiRefNet's F16 source was not promoted and passed with IoU 0.994884
  and boundary F-score 1.0. Depth F16 achieved three-scene Spearman
  0.999982-0.999985 and scale-shift AbsRel 0.000853-0.001219.
- Use the retained result JSONs and the full native table in
  [VISION_STATUS.md](VISION_STATUS.md). Only relative depth is supported; the
  tested synthetic fixtures do not establish real-photo quality.
- Conversion runs in a dedicated Python 3.12.16 environment under the runtime
  tool directory, with CPU Torch wheels. Keep this environment isolated from
  AgentQuantix's Python floor.

### Existing pipeline integration

- CLI and agent/MCP execution establish runtime sessions and leases. Internal
  low-level helpers keep their legacy no-session behavior for library callers;
  public execution entry points must retain their session boundaries.
- Text jobs persist `runtime_snapshot`; archived snapshots are resolved before
  binding an active runtime in both CLI and MCP resume paths.
- Voice stores conversion/scoring engine snapshots in each work directory's
  `runtime-snapshots.json`. Selected audio.cpp model specs are hash-checked and
  scoped to the runtime identity rather than the old bundled catalog.
- Media planning consults the installed revision's catalog without building.
  Plans include that runtime snapshot. Execution checks plan hash/fixtures/
  blockers **before** runtime setup, then checks revision/catalog compatibility
  after binding. Dynamic revisions propagate into conversion and quality data.
- Probe/doctor and architecture support can find managed installations rather
  than looking only at the legacy checkout paths.
- Agent instructions were regenerated from `agent/prompt.py`; do not hand-edit
  `.claude/skills/quantix/SKILL.md` or `adapters/AGENTS.md`.

### Earlier fixes included in the same working changes

- Recognize `StableDiffusionXLPipeline` and resolve the standard SDXL checkpoint
  and repository-declared VAE automatically.
- Reject blocked media plans before requesting execution approval.
- A successful Git command was falsely rejected because the numerical-error
  regex matched `INF` in `Documents/INF/...`. Paths containing `INF`/`nan` now
  pass; actual non-finite values and assertions still fail.

## Validation and local environment

Last full offline run: **434 passed** on Windows / Python 3.14. Ruff correctness
and unused-import checks passed. Wheel/sdist building and extracted-wheel CLI
help/runtime inventory plus stdio MCP initialization/tool discovery passed during
development. The last packaging smoke preceded final control-flow hardening;
rebuild distribution artifacts before any release/install acceptance claim.

The original `.venv` launcher could not execute in this environment. A working
isolated development environment was created at `.venv-dev`, with `[dev,media]`
dependencies and Ruff. `.venv-dev`, `.uv-cache`, `.cache` and `.test-*`
directories are ignored; they are not handover artifacts or release outputs.

Run from the repository root in PowerShell:

```powershell
.\.venv-dev\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp .test-next-agent-tmp
.\.venv-dev\Scripts\ruff.exe check --no-cache --select E9,F63,F7,F82,F401 src tests scripts
.\.venv-dev\Scripts\python.exe scripts/sync_adapters.py
git diff --check
```

For a fresh environment, install `.[dev,media]` and Ruff using the project's
Python 3.10+ requirements. Existing CI exercises Python 3.10/3.13 on Windows
and Linux; those remote jobs have not been run by this session. The default
tests isolate runtime state and use controlled Git/native boundaries. One
Windows build-wrapper test executes a real batch subprocess with spaces and
`&` in its directory names. None of this establishes real-model inference.

New focused tests are `test_runtime_manager.py`, `test_runtime_builders.py`,
`test_runtime_surface.py`, `test_runtime_integration.py` and
`test_runtime_voice_catalog.py`. Keep the regression cases for cancellation,
frozen MCP resumes, read-only catalogs, early plan validation and offline use.

## 0.5.2 implementation slice (historical)

1. Build a shared task/capability registry and typed component/release model.
   Preserve independent load/convert/quantize/package and platform facts.
2. Route `aqx inspect`, `plan`, `show` and `run` by metadata/source contracts.
   Keep genuine ambiguity as a short choice; avoid model-name conditionals in
   the core pipeline. Existing text task filters must not discard all other
   tasks before the shared planner sees them.
3. Implement shared review, verify, publish and card actions, plus run-ID resume
   and migration of existing voice/media records. Preserve task-specific gates
   and existing publication behavior explicitly in each plan.
4. Remove the modality CLI groups only when all shared replacements work.
   Update help, completions, docs, examples, generated instructions and tests
   together. Preserve internal engine adapters rather than deleting capability.
5. Unify the agent action registry alongside the CLI; retain structured runtime
   decisions and exact plan/runtime approval boundaries.
6. Perform early feasibility/build/converter audits for all seven new engines,
   especially SAM3 and forecasting, so late upstream blockers are found early.

## 0.5.4 implementation slice

The implementation adds libmtmd-backed multimodal understanding in llama.cpp:
image, audio, video/frame, and document inputs for the model families the
pinned runtime supports. Keep companion projectors, preprocessing,
tokenizer/prompt behavior, and supported task boundaries explicit. The
read-only feasibility spike and its report are under
`/home/ubuntu/spike/0.5.4b/`; do not treat the spike as pipeline
implementation or native acceptance.

The sequential Linux CPU results are recorded in
[multimodal status](MULTIMODAL_STATUS.md). The approved plan continues with
expanded audio (0.5.5), vision.cpp (0.5.6),
YOLO/RF-DETR (0.5.7), dedicated depth/SAM3 (0.5.8), forecasting/hardening
(0.5.9), and the consolidated 0.6.0 release. Do not claim the full roadmap is
implemented.

## Native acceptance and upstream cautions

- Preserve `docs/RELEASE_VALIDATION.md` and the original media release plan's
  requirements. 0.5.3 native acceptance is complete for the recorded Linux CPU
  model/precision matrix only. Windows/MSVC and GPU/Metal remain untested, and
  real Hub publication was not exercised; the acceptance runs used
  `--no-publish`.
- 0.5.4 native results are model-specific Linux CPU evidence transcribed from
  the implementation result JSONs. Both tested models passed their evaluated
  text-quant matrices. The feasibility spike is not implementation acceptance.
- Select exact upstream SHAs during new-engine audits; README support tables
  alone do not establish acceptance. Track engine revision × family × task ×
  precision × platform/backend.
- vision.cpp currently needs Python 3.12+ for conversion and has floating/F16
  conversion paths rather than a llama-style low-bit sweep. Keep its conversion
  environment isolated instead of raising AgentQuantix's global Python floor.
- SAM3 conversion is documented from an MLX-style layout; official-source
  preparation needs verification. Its CMake configuration contradicted README
  CUDA/Vulkan claims during the audit; check actual source and native probes.
- Depth Anything nested/pose routes may require multiple artifacts and exact
  unit/camera conventions. clip.cpp preprocessing differs from PIL/PyTorch.
- timeseries.cpp is early; unfinished advertised families must remain unaccepted
  until a real path exists. Whisper retains its native format; do not relabel
  its GGML binaries as GGUF.
- Keep code/model license provenance and complete bundle dependencies separate.
  A successful compile or package listing is not model-quality acceptance.

No additional approval is needed to continue the already approved implementation
roadmap. Actual release publication remains a separate operation. Do not
publish models or claim acceptance beyond the documented matrix.
