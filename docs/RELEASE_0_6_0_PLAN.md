# AgentQuantix 0.5.1–0.6.0 implementation plan

Status: approved by the user, 25 October 2026; 0.5.1 implementation in progress.
This document plans implementation;
it does not change the package version or claim a runtime/model has passed
native acceptance. Current source remains 0.4.5, with the existing 0.5.0 media
track and its outstanding acceptance work documented separately.

## Outcome and agreed decisions

By 0.6.0, a normal user can run `aqx run org/model`, let AgentQuantix identify
the task and native engine, review the resolved plan, and build, evaluate, resume,
and publish the complete artifact bundle. Advanced family/component arguments
remain available for ambiguity, rather than being required for standard layouts.
Remove the public `aqx media` and `aqx voice` command groups in 0.5.2, when the
shared router is ready. Do not retain compatibility aliases or alternate
modality-specific execution commands. Task-specific engines remain internal
adapters behind the shared CLI.

The release adds every requested runtime adapter and expands the existing
engines. Support follows verified upstream architecture and task contracts,
not a short AgentQuantix model-name allowlist. This does not mean every Hugging
Face model, task, precision, or device is executable in every engine.

The following user decisions are settled:

- Prefer the latest stable upstream release. If that engine has no stable
  release, resolve its upstream default-branch commit.
- Before each execution invocation, check the required runtime(s) for updates
  and ask before building a newer revision. Declining the update uses the
  installed revision for that run.
- Ask whether an original runtime should be kept for regular use. Acceptance
  means persistent installation outside `temp`; declining means build that
  original runtime in `temp` for this use. This is the explicit exception to
  the original persistent-storage proposal.
- Compatibility forks and PR builds remain in `temp`, independently of the
  original-runtime preference.
- When upstream cannot be checked, continue with an available installed
  revision and disclose that the check was unavailable. If no usable runtime
  or complete local source/build is available, report what is missing.

No further product decision blocks implementation. Detailed command names and
schemas below are proposals, not existing interfaces.

## Current implementation and compatibility baseline

- `config.py` already places llama.cpp, audio.cpp, and whisper.cpp at
  `WORK_ROOT/<engine>`, outside `temp`.
- `pipeline/build.py` adopts/builds llama.cpp and keeps compatibility forks in
  `TEMP_DIR/aqx-forks`. There is no shared pre-run update/consent manager.
- `pipeline/voice_release.py` has independent audio.cpp/whisper.cpp setup.
  audio.cpp initially clones a tag, but existing binaries are reused without a
  universal source/build provenance check. Auxiliary OpenAI Whisper conversion
  assets currently live under the temporary voice-backend directory.
- `pipeline/media_release.py` builds original stable-diffusion.cpp in
  `TEMP_DIR/aqx-media-backends/<identity>`. Move regular-use installations to the
  persistent lifecycle while retaining the chosen temporary-use option.
- Text jobs freeze assessments and quantization choices but not the complete
  immutable runtime identity. Media plans bind more strongly to runtime/catalog
  revisions; preserve and generalize that protection.
- Keep existing state, Hugging Face target repositories, resume records,
  concurrency/disk limits, and fork cleanup behavior compatible.
- Preserve existing pipeline capabilities through shared commands, while making
  the deliberate CLI break that removes the media/voice command groups. Upgrade
  documentation and scripts together in 0.5.2; historical release records can
  retain the command syntax that applied to those releases.
- Keep the 0.5.0 native acceptance requirements visible. Planning 0.6.0 neither
  closes those requirements nor implicitly publishes 0.5.0.

## Runtime coverage

There are eleven engine identities including the retained Whisper and media
engines. libmtmd is a llama.cpp capability, not a second installation.

| Engine / designated upstream | Planned scope | Integration boundary |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Existing text/TTS plus embeddings, similarity, reranking, selected visual retrieval | Family-specific pooling, normalization, templates, heads, calibration and evaluation |
| [llama.cpp / libmtmd](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) | Image/audio/video input, captioning, VQA, document QA/OCR, selected multimodal understanding | Mandatory projectors, preprocessing, audio/frame inputs, prompt and tokenizer contracts |
| [audio.cpp](https://github.com/0xShug0/audio.cpp) | Existing TTS/ASR plus verified generation, music/SFX, VC/S2S, separation/enhancement, classification, VAD, alignment/diarization | Derive tasks, source preparation, packages, companions and precision from upstream specs |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Existing ASR and evaluation dependency; verified VAD paths where applicable | Preserve Whisper's actual native artifact format; do not relabel GGML binaries as GGUF |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | Existing image/edit/video/audio-video workflows | Preserve component bundles, pinned catalog, quality and human-review gates |
| [vision.cpp](https://github.com/Acly/vision.cpp) | MobileSAM, BiRefNet, Depth Anything V2, MI-GAN, ESRGAN | Segmentation/masks, depth, inpainting, super-resolution; only converter/runtime-supported precision |
| [ultralytics-ggml](https://github.com/Asher-1/ultralytics-ggml) | Verified YOLO detection, instance/semantic segmentation, classification, pose, depth and open vocabulary | `cpp_ggml` build subtree; labels, preprocessing/NMS, keypoint topology and encoder companions |
| [rf-detr.cpp](https://github.com/localai-org/rf-detr.cpp) | RF-DETR detection and instance segmentation | Dedicated converter/quantizer, preprocessing, decoding and mask contracts |
| [depth-anything.cpp](https://github.com/localai-org/depth-anything.cpp) | Verified DA2/DA3 depth and available confidence, pose/reconstruction outputs | Relative/metric units, scale, nested/multiple-model dependencies and camera conventions |
| [clip.cpp](https://github.com/monatis/clip.cpp) | Compatible CLIP image/text features, zero-shot classification and image retrieval | Exact tokenizer, resize/interpolation, normalization and similarity scaling |
| [sam3cpp](https://github.com/ropoctl/sam3cpp) | SAM 3 prompted detection/segmentation and mask generation | Source preparation, tokenizer resources, supported prompts, masks and actual compiled backends |
| [timeseries.cpp](https://github.com/liodon-ai/timeseries.cpp) | Implemented Chronos/TimesFM forecasting families | Numeric input, normalization, horizon, quantiles, missing data and temporal evaluation |

"Original" means the designated upstream in this registry. ultralytics-ggml is
itself a GitHub fork, but it is the original provider of its registered C++
engine; its compatibility forks are temporary. Do not classify installations
using GitHub's `fork` flag. Record canonical repository identity and approved
redirects/aliases, including LocalAI repositories with older owner URLs in docs.

## Runtime storage and ownership

Separate `origin_kind` (`upstream`, `fork`, `pr`) from `storage_policy`
(`persistent`, `temporary`) and `ownership_kind` (`managed`, `external-adopted`).
A temporary original is still an upstream runtime, not a compatibility fork.
An adopted external installation remains outside managed automatic cleanup.

Proposed managed layout under the existing configurable work root:

```text
WORK_ROOT/
  llama.cpp/                         existing original checkout, adopted in place
  audio.cpp/                         existing original checkout, adopted in place
  whisper.cpp/                       existing original checkout, adopted in place
  stable-diffusion.cpp/               regular-use original checkout
  vision.cpp/                        other regular-use originals follow this pattern
  ...
  .aqx-runtime-builds/<engine>/<sha>/<build-key>/
    source/                          immutable managed snapshot, where needed
    build/                           side-by-side native output
    env/                             isolated converter dependencies, where needed
    logs/                            setup/build/probe provenance
  temp/
    aqx-runtime-sessions/<engine>/<sha>/<build-key>/
                                     originals when regular-use storage is declined
    aqx-forks/<engine>/<repo-ref>/<sha>/<build-key>/
                                     compatibility forks and PRs
    ...                              existing model-run scratch and conversion outputs

AQX_HOME/state/runtimes/
  registry.json                      path, preference, active version and ownership
  <engine>/<identity>.json            immutable installation/build/probe records
```

Exact directory names can be shortened for Windows path limits. All paths are
absolute, configurable, and validated before moving, removing or reusing files.
Regular-use source, native build, build staging and converter environments stay
outside `temp`. Model scratch files may continue to use `temp` as today.

Adopt existing installations without forcing relocation or rebuilding known-good
binaries. Inventory commit, dirty state, submodules, build flags and executable
hashes before treating an adopted installation as verified. Never reset, clean,
overwrite or delete a user-owned checkout as an implicit update operation.

First-use prompt: "Keep <runtime> installed for regular use at <path>? [Y/n]",
with the explanation that `n` selects a temporary build. Display selected
revision, hardware mode and estimated build/disk cost before the build begins.
Record a persistent choice; a temporary choice applies to this execution unless
the user explicitly saves that preference. A later `runtime install` or promote
operation can adopt a verified temporary build into persistent storage.

Temporary originals and forks can be reused within the same batch. Do not
remove a runtime while another process is using it. Proposed cleanup default:
remove managed temporary originals after a successful batch releases its last
lease; retain failed/interrupted installations until resume or explicit discard/
prune. Preserve logs/provenance in state before cleanup. Existing fork retention
preferences, including `KEEP_FORK`, remain effective. Record exact SHA and build
identity so a cleaned temporary runtime can be reconstructed for resume; pruning
referenced builds must disclose that resume will require reconstruction.

Migration for existing stable-diffusion.cpp caches:

1. Inventory source, commit, submodules, build manifest and binary hashes.
2. Ask the regular-use storage question, showing the existing usable build.
3. For persistent storage, stage-copy/adopt outside `temp`, verify content and
   executable behavior, and atomically register the persistent location.
4. Reprobe path-sensitive converter caches and native resource dependencies.
   Moving an executable can invalidate CMake/RPATH/embedded-resource assumptions;
   rebuild when relocation cannot be verified.
5. Preserve old caches until verified migration succeeds. Do not silently delete
   old directories or rewrite approved plans to another runtime revision.
6. Migrate retained original conversion-tool code, such as OpenAI Whisper assets,
   consistently with the selected installation's storage policy.

## Pre-run updates and reproducibility

Perform this once per execution invocation for each required engine, deduplicated
across a multi-model batch and including evaluator engines. For example, a TTS
release may also require whisper.cpp for scoring. Do not check on every quant,
prompt, image, subprocess or upload. Read-only inspect/research commands do not
install, update or build runtimes.

1. Inspect model metadata and identify task, candidate engines and dependencies.
2. Resolve installed runtimes and storage preference; show unresolved task/engine
   ambiguity before installing unrelated engines.
3. Perform a bounded upstream metadata check. Resolve the latest non-draft,
   non-prerelease stable release according to the engine's documented policy;
   if none exists, resolve the default branch. Store tag plus exact commit SHA,
   check time and comparison result. Never sort version strings lexically or
   assume the default branch is named `main`.
4. If a different upstream target exists, offer "Build and validate update",
   "Use installed version", and "Cancel". Show current/target revisions and
   whether the target is behind, ahead, divergent or merely different; an older
   stable release must not silently downgrade a newer installed commit.
5. If unreachable/rate-limited, disclose the unavailable check and continue with
   the installed version. If first installation is needed, use a complete,
   verified local source snapshot when available; otherwise stop with the exact
   missing dependency. Do not claim that a failed check found no update.
   Record the failed attempt's time/error separately from the last successfully
   resolved target. An unavailable check cannot authorize a fetch/build from
   an unverified moving branch or tag.
6. An accepted update resolves to an immutable SHA, uses the installation's
   storage policy, and builds beside the active version. Fetch precisely the
   approved revision; changed refs trigger a new comparison rather than drift.
7. Probe converter/quantizer/CLI contracts and required runtime assets. Generate
   a capability snapshot for that exact engine revision. Incompatible adapters,
   unavailable dependencies or failed probes prevent candidate activation.
8. Activate atomically only after runtime checks succeed. Build failure preserves
   the current installation; report failure and offer continued use of the old
   version. Do not call a successful compile model-quality acceptance.
9. Generate/reprice the model plan using the selected engine, show any changed
   components/quant list/costs, then obtain approval for that exact plan.
10. Freeze the engine source/build/catalog/resource identity into the job and
    run without changing it. Another process may activate a newer version, but
    active jobs retain their immutable runtime handles.

A recorded build identity includes source commit, submodule commits, adapter
schema, compiler/toolchain, OS/architecture, compute backend, flags, converter
environment lock, executable/resource hashes, and capability/probe fingerprints.
Keep active and previous tested installations for rollback. Use per-engine
interprocess install locks, build locks and active-run leases. Cleanup must
respect active jobs and retained resume references; expose disk inventory and
explicit pruning rather than accumulating unbounded versions.

Updating upstream must not overwrite the package's checked-in catalogs. Store
revision-specific local catalogs; use the shipped reviewed snapshot as a
baseline. A changed upstream layout that the extractor cannot verify is an
adapter incompatibility, not evidence of newly supported models.

On resume, still perform the required update check, but keep the job's frozen
runtime by default. A user may install the newer engine for future work. Switching
the resumed job to it requires a new plan and invalidates affected conversion,
baseline, quality and human-review caches. Path-only relocation can reuse
content-identical artifacts only after provenance verification.

CLI and MCP/agent harnesses must use the same lifecycle decisions. A tool call
returns structured setup/update requirements and exact target IDs; it cannot
block on `input()`. Model `--yes` approval does not silently authorize a runtime
upgrade. Unattended runs use an explicitly configured runtime-install/update
policy or an approved installed version; absent consent returns an actionable
requirement. Offer simple interactive defaults so ordinary users still need
only `aqx run org/model`.

Proposed maintenance commands: `aqx runtime list`, `show`, `install`, `check`,
`update`, `adopt`, `rollback`, `remove` and `prune`. These are optional controls;
routine execution performs the required preflight itself.

## Shared architecture

Introduce a runtime registry and lifecycle manager instead of adding update
logic separately to every build function. Suggested modules:

- `runtimes/spec.py`: engine identity, installation, build/probe/capability schemas.
- `runtimes/manager.py`: storage, ownership, install/adopt/promote/migrate,
  update checks, locks, activation, rollback and pruning.
- `runtimes/adapters/`: source inspection, build targets, conversion, quantization,
  runtime commands and evidence extraction for each engine.
- `tasks/`: task-specific fixtures, calibration, validators, metrics and output
  contracts. A runtime can implement multiple task adapters.
- Shared routing/planning and runtime-management tools used by the CLI and
  `agent/tools.py`; preserve `mcp_server.py` as the transport layer.

Generalize existing media component graphs and journals for text, retrieval,
vision, audio and forecasting. Keep each source format honest: GGUF loading,
source conversion, quantization and pretrained package availability are
independent facts. There is no universal GGUF converter or universal quant sweep.

Each capability records family, tasks, engine revision, source matchers,
converter/preparation requirements, required companions, precision/tensor policy,
preprocessing, runtime command schema, output contract, hardware evidence,
license provenance and quality profile. Record:

`engine revision × architecture × task × precision × platform/compute backend`.

Catalog evidence, executable probes and real-model acceptance are distinct
statuses. Expose all discovered families with accurate states; only advertise
cells as supported after their promised path passes acceptance. Prebuilt-only
loading is useful but must not be described as source conversion support.

When several engines support a task, choose an installed validated compatible
engine using saved preferences and source contract evidence. For unresolved
choices, present a short engine/task selection. Never dispatch an arbitrary
vision or retrieval model into the text-generation imatrix pipeline merely
because it has a Transformer config or a GGUF extension.

## One CLI for every task

The command describes the action; model metadata and the saved plan determine
the task and engine. A user does not need to choose a media/voice/retrieval/vision
namespace before asking AgentQuantix to inspect or execute a model.

| Shared command | Purpose and replacement |
|---|---|
| `aqx inspect org/model` | Read-only source metadata; replaces `aqx media inspect` and works across tasks |
| `aqx plan org/model` | Resolve task, engine, components, precision, feasibility and quality requirements; replaces media/voice planning |
| `aqx show org/model` | Show a detailed existing assessment/plan across supported tasks |
| `aqx run org/model` | Plan, obtain necessary setup/update and model approvals, then execute the resolved pipeline; replaces media/voice execution |
| `aqx run --resume RUN_ID` | Resume the recorded task, plan and runtime without asking the user to select a modality |
| `aqx review RUN_ID --quant QUANT --accept --reviewer NAME` | Record task-appropriate human review, including listening/playback attestations when required |
| `aqx verify TARGET` | Verify a local manifest/bundle or published repository using its recorded task and artifact contract |
| `aqx publish RUN_ID --quant QUANT` | Publish a validated candidate according to its saved release policy and applicable review gates |
| `aqx card TARGET` | Generate the appropriate card from verified artifacts and task evidence |
| `aqx runtime list` / `aqx runtime show ENGINE` | Engine inventory and capabilities; replaces voice list/media backends alongside the existing proposed runtime-management commands |

Keep shared research/probe/doctor/bootstrap commands. Preserve task-specific
controls such as conditioning images, speaker references and forecast settings
as optional arguments or saved-plan fields on the common actions. Help should
group those options by applicability; users should not need a long command for
an unambiguous standard repository. Explicit `--task`/`--engine` choices resolve
ambiguity without creating separate execution commands.

Local manifest, run ID and repository targets must be resolved deterministically;
return a concise choice when an identifier is ambiguous. Provide explicit target
options for automation instead of guessing. Migrate voice/media run identifiers
and review records so existing work remains reachable through the shared actions.

Remove the old parser registrations, public CLI handlers and adapter-install
instructions that expose modality command groups once their shared replacements
are complete. Preserve reusable engine/pipeline internals. Update README, USAGE,
media/voice usage documentation, examples, shell completion, agent prompts,
installer-generated adapters and CLI tests in the same milestone. Publish a
short old-to-new command mapping in the 0.5.2 release notes; removed commands
must not silently forward through hidden aliases.

Use the same unified action registry in MCP and agent harnesses. Transport tools
should route through shared inspect/plan/start/review/verify/publish actions,
with task-specific requirements represented in their schemas and returned plans.
Namespace removal does not change artifact formats, publication policy, quality
requirements or approval scope; those remain explicit in each release plan.

## Phased releases

Versions are delivery milestones, not calendar commitments. Complete each
acceptance gate before assigning its release tag; feasibility spikes can run
in parallel with foundation work.

| Version | Deliverables | Required exit evidence |
|---|---|---|
| **0.5.1** | Shared runtime lifecycle for existing llama/audio/whisper/media engines; storage choice, pre-run update prompts, migration, immutable manifests, rollback and offline behavior | Existing text/voice/media regressions; persistent and temporary-original paths; failed updates preserve current runtime; resume uses its frozen version |
| **0.5.2** | Unified task/capability registry, component bundles, lifecycle hooks in CLI/MCP, shared action commands and metadata routing through `aqx run`; remove `aqx media`/`aqx voice`; early build/conversion spikes for every new engine | Standard repositories route with one command; all former media/voice actions have shared replacements; old namespaces/aliases are absent; existing records remain usable; audited SHA/build/API/source contracts recorded for all engines |
| **0.5.3** | llama.cpp retrieval and clip.cpp: embeddings, similarity, reranking, image features, zero-shot classification and semantic retrieval | Real conversion/inference plus dimension, pooling, normalization, preprocessing and retrieval-quality checks |
| **0.5.4** | libmtmd multimodal understanding: visual/audio/video input, captioning, VQA, document/OCR tasks, supported multimodal retrieval | Complete companion bundles; missing projector fails closed; real image/audio/ordered-frame/document tests and task-quality comparisons |
| **0.5.5** | Expand audio.cpp task adapters beyond TTS/ASR using the selected upstream model specs | Native execution and appropriate metrics for each advertised generation/VC/S2S/separation/enhancement/classification/VAD/analysis task; existing TTS/ASR remains green |
| **0.5.6** | vision.cpp segmentation, masks, depth, inpainting and super-resolution | Conversion and actual task outputs for each advertised family; preserve floating precision where native low-bit conversion is absent |
| **0.5.7** | ultralytics-ggml and rf-detr.cpp detection, segmentation, classification, pose and verified open-vocabulary/depth routes | Real native conversions/quants; labels/coordinates/NMS/masks/keypoints and task-quality parity; encoder companions included |
| **0.5.8** | Dedicated depth-anything.cpp and sam3cpp integrations | Multi-component DA variants and camera semantics verified; SAM source preparation, tokenizer resources and prompted masks validated on actual supported hardware |
| **0.5.9** | timeseries.cpp forecasting; all-adapter feature freeze, migration rehearsal, compatibility hardening | Real Chronos/TimesFM paths supported by the chosen revision; chronological holdout metrics; every requested engine has an accepted runnable path |
| **0.6.0** | Consolidated release with all requested adapters, published support matrix, unified docs, installation/upgrade and re-download acceptance | Windows/Linux advertised matrix passes; 0.6.0 checklist below complete; no placeholder adapter counted as integration |

Audit SAM3 and forecasting during 0.5.1/0.5.2 rather than discovering blockers
late. If a designated upstream cannot execute its documented path, resolve the
problem with a compatible revision or reviewed patch before claiming support.
Temporary PR/fork execution is available for investigation; a failed prototype
is not a completed 0.6.0 adapter. If that threatens scope, explicitly revise the
release plan with the user rather than silently omitting an engine.

## Engine-specific engineering constraints

The following are audit findings, not final release pins. Select and record exact
source SHAs during 0.5.2; repeat contract checks when approving a newer revision.

- **llama.cpp / libmtmd:** share one lifecycle across text, retrieval, TTS and
  multimodal targets. Protect mandatory projector export and validate the
  modality rather than accepting a text-only result as a full multimodal release.
- **audio.cpp:** package specs and embedded metadata are the family contract.
  Upgrade beyond the current initial tag only through the lifecycle manager;
  specs alone do not establish working conversion or validated precision.
- **vision.cpp:** its converter currently requires Python 3.12+ and supports
  source floating/F16 paths; do not promise a llama-style low-bit sweep. Keep
  converter dependencies isolated instead of raising the whole app's Python floor.
- **clip.cpp:** documented resize/interpolation differences from PIL/PyTorch
  require explicit preprocessing and quality tolerances.
- **YOLO/RF-DETR:** distinguish conversion output types from published package
  types and runtime-accepted types. Use upstream task-specific decoding and
  evaluation, not one generic image validator.
- **depth-anything.cpp:** nested metric and auxiliary pose paths can require
  multiple artifacts. Preserve units, max-depth/scaling and camera conventions.
- **sam3cpp:** documented conversion expects MLX-layout safetensors plus an
  index. Establish a reproducible preparation contract from official sources or
  disclose a prebuilt-only path. Current CMake disables CUDA/Vulkan despite README
  claims; require source/probe evidence before enabling those claims. Capture
  ZLIB/BPE/tokenizer resource dependencies as part of the installation.
- **timeseries.cpp:** early source tree; distinguish implemented Chronos/TimesFM
  routes from unfinished architectures. Prove conversion, Windows build and
  numerical parity before committing a supported cell.
- **Whisper:** retain the existing format-specific ASR pipeline and task-specific
  baseline; its presence in the runtime manager does not imply GGUF output.

Sources for these details include the linked upstream repositories,
[audio.cpp GGUF contracts](https://github.com/0xShug0/audio.cpp/blob/main/docs/gguf.md),
[vision.cpp converter](https://github.com/Acly/vision.cpp/blob/main/scripts/convert.py),
and [sam3cpp build configuration](https://github.com/ropoctl/sam3cpp/blob/master/CMakeLists.txt).

## Validation and publication

Keep offline orchestration tests separate from real native model acceptance.
Mocked binaries verify control flow, not architecture support or model quality.
Use deterministic, licensed fixtures and a fixed high-precision baseline from
the same model source revision. Document hardware and precision-specific
tolerances; calibrate thresholds before advertising publication eligibility.

| Task | Required checks beyond successful conversion/load |
|---|---|
| Embeddings/retrieval/reranking | Vector dimensions/norm/finiteness, cosine drift, Recall@k, MRR/NDCG, pooling and query/document templates |
| Multimodal/OCR | Mandatory companions, image preprocessing, VQA/caption correctness, OCR CER/WER, audio understanding, frame order/timestamps |
| Audio | Waveform validity plus task metrics: WER/CER, classification F1, VAD boundary/segment F1, separation SI-SDR, relevant human listening |
| Detection/classification/pose | Dataset-level mAP/top-k/OKS or PCK, labels, coordinates/scaling, NMS and topology |
| Segmentation/masks | IoU/Dice, prompt behavior, mask shape/alignment, boxes and empty-output behavior |
| Depth/pose/reconstruction | AbsRel/RMSE/delta metrics with valid masks, relative vs metric scale, confidence, units and camera conventions |
| Inpainting/super-resolution | Mask preservation, output geometry, unchanged-region behavior and relevant perceptual/baseline comparisons |
| Forecasting | Horizon/quantile shapes, finiteness, chronological holdouts, no future leakage, MAE/MASE/pinball loss and seed behavior |

Publish complete bundles with preprocessing/config/tokenizer/label/companion
assets, runtime revision and installation instructions, exact runnable commands,
precision policy and measured limitations. Record code licenses separately from
model redistribution terms. A runtime's permissive license does not establish
permission to redistribute all of its model weights.

Reuse atomic journals, remote size/hash verification, resource reservations and
resume semantics. Publication requires the task's complete applicable quality
profile; structural-only local experiments remain possible and clearly marked.
After publication, re-download the bundle in a clean directory and execute its
documented commands with the recorded compatible engine.

## Runtime lifecycle acceptance matrix

Tests must cover:

- First-use persistent consent, refusal selecting a temporary original, reuse,
  promotion, path adoption and storage preferences across CLI and MCP.
- Regular-use original source/build/staging outside `temp`; temporary-original
  and compatibility-fork cleanup confined to their respective roots.
- Stable-release selection, branch fallback, prereleases, no update, declined
  update, different/divergent revisions and changed refs between check/fetch.
- Offline, timeout, authentication/rate-limit failures with honest reporting and
  continued installed-version use; actionable failure when no runtime is usable.
- Failed clone/configure/compile/probe/activation and rollback without damaging
  the active version or a dirty external checkout.
- Concurrent runs/updates, lock recovery after crashes and no deletion of in-use
  builds; batch update checks deduplicated once per required engine.
- Resume with its original runtime after activation of a newer one; affected
  cache invalidation when intentionally switching revisions.
- Catalog drift, binary/resource tampering, changed submodule/build identities,
  and original-runtime/fork identity isolation.
- Stable-diffusion.cpp migration, path-sensitive resource relocation, Windows
  paths with spaces/long paths, and the `INF` diagnostic regression.
- Single-command routing, ambiguity, unknown tasks, missing companions and
  noninteractive consent behavior without hidden terminal prompts.
- Every former media/voice action is available through the shared CLI; old
  command groups are absent from help/completion and rejected by the parser
  without forwarding. Saved voice/media plans, reviews and runs remain usable.

## 0.6.0 completion checklist

- Every requested engine has an adapter, lifecycle support and at least one
  accepted real source/preparation-to-runtime path; any legitimate prebuilt-only
  family is separately labeled and does not masquerade as source conversion.
- Inventory every family exposed by each selected upstream; record independent
  load/convert/quantize/task/platform states and evidence for every promised cell.
  Catalog-only and unfinished upstream entries remain explicitly unaccepted.
- All advertised architecture/task families pass representative native tests;
  every published precision is validated during its actual release workflow.
- Windows and Linux checks pass for advertised CPU paths; GPU/Metal support is
  claimed only for exercised hardware/backend cells, not copied from README text.
- Existing text, voice, media, fork/PR and resume behavior remains covered.
- Original-runtime persistence/refusal/update/offline decisions match the user's
  choices and work through every supported harness.
- Clean-install, update, migration, rollback, interrupted-run resume, package
  metadata, CLI/MCP and published-bundle re-download checks pass.
- Documentation teaches `aqx run org/model` first, includes the support matrix,
  and explains only the unresolved choices a user needs to make.
- Close the applicable outstanding 0.5.0 acceptance work and publish accurate
  release notes. Version/tag/publication happen only after release approval.

## First implementation slice

Begin with 0.5.1: define runtime/install/build identities and decision results;
adopt current original roots; move regular-use media builds out of `temp`; add
the temporary-original option; implement update discovery, candidate build,
activation/rollback and runtime locks; then connect text, voice, media and MCP
to that shared path. Run targeted migration/failure/resume tests before adding
new engine adapters. No runtime downloads, rebuilds, migrations or package
version changes are implied by approval of this plan. Implementation status and
native acceptance evidence are tracked separately in RUNTIME_LIFECYCLE_STATUS.md.
