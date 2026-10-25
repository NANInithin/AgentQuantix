# Agent handover — 25 October 2026

## Start here

The user approved the [0.5.1–0.6.0 implementation plan](RELEASE_0_6_0_PLAN.md).
The 0.5.1 shared runtime foundation is implemented in local source and covered
by offline tests. Continue with the **0.5.2 shared task router and unified CLI**,
while keeping native acceptance outstanding and accurately reported.

Read the approved plan and [runtime implementation status](RUNTIME_LIFECYCLE_STATUS.md)
before editing. Source/package version is still **0.4.5**. No release, tag,
publication, real runtime installation, or external runtime migration has been
performed for this development milestone. This handover is included in the
implementation commit on `main`; find that commit with `git log -1 --oneline`.

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
- `runtime_builders.py`: actual build/probe contracts for the **four existing
  engines only**: llama.cpp, audio.cpp, whisper.cpp, stable-diffusion.cpp.
  Captures toolchain, resources, shared libraries, binary hashes and catalogs.
- `runtime_control.py`: interactive policy and per-execution binding; required
  binary scoping; existing media-cache discovery; voice runtime journals;
  explicit failed-update fallback and future-only updates for frozen resumes.
- `runtime_cli.py` and `agent/runtime_tools.py`: inspect/check/install/update/
  adopt/rollback/prune controls. MCP tools return structured requirements and
  never read terminal input. Runtime preparation requires an exact target SHA
  plus runtime-specific approval.

Registry identities exist for all eleven planned engines. Seven new engine
builders and their task pipelines **are not implemented yet**.

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

## Next implementation slice: 0.5.2

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

The approved plan assigns later milestones to retrieval/CLIP (0.5.3), libmtmd
understanding (0.5.4), expanded audio (0.5.5), vision.cpp (0.5.6), YOLO/RF-DETR
(0.5.7), dedicated depth/SAM3 (0.5.8), forecasting/hardening (0.5.9), and the
consolidated 0.6.0 release. Do not claim the full roadmap is implemented.

## Native acceptance and upstream cautions

- Real Windows/Linux clean installs, migration, update/recovery, conversion,
  inference and quality acceptance remain outstanding for the existing engines,
  including the earlier 0.5.0 media gates. Preserve
  `docs/RELEASE_VALIDATION.md` and the original media release plan's requirements.
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
roadmap. Actual release publication remains a separate operation. Do not push
commits, publish models, or announce acceptance that has not been established.
