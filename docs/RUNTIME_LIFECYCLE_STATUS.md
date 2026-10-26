# Runtime lifecycle implementation (0.5.1 development)

The [0.5.1–0.6.0 plan](RELEASE_0_6_0_PLAN.md) was approved on 25 October 2026.
The first implementation milestone adds shared runtime management for the four
existing native engines. Source version remains 0.4.5; this work does not publish
0.5.0/0.5.1 or establish native model acceptance.

## Implemented

- Shared identities, manifests, upstream update discovery, storage decisions,
  adoption, candidate builds, activation, rollback, locks, leases and pruning.
- Native builders for llama.cpp, audio.cpp, whisper.cpp and stable-diffusion.cpp.
  Other engines in the roadmap have registry identities only; their builders
  and task pipelines remain later milestones.
- First-use regular installation outside `temp`, or temporary original builds
  when the user declines regular use. Compatibility forks/PRs retain their
  existing temporary lifecycle.
- Stable-release discovery, default-branch fallback when no stable release
  exists, exact commit resolution and a runtime-specific update decision.
- One metadata check per required engine per execution session, including voice
  scoring dependencies. An unavailable check is disclosed and installed-version
  use continues. Complete clean local source can support offline initial setup.
- Side-by-side build candidates. Failed source setup, native build or probes do
  not overwrite the active installation. Interactive execution can continue with
  the previous installed version after the user chooses that fallback.
- Adoption of existing original checkouts without resetting or modifying them.
  Existing media caches can be copied and verified outside temporary storage;
  source-bound build metadata triggers a candidate rebuild when needed.
- Frozen text/voice/media runtime snapshots, binary/resource/source integrity
  checks, retained catalog resources, and execution leases. Voice journals bind
  both conversion and ASR-scoring runtimes.
- Resume keeps its recorded engine. A newer revision may be installed for future
  work without changing the resumed job. If a temporary runtime was cleaned,
  reconstruction uses its exact SHA and must match the frozen artifacts; a
  nonidentical rebuild requires replanning.
- Revision-specific stable-diffusion.cpp catalogs and audio.cpp model specs.
  Updates cannot silently keep using another revision's bundled metadata.
- CLI runtime maintenance and equivalent MCP tools, with structured decisions
  rather than terminal prompts in MCP. Model `--yes` does not authorize engine
  updates or a new runtime-storage preference.
- Existing SDXL detection and path-diagnostic fixes remain covered.

## User workflow

Normal model execution still uses the commands available in this milestone.
Before required engine setup/update, the CLI explains the selected revision and
asks for the runtime decision. Persistent installations are remembered. A
temporary-original preference applies to the invocation unless explicitly saved.

Runtime maintenance is optional:

```sh
aqx runtime list
aqx runtime show llama.cpp
aqx runtime check stable-diffusion.cpp
aqx runtime install stable-diffusion.cpp --persistent
aqx runtime update audio.cpp
aqx runtime adopt llama.cpp /path/to/existing/llama.cpp
aqx runtime rollback audio.cpp
aqx runtime prune stable-diffusion.cpp
```

`install` also accepts `--temporary`. Runtime `--yes` approves that explicit
maintenance action; it is separate from model `--yes`. Pruning defaults to an
inventory; `--apply` selects removal and asks for approval. Adopted external
checkouts, active/previous installations and leased builds are protected.

For noninteractive or MCP use, read runtime status and check upstream first.
Show the exact target SHA and storage choice to the user, then call
`prepare_runtime` with that target and the corresponding explicit approval.
Separate tools provide adoption, rollback and prune inventory/apply actions.

## Storage

Managed persistent candidates live under
`WORK_ROOT/.aqx-runtime-builds/<engine>/<sha>/<configuration>/`.
Existing original checkouts can be adopted at their present location.
Temporary originals live under
`TEMP_DIR/aqx-runtime-sessions/<engine>/<sha>/<configuration>/`.
Existing compatibility-fork directories retain their original behavior.

Installation records, checks, leases and retained external catalog resources
live under `AQX_HOME/state/runtimes/`. Native conversion assets follow the
selected installation's persistent/temporary policy. Complete successful
sessions can clean managed temporary originals; interrupted or failed sessions
retain them for resume. Explicit pruning may require later reconstruction.

## Validation boundary and remaining work

Offline tests exercise local source fixtures and controlled Git/native
subprocess boundaries, including failure injection, temporary cleanup, migration,
offline setup, approval decisions, frozen resume and resource/catalog drift.
The Windows build wrapper is exercised with a real batch subprocess using paths
containing spaces and `&`. These checks do not constitute native model inference
or the cross-platform acceptance matrix.

Development verification on Windows / Python 3.14:

- Full offline suite: **434 passed**.
- Ruff correctness/unused-import checks: passed.
- Wheel and source distribution built successfully into an isolated test
  directory; both include the new runtime modules and exclude local environments.
- Extracted-wheel CLI help/runtime inventory and stdio MCP initialization/tool
  discovery passed using isolated runtime state.

Still required before a release claim:

- Actual clean install, migration and update/recovery runs with each existing
  engine on Windows and Linux, and the hardware modes being advertised.
- Real conversion/inference and quality acceptance for text, TTS, ASR and media,
  including the outstanding 0.5.0 media gates.
- Packaging/install verification and full recorded release evidence.

## 0.5.2 shared task router and commands

Implemented in local source: `agentquantix.tasks` (task registry, capability
cells with independent load/convert/quantize/package/acceptance facts, metadata
router), `agentquantix.runs` (text:/voice:/media: run ids over the existing
records) and `agentquantix.actions` (inspect/plan/start/review/publish/verify)
shared by the CLI and every agent harness. `aqx voice` and `aqx media` were
removed without aliases; `aqx runtime show` carries the engine catalogs. The
modality-specific agent tools were replaced by `inspect_model`, `plan_release`,
`start_release`, `record_review`, `publish_release`, `list_runs` and
`verify_release`. Routing was checked offline and against live Hub metadata for
text, voice, media, embedding, reranker, detection, depth, CLIP and forecasting
repositories. Early audits of the seven new engines are recorded in
[ENGINE_AUDIT_0_5_2.md](ENGINE_AUDIT_0_5_2.md).

The new runtime/task adapters remain in 0.5.3–0.6.0 as specified in the
approved plan.
