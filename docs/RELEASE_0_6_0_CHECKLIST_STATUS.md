# 0.6.0 completion checklist status

Status against the [0.6.0 plan](RELEASE_0_6_0_PLAN.md). "Met" is limited to
the stated checklist item and its evidence; it is not a release approval.
Accepted model evidence is limited to Linux x86_64 CPU and the task/family/
precision cells in [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md).

## 0.6.0 completion checklist

| Plan item | State | Evidence or missing piece |
|---|---|---|
| Every requested engine has an adapter, lifecycle support, and at least one accepted real source/preparation-to-runtime path; prebuilt-only families are labeled. | partially met | The support matrix and task status documents identify accepted paths and rejected/blocked families. Linux native evidence exists for selected engines and families. Text generation, TTS, stable-diffusion.cpp media, SAM3, and forecasting lack an accepted path; Windows/MSVC build and acceptance for the new engines is missing. See [runtime lifecycle status](RUNTIME_LIFECYCLE_STATUS.md) and each linked status document in the matrix. |
| Inventory every family exposed by each selected upstream, recording load/convert/quantize/task/platform states and evidence; keep catalog-only entries unaccepted. | partially met | [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) covers all registered TaskSpecs and named family evidence; the task status documents contain engine/catalog detail and explicitly distinguish catalog presence from acceptance. A complete independent load/convert/quantize/platform evidence cell for every exposed family across every upstream is not established. |
| All advertised architecture/task families pass representative native tests, and every published precision is validated in its release workflow. | partially met | Named accepted cells have native results in the status docs and `/home/ubuntu/spike/acceptance/`. Text-generation/TTS have no accepted native evidence; all stable-diffusion.cpp media families are experimental without real-model acceptance; rejected, blocked, and untested catalog families remain listed rather than accepted. No published candidates were used to establish release validation. |
| Windows and Linux checks pass for advertised CPU paths; GPU/Metal is claimed only for exercised hardware/backend cells. | not met | Linux x86_64 CPU is the only proven native acceptance platform. New-engine Windows/MSVC builds and native acceptance are missing. No GPU or Metal acceptance is recorded. |
| Existing text, voice, media, fork/PR, and resume behavior remains covered. | partially met | Shared CLI and router tests, the resume round-trip tests in `tests/test_shared_actions.py`, status docs, and the YOLO interrupted-resume log `/home/ubuntu/spike/0.5.9-hardening/logs/phase2-yolo26n-cls-resume-followup-verified.log` cover portions of the flows. A complete native and cross-platform matrix, especially real media acceptance, is missing. |
| Original-runtime persistence, refusal, update, and offline decisions match user choices through every supported harness. | partially met | [Runtime lifecycle status](RUNTIME_LIFECYCLE_STATUS.md) records CLI/MCP decisions, frozen snapshots, locks, rollback, and offline behavior. Cross-platform end-to-end validation through every harness remains outstanding. |
| Clean install, update, migration, rollback, interrupted-run resume, package metadata, CLI/MCP, and published-bundle re-download checks pass. | partially met | Hardening rehearsal evidence covers fresh wheel/sdist installation, version 0.4.5, CLI/MCP smoke, synthetic migration, runtime lifecycle controls, and interrupted YOLO resume under `/home/ubuntu/spike/0.5.9-hardening/`. Real Hub publication and re-download/verification are missing; new-engine Windows/MSVC acceptance and an end-to-end release matrix are also missing. |
| Documentation leads with `aqx run org/model`, includes the support matrix, and explains unresolved user choices. | met | [README.md](../README.md), [USAGE.md](../USAGE.md), [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md), and this draft checklist document the shared entry point, evidence boundary, per-pipeline commands, and open release choices. |
| Close applicable outstanding 0.5.0 acceptance work and publish accurate release notes; version/tag/publication wait for approval. | not met | The outstanding 0.5.0 items below remain open. [0.6.0 release notes](RELEASE_NOTES_0_6_0.md) are explicitly a draft. Owner release approval, real Hub publication, versioning, and tagging have not occurred. |

## Outstanding 0.5.0 acceptance work

Items are drawn from [AGENT_HANDOVER.md](AGENT_HANDOVER.md),
[RUNTIME_LIFECYCLE_STATUS.md](RUNTIME_LIFECYCLE_STATUS.md),
[RELEASE_VALIDATION.md](RELEASE_VALIDATION.md), and the original
[stable-diffusion.cpp release plan](STABLE_DIFFUSION_CPP_RELEASE_PLAN.md).

| Acceptance item | State | Evidence or missing piece |
|---|---|---|
| Route supported image/video repositories to stable-diffusion.cpp without per-model core allowlists. | met | Shared metadata/catalog routing and the family capability catalog are implemented; offline routing and catalog tests are present in the source test suite. See [media implementation status](MEDIA_IMPLEMENTATION_STATUS.md). This is routing evidence, not model acceptance. |
| Generate and pin the capability manifest to an exact upstream revision. | met | The media status records stable-diffusion.cpp revision `8cd731e7b4667eb8f40a153315b9a7125d09a58d`, 56 catalog families, source hashes, and the CI catalog-drift check. |
| Distinguish load, conversion, and quantization support in plans. | met | Media planning records these capabilities independently; see [media implementation status](MEDIA_IMPLEMENTATION_STATUS.md) and the related offline plan tests. |
| Resume, verify, and publish multi-component bundles atomically. | partially met | Journals, checksums, frozen plans, resume, verification, and atomic publication are implemented and covered by offline tests. A full real-model workflow through real Hub publication and re-download is missing. |
| Derive quant choices from the selected backend and preserve high-precision components when required. | partially met | Native converter probes, per-component precision policy, and component contracts are implemented. Real stable-diffusion.cpp model acceptance proving these choices in release workflows is missing. |
| Require real `sd-cli` inference for every published candidate. | not met | No real stable-diffusion.cpp candidate has passed native model acceptance or been published. |
| Pass image, video, and audio-video structural and calibrated quality gates against a high-precision baseline. | not met | Structural checks and evaluator interfaces exist, but real-model family/task calibration and native baseline evidence are outstanding. |
| Complete MiniMax-H3 as a verified GGUF-plus-companions bundle with human-reviewed video and audio. | not met | MiniMax-H3 source/model acceptance, bundle verification, human playback/listening review, and publication evidence are missing. |
| Add a new upstream family by pinning/upgrading the backend and regenerating capability data, without model-specific pipeline code. | partially met | The catalog generator and drift check are implemented and cover the pinned catalog. An end-to-end upstream revision upgrade and family-addition acceptance is not recorded. |
| Run the published card command after downloading only files listed in the bundle manifest. | not met | No published media bundle has been re-downloaded and used to verify the card command. |
| Complete text-generation and TTS native acceptance tracks. | not met | `SUPPORT_MATRIX.md` records no accepted text-generation or TTS family/precision. `RELEASE_VALIDATION.md` requires text and llama-TTS native recipes with expected-output and track-specific gates. |
| Complete native acceptance on Linux and Windows for the required text, TTS, ASR, and media tracks. | partially met | Selected Linux CPU ASR, retrieval, multimodal, vision, detection, and depth cells are accepted. Text-generation/TTS and media acceptance are absent; Windows/MSVC acceptance is not recorded. |
| Complete Windows/MSVC builds and acceptance for the new engine adapters; exercise GPU/Metal if claimed. | not met | No new-engine Windows/MSVC acceptance, GPU, or Metal evidence is present. |
| Exercise real Hub credentials, publication, atomic commits, and re-download verification. | not met | Native acceptance used `--no-publish`; there is no real Hub publication/re-download evidence. Owner-provided Hub authentication is required for that release operation. |
| Obtain owner release approval before version, tag, or publication changes. | not met | This branch is documentation-only; the package version is 0.4.5 and the release notes remain a draft. Owner approval has not been granted. |
