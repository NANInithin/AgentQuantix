# Changelog

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
