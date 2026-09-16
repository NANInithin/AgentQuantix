# Voice-model release plan

## Product boundary

AgentQuantix currently quantizes text-generation GGUFs through `llama-quantize`
and validates them by loading them through `llama-imatrix`. Voice models are a
different product surface: a usable release can contain a language-model GGUF,
one or more audio codec/projector GGUFs, and runtime-specific configuration.
The pipeline must publish and validate that **bundle**, never a language tower
in isolation.

The support tiers describe evidence, not marketing:

| Tier | Meaning | Release position |
| --- | --- | --- |
| 1 | Upstream runtime and repeatable end-to-end path exist; AgentQuantix may automate it. | Qwen3-TTS in v0.3.0; Pocket TTS in v0.3.1 |
| 2 | Upstream audio-input path exists; it is a separate ASR/audio-understanding product. | v0.4.0, beginning with Voxtral and Qwen3-ASR |
| 3 | No supported end-to-end path yet; retain a researched adapter record only. | No release commitment |

Tier 3 is intentionally **not** enabled in v0.3.0. “Experimental” must not
mean that an unproven model can enter an unattended upload pipeline.

## v0.3.0 — Qwen3-TTS (Tier 1)

### Goal

Publish a complete Qwen3-TTS quant bundle only after the runtime generates a
valid non-empty WAV file from it. This release supports TTS output only; it
does not add ASR, arbitrary voice families, or subjective quality ranking.

### Implementation work

1. Add the voice capability registry and make Qwen3-TTS the only enabled
   v0.3.0 family. Keep Pocket TTS, Voxtral, and Qwen3-ASR visible as planned
   capabilities but reject them from the execution path.
2. Define a persisted bundle manifest: family, source revision, primary GGUF,
   every required companion, byte size, SHA-256, and each file's Hub path.
3. Adopt and pin the maintained MIT `predict-woo/qwen3-tts.cpp` converter and
   runtime at `b3ba14077cf1b3e11b86e5f84aa9184605c89b28`. Its audited scope is
   exactly `Qwen/Qwen3-TTS-12Hz-0.6B-Base`; the bundle is a talker GGUF plus an
   F16 tokenizer/vocoder GGUF, not an assumed `BF16 + mmproj` layout.
4. Build/locate `qwen3-tts-cli` with its vendored GGML DLLs and record the
   pinned backend revision in the job record. This is a dedicated runtime,
   rather than the evolving llama.cpp TTS path.
5. After conversion, run a fixed smoke prompt with `qwen3-tts-cli`; require a
   successful exit, a created WAV, a supported PCM header, non-zero frames,
   and a plausible duration before any bundle member is uploaded.
6. Upload the manifest and all bundle members resumably. Verify exact names,
   sizes, and checksums from the Hub listing before rendering the card.
7. Generate a voice-specific card: languages, sample rate, companion files,
   exact `qwen3-tts-cli` usage, licence/provenance, and a clear notice that the
   smoke validation is not a perceptual-quality score.

### Acceptance criteria

- A supported Qwen3-TTS source produces a complete manifest and no file is
  published when the runtime smoke test fails.
- Resume uses the manifest and skips only members whose remote size/checksum
  match; it never treats a language GGUF alone as a finished voice release.
- Offline tests cover capability gating, bundle validation, manifest stability,
  failed runtime execution, invalid/empty WAV output, card facts, and resume.
- One real end-to-end run is recorded against the pinned upstream revision on
  Windows and Linux before tagging v0.3.0.

## v0.3.1 — Pocket TTS and automated quality scoring (Tier 1)

### Goal

Add Pocket TTS without weakening the Qwen3-TTS contract, then rank or reject
quants using repeatable audio checks in addition to v0.3.0's runtime smoke test.

### Implementation work

1. Add Pocket TTS's required language directory and mmproj/codec members to
   the same manifest contract; validate its required speaker-reference input.
2. Create a versioned, licensed benchmark pack of prompts and reference audio;
   keep it separate from production jobs and record its revision in results.
3. Record deterministic metrics: successful-generation rate, duration error,
   silence/clipping checks, and ASR round-trip intelligibility where available.
4. Establish per-family/per-language regression thresholds from BF16 baselines;
   label metrics as automated proxies, not MOS or speaker-similarity claims.
5. Publish a machine-readable `quality.json` with every bundle and add a
   concise quality table to the model card.

### Acceptance criteria

- Both Tier 1 families pass the same bundle and runtime contract.
- A deliberately degraded output is rejected by the scoring gate.
- Scores are reproducible from a pinned benchmark pack and runtime revision.

## v0.4.0 — ASR/audio-understanding backend (Tier 2)

### Goal

Support Voxtral and Qwen3-ASR as audio-input models through a separate
`llama-mtmd-cli`/`libmtmd` adapter. Do not route them through the TTS pipeline.

### Implementation work

1. Introduce an ASR bundle type with its audio encoder/projector and prompt
   template, separate from TTS codecs and output WAV checks.
2. Use a licensed transcription fixture set; validate normalized WER, language
   handling, long-audio chunking, and no-audio/silence failure behaviour.
3. Estimate feasibility by audio seconds, context/audio-token consumption,
   encoder memory, and real-time factor rather than text imatrix cost.
4. Add ASR-specific cards: transcription commands, supported formats,
   language/prompt settings, benchmark corpus, WER, and known limitations.
5. Keep streaming/server support explicitly experimental until it has its own
   integration tests; batch CLI validation is the v0.4.0 release gate.

### Acceptance criteria

- A known fixture reaches the model-specific WER threshold after quantization.
- Missing encoder/projector, invalid audio, and overlong audio fail before upload.
- TTS and ASR jobs cannot be mistaken for one another in state, manifests, or cards.

## Tier 3 admission process

An experimental family becomes Tier 1 or Tier 2 only after all of these are
available: an upstream or maintained adapter, a reproducible conversion path,
a complete bundle definition, an actual runtime command, a small licensed test
fixture, and a family-specific quality gate. Until then AgentQuantix may report
the family as researched, but it must not offer `run`.

## Delivery sequence and ownership boundaries

1. **Foundation (started now):** capability registry, bundle contract, WAV
   validation, and unit tests. These are pure Python and do not alter current
   text-model runs.
2. **Qwen adapter:** source discovery/conversion, `qwen3-tts-cli` build lookup,
   staging and manifest persistence.
3. **Publication:** resumable bundle upload, remote verification, and cards.
4. **Operational proof:** Windows/Linux end-to-end smoke runs, then v0.3.0 tag.

### Qwen conversion decision gate

Upstream llama.cpp documents `llama-tts` inference from published Qwen3-TTS
GGUFs but does not document a raw-checkpoint converter. AgentQuantix has now
adopted and pinned the separately audited `predict-woo/qwen3-tts.cpp` converter/
runtime for the exact 0.6B Base source. The source must still pass a real
Windows and Linux conversion-plus-generation gate before v0.3.0 is tagged; no
other Qwen3-TTS variant inherits this support boundary.

Never share the text `imatrix` path with a voice family merely because both
produce GGUF. A family must explicitly supply its own calibration or explain
why a normal quantization path is safe.
