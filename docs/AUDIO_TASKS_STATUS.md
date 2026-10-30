# Audio task status - 0.5.5

The audio.cpp pin is v0.8.0 (`4af143229384fb6da3f373dc87de145ae954609b`).
Tasks extend the shared voice release pipeline. Catalog `tasks` declarations
are the backend contract; catalog presence alone does not establish acceptance.
All native acceptance is Linux CPU, `--no-publish`, and model-specific.

## Capability and evidence matrix

| Category | Status | Family / license | Evidence and limits |
|---|---|---|---|
| Separation | Accepted | BS-RoFormer, Apache-2.0 per package README (checkpoint source undocumented) | Q8_0 passes both SI-SDR improvement gates. HTDemucs MIT runs at F16 and Q8_0 but neither improves the vocal stem enough to pass. |
| Voice conversion | Accepted | MeanVC2, Apache-2.0 | F32 and Q4_K pass waveform and round-trip WER gates. Speaker similarity was not measured. |
| Forced alignment | Accepted | Qwen3 Forced Aligner, Apache-2.0 | Source conversion passes all nine native precisions. There is no independent timing reference. |
| Enhancement / s2s | Rejected | RNNoise BSD-3-Clause; ZipEnhancer asset license not identified | Both native paths worsened SNR; `s2s` is not offered. |
| Music / SFX generation | Not tested | ACE-Step MIT; Stable Audio Stability AI Community License | The smallest ACE-Step Q8 package is 6,185,468,096 bytes, above the evaluation size preference; Stable Audio is not permissively licensed. |
| Diarization | Not tested | Sortformer v1 CC-BY-NC-4.0; v2 local-only | Non-commercial license or redistribution approval prevents an eligible package trial. |
| MIDI | Not tested | MuScriptor and SheetSage2 CC-BY-NC-4.0 | Candidate packages are non-commercial. |
| Classification | Not supported | — | The pinned audio.cpp CLI and inspected docs expose no standalone classification task. |
| Standalone VAD | Not supported for release | MarbleNet is a bundled runtime asset | Audio.cpp has a native VAD task, but its model is a bundled asset rather than a releasable model source. |

The feasibility spike also tested standalone MarbleNet VAD: five segments,
10 ms frame F1 `0.8732` against the padded clip-span labels. Whisper.cpp
documents a `whisper-cli --vad --vad-model <silero-model>` route, but no pinned
whisper.cpp checkout or standalone AgentQuantix VAD route was verified.

## Gates

- **Separation:** vocal is the stem with `vocal` in its filename.
  Accompaniment is an `instrumental`/`accompaniment` stem when present, or the
  sum of all non-vocal stems. Resampling and length alignment precede SI-SDR.
  F16/base precision requires at least 6 dB improvement over the mixture for
  both stems; every other precision must be within 1 dB of the baseline per
  stem.
- **Voice conversion:** output samples must be finite and non-silent (RMS above
  `1e-3`), and duration must be within 20% of the input. TTS's Whisper
  round-trip ASR is reused: baseline WER at most `0.35`, other precisions at
  most baseline plus `0.10`. Speaker similarity is not measured.
- **Forced alignment:** normalized word count must match the transcript;
  starts and ends must be monotonic and each interval must satisfy
  `0 <= start < end <= duration`. Other precisions must be within mean
  boundary delta `0.05 s` and maximum `0.20 s` of the baseline. There is no
  independent timing reference.

Every precision records metrics, output paths, command timing and gate status.
Accepted precision files are published together as one atomic bundle. New task
publication requires a known source license; `--no-publish` remains available
when license resolution fails.

## Fixtures

`scripts/generate_audio_task_fixtures.py` creates deterministic fixtures under
`src/agentquantix/fixtures/voice/`. `separation/mixture.wav` is JFK speech plus
a synthetic chord; `speech.wav` and `accompaniment.wav` are the reference
stems. `separation/manifest.json` records the hashes. The script also creates
`vc-reference.wav`, a deterministic +4-semitone JFK pitch shift trimmed to
four seconds. Alignment reuses `jfk.wav` and `asr.json`. The source JFK clip is
public domain; derived fixtures are CC0.

## Native acceptance

All runs below used the lifecycle-installed audio.cpp revision
`4af143229384fb6da3f373dc87de145ae954609b` (v0.8.0), Linux CPU,
`--no-publish`, and the repository fixtures. Timings are per-task inference
seconds unless marked as preparation time.

### Source separation

SI-SDR improvement is estimated-stem SI-SDR minus mixture SI-SDR. HTDemucs
rows are independent runs and each precision is its own baseline because F16
did not pass.

| Family | Precision | Status | Vocal improvement | Accompaniment improvement | Inference | Result |
|---|---|---|---:|---:|---:|---|
| BS-RoFormer | Q8_0 | Accepted | +20.690914 dB (26.198514 vs 5.507600 dB mixture) | +26.184860 dB (20.670889 vs -5.513971 dB mixture) | 379.9532 s | `/home/ubuntu/spike/acceptance/audio-bs-roformer-results.json` |
| HTDemucs | F16 | Rejected | -8.292766 dB (-2.785165 vs 5.507600 dB mixture) | +0.119774 dB (-5.394196 vs -5.513971 dB mixture) | 14.7828 s | `/home/ubuntu/spike/acceptance/audio-htdemucs-f16-results.json` |
| HTDemucs | Q8_0 | Rejected | -8.279426 dB (-2.771825 vs 5.507600 dB mixture) | +0.119575 dB (-5.394396 vs -5.513971 dB mixture) | 14.6356 s | `/home/ubuntu/spike/acceptance/audio-htdemucs-q8_0-results.json` |

### Voice conversion

Both precisions use F32 as the baseline. Q4_K's WER increase is `0.045454`,
below the allowed `+0.10`; both outputs are finite, non-silent 16 kHz audio of
10.91 s from an 11.0 s input.

| Family | Precision | Status | Round-trip WER | RMS | Inference / ASR | Result |
|---|---|---|---:|---:|---:|---|
| MeanVC2 | F32 | Accepted baseline | 0.136364 | 0.098022 | 6.0722 / 0.6453 s | `/home/ubuntu/spike/acceptance/audio-meanvc2-f32-q4_k-combined-results.json` |
| MeanVC2 | Q4_K | Accepted | 0.181818 (baseline +0.045454) | 0.107829 | 4.0380 / 0.5651 s | `/home/ubuntu/spike/acceptance/audio-meanvc2-f32-q4_k-combined-results.json` |

The separately saved F32 and Q4_K runs also passed their absolute gates. The
combined run above records the required cross-precision comparison.

### Forced alignment

Qwen3 was converted from Apache-2.0 source revision
`c7cbfc2048c462b0d63a45797104fc9db3ad62b7`. BF16 is the baseline. Every
precision returned 22 valid word intervals over the 11.0 s clip. The other
precisions meet mean delta `<= 0.05 s` and maximum delta `<= 0.20 s`.

| Precision | Status | Words | Mean / max boundary delta | Inference / preparation | Result |
|---|---|---:|---:|---:|---|
| BF16 | Accepted baseline | 22 | baseline | 3.8727 / 1.9909 s | `/home/ubuntu/spike/acceptance/audio-qwen3-forced-aligner-source-results.json` |
| F16 | Accepted | 22 | 0.000000 / 0.000000 s | 4.0725 / 4.5645 s | same |
| ORIG | Accepted | 22 | 0.000000 / 0.000000 s | 4.0548 / 1.4766 s | same |
| Q2_K | Accepted | 22 | 0.005455 / 0.080000 s | 2.4603 / 45.5017 s | same |
| Q3_K | Accepted | 22 | 0.000000 / 0.000000 s | 3.0237 / 11.7198 s | same |
| Q4_K | Accepted | 22 | 0.000000 / 0.000000 s | 2.7234 / 49.4217 s | same |
| Q5_K | Accepted | 22 | 0.000000 / 0.000000 s | 3.5019 / 47.0825 s | same |
| Q6_K | Accepted | 22 | 0.000000 / 0.000000 s | 3.1510 / 23.7003 s | same |
| Q8_0 | Accepted | 22 | 0.000000 / 0.000000 s | 2.8841 / 5.8890 s | same |

The first BF16 source-conversion attempt exposed a namespace mismatch:
AgentQuantix passed `--input weights=<model.safetensors>`, while the pinned
Qwen GGUF contract requires the root namespace. The exact failure is preserved
in `/home/ubuntu/spike/acceptance/audio-qwen3-align-source-v0.8.0.log` and its
JSON record. The converter now derives input namespaces from the GGUF tensor
prefixes for SEP/VC/ALIGN only; regression coverage is in commit `8b83640`.
The rerun converted and accepted all nine types. The package fallback was not
needed.

### Existing ASR regression

Moonshine Streaming Tiny Q8_0 (MIT, 60,409,088 bytes) passed on the pinned
runtime. The 11 s JFK fixture produced WER 0.0; elapsed inference was 0.3647 s
(119.338 ms reported by audio.cpp), with RTF 0.0332. The single available
precision is its own baseline, so the measured WER regression is 0.0 against
the allowed 0.05.

The first run exposed that the generic ASR command sent `--language en` to a
single-language Moonshine model, whose native request parser rejects it. The
command now omits a redundant language flag when the catalog declares exactly
one language; multilingual commands such as Qwen3-ASR still pass the requested
language. The initial failure is preserved in
`/home/ubuntu/spike/acceptance/audio-moonshine-tiny-q8_0-v0.8.0.log`.
The successful result and full command are in
`/home/ubuntu/spike/acceptance/audio-moonshine-tiny-q8_0-results.json`;
the retry log is
`/home/ubuntu/spike/acceptance/audio-moonshine-tiny-q8_0-v0.8.0-after-language-fix.log`.

### Feasibility-spike diagnostics

| Family | Precision | Result |
|---|---|---|
| BS-RoFormer | Q8_0 | Vocal SI-SDR `26.1854 dB` vs mixture `5.5076 dB`; accompaniment `20.6578 dB` vs `-5.5140 dB`; wall `448.31 s`. |
| HTDemucs | F16 | Vocal `-2.7864 dB` vs mixture `5.5076 dB`; accompaniment `-5.3942 dB` vs `-5.5140 dB`; wall `13.46 s`. |
| HTDemucs | Q8_0 | Vocal `-2.7731 dB`; accompaniment `-5.3944 dB`; wall `13.33 s`. Neither precision improved both stems over the mixture. |
| MeanVC2 | Q4_K | Valid finite/non-silent mono 16 kHz, `10.91 s`, RMS `0.14235`; wall `7.94 s`; no speaker-similarity score. |
| Qwen3 Forced Aligner | Q8_0 | 22 monotonic/in-range words over 11 s; runtime `3.617 s`, process `7.54 s`; no independent reference. |

Enhancement diagnostics: RNNoise SNR `-2.9361 dB` vs noisy-input `19.7126 dB`
(change `-22.6487 dB`); ZipEnhancer `9.1246 dB` vs `19.5559 dB` (change
`-10.4313 dB`). These failures are why `s2s` is not offered.

## Artifacts and remaining limits

- Feasibility evidence: `/home/ubuntu/spike/0.5.5/REPORT.md` and
  `/home/ubuntu/spike/0.5.5/results/results.json`.
- Native implementation results: `/home/ubuntu/spike/acceptance/`.
  SHA-256: BS-RoFormer `69c47cd68e9298c59a7c543a5a6262d6331cadf11188ee47600445d4f1711d4d`;
  HTDemucs F16 `222443575b9cd5607cd5260537f0edc3cea62ce4e403aee2b0cf9789ff4f42fa`;
  HTDemucs Q8_0 `2f8c9795cde83786b9487c5189ea1de9959ff77ef5af320efa25ce2adab64dc4`;
  MeanVC2 combined `97438dfc1374f01e6cbec99fab6ab27a954687030dce88e77a03b0eb1a2d4381`;
  Qwen3 source `f760c12392661044968b118c77902835c9c8f6805229c32a5a52a7f1d3d0f54d`;
  Moonshine ASR `4fde018de8e11e2447d313b5c96ea82cf9ac524ce89838f8fcf2c092e98ef8bb`.
  All run logs remain in the same directory.
- Windows/MSVC, GPU/Metal, real Hub publication and human review are not part
  of this Linux CPU acceptance.
