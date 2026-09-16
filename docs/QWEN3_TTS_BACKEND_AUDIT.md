# Qwen3-TTS external-backend audit

**Status:** adopted for the v0.3.0 implementation path, pending Windows and
Linux end-to-end acceptance runs. This document is the boundary of that
adoption: AgentQuantix pins and verifies the checkout; it does not vendor its
code or silently follow the backend's `main` branch.

## Component pinned

| Field | Value |
| --- | --- |
| Backend | `predict-woo/qwen3-tts.cpp` |
| Repository | `https://github.com/predict-woo/qwen3-tts.cpp` |
| Pinned commit | `b3ba14077cf1b3e11b86e5f84aa9184605c89b28` |
| Licence | MIT |
| Runtime | `qwen3-tts-cli` |
| Build | CMake + the repository's vendored GGML submodule |
| Windows verification | Built and launched from the pinned checkout on 2026-09-16 |

The adapter refuses a checkout whose `HEAD` is not that exact commit. A later
backend upgrade is a new audit, benchmark, and pin—not an unattended change.

## Verified interface

The backend documents two conversion commands and a local runtime command:

```text
convert_tts_to_gguf.py       → qwen3-tts-0.6b-<f16|q8_0|q4_k>.gguf
convert_tokenizer_to_gguf.py → qwen3-tts-tokenizer-f16.gguf
qwen3-tts-cli -m <bundle-dir> -t <prompt> -o <fresh-output.wav>
```

The two GGUF files are one release bundle. The talker may be F16, Q8_0, or
Q4_K; the audio tokenizer/vocoder remains F16. AgentQuantix records both
members, their SHA-256 values, roles, source identity, and adapter revision in
the bundle manifest.

## Supported scope

Only `Qwen/Qwen3-TTS-12Hz-0.6B-Base` is admitted. The audited converter names
that model explicitly and contains fixed architecture assumptions for it.
Qwen3-TTS 1.7B, CustomVoice, VoiceDesign, and future checkpoint layouts are
Tier 3 until they have an independent converter/runtime audit and quality gate.

## Constraints and risks

- This is a dedicated GGML runtime, not stock llama.cpp. Do not publish
  `llama-cli` or `llama-tts` commands for these bundles.
- The backend builds its own GGML submodule. It must not share AgentQuantix's
  llama.cpp build directory or ABI.
- On Windows, the executable needs the vendored `ggml/build/bin/Release` DLL
  directory on `PATH`; the adapter provides it for its smoke invocation rather
  than relying on a developer shell.
- The external converter can report unmapped tensors. The operational adapter
  must capture converter output and reject non-zero skipped-tensor counts
  before it is permitted to upload a bundle.
- The backend reports deterministic/reference tests, but those are not a
  substitute for AgentQuantix's fresh output-WAV validation or v0.3.1 quality
  benchmark.
- No unrestricted `git pull`, branch tracking, or automatic model-family
  expansion is allowed in production runs.

## Remaining v0.3.0 gates

1. Run the pinned converter on the admitted source and reject unmapped tensors.
2. Build the pinned runtime on Linux, then generate a fresh valid WAV on both
   Windows and Linux from the two-file bundle before upload. The Windows build
   and `--help` launch have been verified; no model weights were downloaded.
3. Make bundle upload/resume compare remote size and SHA-256, then extend the
   card validator with the runtime and companion-file facts.
4. Record the exact source revision, backend pin, sample rate, and runtime
   smoke facts in the published manifest and model card.
