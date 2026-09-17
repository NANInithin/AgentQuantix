# Qwen3 TTS llama.cpp backend audit

## Decision

AgentQuantix supports Qwen3-TTS through upstream llama.cpp. The former pinned
`predict-woo/qwen3-tts.cpp` adapter and `qwen3-tts-cli` runtime are not part of
the release architecture. The initial validated source is exactly
`Qwen/Qwen3-TTS-12Hz-1.7B-Base`; family-name and size-only aliases are rejected
because several upstream 1.7B variants exist and have different semantics.

| Property | Contract |
| --- | --- |
| Backend | `llama.cpp` |
| Converter | `convert_hf_to_gguf.py` |
| Runtime | `llama-tts` |
| Primary format | GGUF |
| Required companion | audio tokenizer/projector `mmproj` GGUF |
| Initial primary quants | Q8_0, Q6_K, Q5_K_M, Q4_K_M |
| Companion quant | Q8_0 |
| Output gate | valid PCM WAV, plausible duration, non-silent, not clipped |
| Intelligibility gate | Whisper ASR round-trip WER |
| Final gate | explicit human listening review |

## Conversion and runtime path

The pipeline downloads a pinned source revision, converts the primary model and
mmproj with the converter from the same llama.cpp checkout, then quantizes only
the primary model through the conservative TTS quant set. It validates every
candidate through the actual runtime:

```bash
llama-tts -m Qwen3-TTS-Q4_K_M.gguf \
  -mm mmproj-Qwen3-TTS-Q8_0.gguf \
  -p "Hello from AgentQuantix." \
  --tts-lang en \
  --tts-speaker-file speaker.wav \
  --output out.wav
```

The exact llama.cpp revision is recorded in the bundle configuration. A bundle
is not publishable when conversion omits its mmproj, runtime inference fails,
the WAV gate fails, ASR round-trip WER exceeds the configured threshold, or the
human review rejects the candidate quant.

The round-trip gate requires only whisper.cpp's `whisper-cli`. It does not
build the ASR-only quantizer; current whisper.cpp names that CMake target and
binary `whisper-quantize`, while older checkouts used `quantize`.

## Publication contract

All model members, `bundle.json`, `quality.json`, and the voice-specific model
card are submitted in one Hub commit. Resume skips a member only when its
remote size and SHA-256 match the manifest. Publication is followed by a fresh
remote verification of every required bundle member.

## Boundaries

- Pocket TTS uses the same llama.cpp runtime but has a different source layout
  and requires a speaker reference.
- Whisper ASR is not routed through llama.cpp. It uses an independently cached
  whisper.cpp build, `whisper-cli`, a 16 kHz 16-bit WAV corpus, native Whisper
  quant types, and a WER regression gate against the base model.
- Loading a GGUF is necessary but insufficient. The release gate is real audio
  inference plus the track-specific quality suite.

References: [llama.cpp TTS](https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md)
and [whisper.cpp](https://github.com/ggml-org/whisper.cpp).
