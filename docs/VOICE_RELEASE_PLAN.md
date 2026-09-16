# Voice release plan

## Direction

Support text-to-speech first through llama.cpp, then add automatic speech
recognition as a separate whisper.cpp backend. llama.cpp already exposes TTS
flows for Qwen3-TTS and Pocket TTS, while Whisper uses different artifacts and
runtime semantics.

References: [llama.cpp TTS](https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md)
and [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

## Plan

1. **Split voice into TTS and ASR tracks**

   Do not treat voice as one architecture: TTS emits audio, while ASR consumes
   it. Give each track a distinct converter, runtime, quality gate, and
   feasibility model.

2. **Add a `VoiceBackend` capability registry**

   Extend architecture support with the backend, converter, required companion
   files, and supported quant types. This prevents the core pipeline from
   becoming a chain of model-family conditionals.

3. **Make a model bundle the unit of publication**

   A voice release may include the primary GGUF, codec or mmproj, tokenizer,
   speaker-reference requirements, and configuration. Track, upload, resume,
   verify, and document all bundle members atomically.

4. **Start with two supported TTS families only**

   Target Qwen3-TTS and Pocket TTS first because llama.cpp already provides
   documented inference paths for them. Add new families only after they pass
   the same end-to-end acceptance suite.

5. **Build an audio-aware calibration pipeline**

   Replace wiki-text-only calibration with a small, licensed multilingual
   speech and prompt fixture set. Cover short and long utterances, punctuation,
   numbers, non-English text, and speaker-conditioning cases.

6. **Use quality gates, not just GGUF load checks**

   For TTS, generate fixed prompts and reject silence, clipping, invalid WAV
   headers, or wildly incorrect duration. For ASR, run a fixed audio corpus and
   set a maximum word-error-rate regression relative to BF16.

7. **Use a two-stage TTS quality assessment**

   Automate intelligibility checks with ASR round-trip word error rate, audio
   duration, and silence checks. Require a brief human listening review only
   for candidate quant types that pass automation.

8. **Quantize conservatively before expanding the sweep**

   Begin with a small voice-safe set: Q8_0, Q6_K, Q5_K_M, and Q4_K_M. Add
   lower-bit types only after measured intelligibility and speaker-similarity
   results are acceptable.

9. **Model feasibility in seconds of audio, not only parameters**

   Estimate real-time factor, tokens or frames per second, first-audio latency,
   VRAM or RAM, and bundle disk size. Report minutes required to synthesize one
   hour of audio alongside the existing quantization estimates.

10. **Validate through the real inference binary**

    Extend `sanity.py` to invoke `llama-tts` or `whisper-cli`, rather than merely
    inspecting metadata. This mirrors the recent GGUF load validation and
    catches missing companion assets or configuration.

11. **Add voice-specific model-card sections**

    Publish supported languages, sample rate, required files, the recommended
    runtime command, and quality results for each quant. Prominently include
    voice-cloning and consent limitations and the upstream model's license.

12. **Keep ASR integration separate from the llama.cpp build**

    Build and cache whisper.cpp independently, with its own architecture and
    model-format checks. It supports offline CPU and GPU inference and VAD,
    making it the natural second milestone. See the
    [whisper.cpp capabilities](https://github.com/ggml-org/whisper.cpp).

13. **Create deterministic audio fixtures for CI**

    Commit tiny, licensed WAV clips with expected transcript and audio-header
    assertions. Keep full quality benchmarks opt-in, but make conversion,
    bundle integrity, and smoke inference mandatory.

14. **Release in three milestones**

    - **v0.3.0:** TTS bundle handling, Qwen3-TTS, runtime validation, and cards.
    - **v0.3.1:** Pocket TTS and automated quality scoring.
    - **v0.4.0:** A separate Whisper and ASR backend.
