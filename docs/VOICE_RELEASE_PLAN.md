# Voice release plan

## Direction

Support text-to-speech and automatic speech recognition through explicit
backend capabilities. llama.cpp exposes TTS flows for Qwen3-TTS and Pocket
TTS; audio.cpp contributes every TTS and ASR family declared by its pinned
upstream model-spec catalog; Whisper models use the separate whisper.cpp
artifacts and runtime semantics.

References: [llama.cpp TTS](https://github.com/ggml-org/llama.cpp/blob/master/tools/tts/README.md),
[audio.cpp](https://github.com/0xShug0/audio.cpp), and
[whisper.cpp](https://github.com/ggml-org/whisper.cpp).

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

4. **Treat each backend catalog as the support boundary**

   Qwen3-TTS and Pocket TTS use llama.cpp. For audio.cpp, ingest its complete
   `model_specs` directory and derive families, tasks, packages, source tensor
   mappings, languages, speaker requirements, and precisions from those files.
   Never add a per-model AgentQuantix branch for an already supported backend;
   upgrading and syncing the backend catalog exposes new families.

5. **Build an audio-aware calibration pipeline**

   Replace wiki-text-only calibration with a small, licensed multilingual
   speech and prompt fixture set. Cover short and long utterances, punctuation,
   numbers, non-English text, and speaker-conditioning cases.

6. **Use quality gates, not just GGUF load checks**

   For TTS, generate fixed prompts and reject silence, clipping, invalid WAV
   headers, or wildly incorrect duration. For ASR, run a fixed audio corpus and
   set a maximum WER regression relative to BF16 for whitespace-delimited
   languages and a CER regression for Chinese/Japanese.

7. **Use a two-stage TTS quality assessment**

   Automate intelligibility checks with ASR round-trip word error rate, audio
   duration, and silence checks. Require a brief human listening review only
   for candidate quant types that pass automation.

8. **Expose every quant supported by the selected runtime**

   Derive quant availability from the installed converter or quantizer rather
   than maintaining an AgentQuantix shortlist. Distinguish direct source
   conversion from prebuilt packages: sources get every converter output type,
   while a package reference gets only the precisions that actually exist.
   Build a multilingual TTS importance matrix for llama.cpp low-bit types and
   let the same intelligibility and speaker-similarity gates decide which
   candidates are publishable.

9. **Model feasibility in seconds of audio, not only parameters**

   Estimate real-time factor, tokens or frames per second, first-audio latency,
   VRAM or RAM, and bundle disk size. Report minutes required to synthesize one
   hour of audio alongside the existing quantization estimates.

10. **Validate through the real inference binary**

    Extend `sanity.py` to invoke `llama-tts`, `audiocpp_cli`, or `whisper-cli`,
    rather than merely inspecting metadata. This catches missing companion
    assets, embedded package specifications, or runtime configuration.

11. **Add voice-specific model-card sections**

    Publish supported languages, sample rate, required files, the recommended
    runtime command, and quality results for each quant. Prominently include
    voice-cloning and consent limitations and the upstream model's license.

12. **Keep each ASR runtime separate from the llama.cpp build**

    Build and cache whisper.cpp independently, with its own architecture and
    model-format checks. Route audio.cpp ASR families through the same catalog
    mechanism as audio.cpp TTS, but with ASR fixtures and WER/CER gates. Whisper
    supports offline CPU and GPU inference and VAD. See the
    [whisper.cpp capabilities](https://github.com/ggml-org/whisper.cpp).

13. **Create deterministic audio fixtures for CI**

    Commit tiny, licensed WAV clips with expected transcript and audio-header
    assertions. Keep full quality benchmarks opt-in, but make conversion,
    bundle integrity, and smoke inference mandatory.

14. **Release in three milestones**

    - **v0.3.0:** TTS bundle handling, Qwen3-TTS, runtime validation, and cards.
    - **v0.3.1:** Pocket TTS and automated quality scoring.
    - **v0.3.2:** catalog-driven audio.cpp backend for all declared TTS/ASR
      families, with generic conversion and runtime validation.
    - **v0.4.0:** A separate Whisper and ASR backend.

15. **Hunt backend forks for unresolved voice architectures**

    Search publisher-owned forks and open pull requests across llama.cpp,
    audio.cpp, whisper.cpp, and NeMo-Speech.cpp, using the same cached evidence
    model as the text pipeline. Report leads in a structured blocked plan; do
    not treat a matching branch name as executable until its converter, model
    spec, bundle contract, and smoke inference have been verified.
