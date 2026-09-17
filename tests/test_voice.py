"""Voice registry, bundle, audio, quality, and card contracts."""

import json
import math
import wave

import pytest

from agentquantix import voice


def _wav(path, *, seconds=0.25, rate=16_000, amplitude=0.25):
    frames = int(seconds * rate)
    samples = bytearray()
    for index in range(frames):
        value = int(amplitude * 32767 * math.sin(2 * math.pi * 440 * index / rate))
        samples.extend(value.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(bytes(samples))


def _bundle(tmp_path, family="qwen3-tts", quant="Q4_K_M"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    backend = next(item for item in voice.BACKENDS if item.family == family)
    primary = tmp_path / f"model-{quant}.gguf"
    primary.write_bytes(b"primary")
    companion = tmp_path / "mmproj-Q8_0.gguf"
    companion.write_bytes(b"companion")
    return voice.VoiceBundle(
        backend=backend,
        primary=voice.BundleMember(primary, "primary", quant=quant),
        companions=(voice.BundleMember(companion, "mmproj", quant="Q8_0"),),
        source_repo="org/model", source_revision="abc", quant=quant)


def test_registry_splits_tts_and_asr_and_uses_native_quants():
    qwen = voice.backend_for("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    pocket = voice.backend_for("kyutai/pocket-tts")
    whisper = voice.backend_for("openai/whisper-small")

    assert qwen.track == pocket.track == voice.TTS
    assert qwen.runtime == pocket.runtime == "llama-tts"
    assert qwen.supported_quants == voice.TTS_QUANTS
    assert whisper.track == voice.ASR and whisper.runtime == "whisper-cli"
    assert whisper.model_format == "whisper-ggml-bin"
    assert whisper.supported_quants == voice.WHISPER_QUANTS
    assert voice.backend_for("ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF") is None
    assert voice.backend_for("ggerganov/whisper.cpp") is None
    assert voice.backend_for("Qwen/Qwen3-TTS-1.7B") is None
    assert voice.advisory_catalog()["candidates"][0]["repo_id"] == (
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base")


def test_unknown_family_is_not_allowed_to_fall_through():
    allowed, reason, backend = voice.execution_gate("org/unknown-speech")
    assert not allowed and backend is None
    assert "converter" in reason and "quality gate" in reason

    allowed, reason, backend = voice.execution_gate("Qwen/Qwen3-TTS-1.7B")
    assert not allowed and backend is None
    assert "Qwen/Qwen3-TTS-12Hz-1.7B-Base" in reason


def test_bundle_requires_companions_and_is_stable(tmp_path):
    bundle = _bundle(tmp_path)
    first = bundle.manifest()
    second = bundle.manifest()
    assert first == second
    assert first["format"] == 2
    assert [member["role"] for member in first["members"]] == ["primary", "mmproj"]
    assert len(first["members"][0]["sha256"]) == 64

    broken = voice.VoiceBundle(
        backend=bundle.backend, primary=bundle.primary, quant="Q4_K_M")
    with pytest.raises(voice.VoiceValidationError, match="mmproj"):
        broken.manifest()


def test_manifest_roundtrip_and_local_verification(tmp_path):
    bundle = _bundle(tmp_path)
    path = voice.write_manifest(bundle, tmp_path / "bundle.json")
    manifest = voice.load_manifest(path)
    assert voice.verify_manifest_files(manifest, tmp_path) == []
    bundle.primary.path.write_bytes(b"changed")
    assert "mismatch" in voice.verify_manifest_files(manifest, tmp_path)[0]


def test_remote_verification_checks_size_and_checksum(tmp_path):
    manifest = _bundle(tmp_path).manifest()
    remote = {member["path"]: {"bytes": member["bytes"],
                                "sha256": member["sha256"]}
              for member in manifest["members"]}
    assert voice.remote_bundle_problems(manifest, remote) == []
    remote.pop(manifest["members"][0]["path"])
    assert "missing" in voice.remote_bundle_problems(manifest, remote)[0]


def test_tts_commands_apply_family_requirements(tmp_path):
    qwen = _bundle(tmp_path / "qwen")
    command = voice.llama_tts_command(
        "llama-tts", qwen, "Hello", tmp_path / "out.wav", language="en")
    assert command[:3] == ["llama-tts", "-m", str(qwen.primary.path)]
    assert "-mm" in command and "--tts-lang" in command

    pocket_path = tmp_path / "pocket"
    pocket_path.mkdir()
    pocket = _bundle(pocket_path, family="pocket-tts")
    with pytest.raises(voice.VoiceValidationError, match="speaker"):
        voice.llama_tts_command("llama-tts", pocket, "Hi", tmp_path / "p.wav")


def test_whisper_command_uses_separate_cli_and_output_file(tmp_path):
    command = voice.whisper_command(
        "whisper-cli", "model.bin", "input.wav", tmp_path / "transcript",
        language="en", vad=True)
    assert command[:5] == ["whisper-cli", "-m", "model.bin", "-f", "input.wav"]
    assert "--output-txt" in command and "--vad" in command


def test_tts_smoke_runs_inference_and_enforces_audio_contract(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "bundle")
    output = tmp_path / "generated.wav"

    def fake_run(command, *, output, timeout):
        _wav(output, seconds=1, rate=24_000)
        return {"command": command, "elapsed_seconds": 0.5,
                "first_output_seconds": 0.1, "stdout": ""}

    monkeypatch.setattr(voice, "run_checked", fake_run)
    result = voice.run_tts_smoke(
        "llama-tts", bundle, "Hello from the smoke test.", output,
        language="en")
    assert result["sample_rate"] == 24_000
    assert result["real_time_factor"] == pytest.approx(0.5)
    assert result["minutes_per_audio_hour"] == pytest.approx(30)


def test_asr_smoke_runs_inference_and_scores_transcript(tmp_path, monkeypatch):
    audio = tmp_path / "input.wav"
    _wav(audio, seconds=1, rate=16_000)
    prefix = tmp_path / "transcript"

    def fake_run(command, *, timeout):
        (tmp_path / "transcript.txt").write_text(
            "hello deterministic audio\n", encoding="utf-8")
        return {"command": command, "elapsed_seconds": 0.25,
                "first_output_seconds": 0.25, "stdout": ""}

    monkeypatch.setattr(voice, "run_checked", fake_run)
    result = voice.run_asr_smoke(
        "whisper-cli", "model.bin", audio, "hello deterministic audio",
        prefix, language="en")
    assert result["wer"] == 0
    assert result["audio_seconds_per_second"] == pytest.approx(4)


def test_audio_gate_accepts_signal_and_rejects_silence_and_clipping(tmp_path):
    good = tmp_path / "good.wav"
    _wav(good)
    facts = voice.validate_tts_audio(good)
    assert facts["sample_rate"] == 16_000
    assert facts["silence_ratio"] < 0.1

    silence = tmp_path / "silence.wav"
    _wav(silence, amplitude=0)
    with pytest.raises(voice.VoiceValidationError, match="silent"):
        voice.validate_tts_audio(silence)

    clipped = tmp_path / "clipped.wav"
    with wave.open(str(clipped), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes((32767).to_bytes(2, "little", signed=True) * 4000)
    with pytest.raises(voice.VoiceValidationError, match="clipped"):
        voice.validate_tts_audio(clipped)


def test_tts_audio_is_resampled_to_whisper_contract(tmp_path):
    source, destination = tmp_path / "tts.wav", tmp_path / "asr.wav"
    _wav(source, seconds=1, rate=24_000)
    voice.resample_pcm16_wav(source, destination)
    facts = voice.inspect_wav(destination)
    assert facts["sample_rate"] == 16_000
    assert facts["sample_width"] == 2
    assert facts["channels"] == 1
    assert facts["seconds"] == pytest.approx(1, abs=0.001)


def test_wer_and_regression_gate_are_deterministic():
    assert voice.word_error_rate("Hello, world!", "hello world") == 0
    assert voice.word_error_rate("one two three", "one four three") == pytest.approx(1 / 3)
    assert voice.asr_regression(0.14, 0.10, 0.05)["passed"]
    assert not voice.asr_regression(0.16, 0.10, 0.05)["passed"]


def test_human_review_is_explicit_and_machine_readable(tmp_path):
    path = voice.write_human_review(
        tmp_path / "review.json", quant="Q4_K_M", accepted=True,
        reviewer="Nithin", notes="Clear and correctly paced")
    review = json.loads(path.read_text(encoding="utf-8"))
    assert review["accepted"] is True and review["quant"] == "Q4_K_M"
    with pytest.raises(voice.VoiceValidationError, match="reviewer"):
        voice.write_human_review(tmp_path / "bad.json", quant="Q8_0",
                                 accepted=True, reviewer="")


def test_voice_card_contains_runtime_files_quality_and_consent(tmp_path):
    bundle = _bundle(tmp_path)
    bundle = voice.VoiceBundle(**{
        **bundle.__dict__,
        "quality": {"roundtrip_wer": 0.1, "minutes_per_audio_hour": 25.0},
    })
    card = voice.render_model_card(bundle, "owner/model-GGUF", "apache-2.0")
    assert "llama-tts" in card
    assert "mmproj-Q8_0.gguf" in card
    assert "roundtrip wer" in card
    assert "explicit, lawful consent" in card
    assert "24000 Hz" in card


def test_committed_asr_fixture_is_valid_16khz_pcm():
    from agentquantix import config
    fixture = config.VOICE_FIXTURES_DIR / "jfk.wav"
    facts = voice.inspect_wav(fixture)
    assert facts["sample_rate"] == 16_000
    assert facts["sample_width"] == 2
    assert facts["seconds"] > 5
