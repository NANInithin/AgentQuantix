"""The voice boundary: planned families cannot enter the v0.3.0 runtime."""

import subprocess
import wave

import pytest

from agentquantix import voice


def _wav(path, frames=32):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"\0\0" * frames)


def test_only_qwen3_tts_is_enabled_for_v030():
    allowed, _, family = voice.execution_gate("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    assert allowed and family.name == "qwen3-tts"

    for repo in ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "kyutai/Pocket-TTS",
                 "mistralai/Voxtral-Mini-3B-2507", "Qwen/Qwen3-ASR-1.7B"):
        allowed, reason, _ = voice.execution_gate(repo)
        assert not allowed
        assert "not runnable" in reason or "not an approved" in reason


def test_advisor_shows_voice_without_claiming_agent_run_support():
    catalog = voice.advisory_catalog()
    by_family = {entry["family"]: entry for entry in catalog["candidates"]}

    assert by_family["qwen3-tts"]["status"] == "preview"
    assert by_family["qwen3-tts"]["supported_quants"] == ["f16", "q8_0", "q4_k"]
    assert by_family["qwen3-tts"]["agent_run_available"] is False
    assert by_family["pocket-tts"]["status"] == "planned"


def test_unknown_voice_models_are_not_an_execution_fallback():
    allowed, reason, family = voice.execution_gate("org/Interesting-TTS")
    assert not allowed and family is None
    assert "converter" in reason


def test_bundle_manifest_requires_every_required_member(tmp_path):
    primary = tmp_path / "model.gguf"
    primary.write_bytes(b"weights")
    missing = tmp_path / "codec.gguf"
    bundle = voice.VoiceBundle(
        voice.family_for("Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        voice.BundleMember(primary, "language-model"),
        (voice.BundleMember(missing, "audio-codec"),),
    )
    assert "audio-codec" in bundle.problems()[0]
    with pytest.raises(voice.VoiceValidationError, match="incomplete"):
        bundle.manifest()


def test_bundle_manifest_is_complete_and_describes_roles(tmp_path):
    primary, codec = tmp_path / "model.gguf", tmp_path / "codec.gguf"
    primary.write_bytes(b"weights")
    codec.write_bytes(b"codec")
    bundle = voice.VoiceBundle(
        voice.family_for("Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        voice.BundleMember(primary, "language-model"),
        (voice.BundleMember(codec, "audio-codec", hub_path="codec/codec.gguf"),),
        source_repo="Qwen/Qwen3-TTS-12Hz-0.6B-Base", source_revision="abc123",
    )
    manifest = bundle.manifest()
    assert manifest["family"] == "qwen3-tts"
    assert [member["role"] for member in manifest["members"]] == [
        "language-model", "audio-codec"]
    assert manifest["members"][1]["hub_path"] == "codec/codec.gguf"
    assert len(manifest["members"][0]["sha256"]) == 64


def test_manifest_persists_complete_bundle_and_rejects_corruption(tmp_path):
    primary = tmp_path / "model.gguf"
    primary.write_bytes(b"weights")
    bundle = voice.VoiceBundle(
        voice.family_for("Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        voice.BundleMember(primary, "language-model"),
    )
    manifest_path = voice.write_manifest(bundle, tmp_path / "bundle.json")
    assert voice.load_manifest(manifest_path)["members"][0]["path"] == "model.gguf"
    manifest_path.write_text("not json", encoding="utf-8")
    with pytest.raises(voice.VoiceValidationError, match="could not read"):
        voice.load_manifest(manifest_path)


def test_wav_validation_rejects_empty_and_accepts_audio(tmp_path):
    empty = tmp_path / "empty.wav"
    _wav(empty, frames=0)
    with pytest.raises(voice.VoiceValidationError, match="empty"):
        voice.validate_wav(empty)

    good = tmp_path / "good.wav"
    _wav(good)
    facts = voice.validate_wav(good)
    assert facts["sample_rate"] == 24_000 and facts["frames"] == 32


def test_qwen_command_uses_the_audited_external_runtime_path(tmp_path):
    command = voice.qwen3_tts_smoke_command(
        "qwen3-tts-cli", tmp_path, tmp_path / "out.wav")
    assert command[:3] == ["qwen3-tts-cli", "-m", str(tmp_path)]
    assert "-o" in command


def test_qwen_runtime_environment_adds_the_vendored_windows_ggml_dll_dir(
        tmp_path, monkeypatch):
    runtime = tmp_path / "build" / "Release" / "qwen3-tts-cli.exe"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"runtime")
    ggml = tmp_path / "ggml" / "build" / "bin" / "Release"
    ggml.mkdir(parents=True)
    monkeypatch.setattr(voice.os, "name", "nt")
    monkeypatch.setenv("PATH", "original-path")

    environment = voice.qwen3_tts_runtime_environment(runtime)

    assert str(ggml) in environment["PATH"].split(voice.os.pathsep)
    assert environment["PATH"].endswith("original-path")


def test_smoke_requires_a_successful_runtime_and_valid_output(tmp_path, monkeypatch):
    output = tmp_path / "out.wav"
    def success(command, **_kwargs):
        _wav(output)
        return subprocess.CompletedProcess(command, 0, "ok")

    monkeypatch.setattr(voice.subprocess, "run", success)
    facts = voice.run_qwen3_tts_smoke(
        "qwen3-tts-cli", tmp_path, output)
    assert facts["seconds"] > 0

    failed_output = tmp_path / "failed.wav"
    monkeypatch.setattr(voice.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 1, "fatal failure"))
    with pytest.raises(voice.VoiceValidationError, match="fatal failure"):
        voice.run_qwen3_tts_smoke(
            "qwen3-tts-cli", tmp_path, failed_output)


def test_smoke_refuses_to_reuse_an_old_output(tmp_path):
    output = tmp_path / "old.wav"
    _wav(output)
    with pytest.raises(voice.VoiceValidationError, match="already exists"):
        voice.run_qwen3_tts_smoke(
            "qwen3-tts-cli", tmp_path, output)
