"""Offline regression tests for GGUF-only audio.cpp source quants."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from agentquantix import actions, cli, runtime_control, voice
from agentquantix.pipeline import voice_release


SOURCE_REPO = "BreezeBlue/Breeze-TTS-2"


@pytest.fixture
def breeze_catalog(monkeypatch):
    spec = {
        "family": "breeze_tts",
        "display_name": "Breeze TTS",
        "category": "tts",
        "status": "supported",
        "tasks": ["tts"],
        "sources": [{"format": "gguf", "repo": SOURCE_REPO}],
        "packages": [
            {"id": "breeze_tts_2_q4_0", "format": "gguf",
             "precision": "q4_0",
             "download": {"repo": "audio-cpp/audio.cpp-gguf"}},
            {"id": "breeze_tts_2_q8_0", "format": "gguf",
             "precision": "q8_0",
             "download": {"repo": "audio-cpp/audio.cpp-gguf"}},
            {"id": "breeze_tts_2_bf16", "format": "gguf",
             "precision": "bf16",
             "download": {"repo": "audio-cpp/audio.cpp-gguf"}},
        ],
    }
    backend = voice._audio_cpp_backends([spec])[0]
    catalog = {
        "registry": {spec["family"]: spec},
        "package_repos": {spec["family"]: ("audio-cpp/audio.cpp-gguf",)},
    }
    monkeypatch.setattr(voice, "audio_cpp_catalog", lambda: catalog)
    monkeypatch.setattr(voice, "execution_gate",
                        lambda *_args, **_kwargs: (True, "", backend))
    monkeypatch.setattr(actions, "_audio_cpp_catalog_blocker", lambda *_args: None)
    return backend


def test_breeze_source_plan_excludes_package_only_quants(
        breeze_catalog, monkeypatch):
    backend = breeze_catalog
    monkeypatch.setattr(
        voice_release, "source_metadata",
        lambda *_args, **_kwargs: {
            "source_files": [], "revision": "source-revision",
            "source_bytes": 1024, "gated": False})
    monkeypatch.setattr(voice_release, "audiocpp_source_plan",
                        lambda *_args, **_kwargs: [])

    source_quants = voice.available_quants(SOURCE_REPO, backend)
    plan = actions.voice_plan(
        SOURCE_REPO, family=backend.family, target_repo="test/Breeze-GGUF",
        hunt_forks=False)

    assert source_quants == ("Q8_0", "BF16")
    assert plan["available_quants"] == ["Q8_0", "BF16"]
    assert plan["quants"] == ["Q8_0", "BF16"]
    assert voice.available_quants("breeze_tts_2_q4_0", backend) == ("Q4_0",)
    assert voice.available_quants("audio.cpp:breeze_tts", backend) == (
        "Q4_0", "Q8_0", "BF16")


def test_explicit_unsupported_source_quant_fails_before_download_or_preparation(
        breeze_catalog, monkeypatch):
    monkeypatch.setattr(
        voice_release, "source_metadata",
        lambda *_args, **_kwargs: pytest.fail("source metadata was fetched"))
    monkeypatch.setattr(
        voice_release, "prepare_audiocpp_source",
        lambda *_args, **_kwargs: pytest.fail("audio.cpp source was prepared"))

    with pytest.raises(ValueError, match="does not support.*Q4_0"):
        actions.voice_plan(
            SOURCE_REPO, family=breeze_catalog.family, quants=["q4_0"],
            target_repo="test/Breeze-GGUF", hunt_forks=False)


def test_cli_reports_voice_validation_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(cli.config, "migrate_legacy_state", lambda: [])
    monkeypatch.setattr(cli, "_ACTIVE_RESUME_IDS", [])
    monkeypatch.setattr(
        runtime_control, "execution_session",
        lambda **_kwargs: nullcontext(SimpleNamespace(mark_failed=lambda: None)))

    def fail(_args):
        cli._ACTIVE_RESUME_IDS.append("voice:breeze-tts")
        raise voice.VoiceValidationError("unsupported audio.cpp quant Q4_0")

    parser = SimpleNamespace(parse_args=lambda _argv: SimpleNamespace(func=fail))
    monkeypatch.setattr(cli, "build_parser", lambda: parser)

    assert cli.main([]) == 1
    output = capsys.readouterr().out
    assert "aqx: unsupported audio.cpp quant Q4_0" in output
    assert "Resume with: `aqx run --resume voice:breeze-tts`" in output
    assert "Traceback" not in output
