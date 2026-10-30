"""Offline regression tests for GGUF-only audio.cpp source quants."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from agentquantix import actions, cli, runtime_control, voice
from agentquantix.pipeline import voice_release


SOURCE_REPO = "BreezeBlue/Breeze-TTS-2"


@pytest.fixture
def breeze_catalog(monkeypatch):
    breeze = {
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
    convertible = {
        "family": "convertible_tts",
        "display_name": "Convertible TTS",
        "category": "tts",
        "status": "supported",
        "tasks": ["tts"],
        "sources": [{"format": "safetensors"}],
        "packages": [
            {"id": "convertible_tts_q8_0", "format": "gguf",
             "precision": "q8_0",
             "download": {"repo": "audio-cpp/convertible-gguf"}},
        ],
    }
    specs = (breeze, convertible)
    backends = {
        spec["family"]: voice._audio_cpp_backends([spec])[0]
        for spec in specs
    }
    catalog = {
        "registry": {spec["family"]: spec for spec in specs},
        "package_repos": {
            spec["family"]: voice._audio_cpp_package_repos(spec)
            for spec in specs
        },
    }
    monkeypatch.setattr(voice, "audio_cpp_catalog", lambda: catalog)
    monkeypatch.setattr(
        voice, "execution_gate",
        lambda _repo_id, *, family=None, **_kwargs:
            (True, "", backends[family or "breeze_tts"]))
    monkeypatch.setattr(actions, "_audio_cpp_catalog_blocker", lambda *_args: None)
    return SimpleNamespace(backends=backends)


def test_breeze_gguf_only_source_plan_is_blocked_before_metadata(
        breeze_catalog, monkeypatch):
    backend = breeze_catalog.backends["breeze_tts"]
    monkeypatch.setattr(voice_release, "source_metadata",
                        lambda *_args, **_kwargs:
                            pytest.fail("source metadata was fetched"))
    monkeypatch.setattr(
        voice_release, "prepare_audiocpp_source",
        lambda *_args, **_kwargs:
            pytest.fail("audio.cpp source was prepared"))

    source_quants = voice.available_quants(SOURCE_REPO, backend)
    plan = actions.voice_plan(
        SOURCE_REPO, family=backend.family, target_repo="test/Breeze-GGUF",
        hunt_forks=False)

    assert source_quants == ("Q8_0", "BF16")
    assert plan["status"] == "blocked"
    assert plan["task"] == "text-to-speech"
    assert plan["blocker"] == voice._audio_cpp_source_blocker("breeze_tts")
    assert plan["next"] == (
        "aqx run audio.cpp:breeze_tts --task text-to-speech --no-publish")

    explicit = actions.voice_plan(
        SOURCE_REPO, family=backend.family, quants=["q4_0"],
        target_repo="test/Breeze-GGUF", hunt_forks=False)
    assert explicit["status"] == "blocked"
    assert explicit["next"] == plan["next"]


def test_audio_cpp_package_reference_remains_ready(
        breeze_catalog, monkeypatch):
    backend = breeze_catalog.backends["breeze_tts"]
    monkeypatch.setattr(
        voice_release, "source_metadata",
        lambda *_args, **_kwargs: {
            "source_files": [], "revision": "package-revision",
            "source_bytes": None, "gated": False})
    monkeypatch.setattr(voice_release, "audiocpp_source_plan",
                        lambda *_args, **_kwargs: [])

    plan = actions.voice_plan(
        "audio.cpp:breeze_tts", family=backend.family,
        target_repo="test/Breeze-GGUF", hunt_forks=False)

    assert plan["status"] == "ready"
    assert voice.available_quants("breeze_tts_2_q4_0", backend) == ("Q4_0",)
    assert voice.available_quants("audio.cpp:breeze_tts", backend) == (
        "Q4_0", "Q8_0", "BF16")


def test_safetensors_convertible_audio_cpp_family_remains_ready(
        breeze_catalog, monkeypatch):
    backend = breeze_catalog.backends["convertible_tts"]
    monkeypatch.setattr(
        voice_release, "source_metadata",
        lambda *_args, **_kwargs: {
            "source_files": ["model.safetensors"],
            "revision": "convertible-revision",
            "source_bytes": 1024, "gated": False})
    monkeypatch.setattr(voice_release, "audiocpp_source_plan",
                        lambda *_args, **_kwargs: [])

    plan = actions.voice_plan(
        "org/convertible-tts", family=backend.family,
        target_repo="test/Convertible-GGUF", hunt_forks=False)

    assert plan["status"] == "ready"
    assert plan["family"] == "convertible_tts"


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
