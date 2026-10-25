"""A selected audio.cpp revision controls planning without global mutations."""
import json
from types import SimpleNamespace

import pytest

from agentquantix import voice
from agentquantix.runtimes import RuntimeSession


def _handle(tmp_path, family, *, package_quant="Q8_0", content=None):
    source = tmp_path / family
    specs = source / "model_specs"
    specs.mkdir(parents=True)
    spec = {"family": family, "display_name": family, "tasks": ["tts"],
            "category": "tts", "status": "testing", "languages": ["en"],
            "packages": [{"id": family + "-source", "format": "safetensors",
                          "download": {"repo": f"publisher/{family}"}},
                         {"id": family + "-package", "format": "gguf",
                          "precision": package_quant, "download": {"repo": f"publisher/{family}-GGUF"}}],
            "sources": [{"format": "safetensors"}]}
    path = specs / f"{family}.json"
    path.write_text(json.dumps(spec) if content is None else content, encoding="utf-8")
    return SimpleNamespace(source=source, revision=("a" if family == "first" else "b") * 40,
                           identity=family,
                           manifest={"resources": {f"model_specs/{family}.json":
                               {"path": str(path), "sha256": voice.sha256(path)}}})


def test_two_selected_audio_catalogs_are_isolated_and_restore_offline_baseline(tmp_path):
    first, second = _handle(tmp_path, "first"), _handle(tmp_path, "second", package_quant="Q4_0")
    baseline = voice.audio_cpp_catalog()
    legacy_registry = dict(voice.AUDIOCPP_SPEC_REGISTRY)
    with RuntimeSession(manager=object()) as session:
        session.handles["audio.cpp"] = first
        backend = voice.backend_for("publisher/first")
        assert backend.family == "first"
        assert voice.backend_named(backend.id) == backend
        assert voice.available_quants("first-package", backend) == ("Q8_0",)
        assert voice.available_quants("publisher/first", backend) == voice.AUDIOCPP_QUANTS
        assert voice.advisory_catalog()["audio_cpp_catalog"]["revision_source"] == first.revision
        manifest = tmp_path / "bundle.json"
        manifest.write_text(json.dumps({"format": 2, "backend": backend.id,
                                        "members": [{"role": "primary"}]}), encoding="utf-8")
        assert voice.load_manifest(manifest)["backend"] == backend.id
        voice.audio_cpp_catalog()["registry"]["first"]["family"] = "mutated-plan"
        assert voice.audio_cpp_catalog()["registry"]["first"]["family"] == "first"
        with RuntimeSession(manager=object()) as nested:
            nested.handles["audio.cpp"] = second
            selected = voice.backend_for("publisher/second")
            assert selected.family == "second"
            assert voice.available_quants("second-package", selected) == ("Q4_0",)
            with pytest.raises(voice.VoiceValidationError, match="unknown voice backend"):
                voice.backend_named(backend.id)
        assert voice.backend_named(backend.id) == backend
    assert voice.audio_cpp_catalog()["specs"] == baseline["specs"]
    assert voice.AUDIOCPP_SPEC_REGISTRY == legacy_registry
    assert voice.backend_for("openbmb/VoxCPM2").backend == "audio.cpp"


def test_selected_audio_catalog_cannot_fall_back_after_missing_or_tampered_specs(tmp_path):
    handle = _handle(tmp_path, "first")
    with RuntimeSession(manager=object()) as session:
        session.handles["audio.cpp"] = handle
        assert voice.audio_cpp_catalog()["registry"]["first"]
        path = handle.source / "model_specs/first.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(voice.VoiceValidationError, match="changed"):
            voice.backend_for("publisher/first")
        path.unlink()
        with pytest.raises(voice.VoiceValidationError, match="no model_specs"):
            voice.advisory_catalog()


@pytest.mark.parametrize("content", ["not JSON", '{"family": "first", "packages": "bad"}'])
def test_selected_audio_specs_must_be_parseable_contracts(tmp_path, content):
    handle = _handle(tmp_path, "first", content=content)
    with RuntimeSession(manager=object()) as session:
        session.handles["audio.cpp"] = handle
        with pytest.raises(voice.VoiceValidationError, match="model spec"):
            voice.audio_cpp_catalog()


def test_unrecorded_spec_cannot_extend_frozen_audio_catalog(tmp_path):
    handle = _handle(tmp_path, "first")
    path = handle.source / "model_specs/unrecorded.json"
    path.write_text('{"family":"unrecorded"}', encoding="utf-8")
    with RuntimeSession(manager=object()) as session:
        session.handles["audio.cpp"] = handle
        with pytest.raises(voice.VoiceValidationError, match="frozen runtime resources"):
            voice.audio_cpp_catalog()
