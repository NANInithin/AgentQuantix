"""Audit boundaries for the pinned external Qwen3-TTS backend."""

import pytest

from agentquantix import voice
from agentquantix.pipeline import qwen3_tts


def _source(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"talker")
    tokenizer = tmp_path / "speech_tokenizer"
    tokenizer.mkdir()
    (tokenizer / "config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "model.safetensors").write_bytes(b"tokenizer")
    return tmp_path


def test_audit_is_pinned_and_records_its_runtime_boundary():
    audit = qwen3_tts.backend_audit()
    assert len(audit["revision"]) == 40
    assert audit["license"] == "MIT"
    assert audit["runtime"] == "qwen3-tts-cli"
    assert "stock llama.cpp" in audit["limitations"][-1]


def test_source_shape_rejects_a_missing_tokenizer(tmp_path):
    source = _source(tmp_path)
    for path in (source / "speech_tokenizer").iterdir():
        path.unlink()
    (source / "speech_tokenizer").rmdir()
    assert "speech_tokenizer" in " ".join(qwen3_tts.source_problems(source))


def test_conversion_plan_requires_the_audited_base_variant(tmp_path):
    source = _source(tmp_path / "source")
    plan = qwen3_tts.conversion_plan(
        qwen3_tts.SUPPORTED_SOURCE, source, tmp_path / "output", "q4_k")
    assert plan["talker"].name == "qwen3-tts-0.6b-q4_k.gguf"
    assert plan["tokenizer"].name == "qwen3-tts-tokenizer-f16.gguf"
    assert "convert_tts_to_gguf.py" in str(plan["commands"][0][1])
    assert "convert_tokenizer_to_gguf.py" in str(plan["commands"][1][1])

    with pytest.raises(voice.VoiceValidationError, match="only"):
        qwen3_tts.conversion_plan(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base", source, tmp_path / "output")


def test_conversion_plan_rejects_unsupported_talker_quants(tmp_path):
    source = _source(tmp_path / "source")
    with pytest.raises(voice.VoiceValidationError, match="only f16"):
        qwen3_tts.conversion_plan(
            qwen3_tts.SUPPORTED_SOURCE, source, tmp_path / "output", "iq4_xs")


def test_converter_summary_rejects_unmapped_or_unrecognised_output():
    assert qwen3_tts._converter_summary("Converted 200 tensors, skipped 0") == (200, 0)
    with pytest.raises(voice.VoiceValidationError, match="skipped 2"):
        qwen3_tts._converter_summary("Converted 198 tensors, skipped 2")
    with pytest.raises(voice.VoiceValidationError, match="did not print"):
        qwen3_tts._converter_summary("conversion completed")


def test_backend_directory_is_revision_addressed(monkeypatch, tmp_path):
    monkeypatch.setattr(qwen3_tts.config, "TEMP_DIR", tmp_path)
    assert qwen3_tts.BACKEND_REVISION[:12] in qwen3_tts.backend_dir().name
