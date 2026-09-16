"""Offline orchestration tests for the llama.cpp and whisper.cpp tracks."""

import pytest

from agentquantix import voice
from agentquantix.pipeline import voice_release


def test_fixture_pack_covers_required_tts_cases():
    fixtures = voice_release.load_fixtures(voice.TTS)
    text = " ".join(item["text"] for item in fixtures)
    assert len(fixtures) >= 4
    assert any(item["language"] != "en" for item in fixtures)
    assert any(char.isdigit() for char in text)
    assert any(len(item["text"]) > 100 for item in fixtures)


def test_asr_fixture_has_expected_transcript_and_audio():
    fixture = voice_release.load_fixtures(voice.ASR)[0]
    assert fixture["transcript"].startswith("And so my fellow Americans")
    assert (voice_release.config.VOICE_FIXTURES_DIR / fixture["audio"]).is_file()


def test_tts_quantizer_rejects_lower_bit_sweep(tmp_path, monkeypatch):
    base = tmp_path / "model-BF16.gguf"
    base.write_bytes(b"base")
    monkeypatch.setattr(voice_release, "_run", lambda *args, **kwargs: None)
    with pytest.raises(voice.VoiceValidationError, match="voice-safe"):
        voice_release.quantize_tts(base, tmp_path / "quantize", "IQ2_XXS")


def test_tts_quantizer_uses_requested_conservative_quant(tmp_path, monkeypatch):
    base = tmp_path / "model-BF16.gguf"
    base.write_bytes(b"base")
    commands = []

    def run(command, _label):
        commands.append(command)
        command[2].write_bytes(b"quantized")

    monkeypatch.setattr(voice_release, "_run", run)
    output = voice_release.quantize_tts(base, tmp_path / "llama-quantize", "Q6_K")
    assert output.name == "model-Q6_K.gguf"
    assert commands[0][-1] == "Q6_K"


def test_qwen_conversion_produces_primary_and_companion(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir()
    runtime = llama_dir / "llama-tts"
    quantizer = llama_dir / "llama-quantize"
    commands = []

    monkeypatch.setattr(voice_release, "_converter_requirements", lambda: [])
    monkeypatch.setattr(
        voice_release, "ensure_llama_tts",
        lambda: (llama_dir, runtime, quantizer))
    monkeypatch.setattr(
        voice_release, "_snapshot",
        lambda repo_id, destination: (source, "source-revision"))
    monkeypatch.setattr(
        voice_release, "work_dir", lambda repo_id: tmp_path / "work")

    def fake_run(command, label):
        commands.append((command, label))
        output = voice_release.Path(command[command.index("--outfile") + 1])
        output.write_bytes(b"converted")

    monkeypatch.setattr(voice_release, "_run", fake_run)
    backend, primary, mmproj, revision, *_ = voice_release.convert_tts_source(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", voice_release.VoiceReleaseOptions())
    assert backend.family == "qwen3-tts"
    assert primary.is_file() and mmproj.is_file()
    assert revision == "source-revision"
    assert len(commands) == 2 and "--mmproj" in commands[1][0]


def test_whisper_quantizer_keeps_native_format(tmp_path, monkeypatch):
    base = tmp_path / "ggml-tiny.en.bin"
    base.write_bytes(b"base")

    def run(command, _label):
        command[2].write_bytes(b"quantized")

    monkeypatch.setattr(voice_release, "_run", run)
    output = voice_release.quantize_whisper(base, tmp_path / "quantize", "q5_0")
    assert output.name == "ggml-tiny.en-q5_0.bin"
    with pytest.raises(voice.VoiceValidationError, match="whisper.cpp quants"):
        voice_release.quantize_whisper(base, tmp_path / "quantize", "Q5_K_M")


def test_feasibility_reports_minutes_per_audio_hour(tmp_path):
    backend = voice.backend_for("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    primary, mmproj = tmp_path / "model.gguf", tmp_path / "mmproj.gguf"
    primary.write_bytes(b"x" * 100)
    mmproj.write_bytes(b"x" * 20)
    bundle = voice.VoiceBundle(
        backend, voice.BundleMember(primary, "primary"),
        (voice.BundleMember(mmproj, "mmproj"),), quant="Q8_0")
    result = voice_release.feasibility(
        bundle, {"real_time_factor": 0.5, "first_audio_seconds": 0.2,
                 "minutes_per_audio_hour": 30})
    assert result["minutes_per_audio_hour"] == 30
    assert result["bundle_disk_bytes"] == 120


class _Sibling:
    def __init__(self, name, size, digest):
        self.rfilename = name
        self.size = size
        self.lfs = {"sha256": digest}


class _Info:
    def __init__(self, siblings):
        self.siblings = siblings


class _Api:
    def __init__(self):
        self.files = {}
        self.commits = []

    def create_repo(self, **_kwargs):
        return None

    def model_info(self, *_args, **_kwargs):
        return _Info([_Sibling(name, value["bytes"], value["sha256"])
                      for name, value in self.files.items()])

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        for operation in kwargs["operations"]:
            path = getattr(operation, "path_or_fileobj")
            repo_path = getattr(operation, "path_in_repo")
            if repo_path in ("bundle.json", "quality.json", "README.md"):
                continue
            local = voice_release.Path(path)
            self.files[repo_path] = {
                "bytes": local.stat().st_size, "sha256": voice.sha256(local)}


def test_publication_is_one_commit_and_resume_skips_matching_members(
        tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    backend = voice.backend_for("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    primary, mmproj = tmp_path / "model.gguf", tmp_path / "mmproj.gguf"
    primary.write_bytes(b"primary")
    mmproj.write_bytes(b"mmproj")
    bundle = voice.VoiceBundle(
        backend, voice.BundleMember(primary, "primary"),
        (voice.BundleMember(mmproj, "mmproj"),), quant="Q8_0",
        quality={"passed": True})
    api = _Api()

    result = voice_release.publish_bundle(bundle, "owner/repo", api=api)
    assert result["verified"] and len(api.commits) == 1
    first_paths = {op.path_in_repo for op in api.commits[0]["operations"]}
    assert {"model.gguf", "mmproj.gguf", "bundle.json", "quality.json",
            "README.md"} <= first_paths

    voice_release.publish_bundle(bundle, "owner/repo", api=api)
    second_paths = {op.path_in_repo for op in api.commits[1]["operations"]}
    assert second_paths == {"bundle.json", "quality.json", "README.md"}
