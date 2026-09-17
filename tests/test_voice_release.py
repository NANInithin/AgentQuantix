"""Offline orchestration tests for the llama.cpp and whisper.cpp tracks."""

import math
import wave

import pytest

from agentquantix import voice
from agentquantix.pipeline import voice_release


def _wav(path, *, seconds=0.5, rate=16_000, amplitude=0.25):
    frames = int(seconds * rate)
    samples = bytearray()
    for index in range(frames):
        value = int(amplitude * 32767 * math.sin(
            2 * math.pi * 440 * index / rate))
        samples.extend(value.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(bytes(samples))


def test_fixture_pack_covers_required_tts_cases():
    fixtures = voice_release.load_fixtures(voice.TTS)
    text = " ".join(item["text"] for item in fixtures)
    assert len(fixtures) >= 4
    assert any(item["language"] != "en" for item in fixtures)
    assert any(char.isdigit() for char in text)
    assert any(len(item["text"]) > 100 for item in fixtures)
    by_id = {item["id"]: item["text"] for item in fixtures}
    assert by_id["short-ja"] == "こんにちは、これは音声合成のテストです。"
    assert by_id["short-zh"] == "你好，这是一个语音合成测试。"


def test_asr_fixture_has_expected_transcript_and_audio():
    fixture = voice_release.load_fixtures(voice.ASR)[0]
    assert fixture["transcript"].startswith("And so my fellow Americans")
    assert (voice_release.config.VOICE_FIXTURES_DIR / fixture["audio"]).is_file()


def test_recorded_reviews_are_auto_discovered(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_release.config, "TEMP_DIR", tmp_path)
    options = voice_release.VoiceReleaseOptions()
    assert voice_release._with_recorded_reviews("org/model", options) is options

    reviews = voice_release.work_dir("org/model") / "reviews"
    voice.write_human_review(
        reviews / "Q4_K.json", quant="Q4_K", accepted=True,
        reviewer="Nithin")
    discovered = voice_release._with_recorded_reviews("org/model", options)
    assert discovered.human_review == reviews


def test_reviewed_legacy_tts_artifacts_resume_without_inference(
        tmp_path, monkeypatch):
    backend = voice.backend_for("openbmb/VoxCPM2")
    models = tmp_path / "models"
    models.mkdir()
    primary = models / "voxcpm2-Q4_K.gguf"
    primary.write_bytes(b"model")
    bundle = voice.VoiceBundle(
        backend=backend,
        primary=voice.BundleMember(primary, "primary", quant="Q4_K"),
        source_repo="openbmb/VoxCPM2", source_revision="abc", quant="Q4_K")
    quality = models / "quality" / "Q4_K"
    quality.mkdir(parents=True)
    _wav(quality / "00-short-en.wav", rate=24_000)
    _wav(quality / "00-short-en-16khz.wav", rate=16_000)
    (quality / "00-short-en-asr.txt").write_text(
        "Hello world\n", encoding="utf-8")
    runtime, asr_runtime, asr_model = (
        tmp_path / "audiocpp_cli", tmp_path / "whisper-cli",
        tmp_path / "ggml-tiny.bin")
    for path in (runtime, asr_runtime, asr_model):
        path.write_bytes(b"tool")

    monkeypatch.setattr(
        voice_release.sanity, "validate_tts_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reviewed resume reran TTS inference")))
    result = voice_release.score_tts(
        bundle, runtime, voice_release.VoiceReleaseOptions(),
        fixtures=[{"id": "short-en", "text": "Hello world", "language": "en"}],
        asr_runtime=asr_runtime, asr_model=asr_model,
        existing_review={"accepted": True, "notes": "Automated gate: RTF 5.0"})

    assert result["passed"] is True
    assert result["roundtrip_wer"] == 0
    assert result["resumed_from_artifacts"] is True
    assert result["real_time_factor"] == 5.0
    assert result["minutes_per_audio_hour"] == 300.0
    assert result["performance_from_review"] is True


def test_tts_quality_cache_skips_repeated_inference(tmp_path, monkeypatch):
    backend = voice.backend_for("openbmb/VoxCPM2")
    models = tmp_path / "models"
    models.mkdir()
    primary = models / "voxcpm2-Q8_0.gguf"
    primary.write_bytes(b"model")
    bundle = voice.VoiceBundle(
        backend=backend,
        primary=voice.BundleMember(primary, "primary", quant="Q8_0"),
        source_repo="openbmb/VoxCPM2", source_revision="abc", quant="Q8_0")
    runtime, asr_runtime, asr_model = (
        tmp_path / "audiocpp_cli", tmp_path / "whisper-cli",
        tmp_path / "ggml-tiny.bin")
    for path in (runtime, asr_runtime, asr_model):
        path.write_bytes(b"tool")
    calls = {"tts": 0, "asr": 0}

    def synthesize(_runtime, _bundle, _text, output, **_kwargs):
        calls["tts"] += 1
        _wav(output, rate=24_000)
        return {"elapsed_seconds": 1.0, "first_output_seconds": 0.1,
                "seconds": 0.5, "sample_rate": 24_000, "sample_width": 2,
                "channels": 1, "frames": 12_000, "rms": 0.1,
                "silence_ratio": 0.0, "clipping_ratio": 0.0,
                "real_time_factor": 2.0, "minutes_per_audio_hour": 120.0,
                "frames_per_second": 12_000.0}

    def transcribe(*_args, **_kwargs):
        calls["asr"] += 1
        return {"transcript": "Hello world", "wer": 0.0}

    monkeypatch.setattr(voice_release.sanity, "validate_tts_runtime", synthesize)
    monkeypatch.setattr(voice_release.sanity, "validate_asr_runtime", transcribe)
    kwargs = {
        "fixtures": [{"id": "short-en", "text": "Hello world",
                      "language": "en"}],
        "asr_runtime": asr_runtime,
        "asr_model": asr_model,
    }
    first = voice_release.score_tts(
        bundle, runtime, voice_release.VoiceReleaseOptions(), **kwargs)
    second = voice_release.score_tts(
        bundle, runtime, voice_release.VoiceReleaseOptions(), **kwargs)

    assert first == second
    assert calls == {"tts": 1, "asr": 1}


def test_source_preflight_returns_revision_and_size():
    class Sibling:
        size = 42
        lfs = None

        def __init__(self, name):
            self.rfilename = name

    class Info:
        sha = "revision"
        gated = False
        private = False
        siblings = [Sibling("model.safetensors"), Sibling("config.json")]

    class Api:
        def model_info(self, *_args, **_kwargs):
            return Info()

    result = voice_release.source_metadata(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", api=Api())
    assert result["revision"] == "revision"
    assert result["source_bytes"] == 84
    assert result["source_files"] == ["config.json", "model.safetensors"]


def test_source_preflight_reports_inaccessible_registered_repo():
    class Api:
        def model_info(self, *_args, **_kwargs):
            raise RuntimeError("404 Repository Not Found")

    with pytest.raises(voice.VoiceValidationError, match="not accessible"):
        voice_release.source_metadata("openai/whisper-small", api=Api())


def test_tts_whisper_setup_requires_only_runtime(tmp_path, monkeypatch):
    runtime = tmp_path / "build" / "bin" / "whisper-cli"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"runtime")
    monkeypatch.setattr(voice_release.config, "UPSTREAM_WHISPER", tmp_path)
    monkeypatch.setattr(
        voice_release.build_mod, "find_binary",
        lambda directory, name: runtime if name == "whisper-cli" else None)
    monkeypatch.setattr(
        voice_release, "_cmake_build",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("runtime-only setup tried to build a quantizer")))
    directory, found = voice_release.ensure_whisper_runtime()
    assert directory == tmp_path and found == runtime


def test_audiocpp_build_enables_the_full_backend_catalog(monkeypatch):
    monkeypatch.setattr(voice_release.build_mod, "has_cuda_toolkit", lambda: False)
    defines = voice_release._audio_cpp_defines()
    assert "-DAUDIOCPP_MODEL_SET=full" in defines
    assert not any(value.startswith("-DAUDIOCPP_MODELS=") for value in defines)


def test_audio_cpp_virtual_family_preflight_needs_no_hub_call():
    class Api:
        def model_info(self, *_args, **_kwargs):
            raise AssertionError("catalog package preflight contacted the Hub")

    result = voice_release.source_metadata(
        "audio.cpp:fish_audio", api=Api())
    assert result["revision"].startswith("audio.cpp-v")
    assert result["source_bytes"] is None


def test_asr_setup_uses_current_whisper_quantizer_target(tmp_path, monkeypatch):
    cmake = tmp_path / "examples" / "quantize" / "CMakeLists.txt"
    cmake.parent.mkdir(parents=True)
    cmake.write_text("set(TARGET whisper-quantize)\n", encoding="utf-8")
    runtime, quantizer = tmp_path / "whisper-cli", tmp_path / "whisper-quantize"
    built = {"quantizer": False}
    monkeypatch.setattr(
        voice_release, "ensure_whisper_runtime", lambda: (tmp_path, runtime))

    def find_binary(directory, name):
        if name == "whisper-quantize" and built["quantizer"]:
            return quantizer
        return None

    def build(directory, targets):
        assert directory == tmp_path
        assert targets == ("whisper-quantize",)
        built["quantizer"] = True

    monkeypatch.setattr(voice_release.build_mod, "find_binary", find_binary)
    monkeypatch.setattr(voice_release, "_cmake_build", build)
    assert voice_release.ensure_whisper_tools() == (tmp_path, runtime, quantizer)


def test_tts_quantizer_requires_imatrix_for_required_low_bit_type(
        tmp_path, monkeypatch):
    base = tmp_path / "model-BF16.gguf"
    base.write_bytes(b"base")
    monkeypatch.setattr(voice_release, "_run", lambda *args, **kwargs: None)
    with pytest.raises(voice.VoiceValidationError, match="importance matrix"):
        voice_release.quantize_tts(base, tmp_path / "quantize", "IQ2_XXS")


def test_tts_quantizer_passes_imatrix_to_low_bit_type(tmp_path, monkeypatch):
    base = tmp_path / "model-BF16.gguf"
    matrix = tmp_path / "imatrix.dat"
    base.write_bytes(b"base")
    matrix.write_bytes(b"matrix")
    commands = []

    def run(command, _label):
        commands.append(command)
        command[-2].write_bytes(b"quantized")

    monkeypatch.setattr(voice_release, "_run", run)
    output = voice_release.quantize_tts(
        base, tmp_path / "quantize", "IQ2_XXS", imatrix=matrix)
    assert output.name == "model-IQ2_XXS.gguf"
    assert commands[0][:3] == [tmp_path / "quantize", "--imatrix", matrix]


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


def test_audiocpp_conversion_uses_namespaced_sources_and_inspects(
        tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "audiovae.safetensors").write_bytes(b"audio")
    audio_dir = tmp_path / "audio.cpp"
    spec = audio_dir / "model_specs" / "voxcpm2.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        (voice.AUDIOCPP_SPECS_DIR / "voxcpm2.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    converter = audio_dir / "audiocpp_gguf"
    commands = []

    def run(command, _label):
        commands.append(command)
        if "--output" in command:
            output = voice_release.Path(command[command.index("--output") + 1])
            output.write_bytes(b"x" * 2048)

    monkeypatch.setattr(voice_release, "_run", run)
    backend = voice.backend_for("openbmb/VoxCPM2")
    output = voice_release.convert_audiocpp_model(
        source, backend, converter, audio_dir, tmp_path / "models", "Q8_0")
    assert output.name == "voxcpm2-Q8_0.gguf"
    assert "--family" in commands[0] and "voxcpm2" in commands[0]
    inputs = [str(commands[0][index + 1]) for index, value in
              enumerate(commands[0]) if value == "--input"]
    assert any(value.startswith("weights=") for value in inputs)
    assert any(value.startswith("audiovae_weights=") for value in inputs)
    assert commands[1][:2] == [converter, "--inspect"]


def test_audiocpp_prepares_official_pytorch_component_with_pinned_utility(
        tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "audiovae.pth").write_bytes(b"official audio vae")
    audio_dir = tmp_path / "audio.cpp"
    spec_path = audio_dir / "model_specs" / "voxcpm2.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(
        (voice.AUDIOCPP_SPECS_DIR / "voxcpm2.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    manager = audio_dir / "tools" / "model_manager_deprecated.py"
    manager.parent.mkdir(parents=True)
    manager.write_text("# pinned audio.cpp utility\n", encoding="utf-8")
    converter = audio_dir / "audiocpp_gguf"
    commands = []

    def run(command, _label):
        commands.append(command)
        if "--output-file" in command:
            output = voice_release.Path(
                command[command.index("--output-file") + 1])
            output.write_bytes(b"s" * 2048)
        if "--output" in command:
            output = voice_release.Path(command[command.index("--output") + 1])
            output.write_bytes(b"g" * 2048)

    monkeypatch.setattr(voice_release, "_run", run)
    backend = voice.backend_for("openbmb/VoxCPM2")
    output = voice_release.convert_audiocpp_model(
        source, backend, converter, audio_dir, tmp_path / "models", "Q8_0")

    assert output.is_file()
    assert (source / "audiovae.safetensors").is_file()
    prepare = commands[0]
    assert prepare[1:4] == [manager, "install", "voxcpm2_audiovae"]
    assert prepare[prepare.index("--source-file") + 1] == (
        source / "audiovae.pth")
    inputs = [str(commands[1][index + 1]) for index, value in
              enumerate(commands[1]) if value == "--input"]
    assert any(value.startswith("audiovae_weights=") and
               value.endswith("audiovae.safetensors") for value in inputs)


def test_audiocpp_source_plan_discloses_prepared_inputs():
    backend = voice.backend_for("openbmb/VoxCPM2")
    inputs = voice_release.audiocpp_source_plan(
        "openbmb/VoxCPM2", backend,
        ["config.json", "model.safetensors", "audiovae.pth"])

    assert [item["namespace"] for item in inputs] == [
        "weights", "audiovae_weights"]
    assert inputs[0]["status"] == "ready"
    assert inputs[1]["status"] == "needs_preparation"
    assert inputs[1]["preparation"]["package"] == "voxcpm2_audiovae"


def test_audiocpp_conversion_contract_is_not_family_hardcoded(
        tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("ve.safetensors", "s3gen.safetensors", "t3_cfg.safetensors",
                 "t3_mtl23ls_v2.safetensors", "t3_mtl23ls_v3.safetensors"):
        (source / name).write_bytes(b"weights")
    audio_dir = tmp_path / "audio.cpp"
    specs = audio_dir / "model_specs"
    specs.mkdir(parents=True)
    (specs / "chatterbox.json").write_text(
        (voice.AUDIOCPP_SPECS_DIR / "chatterbox.json").read_text(
            encoding="utf-8"), encoding="utf-8")
    commands = []

    def run(command, _label):
        commands.append(command)
        if "--output" in command:
            output = voice_release.Path(command[command.index("--output") + 1])
            output.write_bytes(b"x" * 2048)

    monkeypatch.setattr(voice_release, "_run", run)
    backend = voice.backend_for("ResembleAI/chatterbox")
    voice_release.convert_audiocpp_model(
        source, backend, audio_dir / "audiocpp_gguf", audio_dir,
        tmp_path / "models", "Q8_0")
    inputs = [str(commands[0][index + 1]) for index, value in
              enumerate(commands[0]) if value == "--input"]
    assert any(value.startswith("voice_encoder_weights=") for value in inputs)
    assert any(value.startswith("s3gen_weights=") for value in inputs)


def test_audiocpp_package_selection_follows_recommended_variant():
    spec = voice.AUDIOCPP_SPEC_REGISTRY["qwen3_tts"]
    selected = voice_release._select_audiocpp_package(
        spec, "audio.cpp:qwen3_tts", "BF16")
    assert selected["id"] == "qwen3_tts_1_7b_base_bf16"

    exact = voice_release._select_audiocpp_package(
        spec, "qwen3_tts_0_6b_base_q8_0", "Q8_0")
    assert exact["id"] == "qwen3_tts_0_6b_base_q8_0"


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
