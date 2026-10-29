"""Offline contracts for the multimodal pipeline."""
import base64
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agentquantix import actions, config, multimodal, runs
from agentquantix.pipeline import multimodal_release as release
from agentquantix.pipeline import native_common
from agentquantix.tasks import MULTIMODAL, TASKS, route


def test_explicit_multimodal_routes_override_hub_tags_but_default_image_text_stays_text():
    image = route("HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                  metadata={"pipeline_tag": "image-text-to-text", "tags": []})
    audio = route("Qwen/Qwen3-ASR-0.6B",
                  metadata={"pipeline_tag": "automatic-speech-recognition", "tags": []})
    explicit_image = route("HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                           task="multimodal-understanding",
                           metadata={"pipeline_tag": "image-text-to-text", "tags": []})
    explicit_audio = route("Qwen/Qwen3-ASR-0.6B", task="multimodal-understanding",
                           metadata={"pipeline_tag": "automatic-speech-recognition", "tags": []})
    assert image.pipeline == "text" and image.task == "text-generation"
    assert audio.pipeline == "voice"
    assert (explicit_image.pipeline, explicit_image.engine) == (MULTIMODAL, "llama.cpp")
    assert (explicit_audio.pipeline, explicit_audio.engine) == (MULTIMODAL, "llama.cpp")
    spec = TASKS["multimodal-understanding"]
    assert set(spec.hub_tags) == {
        "visual-question-answering", "document-question-answering", "image-to-text",
        "video-text-to-text", "audio-text-to-text"}
    assert spec.review == "none"
    assert "F16 projector" in spec.publication


def test_source_contract_claims_modalities_and_blocks_non_multimodal_sources():
    data = {"config.json": {"architectures": ["VisionAudioModel"], "model_type": "demo",
                            "vision_config": {}, "audio_config": {}},
            "tokenizer_config.json": {"chat_template": "template"}}
    result = multimodal.source_contract(
        "org/demo", ["config.json", "tokenizer_config.json"],
        lambda name: data[name], license="apache-2.0")
    assert result["contract"]["claimed_modalities"] == ["vision", "audio"]
    assert result["contract"]["architectures"] == ["VisionAudioModel"]
    assert result["contract"]["model_type"] == "demo"
    assert result["contract"]["license"] == "apache-2.0"
    assert result["contract"]["chat_template"] is True
    empty = multimodal.source_contract("org/text", [], lambda _: {})
    assert empty["blockers"] == [
        "no vision or audio encoder declared; not a multimodal-understanding source"]


def test_plan_uses_default_text_quants_f16_projector_and_hub_license(monkeypatch, tmp_path):
    import huggingface_hub

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "architectures": ["SmolVLMForConditionalGeneration"], "model_type": "smolvlm",
        "vision_config": {}}), encoding="utf-8")
    info = SimpleNamespace(
        sha="s" * 40,
        siblings=[SimpleNamespace(rfilename="config.json", size=config_file.stat().st_size)],
        gated=False, card_data=SimpleNamespace(license="apache-2.0"))

    class Api:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id, *, files_metadata):
            return info

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: str(config_file))
    monkeypatch.setattr("agentquantix.archsupport.supported_quants",
                        lambda: ["F16", *multimodal.TEXT_QUANTS])
    monkeypatch.setattr("agentquantix.runtimes.RuntimeManager.inspect",
                        lambda *a, **k: None)
    plan = multimodal.plan_release("org/smolvlm")
    assert plan["status"] == "ready"
    assert plan["quants"] == list(multimodal.TEXT_QUANTS)
    assert plan["projector_precision"] == "F16"
    assert plan["license"] == "apache-2.0"
    assert plan["contract"]["claimed_modalities"] == ["vision"]


def test_asr_architecture_claims_audio_without_audio_config():
    result = multimodal.source_contract(
        "org/asr", ["config.json"],
        lambda _: {"architectures": ["Qwen3ASRForConditionalGeneration"]})
    assert result["contract"]["claimed_modalities"] == ["audio"]
    assert not result["blockers"]


def test_cer_wer_whitespace_punctuation_and_casefold():
    assert multimodal.cer(" a  b\n", "a b") == 0
    assert multimodal.wer("HELLO,   World!", "hello world") == 0
    assert multimodal.wer("don't stop", "dont stop") == 0
    assert multimodal.wer("one two", "one three") == 0.5


def test_answer_synonyms_are_whole_words_and_order_uses_earliest_synonym():
    expected = [["circle", "round"], ["red"]]
    assert multimodal.answer_hits("A RED round.", expected)
    assert not multimodal.answer_hits("A reddish circle.", expected)
    assert multimodal.order_ok("A red circle, then a blue square.", ["red", "circle"],
                                ["blue", "square"])
    assert not multimodal.order_ok("A blue square, then a red circle.", ["red", "circle"],
                                   ["blue", "square"])
    assert not multimodal.order_ok("red appears but second is absent", ["red"], ["blue"])


def test_strip_markup_is_generic():
    assert multimodal.strip_markup("<asr_text>recognized words</asr_text>") == "recognized words"
    assert multimodal.strip_markup("language English<asr_text>recognized words") == \
        "recognized words"
    assert multimodal.strip_markup("before <tag>after") == "before after"
    assert multimodal.strip_markup("plain <b>text</b>") == "plain text"


def test_fixture_cases_reuse_existing_retrieval_and_voice_assets():
    cases = json.loads((multimodal.FIXTURES / "cases.json").read_text(encoding="utf-8"))
    assert len(cases["vqa"]) == 5
    assert all((multimodal.FIXTURES / case["image"]).is_file() for case in cases["vqa"])
    assert all("reference.png" not in case["image"] for case in cases["vqa"])
    assert (multimodal.FIXTURES / cases["asr"]["audio"]).is_file()
    assert (multimodal.FIXTURES / cases["asr"]["transcript_source"]).is_file()


def test_snapshot_patterns_exclude_bin_when_safetensors_are_present():
    safetensors_allow, _ = release._snapshot_patterns({
        "files": ["model.safetensors", "pytorch_model.bin"]})
    bin_allow, _ = release._snapshot_patterns({"files": ["pytorch_model.bin"]})
    uppercase_allow, _ = release._snapshot_patterns({"files": ["MODEL.SAFETENSORS"]})
    assert "*.safetensors" in safetensors_allow
    assert "*.bin" not in safetensors_allow
    assert "*.bin" in bin_allow
    assert "*.bin" not in uppercase_allow


def test_f16_and_quant_gates_skip_missing_video_and_fail_without_applicable_modalities():
    quality = {"metrics": {"vqa_accuracy": 0.8, "ocr_cer": 0.05,
                           "video_order_ok": None, "asr_wer": None}}
    assert release._quality_gate(quality, ["vision"], False)[0]
    assert release._quality_gate(
        {"metrics": {"vqa_accuracy": 0.6, "ocr_cer": 0.10, "video_order_ok": None}},
        ["vision"], False, baseline=quality)[0]
    assert release._quality_gate(
        {"metrics": {"vqa_accuracy": 0.5, "ocr_cer": 0.2}},
        ["vision"], False, baseline=quality)[0] is False
    assert release._quality_gate({"metrics": {}}, [], False) == (
        False, ["no applicable multimodal quality gates"])
    assert not release._quality_gate(
        {"metrics": {"asr_wer": 0.11}}, ["audio"], False)[0]
    failed_video = {"metrics": {"vqa_accuracy": 1.0, "ocr_cer": 0.0,
                                "video_order_ok": False}}
    assert not release._quality_gate(failed_video, ["vision"], True)[0]
    assert not release._quality_gate(
        {"metrics": {"vqa_accuracy": 0.5, "ocr_cer": 0.0, "video_order_ok": False}},
        ["vision"], True, baseline={"metrics": {
            "vqa_accuracy": 0.5, "ocr_cer": 0.0, "video_order_ok": True}})[0]
    assert not release._quality_gate(
        {"metrics": {"vqa_accuracy": 0.5, "ocr_cer": 0.0, "video_order_ok": None}},
        ["vision"], False,
        reference={"available": True, "vqa_accuracy": 0.8})[0]


def test_video_requires_both_cli_help_flag_and_ffmpeg(monkeypatch):
    monkeypatch.setattr(release.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="supports --video", stderr="", returncode=0))
    assert release._video_support("mtmd")
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout="no video", stderr="", returncode=0))
    assert not release._video_support("mtmd")
    monkeypatch.setattr(release.shutil, "which", lambda name: None)
    assert not release._video_support("mtmd")


def test_video_answer_uses_verified_frame_rate_and_timestamp_interval(monkeypatch, tmp_path):
    calls = {}

    def run(argv, **kwargs):
        calls["argv"] = argv
        calls["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="red circle first, blue square next", stderr="")

    monkeypatch.setattr(release.subprocess, "run", run)
    answer, _ = release._video_answer("mtmd", "text.gguf", "mmproj.gguf", tmp_path)

    assert answer == "red circle first, blue square next"
    assert calls["argv"][calls["argv"].index("--video-fps") + 1] == "1"
    assert calls["argv"][calls["argv"].index("--video-timestamp-interval") + 1] == "1000"
    assert calls["kwargs"]["cwd"] == tmp_path


def test_chat_requests_encode_image_and_audio_shapes(monkeypatch, tmp_path):
    image, audio = tmp_path / "sample.png", tmp_path / "sample.wav"
    image.write_bytes(b"image-bytes")
    audio.write_bytes(b"audio-bytes")
    requests = []

    class Response(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    def open_url(request, timeout):
        requests.append(json.loads(request.data))
        return Response(b'{"choices":[{"message":{"content":"ok"}}]}')

    monkeypatch.setattr(release, "urlopen", open_url)
    assert release._chat_request("http://localhost", "describe", image=image) == "ok"
    assert release._chat_request("http://localhost", "transcribe", audio=audio) == "ok"
    image_part = requests[0]["messages"][0]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(b"image-bytes").decode())
    audio_part = requests[1]["messages"][0]["content"][1]
    assert audio_part == {"type": "input_audio", "input_audio": {
        "data": base64.b64encode(b"audio-bytes").decode(), "format": "wav"}}
    assert all((body["temperature"], body["seed"], body["max_tokens"]) == (0, 42, 96)
               for body in requests)


def test_shared_server_terminates_on_exception(monkeypatch, tmp_path):
    terminated = []

    class Process:
        def poll(self):
            return None

        def terminate(self):
            terminated.append(True)

        def wait(self, timeout=None):
            return 0

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(native_common.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(native_common, "_free_port", lambda: 5555)
    monkeypatch.setattr(native_common, "urlopen", lambda *a, **k: Response())
    with pytest.raises(RuntimeError, match="request failed"):
        with native_common._server("server", "model.gguf", tmp_path, []):
            raise RuntimeError("request failed")
    assert terminated == [True]


def test_missing_projector_conversion_fails_closed(monkeypatch, tmp_path):
    class Handle:
        source = tmp_path / "runtime"

    def convert(argv, **kwargs):
        if "--mmproj" not in argv:
            Path(argv[argv.index("--outfile") + 1]).write_bytes(b"text")

    monkeypatch.setattr(release, "_command", convert)
    with pytest.raises(RuntimeError, match="projector conversion produced no file"):
        release._convert({"model": "org/demo"}, tmp_path / "source", Handle(),
                         tmp_path / "run", {}, tmp_path / "journal.json")


def test_projector_modality_mismatch_is_recorded_and_blocks(monkeypatch, tmp_path):
    class Handle:
        source = tmp_path / "runtime"

    def convert(argv, **kwargs):
        Path(argv[argv.index("--outfile") + 1]).write_bytes(b"gguf")

    journal = {}
    monkeypatch.setattr(release, "_command", convert)
    monkeypatch.setattr(release, "read_projector_metadata", lambda *_: {
        "general.type": "mmproj", "clip.has_vision_encoder": True,
        "clip.has_audio_encoder": False})
    with pytest.raises(RuntimeError, match="does not cover claimed modalities"):
        release._convert({"model": "org/demo", "contract": {
            "claimed_modalities": ["audio"]}}, tmp_path / "source", Handle(),
            tmp_path / "run", journal, tmp_path / "journal.json")
    assert "blocker" in journal


def test_projector_metadata_sets_actual_modalities_and_requires_mmproj_type(tmp_path):
    metadata = {"clip.has_vision_encoder": 1, "clip.has_audio_encoder": False}
    assert release._actual_modalities(metadata) == ["vision"]
    package = tmp_path / "runtime" / "gguf-py" / "gguf"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "class Field:\n"
        "    def __init__(self, value): self.value = value\n"
        "    def contents(self): return self.value\n"
        "class GGUFReader:\n"
        "    def __init__(self, path): pass\n"
        "    def get_field(self, key):\n"
        "        return Field(b'clip') if key == 'general.type' else None\n",
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="general.type must be 'mmproj'"):
        release.read_projector_metadata(tmp_path / "runtime", "/missing/projector.gguf")


def test_projector_metadata_prefers_runtime_gguf_over_installed_package(tmp_path, monkeypatch):
    installed_package = ModuleType("gguf")
    installed_constants = ModuleType("gguf.constants")
    installed_constants.RUNTIME_TYPE = b"installed"
    monkeypatch.setitem(sys.modules, "gguf", installed_package)
    monkeypatch.setitem(sys.modules, "gguf.constants", installed_constants)

    package = tmp_path / "runtime" / "gguf-py" / "gguf"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from .gguf_reader import GGUFReader\n", encoding="utf-8")
    (package / "constants.py").write_text(
        "RUNTIME_TYPE = b'mmproj'\n", encoding="utf-8")
    (package / "gguf_reader.py").write_text(
        "from gguf.constants import RUNTIME_TYPE\n"
        "class Field:\n"
        "    def contents(self): return RUNTIME_TYPE\n"
        "class GGUFReader:\n"
        "    def __init__(self, path): pass\n"
        "    def get_field(self, key):\n"
        "        return Field() if key == 'general.type' else None\n",
        encoding="utf-8")

    metadata = release.read_projector_metadata(
        tmp_path / "runtime", "/missing/projector.gguf")

    assert metadata["general.type"] == "mmproj"
    assert sys.modules["gguf"] is installed_package
    assert sys.modules["gguf.constants"] is installed_constants


def test_reference_unavailable_is_cached(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "transformers", None)
    journal, journal_path = {}, tmp_path / "journal.json"
    plan = {"model": "org/unsupported", "contract": {"claimed_modalities": ["vision"]}}
    first = release._reference(plan, tmp_path, journal, journal_path)
    second = release._reference(plan, tmp_path, journal, journal_path)
    assert first == second
    assert first["available"] is False and first["reason"]


def test_multimodal_run_resumes_cached_evaluations(monkeypatch, tmp_path):
    from agentquantix import runtime_control

    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    runtime_snapshots = []

    class Handle:
        revision = "r" * 40

        def snapshot(self):
            return {"revision": self.revision, "source": str(tmp_path / "runtime"),
                    "provenance": {"binary_sha256": {}}}

    handle = Handle()

    def ensure_runtime(engine, *, required_targets, frozen=None):
        assert engine == "llama.cpp"
        assert set(required_targets) == set(release.RUNTIME_TARGETS)
        runtime_snapshots.append(frozen)
        return handle

    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(runtime_control, "ensure_runtime", ensure_runtime)
    monkeypatch.setattr(release, "require_binaries",
                        lambda *_: ["/native/server", "/native/mtmd", "/native/quantize"])
    monkeypatch.setattr(release, "snapshot_download", lambda *a, **k: str(source))

    def convert(plan, source, runtime, root, journal, journal_path):
        models = root / "models"
        models.mkdir(exist_ok=True)
        text, projector = models / "demo-F16.gguf", models / "mmproj-demo-F16.gguf"
        text.write_bytes(b"f16-text")
        projector.write_bytes(b"f16-projector")
        return text, projector, {
            "general.type": "mmproj", "clip.has_vision_encoder": True,
            "clip.has_audio_encoder": False}, ["vision"]

    monkeypatch.setattr(release, "_convert", convert)

    def quantize(binary, text, root, journal, journal_path, quant):
        output = root / "models" / f"demo-{quant}.gguf"
        output.write_bytes(quant.encode())
        return output

    monkeypatch.setattr(release, "_quantize", quantize)
    monkeypatch.setattr(release, "_reference", lambda *a: {
        "available": False, "reason": "unsupported architecture"})
    monkeypatch.setattr(release, "_video_support", lambda _: False)
    evaluations = []

    def evaluate(plan, model, projector, metadata, modalities, binaries, root, video):
        evaluations.append(model.name)
        return {"metrics": {"vqa_accuracy": 1.0, "ocr_cer": 0.0, "video_order_ok": None},
                "answers": {}, "request_seconds": {}}

    monkeypatch.setattr(release, "_evaluate", evaluate)
    plan = {"model": "org/demo", "task": "multimodal-understanding",
            "engine": "llama.cpp", "status": "ready", "blockers": [],
            "source_revision": "s" * 40, "quants": ["Q8_0"],
            "contract": {"claimed_modalities": ["vision"]},
            "quality_profile": multimodal.QUALITY_PROFILE}
    first = release.run_release(plan, user_approved=True)
    second = release.run_release(plan, user_approved=True)
    assert [item["status"] for item in first["candidates"]] == ["passed", "passed"]
    assert [item["status"] for item in second["candidates"]] == ["passed", "passed"]
    assert len(evaluations) == 2
    assert runtime_snapshots[0] is None
    assert runtime_snapshots[1]["revision"] == handle.revision


def test_publish_bundle_contains_only_passing_quants_and_verifies_manifest(
        monkeypatch, tmp_path):
    import huggingface_hub
    from agentquantix.pipeline import voice_release

    class Operation:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class Api:
        def create_repo(self, **kwargs):
            self.repo = kwargs["repo_id"]

        def create_commit(self, *, operations, **kwargs):
            self.operations = operations

    monkeypatch.setattr(huggingface_hub, "CommitOperationAdd", Operation)
    api = Api()

    def remote_files(api, repo_id):
        return {operation.path_in_repo: {
            "bytes": Path(operation.path_or_fileobj).stat().st_size,
            "sha256": release._hash(operation.path_or_fileobj)}
                for operation in api.operations}

    monkeypatch.setattr(voice_release, "_remote_files", remote_files)
    root = tmp_path / "run"
    models = root / "models"
    models.mkdir(parents=True)
    f16 = models / "demo-F16.gguf"
    q8 = models / "demo-Q8_0.gguf"
    rejected = models / "demo-Q4_K_M.gguf"
    projector = models / "mmproj-demo-F16.gguf"
    for path in (f16, q8, rejected, projector):
        path.write_bytes(path.name.encode())
    plan = {"model": "org/demo", "target_repo": "owner/demo",
            "source_revision": "s" * 40, "contract": {"claimed_modalities": ["vision"]},
            "quality_profile": multimodal.QUALITY_PROFILE, "fixtures": {},
            "license": "apache-2.0"}
    result = {"runtime_snapshot": {"revision": "r" * 40, "provenance": {}},
              "projector": {"path": str(projector), "metadata": {
                  "general.type": "mmproj", "clip.has_vision_encoder": True}},
              "modalities": ["vision"], "video": "skipped: runtime lacks video",
              "candidates": [
                  {"quant": "F16", "path": str(f16), "status": "passed",
                   "quality": {"metrics": {"vqa_accuracy": 1.0}}},
                  {"quant": "Q8_0", "path": str(q8), "status": "passed",
                   "quality": {"metrics": {"vqa_accuracy": 1.0}}},
                  {"quant": "Q4_K_M", "path": str(rejected), "status": "quality-rejected",
                   "quality": {"metrics": {"vqa_accuracy": 0.1}}}]}
    published = release.publish_bundle(root, plan, result, api=api)
    manifest_path = root / "publication" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert published["verified"] is True
    assert {item["quant"] for item in manifest["members"]} == {"F16", "Q8_0"}
    assert any(item.get("role") == "projector" for item in manifest["members"])
    assert all("Q4_K_M" not in item["path"] for item in manifest["members"])
    assert release.verify_manifest(manifest, root / "publication") == []


def test_actions_require_exact_plan_approval_and_reject_review(monkeypatch):
    with pytest.raises(ValueError, match="approval of the exact multimodal plan"):
        actions.start_multimodal({"status": "ready", "blockers": []}, user_approved=False)
    with pytest.raises(ValueError, match="multimodal plan is blocked"):
        actions.start_multimodal({"status": "blocked", "blockers": ["missing config"]},
                                 user_approved=True)
    monkeypatch.setattr(runs, "resolve", lambda *_: SimpleNamespace(
        pipeline="multimodal", id="multimodal:demo"))
    with pytest.raises(ValueError, match="no human review"):
        actions.review("multimodal:demo", "F16", accepted=True, reviewer="Reviewer")


def test_saved_plan_cli_and_agent_dispatch_use_pipeline(monkeypatch, tmp_path):
    from agentquantix import cli
    from agentquantix.agent import tools

    plan = {"task": "multimodal-understanding", "engine": "llama.cpp"}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(cli, "_check_applicable", lambda *args: None)
    monkeypatch.setattr(cli, "_run_multimodal",
                        lambda args, route=None, saved=None: ("multimodal", saved))
    result = cli.cmd_run(cli.build_parser().parse_args(
        ["run", "--plan", str(plan_path), "--yes"]))
    assert result == ("multimodal", plan)
    monkeypatch.setattr(tools.actions, "start_multimodal",
                        lambda approved, **kwargs: (approved, kwargs))
    started, options = tools._start({"user_approved": True, "plan": plan})
    assert started is plan
    assert options == {"user_approved": True, "publish": False}


def test_multimodal_cli_and_run_inventory_contract(monkeypatch, tmp_path):
    from agentquantix import cli, config
    args = cli.build_parser().parse_args([
        "run", "org/model", "--task", "multimodal-understanding", "--quant", "Q8_0",
        "--no-publish", "--yes"])
    assert args.quant == ["Q8_0"] and args.publish is False
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    run_dir = runs.multimodal_root() / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"plan": {
        "model": "org/demo", "task": "multimodal-understanding", "engine": "llama.cpp"}}),
        encoding="utf-8")
    assert runs.resolve("multimodal:demo").pipeline == MULTIMODAL
    assert cli.build_parser().parse_args(["runs", "--pipeline", "multimodal"]).pipeline == MULTIMODAL


def test_multimodal_card_and_runtime_targets_include_required_contract():
    metadata = {"clip.projector_type": "qwen2.5_vl", "clip.vision.projector_type": "siglip",
                "clip.vision.image_size": [384, 384],
                "clip.vision.preproc_image_size": [384, 384],
                "clip.vision.image_mean": [0.5, 0.5, 0.5],
                "clip.vision.image_std": [0.5, 0.5, 0.5],
                "clip.vision.patch_size": 14}
    manifest = {"pipeline": "multimodal", "family": "demo", "source": {"repo_id": "org/demo"},
                "modalities": ["vision"], "projector_metadata": metadata,
                "quality_results": {"F16": {"status": "passed", "metrics": {"vqa_accuracy": 1.0}}}}
    card = multimodal.render_card(manifest)
    assert "llama-server -m" in card and "--mmproj" in card and "--jinja" in card
    assert "image_url" in card and "llama-mtmd-cli" in card
    assert "--video-fps 1 --video-timestamp-interval 1000" in card
    assert "projector quantization is unaccepted" in card
    assert release.RUNTIME_TARGETS == ("llama-server", "llama-mtmd-cli", "llama-quantize")


def test_llama_runtime_fingerprint_includes_mtmd_video_and_target():
    from agentquantix import runtime_builders
    fingerprint = runtime_builders.fingerprint("llama.cpp", compute_backend="cpu",
                                               required_targets=release.RUNTIME_TARGETS)
    assert "llama-mtmd-cli" in fingerprint["targets"]
    assert "llama-server" in fingerprint["targets"]
    assert "-DMTMD_VIDEO=ON" in fingerprint["flags"]
