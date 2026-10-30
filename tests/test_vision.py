import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agentquantix import vision
from agentquantix.pipeline import vision_release
from agentquantix.tasks import TASKS, VISION
from agentquantix.tasks.router import route


def _safetensors(path):
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1],
                                    "data_offsets": [0, 4]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
    return path


def test_family_resolution_uses_explicit_repo_then_config():
    assert vision.family_for("unknown/model", {}, family="sam")["family"] == "sam"
    assert vision.family_for("dhkim2810/MobileSAM", {})["family"] == "sam"
    assert vision.family_for("unknown/model", {"model_type": "depth_anything"})[
        "family"] == "depth-anything"


def test_precision_policy_never_promotes():
    assert vision.precision_policy({"F32"})["precisions"] == ["F32", "F16"]
    policy = vision.precision_policy({"F16", "F32"})
    assert policy["precisions"] == ["F16"]
    assert policy["source_dtype"] == "F16"
    assert vision.precision_policy({"F16"})["precisions"] == ["F16"]


def test_source_dtype_reads_safetensors_header(tmp_path):
    weight = _safetensors(tmp_path / "weights.safetensors")
    assert vision.source_dtypes(tmp_path, [weight.name]) == {"F32"}


def test_pt_dtype_inspection_uses_converter_python_and_weights_only(tmp_path):
    weight = tmp_path / "weights.pt"
    weight.write_bytes(b"checkpoint")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout='["float32"]', stderr="", returncode=0)

    assert vision.source_dtypes(tmp_path, [weight.name], python="/converter/python",
                                runner=runner) == {"float32"}
    command, kwargs = calls[0]
    assert command[:2] == ["/converter/python", "-c"]
    assert "weights_only=True" in command[2]
    assert kwargs["check"] is True


def test_python_resolver_prefers_explicit_python():
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="3.12\n")

    found = vision.resolve_python({"AQX_VISION_PYTHON": "/custom/python"},
                                  which=lambda _name: None, run=fake_run)
    assert found == str(Path("/custom/python").resolve())
    assert calls[0][0] == "/custom/python"


def test_converter_command_matches_upstream_cli():
    family = {"converter_arch": "sam"}
    assert vision.converter_command("python", "convert.py", "weights.pt",
                                    "output.gguf", family, "F32") == [
        "python", "convert.py", "sam", "weights.pt", "--output", "output.gguf"]
    assert vision.converter_command("python", "convert.py", "weights.pt",
                                    "output.gguf", family, "F16")[-2:] == [
        "--quantize", "f16"]


def test_runtime_revision_guard_and_router_blockers():
    with pytest.raises(vision_release.VisionValidationError, match="install vision.cpp"):
        vision_release.runtime_guard("wrong")
    route_result = route("unknown/model", task="mask-generation")
    assert not route_result.ready
    assert route_result.blocker == (
        "official facebook/sam3 weights are gated under the SAM License; "
        "no ungated source with verifiable conversion provenance; sam3cpp not accepted")
    assert route("dhkim2810/MobileSAM", task="mask-generation").pipeline == VISION
    configured = route("someone/custom", task="depth-estimation",
                       metadata={"model_type": "depth_anything"})
    assert configured.ready and configured.family == "depth-anything"


def test_placeholder_tasks_keep_quality_note():
    expected = (
        "native MI-GAN/Real-ESRGAN outputs did not beat the naive baseline under "
        "PSNR/SSIM; no perceptual quality gate is established")
    for task in ("inpainting", "super-resolution"):
        assert TASKS[task].note == expected
        assert not TASKS[task].implemented


def test_mask_metrics_and_gate_edges():
    mask = np.zeros((16, 16), dtype=bool)
    mask[3:12, 4:13] = True
    assert vision.iou(mask, mask) == 1.0
    assert vision.boundary_fscore(mask, mask) == 1.0
    assert vision_release.gate(
        "image-segmentation", {"iou": 0.90, "boundary_fscore": 0.80},
        is_baseline=True)
    assert not vision_release.gate(
        "image-segmentation", {"iou": 0.899, "boundary_fscore": 0.80},
        is_baseline=True)
    base = {"point_iou": 0.91, "box_iou": 0.92}
    assert vision_release.gate(
        "mask-generation", {"point_iou": 0.90, "box_iou": 0.91,
                            "pixel_agreement": 0.99},
        base)
    assert not vision_release.gate(
        "mask-generation", {"point_iou": 0.90, "box_iou": 0.91,
                            "pixel_agreement": 0.98}, base)


def test_depth_metrics_and_gate_edges():
    near = np.tile(np.arange(1, 10, dtype=np.float32), (8, 1))
    far = near * 2
    assert vision.depth_baseline_passes([near], np.vstack((near, far)))
    metric = vision.depth_comparison(near, near * 2 + 1)
    assert metric["spearman"] == pytest.approx(1.0)
    assert metric["scale_shift_absrel"] < 1e-8
    assert vision_release.gate("depth-estimation", {
        "depth_comparisons": [{"spearman": 0.999, "scale_shift_absrel": 0.01}]})
    assert not vision_release.gate("depth-estimation", {
        "depth_comparisons": [{"spearman": 0.998, "scale_shift_absrel": 0.01}]})


def test_runtime_catalog_and_cmake_flags(monkeypatch):
    from agentquantix import runtime_builders
    assert runtime_builders._submodules_for("vision.cpp") == ["depend/ggml"]
    monkeypatch.setattr(runtime_builders.native_build, "find_tool",
                        lambda name: "/usr/bin/cmake" if name == "cmake" else None)
    version = {"value": "cmake version 3.27.9"}

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(stdout=version["value"] + "\n")

    monkeypatch.setattr(runtime_builders.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="CMake >= 3.28"):
        runtime_builders._vision_cmake()
    version["value"] = "cmake version 3.28.0"
    fingerprint = runtime_builders.fingerprint("vision.cpp")
    for flag in ("-DCMAKE_BUILD_TYPE=Release", "-DVISP_VULKAN=OFF",
                 "-DVISP_TESTS=OFF", "-DBUILD_SHARED_LIBS=OFF",
                 "-DVISP_FMT_LIB=ON", "-DGGML_CCACHE=OFF"):
        assert flag in fingerprint["flags"]


def test_release_partial_pass_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "converter_venv", lambda *_args: "python")
    monkeypatch.setattr(vision_release, "_requirements_provenance",
                        lambda _python: {"python": "python", "packages": []})
    monkeypatch.setattr(vision_release.config, "TEMP_DIR", tmp_path)
    runtime = SimpleNamespace(
        revision=vision.VISIONCPP_REVISION,
        source=tmp_path / "runtime",
        binaries={"vision-cli": tmp_path / "vision-cli"},
        snapshot=lambda: {"revision": vision.VISIONCPP_REVISION})
    plan = {
        "pipeline": "vision", "engine": "vision.cpp", "task": "image-segmentation",
        "model": "ZhengPeng7/BiRefNet", "family": "birefnet",
        "source_revision": "source-sha", "source_files": ["weights.safetensors"],
        "source_license": "MIT", "license_note": "MIT", "target_repo": "user/model",
        "quants": ["F32", "F16"], "runtime_revision": vision.VISIONCPP_REVISION,
        "status": "ready", "blockers": [],
    }
    conversions = []

    def download(_model, revision, local_dir, allow_patterns):
        assert revision == "source-sha"
        assert allow_patterns == ["weights.safetensors"]
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return _safetensors(Path(local_dir) / "weights.safetensors").parent

    def convert(command):
        conversions.append(command)
        output = Path(command[command.index("--output") + 1])
        output.write_bytes(output.name.encode())

    def evaluate(_task, _family, model, _binary, _root, _baseline):
        output = np.zeros((2, 2), dtype=bool)
        quality = {"iou": 0.95, "boundary_fscore": 0.9, "elapsed_seconds": 1.25,
                   "output_path": f"{model.stem}.png", "outputs": [output]}
        if model.stem == "F16":
            output[0, 0] = True
        return quality

    result = vision_release.run_release(
        plan, user_approved=True, runtime=runtime, download_source=download,
        convert=convert, evaluate=evaluate)
    assert [item["status"] for item in result["candidates"]] == [
        "passed", "quality-rejected"]
    assert [item["quality"]["elapsed_seconds"] for item in result["candidates"]] == [
        1.25, 1.25]
    assert len(conversions) == 2
    resumed = vision_release.run_release(
        plan, user_approved=True, runtime=runtime, download_source=download,
        convert=convert, evaluate=evaluate)
    assert resumed["candidates"] == result["candidates"]
    assert len(conversions) == 2


def test_release_skips_completed_precision_after_interruption(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "converter_venv", lambda *_args: "python")
    monkeypatch.setattr(vision_release, "_requirements_provenance",
                        lambda _python: {"python": "python", "packages": []})
    monkeypatch.setattr(vision_release.config, "TEMP_DIR", tmp_path)
    runtime = SimpleNamespace(
        revision=vision.VISIONCPP_REVISION,
        source=tmp_path / "runtime",
        binaries={"vision-cli": tmp_path / "vision-cli"},
        snapshot=lambda: {"revision": vision.VISIONCPP_REVISION, "identity": "frozen"})
    plan = {
        "pipeline": "vision", "engine": "vision.cpp", "task": "image-segmentation",
        "model": "org/birefnet", "family": "birefnet",
        "source_revision": "source-sha", "source_files": ["weights.safetensors"],
        "source_license": "MIT", "license_note": "MIT", "target_repo": "user/model",
        "quants": ["F32", "F16"], "runtime_revision": vision.VISIONCPP_REVISION,
        "status": "ready", "blockers": [],
    }
    evaluations = []
    conversions = []

    def download(_model, revision, local_dir, allow_patterns):
        assert revision == "source-sha" and allow_patterns == ["weights.safetensors"]
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return _safetensors(Path(local_dir) / "weights.safetensors").parent

    def convert(command):
        conversions.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(b"gguf")

    def evaluate(_task, _family, model, _binary, _root, _baseline):
        evaluations.append(model.stem)
        if model.stem == "F16" and evaluations.count("F16") == 1:
            raise RuntimeError("interrupted")
        return {"iou": 0.95, "boundary_fscore": 0.9,
                "outputs": [np.zeros((2, 2), dtype=bool)]}

    with pytest.raises(RuntimeError, match="interrupted"):
        vision_release.run_release(
            plan, user_approved=True, runtime=runtime, download_source=download,
            convert=convert, evaluate=evaluate)
    result = vision_release.run_release(
        plan, user_approved=True, runtime=runtime, download_source=download,
        convert=convert, evaluate=evaluate)

    assert evaluations == ["F32", "F16", "F16"]
    assert len(conversions) == 2
    assert [candidate["precision"] for candidate in result["candidates"]] == ["F32", "F16"]


def test_bundle_publishes_only_passing_precision_atomically(tmp_path):
    candidate_path = tmp_path / "F32.gguf"
    candidate_path.write_bytes(b"model")
    candidate = {"precision": "F32", "path": str(candidate_path),
                 "sha256": vision_release._hash(candidate_path), "status": "passed"}
    rejected = {"precision": "F16", "path": str(tmp_path / "F16.gguf"),
                "sha256": "unused", "status": "quality-rejected"}
    plan = {"task": "depth-estimation", "family": "depth-anything",
            "source_license": "Apache-2.0", "license_note": "Apache-2.0",
            "source_dtype": "F32", "runtime_revision": vision.VISIONCPP_REVISION,
            "target_repo": "user/model"}

    class FakeApi:
        commits = []

        def create_repo(self, *_args, **_kwargs):
            return None

        def create_commit(self, **kwargs):
            self.commits.append(kwargs)

    api = FakeApi()
    published = vision_release.publish_bundle(
        tmp_path, plan, [candidate, rejected], api=api)
    assert published["precisions"] == ["F32"]
    assert len(api.commits) == 1
    manifest_path = tmp_path / "publication" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert [item["precision"] for item in manifest["candidates"]] == ["F32"]
    assert not vision_release.verify_manifest(manifest, tmp_path / "publication")


def test_cli_resume_and_agent_tool_dispatch_vision(monkeypatch):
    from agentquantix import cli, actions
    from agentquantix.agent import tools

    plan = {"pipeline": "vision", "task": "mask-generation", "status": "ready"}
    record = SimpleNamespace(
        id="vision:MobileSAM", pipeline=VISION, task="mask-generation",
        repo_id="dhkim2810/MobileSAM", location="/unused", details={"plan": plan})
    monkeypatch.setattr(cli.runs, "resolve", lambda _run_id: record)
    monkeypatch.setattr(cli, "_check_applicable", lambda *_args: None)
    calls = []
    monkeypatch.setattr(cli, "_run_vision",
                        lambda _args, saved, **kwargs: calls.append((saved, kwargs)) or 0)
    assert cli._resume(SimpleNamespace(resume=record.id)) == 0
    assert calls == [(plan, {"runtime_snapshot": None})]

    monkeypatch.setattr(actions, "vision_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(actions, "start_vision",
                        lambda *args, **kwargs: {"pipeline": "vision"})
    result = tools._start({"user_approved": True, "models": ["dhkim2810/MobileSAM"],
                           "task": "mask-generation"})
    assert result["pipeline"] == "vision"


def test_start_vision_passes_an_immutable_runtime_target(monkeypatch):
    from agentquantix import actions, runtime_control
    from agentquantix.pipeline import vision_release

    calls = {}
    monkeypatch.setattr(runtime_control, "ensure_runtime",
                        lambda *args, **kwargs: calls.update(kwargs))
    monkeypatch.setattr(vision_release, "run_release",
                        lambda *_args, **_kwargs: {"run_id": "run", "candidates": []})

    result = actions.start_vision({"status": "ready", "blockers": []}, user_approved=True)

    assert calls["revision"] == {
        "revision": vision_release.REVISION, "ref": vision_release.REVISION,
        "channel": "pinned", "repository": "Acly/vision.cpp",
    }
    assert result["run_id"] == "vision:run"
