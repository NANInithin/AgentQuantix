import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentquantix import actions, runtime_builders, vision
from agentquantix.pipeline import depth_release
from agentquantix.tasks.router import route


def test_depth_families_and_metric_relative_resolution():
    relative = vision.family_for("depth-anything/Depth-Anything-V2-Small", task="depth-estimation")
    metric = vision.family_for(
        "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
        task="depth-estimation")
    da3 = vision.family_for("depth-anything/DA3-SMALL", task="depth-estimation")
    assert relative["engine"] == "vision.cpp"
    assert metric["engine"] == "depth-anything.cpp"
    assert da3["engine"] == "depth-anything.cpp"
    assert vision.family_for(
        "unknown/model", {"model_type": "depth_anything",
                          "depth_estimation_type": "metric"},
        task="depth-estimation")["family"] == "da2-metric-hypersim-small"
    assert vision.family_for(
        "unknown/model", {"model_type": "depth_anything",
                          "depth_estimation_type": "relative"},
        task="depth-estimation")["engine"] == "vision.cpp"
    assert route("depth-anything.cpp:da3-small", task="depth-estimation").family == "da3-small"


def test_avx512f_cpu_flag_detection_and_identity(tmp_path, monkeypatch):
    info = tmp_path / "cpuinfo"
    info.write_text("flags : fpu avx2 avx512f\n")
    assert runtime_builders._depth_avx512f_enabled(info, "Linux")
    assert not runtime_builders._depth_avx512f_enabled(info, "Darwin")
    info.write_text("flags : fpu avx2\n")
    assert not runtime_builders._depth_avx512f_enabled(info, "Linux")
    monkeypatch.setattr(runtime_builders, "_depth_avx512f_enabled", lambda: True)
    fingerprint = runtime_builders.fingerprint("depth-anything.cpp")
    assert "-DDA_HAS_AVX512F=ON" in fingerprint["flags"]
    monkeypatch.setattr(runtime_builders, "_depth_avx512f_enabled", lambda: False)
    fingerprint = runtime_builders.fingerprint("depth-anything.cpp")
    assert "-DDA_HAS_AVX512F=OFF" in fingerprint["flags"]
    assert fingerprint["targets"] == ["da3-cli"]


def test_depth_pin_precision_producers_and_argv():
    plan = depth_release.plan_release(
        "depth-anything.cpp:da3-small", task="depth-estimation")
    assert plan["runtime_revision"] == depth_release.RUNTIME_REVISION
    assert plan["precision_producers"] == {
        "F32": "converter", "F16": "native-quantizer",
        "Q8_0": "native-quantizer", "Q4_K": "native-quantizer",
    }
    assert "sentencepiece==0.2.2" in depth_release.REQUIREMENTS
    assert "einops==0.8.2" in depth_release.REQUIREMENTS
    assert "imageio==2.38.1" in depth_release.REQUIREMENTS
    assert "addict==2.4.0" in depth_release.REQUIREMENTS
    assert "evo==1.38.1" in depth_release.REQUIREMENTS
    card = depth_release.runtime_card(plan, {}, {})
    assert card["semantics"] == "relative depth + camera pose/intrinsics"
    assert card["confidence_output"] == "not evaluated"
    assert card["precision_producers"] == plan["precision_producers"]
    assert depth_release.converter_command(
        "python", "/engine", "/ref", "da3-small", "/weights", "/out.gguf")[0:2] == [
            "python", str(Path(depth_release.__file__).with_name("depth_converter.py"))]
    assert depth_release.quantize_command(
        "/bin/da3-cli", "/in.gguf", "/out.gguf", "Q4_K") == [
            "/bin/da3-cli", "quantize", "/in.gguf", "/out.gguf", "q4_k"]
    assert depth_release.native_command(
        "/bin/da3-cli", "da2-metric-hypersim-small", "/m.gguf", "/i.jpg", "/o.pfm") == [
            "/bin/da3-cli", "depth", "--model", "/m.gguf", "--input", "/i.jpg",
            "--pfm", "/o.pfm", "--no-invert", "--threads", "4"]


def test_da3_reference_loads_real_torchvision_before_model_helper():
    source = Path(depth_release.__file__).with_name("depth_reference.py").read_text()
    assert source.index("from torchvision import transforms") < source.index(
        "from scripts.da3_reference import load_model")


def test_depth_revision_guard():
    with pytest.raises(depth_release.DepthValidationError, match="revision mismatch"):
        depth_release.guard_runtime("wrong")


def test_depth_source_hash_is_verified_before_conversion(tmp_path):
    source = {"size": 3, "sha256": "0" * 64}
    path = tmp_path / "weights"
    path.write_bytes(b"bad")
    with pytest.raises(depth_release.DepthValidationError, match="integrity check failed"):
        depth_release.verify_source(path, source)


def test_depth_gate_edges_and_baseline_failure():
    metric = {
        "vs_reference": {"spearman": 0.99},
        "metric_vs_reference": {
            "mean_absrel": 0.05, "median_error_percent_of_reference_median": 5.0,
        },
    }
    assert depth_release.gate_quality(
        "da2-metric-hypersim-small", metric, is_baseline=True)["passed"]
    for key, value in (
            ("spearman", 0.989),):
        failed = {**metric, "vs_reference": {key: value}}
        assert not depth_release.gate_quality(
            "da2-metric-hypersim-small", failed, is_baseline=True)["passed"]
    metric["metric_vs_reference"]["mean_absrel"] = 0.051
    assert not depth_release.gate_quality(
        "da2-metric-hypersim-small", metric, is_baseline=True)["passed"]
    metric["metric_vs_reference"] = {
        "mean_absrel": 0.01, "median_error_percent_of_reference_median": 5.1}
    assert not depth_release.gate_quality(
        "da2-metric-hypersim-small", metric, is_baseline=True)["passed"]
    da3 = {
        "vs_reference": {"spearman": 0.99, "scale_shift_absrel": 0.05},
        "camera_vs_reference": {
            "rotation_mean_error_degrees": 1.0, "translation_mean_l2": 1e-3,
            "intrinsics_mean_relative_error": 0.01,
        },
    }
    assert depth_release.gate_quality("da3-small", da3, is_baseline=True)["passed"]
    da3["camera_vs_reference"]["intrinsics_mean_relative_error"] = 0.0101
    assert not depth_release.gate_quality("da3-small", da3, is_baseline=True)["passed"]
    for key, value in (
            ("rotation_mean_error_degrees", 1.001),
            ("translation_mean_l2", 0.00101)):
        failed = {
            **da3,
            "camera_vs_reference": {
                "rotation_mean_error_degrees": 1.0,
                "translation_mean_l2": 1e-3,
                "intrinsics_mean_relative_error": 0.01,
                key: value,
            },
        }
        assert not depth_release.gate_quality("da3-small", failed, is_baseline=True)["passed"]
    for key, value in (("spearman", 0.989), ("scale_shift_absrel", 0.051)):
        failed = {**da3, "vs_reference": {
            "spearman": 0.99, "scale_shift_absrel": 0.05, key: value}}
        assert not depth_release.gate_quality("da3-small", failed, is_baseline=True)["passed"]
    candidate = {
        "vs_f32": {"spearman": 1.0, "scale_shift_absrel": 0.0},
        "camera_vs_f32": {
            "rotation_mean_error_degrees": 0.0, "translation_mean_l2": 0.0,
            "intrinsics_mean_relative_error": 0.0,
        },
    }
    result = depth_release.gate_quality(
        "da3-small", candidate, is_baseline=False, baseline_passed=False)
    assert not result["passed"]
    assert "baseline F32 failed reference parity" in result["gate_failures"]


def test_depth_camera_comparison_ignores_zero_intrinsics():
    import numpy as np

    reference = {
        "17": {
            "extrinsics": np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
            "intrinsics": np.asarray([[500, 0, 252], [0, 500, 168], [0, 0, 1]]),
        },
    }
    estimate = {
        "17": {
            "extrinsics": np.asarray([[1, 0, 0, 1e-4], [0, 1, 0, 0], [0, 0, 1, 0]]),
            "intrinsics": np.asarray([
                [501, 1e-10, 252], [1e-10, 500, 168], [0, 0, 1]]),
        },
    }

    result = depth_release._camera_comparison(reference, estimate)
    assert result["rotation_mean_error_degrees"] == 0
    assert result["translation_mean_l2"] == pytest.approx(1e-4)
    assert result["intrinsics_mean_relative_error"] < 0.01


def test_depth_build_flag_is_read_from_runtime_snapshot():
    plan = depth_release.plan_release(
        "depth-anything.cpp:da3-small", task="depth-estimation")
    snapshot = {"build_fingerprint": {"flags": ["-DDA_HAS_AVX512F=OFF"]}}

    assert depth_release._avx_choice(SimpleNamespace(snapshot=lambda: snapshot)) == "OFF"
    assert depth_release.runtime_card(plan, snapshot, {})["avx512f_build_flag"] == "OFF"


def test_depth_bundle_publishes_passing_precisions_atomically(tmp_path):
    model = tmp_path / "f32.gguf"
    model.write_bytes(b"gguf")
    candidate = {"precision": "F32", "path": str(model),
                 "sha256": depth_release._hash(model), "status": "passed"}
    plan = depth_release.plan_release("depth-anything.cpp:da3-small",
                                      task="depth-estimation", target_repo="org/test")

    class FakeApi:
        def __init__(self):
            self.commits = []

        def create_repo(self, *_args, **_kwargs):
            return None

        def create_commit(self, **kwargs):
            self.commits.append(kwargs)

    api = FakeApi()
    result = depth_release.publish_bundle(
        tmp_path, plan, [candidate], {"provenance": {"build_fingerprint": {
            "flags": ["-DDA_HAS_AVX512F=OFF"]}}}, {}, api=api)
    assert result["precisions"] == ["F32"]
    assert len(api.commits) == 1
    manifest = json.loads((tmp_path / "publication" / "manifest.json").read_text())
    assert manifest["card"]["precision_producers"]["F16"] == "native-quantizer"
    assert not depth_release.verify_manifest(manifest, tmp_path / "publication")


def test_depth_release_rejects_only_failed_precisions_and_publishes_passes(
        tmp_path, monkeypatch):
    from agentquantix import detection

    plan = depth_release.plan_release(
        "depth-anything.cpp:da3-small", task="depth-estimation")
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"ok")
    plan["source"]["sha256"] = depth_release._hash(source)
    plan["source"]["size"] = source.stat().st_size
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    manifest = {"images": [{"id": 17, "local_path": str(image)}]}
    runtime = SimpleNamespace(
        revision=depth_release.RUNTIME_REVISION, source=runtime_dir,
        binaries={"da3-cli": tmp_path / "da3-cli"},
        snapshot=lambda: {"provenance": {"build_fingerprint": {
            "flags": ["-DDA_HAS_AVX512F=OFF"]}}})
    monkeypatch.setattr(depth_release, "work_dir",
                        lambda *_args: tmp_path / "aqx-vision-runs" / "depth-test")
    monkeypatch.setattr(depth_release, "_checkout_reference", lambda *_args: tmp_path)
    monkeypatch.setattr(vision, "converter_venv", lambda *_args: "python")
    monkeypatch.setattr(depth_release, "_requirements_provenance",
                        lambda _python: {"python": "python", "packages": []})
    monkeypatch.setattr(detection, "download_coco_images", lambda _cache: manifest)
    monkeypatch.setattr(depth_release, "_load_reference",
                        lambda _path: ({}, {"17": {"model_shape": [2, 2]}}))

    def timed(command, log_path, **_kwargs):
        command = list(map(str, command))
        if "depth_reference.py" in command[1]:
            Path(command[-1]).write_bytes(b"reference")
        if "depth" in command:
            Path(command[command.index("--pfm") + 1]).write_bytes(b"pfm")
        return {"wall_seconds": 0.1, "peak_rss_kib": 100}

    def convert(command):
        Path(command[-1]).write_bytes(b"f32")

    def quantize(command):
        Path(command[-2]).write_bytes(command[-1].encode())

    statuses = {"F32": "passed", "F16": "passed",
                "Q8_0": "quality-rejected", "Q4_K": "quality-rejected"}

    def evaluate(_plan, _family, precision, *_args):
        return {"passed": statuses[precision] == "passed",
                "gate_failures": [] if statuses[precision] == "passed" else ["gate"]}

    monkeypatch.setattr(depth_release, "_timed_run", timed)
    published = {}
    monkeypatch.setattr(depth_release, "publish_bundle",
                        lambda _root, _plan, candidates, *_args:
                        published.update(precisions=[
                            item["precision"] for item in candidates
                            if item["status"] == "passed"]) or
                        {"precisions": published["precisions"]})
    result = depth_release.run_release(
        plan, user_approved=True, publish=True, runtime=runtime,
        download_source=lambda _source: (tmp_path, source),
        evaluate=evaluate, convert=convert, quantize=quantize)
    assert result["status"] == "passed"
    assert result["reference_timing"] == {
        "wall_seconds": 0.1, "peak_rss_kib": 100}
    assert result["plan"]["baseline_failed"] is False
    assert result["evaluation_images"] == manifest["images"]
    assert [item["status"] for item in result["candidates"]] == [
        "passed", "passed", "quality-rejected", "quality-rejected"]
    assert published["precisions"] == ["F32", "F16"]


def test_sam3_blocker_is_exact_and_depth_actions_dispatch(monkeypatch):
    blocked = route("unknown/model", task="mask-generation")
    assert blocked.blocker == (
        "official facebook/sam3 weights are gated under the SAM License; no ungated "
        "source with verifiable conversion provenance; sam3cpp not accepted")
    seen = {}
    monkeypatch.setattr(depth_release, "plan_release",
                        lambda *args, **kwargs: seen.update(kwargs) or {"engine": "depth-anything.cpp"})
    assert actions.vision_plan(
        "depth-anything.cpp:da3-small", task="depth-estimation")["engine"] == "depth-anything.cpp"


def test_depth_cli_and_agent_tool_dispatch(monkeypatch):
    from agentquantix import cli
    from agentquantix.agent import tools

    route_value = SimpleNamespace(
        repo_id="depth-anything.cpp:da3-small", task="depth-estimation",
        family="da3-small")
    plan = {"engine": "depth-anything.cpp", "task": "depth-estimation",
            "model": "depth-anything.cpp:da3-small", "family": "da3-small",
            "source": {"sha256": "a" * 64}, "status": "ready", "blockers": []}
    calls = []
    monkeypatch.setattr(cli, "_confirm", lambda *_args: True)
    monkeypatch.setattr(cli, "_json", lambda value: calls.append(value))
    monkeypatch.setattr(actions, "vision_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(actions, "start_vision",
                        lambda *_args, **_kwargs: {
                            "run_id": "vision:depth-run",
                            "candidates": [{"status": "passed"}],
                        })
    args = SimpleNamespace(output=None, dry_run=False, yes=True, publish=False,
                           target=None, quant=None)
    assert cli._run_vision(args, route=route_value) == 0
    assert calls[-1]["run_id"] == "vision:depth-run"
    result = tools._start({"user_approved": True, "plan": plan})
    assert result["run_id"] == "vision:depth-run"
