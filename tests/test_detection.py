import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentquantix import detection, vision
from agentquantix.actions import route_named, vision_plan
from agentquantix.pipeline import detection_release as release
from agentquantix.tasks.router import route


def _asset(data=b"checkpoint"):
    return {"kind": "url", "url": "https://example.test/model.pt",
            "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
            "licence": "AGPL-3.0"}


def test_catalog_sources_and_engine_family_routing():
    catalog = vision.families()
    assert all(item.get("engine") for item in catalog)
    for family_name, task, engine in (
            ("rfdetr-nano", "object-detection", "rf-detr.cpp"),
            ("rfdetr-seg-nano", "image-segmentation", "rf-detr.cpp"),
            ("yolo26n-pose", "keypoint-detection", "ultralytics-ggml"),
            ("yolo26n-cls", "image-classification", "ultralytics-ggml")):
        family = vision.family_for("unknown", family=family_name, engine=engine, task=task)
        assert family["engine"] == engine
        assert detection.family_target(family) == f"{engine}:{family_name}"
        assert detection.source_asset(family)["kind"] == "url"
    assert vision.family_for("ultralytics/assets/yolo26n-cls", task="image-classification")
    assert vision.family_for(
        "unknown", family="rf-detr.cpp:rfdetr-nano", task="object-detection")["engine"] == "rf-detr.cpp"


def test_url_source_download_verifies_size_and_sha(tmp_path):
    data = b"verified model"
    asset = _asset(data)
    path = tmp_path / "model.pt"
    path.write_bytes(data)
    assert detection.verify_source(path, asset) == path
    bad = _asset(data + b"x")
    with pytest.raises(ValueError, match="integrity"):
        detection.verify_source(path, bad)

    family = {"family": "test", "source": asset, "source_files": ["model.pt"]}
    response = io.BytesIO(data)
    downloaded = detection.download_source(family, tmp_path, opener=lambda *_args, **_kwargs: response)
    assert downloaded.read_bytes() == data


def test_source_mismatch_blocks_pickle_conversion(monkeypatch, tmp_path):
    family = vision.family_for("unknown", family="yolo26n-cls", task="image-classification")
    plan = release.plan_release("ultralytics/assets/yolo26n-cls", task="image-classification",
                               family=family["family"])
    monkeypatch.setattr(release, "work_dir", lambda *_args: tmp_path / "run")
    monkeypatch.setattr(detection, "download_source", lambda *_args: tmp_path / "bad.pt")
    (tmp_path / "bad.pt").write_bytes(b"bad")
    monkeypatch.setattr(detection, "verify_source",
                        lambda *_args: (_ for _ in ()).throw(ValueError("hash mismatch")))
    runtime_source = tmp_path / "runtime"
    runtime_source.mkdir()
    runtime = SimpleNamespace(
        revision=detection.RUNTIME_REVISIONS["ultralytics-ggml"],
        source=runtime_source, binaries={"yolo-cli": tmp_path / "yolo-cli"},
        snapshot=lambda: {"revision": "pinned"})
    converted = []
    with pytest.raises(ValueError, match="hash mismatch"):
        release.run_release(plan, user_approved=True, runtime=runtime, convert=converted.append)
    assert converted == []


def test_detection_d25_empty_class_agnostic_and_tie_order():
    assert detection.match_detections([], [])["ref_recall"] == 1.0
    assert detection.match_detections([], [])["native_precision"] == 1.0
    ref = [{"box": [0, 0, 10, 10], "score": 0.9, "class_id": 1}]
    native = [{"box": [0, 0, 10, 10], "score": 0.8, "class_id": 2}]
    result = detection.match_detections(ref, native)
    assert result["matched"] == 1
    assert result["class_agreement"] == 0.0
    duplicate_refs = ref * 2
    duplicate_native = native * 2
    result = detection.match_detections(duplicate_refs, duplicate_native)
    assert [(r["class_id"], n["class_id"]) for r, n, _ in result["matches"]] == [
        (1, 2), (1, 2)]
    assert detection.aggregate_d25([(ref, native)])["matched"] == 1


def test_map_wrapper_and_detection_gate_boundaries():
    assert detection.coco_map50({"coco": {"map_50": 0.5}}) == 0.5
    assert detection.coco_map50({"mAP50": 0.4}) == 0.4
    metrics = {
        "ref_recall": 0.95, "native_precision": 0.95, "class_agreement": 0.98,
        "mean_box_iou": 0.90, "coco": {"map_50": 0.52},
        "reference": {"coco": {"map_50": 0.50}},
        "mean_mask_iou": 0.85,
        "mask": {"map_50": 0.52, "reference": {"map_50": 0.50}},
        "keypoint_error_px": 2.0,
    }
    assert detection.detection_gate(metrics, baseline=True, segmentation=True, pose=True)["passed"]
    failing = {**metrics, "native_precision": 0.949}
    assert not detection.detection_gate(failing, baseline=True)["passed"]
    lower = {
        "ref_recall": 0.8, "native_precision": 0.8, "class_agreement": 0.8,
        "mean_box_iou": 0.8, "d25_vs_f32": {
            "ref_recall": 0.95, "native_precision": 0.95, "class_agreement": 0.98,
            "mean_box_iou": 0.90, "mean_mask_iou": 0.85, "keypoint_error_px": 2.0},
        "coco": {"map_50": 0.53}, "f32": {"coco": {"map_50": 0.50}},
        "mask": {"map_50": 0.53}, "f32_mask": {"map_50": 0.50},
    }
    lower["f32"]["mask"] = {"map_50": 0.50}
    assert detection.detection_gate(
        lower, baseline=False, segmentation=True, pose=True)["passed"]
    lower["d25_vs_f32"]["class_agreement"] = 0.97
    assert not detection.detection_gate(lower, baseline=False)["passed"]


def test_classification_gate_edges():
    assert detection.classification_gate(
        {"top1_agreement": 0.95, "mean_top5_overlap": 0.90})["passed"]
    assert not detection.classification_gate(
        {"top1_agreement": 0.949, "mean_top5_overlap": 0.90})["passed"]
    assert not detection.classification_gate({
        "top1_agreement": 1.0, "mean_top5_overlap": 1.0,
        "f32": {}, "f32_top1_agreement": 0.94,
        "f32_mean_top5_overlap": 1.0})["passed"]
    assert detection.classification_gate({
        "top1_agreement": 0.80, "mean_top5_overlap": 0.80,
        "f32": {}, "f32_top1_agreement": 0.95,
        "f32_mean_top5_overlap": 0.90})["passed"]


def test_runtime_pins_guards_and_converter_argv():
    assert detection.RUNTIME_REVISIONS["ultralytics-ggml"] == (
        "59792c1d1b10a87172c1fdf6eceaa19262c1f11e")
    assert detection.RUNTIME_REVISIONS["rf-detr.cpp"] == (
        "98d0f381b832ef08a608b65c7dd78db066ed8b9a")
    detection.guard_runtime("rf-detr.cpp", detection.RUNTIME_REVISIONS["rf-detr.cpp"])
    with pytest.raises(ValueError, match="aqx runtime install rf-detr.cpp --persistent --revision"):
        detection.guard_runtime("rf-detr.cpp", "wrong")
    family = vision.family_for("unknown", family="rfdetr-nano", task="object-detection")
    assert release.converter_command(
        "python", "convert.py", "weights.pth", "out.gguf", "rf-detr.cpp", "f32", family) == [
            "python", "convert.py", "--variant", "nano", "--dtype", "f32",
            "--output", "out.gguf", "--checkpoint", "weights.pth"]
    assert "--trust-checkpoint" not in release.converter_command(
        "python", "convert.py", "weights.pth", "out.gguf", "rf-detr.cpp", "f32", family)
    assert release.native_command(
        "rfdetr-cli", family, "model.gguf", "image.jpg", "out.json") == [
            "rfdetr-cli", "detect", "--model", "model.gguf", "--input", "image.jpg",
            "--threshold", "0.001", "--topk", "300", "--threads", "4", "--output", "out.json"]


def test_no_family_route_uses_fixed_blocker():
    result = route("unlisted/model", task="object-detection", metadata={})
    assert result.blocker == (
        "ultralytics-ggml YOLO26 detection, instance/semantic segmentation, depth and "
        "YOLO-World open-vocabulary did not meet native parity gates in 0.5.7; "
        "OBB was not evaluated")


def test_plan_selects_detection_engine():
    plan = vision_plan("localai-org/rfdetr-nano", task="object-detection")
    assert plan["pipeline"] == "vision"
    assert plan["engine"] == "rf-detr.cpp"
    assert plan["family_target"] == "rf-detr.cpp:rfdetr-nano"


def test_timed_run_creates_nested_log_directory_before_time(monkeypatch, tmp_path):
    log_path = tmp_path / "logs" / "f32" / "785.log"

    def run(command, **_kwargs):
        assert log_path.parent.is_dir()
        Path(command[command.index("-o") + 1]).write_text(
            "Maximum resident set size (kbytes): 42\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(release.subprocess, "run", run)
    result = release._timed_run(["native-cli"], log_path)
    assert result["peak_rss_kib"] == 42


def test_coco_metric_dependency_error_is_actionable(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_pycoco(name, *args, **kwargs):
        if name.startswith("pycocotools"):
            raise ModuleNotFoundError("No module named 'pycocotools'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pycoco)
    with pytest.raises(release.DetectionValidationError, match=r'agentquantix\[detection\]'):
        release._require_coco_metrics()


def test_engine_family_run_targets_route_without_hub_lookup():
    for target, task, engine, family in (
            ("rf-detr.cpp:rfdetr-nano", "object-detection", "rf-detr.cpp", "rfdetr-nano"),
            ("ultralytics-ggml:yolo26n-pose", "keypoint-detection",
             "ultralytics-ggml", "yolo26n-pose")):
        result = route_named(target, task=task)
        assert result.ready
        assert result.engine == engine
        assert result.family == family


def test_partial_precision_publish_is_atomic_and_resumable(monkeypatch, tmp_path):
    source = tmp_path / "weights.pt"
    source.write_bytes(b"weights")
    runtime_source = tmp_path / "runtime"
    runtime_source.mkdir()
    converter = runtime_source / "cpp_ggml/scripts/convert_yolo_to_gguf.py"
    converter.parent.mkdir(parents=True)
    converter.write_text("", encoding="utf-8")
    runtime = SimpleNamespace(
        revision=detection.RUNTIME_REVISIONS["ultralytics-ggml"],
        source=runtime_source, binaries={"yolo-cli": tmp_path / "yolo-cli"},
        snapshot=lambda: {"revision": "pinned"})
    monkeypatch.setattr(release.config, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(detection, "download_source", lambda *_args: source)
    monkeypatch.setattr(detection, "verify_source", lambda path, _asset: path)
    monkeypatch.setattr(detection, "download_coco_images", lambda *_args: {"images": []})
    monkeypatch.setattr(vision, "converter_venv", lambda *_args: "python")
    monkeypatch.setattr(release, "_requirements_provenance", lambda *_args: {"python": "3.12"})
    monkeypatch.setattr(release, "_require_coco_metrics", lambda: None)
    monkeypatch.setattr(detection, "gguf_labels", lambda *_args: ["person"])
    plan = release.plan_release("ultralytics/assets/yolo26n-pose", task="keypoint-detection",
                               family="yolo26n-pose")
    calls = []
    def convert(command):
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(output.name.encode())
    def evaluate(_plan, _family, path, _binary, _root, _baseline, _python):
        return {"passed": path.stem != "q8_0", "coco": {"map_50": 0.5}}
    result = release.run_release(plan, user_approved=True, runtime=runtime,
                                 convert=convert, evaluate=evaluate)
    assert [item["status"] for item in result["candidates"]] == [
        "passed", "passed", "quality-rejected"]
    assert len(calls) == 3

    class FakeAPI:
        def __init__(self):
            self.commits = []
        def create_repo(self, *args, **kwargs):
            return None
        def create_commit(self, **kwargs):
            self.commits.append(kwargs)
    api = FakeAPI()
    published = release.publish_bundle(
        tmp_path / "aqx-detection-runs" /
        f"ultralytics-assets-yolo26n-pose-yolo26n-pose-{plan['source']['sha256'][:12]}",
        plan, result["candidates"], api=api, provenance={"python": "3.12"})
    assert published["precisions"] == ["f32", "f16"]
    assert len(api.commits) == 1
    assert len(api.commits[0]["operations"]) == 4
    monkeypatch.setattr(release, "publish_bundle", lambda *_args, **_kwargs: {
        "target_repo": plan["target_repo"], "precisions": ["f32", "f16"]})
    resumed = release.run_release(plan, user_approved=True, publish=True, runtime=runtime,
                                   convert=lambda *_: pytest.fail("resume reconverted"),
                                   evaluate=lambda *_: pytest.fail("resume reevaluated"))
    assert resumed["published"] is True
