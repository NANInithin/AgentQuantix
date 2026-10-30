"""URL-pinned detector conversion, evaluation and atomic bundle publication."""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

from .. import config, detection, vision
from ..package_data import package_path
from ..runtimes.spec import RUNTIMES


FIXTURES = package_path("fixtures/detection")


class DetectionValidationError(RuntimeError):
    pass


def _require_coco_metrics():
    try:
        for module in ("numpy", "PIL.Image", "pycocotools.mask",
                       "pycocotools.coco", "pycocotools.cocoeval"):
            importlib.import_module(module)
    except ImportError as error:
        raise DetectionValidationError(
            "COCO evaluation requires Pillow, NumPy and pycocotools; "
            'install them with `pip install "agentquantix[detection]"`') from error


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    pending.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def work_dir(model, family, source_sha=""):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")
    suffix = f"-{source_sha[:12]}" if source_sha else ""
    return config.TEMP_DIR / "aqx-detection-runs" / f"{name}-{family}{suffix}"


def plan_release(model, *, task, family=None, engine=None, target_repo=None,
                 runtime_revision=None, quants=None):
    matched = detection.family_for(model, task, family=family, engine=engine)
    if not matched:
        raise DetectionValidationError(f"no detection family matches {model} for {task}")
    engine = matched["engine"]
    source = detection.source_asset(matched)
    if runtime_revision:
        try:
            detection.guard_runtime(engine, runtime_revision)
        except ValueError as error:
            raise DetectionValidationError(str(error)) from error
    supported = list(detection.precision_types(engine))
    requested = [str(item).casefold() for item in quants or supported]
    if requested != [item.casefold() for item in supported]:
        raise DetectionValidationError("all pinned precisions are evaluated for detection releases")
    runtime_spec = RUNTIMES[engine]
    return {
        "pipeline": "vision",
        "engine": engine,
        "task": task,
        "model": model,
        "family": matched["family"],
        "family_target": detection.family_target(matched),
        "source": source,
        "source_files": matched["source_files"],
        "source_license": source["licence"],
        "engine_license": "AGPL-3.0" if engine == "ultralytics-ggml" else "Apache-2.0",
        "runtime_revision": detection.RUNTIME_REVISIONS[engine],
        "runtime_repository": runtime_spec.repository,
        "converter": (
            "cpp_ggml/scripts/convert_yolo_to_gguf.py"
            if engine == "ultralytics-ggml" else "scripts/convert_rfdetr_to_gguf.py"
        ),
        "quants": [item.upper() for item in supported],
        "target_repo": target_repo or (
            f"{config.namespace()}/{matched['family']}-GGUF"),
        "status": "ready",
        "blockers": [],
        "quality_profile": "detection-0.5.7-d25-fixed-coco-sample",
        "evaluation_version": detection.EVALUATION_VERSION,
    }


def converter_command(python, converter, source, output, engine, precision, family):
    if engine == "ultralytics-ggml":
        return [str(python), str(converter), "--model", str(source),
                "--dtype", precision, "--output", str(output)]
    return [str(python), str(converter), "--variant", family["converter_arch"],
            "--dtype", precision, "--output", str(output),
            "--checkpoint", str(source)]


def native_command(binary, family, model, image, output, *, masks=None):
    if family["engine"] == "ultralytics-ggml":
        args = [str(binary), family["cli_command"], "--model", str(model),
                "--source", str(image)]
        if family["task"] == "image-classification":
            return [*args, "--topk", "5", "--threads", "4", "--raw", str(output)]
        return [*args, "--conf", "0.001", "--iou", "0.7", "--max-det", "300",
                "--threads", "4", "--dets-json", str(output)]
    args = [str(binary), "detect", "--model", str(model), "--input", str(image),
            "--threshold", "0.001", "--topk", "300", "--threads", "4",
            "--output", str(output)]
    if masks:
        args.extend(["--masks", str(masks)])
    return args


def _requirements_provenance(python):
    result = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True,
                            text=True, check=True)
    version = subprocess.run(
        [str(python), "-c", "import sys; print(sys.version.split()[0])"],
        capture_output=True, text=True, check=True).stdout.strip()
    return {"python": str(python), "python_version": version,
            "packages": sorted(result.stdout.splitlines())}


def _runtime_card(plan, provenance, labels):
    return {
        "task": plan["task"],
        "family": plan["family"],
        "weights_license": plan["source_license"],
        "engine_license": plan["engine_license"],
        "source_url": plan["source"]["url"],
        "source_sha256": plan["source"]["sha256"],
        "evaluation": "evaluated on 10 COCO val2017 images; small sample",
        "d25_protocol": (
            "Per image, filter reference and native detections at score >= 0.25; "
            "greedily match class-agnostic box IoU >= 0.5 in descending IoU order."
        ),
        "baseline_gates": {
            "ref_recall": ">= 0.95",
            "native_precision": ">= 0.95",
            "class_agreement": ">= 0.98",
            "mean_box_iou": ">= 0.90",
            "mAP50_delta": "<= 0.02",
            "segmentation_mask_iou": ">= 0.85",
            "pose_keypoint_error_px": "<= 2",
        },
        "lower_precision_gates": "same D(0.25) bars vs F32; mAP50 within 0.03 of F32",
        "runtime_revision": plan["runtime_revision"],
        "converter_provenance": provenance,
        "labels": labels,
        "labels_source": "GGUF metadata",
    }


def run_release(plan, *, user_approved, publish=False, runtime=None,
                download=None, convert=None, evaluate=None):
    if user_approved is not True:
        raise DetectionValidationError("start_release requires approval of the exact detection plan")
    if plan.get("status") != "ready" or plan.get("blockers"):
        raise DetectionValidationError("detection plan is blocked: " +
                                       "; ".join(plan.get("blockers", [])))
    if plan["task"] != "image-classification":
        _require_coco_metrics()
    engine = plan["engine"]
    if runtime is None:
        from ..runtime_control import ensure_runtime
        revision = {"revision": plan["runtime_revision"],
                    "ref": plan["runtime_revision"], "channel": "pinned",
                    "repository": plan["runtime_repository"]}
        runtime = ensure_runtime(engine, revision=revision,
                                 required_targets=("yolo-cli" if engine == "ultralytics-ggml"
                                                   else "rfdetr-cli",))
    try:
        detection.guard_runtime(engine, runtime.revision)
    except ValueError as error:
        raise DetectionValidationError(str(error)) from error
    family = vision.family_for(plan["model"], family=plan["family"], task=plan["task"])
    if family is None:
        raise DetectionValidationError("saved family no longer matches the catalog")
    root = work_dir(plan["model"], plan["family"], plan["source"]["sha256"])
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    run_path = root / "run.json"
    progress = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    result_saved = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    saved = result_saved if result_saved.get("candidates") else progress
    old = saved.get("plan") or {}
    runtime_snapshot = runtime.snapshot()
    saved_runtime_identity = (saved.get("runtime_snapshot") or {}).get("identity")
    same_run = all(old.get(key) == plan.get(key) for key in (
        "model", "task", "family", "source", "quants", "runtime_revision",
        "evaluation_version")) and saved_runtime_identity == runtime_snapshot.get("identity")
    candidates = []
    if same_run:
        candidates = [
            item for item in saved.get("candidates", [])
            if item.get("status") != "conversion-or-evaluation-failed"
            and Path(item.get("path", "")).is_file()
            and item.get("sha256") == _hash(item["path"])]
    if (result_saved and same_run and len(candidates) == len(result_saved.get("candidates", []))
            and all(item.casefold() in {
                saved_item["precision"].casefold() for saved_item in candidates}
                    for item in plan["quants"])):
        if publish and not result_saved.get("published"):
            result_saved["publication"] = publish_bundle(
                root, old, result_saved["candidates"], provenance=result_saved.get("converter"))
            result_saved["published"] = True
            _write(result_path, result_saved)
        return result_saved
    plan["baseline_failed"] = False
    plan["baseline_failures"] = []
    run_created = progress.get("created", time.time())
    _write(run_path, {"created": run_created, "plan": plan,
                      "runtime_snapshot": runtime_snapshot, "candidates": candidates})
    if download:
        source_path = Path(download(family, config.TEMP_DIR))
    else:
        source_path = detection.download_source(family, config.TEMP_DIR)
    source_path = detection.verify_source(source_path, plan["source"])
    if source_path.suffix.casefold() in (".pt", ".pth") and not source_path.is_file():
        raise DetectionValidationError("pickle-format source failed SHA verification")
    _write(root / "source.json", {"path": str(source_path),
                                  "sha256": plan["source"]["sha256"]})
    python = vision.converter_venv(runtime.source, detection.requirements(engine))
    provenance = _requirements_provenance(python)
    coco_manifest = detection.download_coco_images(config.TEMP_DIR)
    image_manifest_path = root / "images.json"
    _write(image_manifest_path, coco_manifest)
    converter = Path(runtime.source) / plan["converter"]
    if not converter.is_file():
        raise DetectionValidationError(f"pinned converter not found: {converter}")
    binary_name = "yolo-cli" if engine == "ultralytics-ggml" else "rfdetr-cli"
    binary = runtime.binaries[binary_name]
    baseline = next((item.get("quality") for item in candidates
                     if item["precision"].casefold() == "f32"), None)
    baseline_failed = bool(baseline is not None and not baseline.get("passed"))
    baseline_failures = list(baseline.get("gate_failures", [])) if baseline else []
    plan["baseline_failed"] = baseline_failed
    plan["baseline_failures"] = baseline_failures
    _write(run_path, {"created": run_created, "plan": plan,
                      "runtime_snapshot": runtime_snapshot, "candidates": candidates})
    for precision in (item.casefold() for item in plan["quants"]):
        previous = next((item for item in candidates
                         if item["precision"].casefold() == precision), None)
        if previous and previous.get("status") != "conversion-or-evaluation-failed":
            continue
        if previous:
            candidates.remove(previous)
        model_path = root / "models" / f"{precision}.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        command = converter_command(python, converter, source_path, model_path,
                                    engine, precision, family)
        if "--trust-checkpoint" in command:
            raise DetectionValidationError("checkpoint trust flags are prohibited")
        try:
            if not model_path.is_file() or model_path.stat().st_size == 0:
                if convert:
                    started = time.monotonic()
                    convert(command)
                    conversion_run = {"wall_seconds": time.monotonic() - started,
                                      "peak_rss_kib": None, "argv": command}
                else:
                    conversion_run = _timed_run(
                        command, root / "logs" / f"convert-{precision}.log",
                        timeout=7200)
            else:
                conversion_run = {"wall_seconds": 0.0, "peak_rss_kib": None,
                                  "argv": command, "reused": True}
            if not model_path.is_file() or model_path.stat().st_size == 0:
                raise DetectionValidationError("converter did not produce a GGUF")
            quality = (evaluate or _evaluate_candidate)(
                plan, family, model_path, binary, root, baseline, python)
            quality["conversion_run"] = conversion_run
            if baseline is None:
                baseline = quality
                baseline_failed = not bool(quality.get("passed"))
                baseline_failures = list(quality.get("gate_failures", []))
                plan["baseline_failed"] = baseline_failed
                plan["baseline_failures"] = baseline_failures
            passed = bool(quality.get("passed"))
            status = "quality-rejected" if baseline_failed else (
                "passed" if passed else "quality-rejected")
            candidates.append({"precision": precision, "path": str(model_path),
                               "sha256": _hash(model_path), "bytes": model_path.stat().st_size,
                               "quality": {key: value for key, value in quality.items()
                                           if key not in ("outputs", "predictions")},
                               "status": status})
        except Exception as error:
            candidates.append({"precision": precision, "path": str(model_path),
                               "status": ("quality-rejected" if baseline_failed else
                                          "conversion-or-evaluation-failed"),
                               "error": str(error)})
            if precision == "f32" and baseline is None:
                baseline_failed = True
                baseline_failures = ["F32 baseline conversion or evaluation failed"]
                plan["baseline_failed"] = True
                plan["baseline_failures"] = baseline_failures
        if baseline_failed:
            for item in candidates:
                item["status"] = "quality-rejected"
        _write(run_path, {"created": run_created, "plan": plan,
                          "runtime_snapshot": runtime_snapshot, "candidates": candidates})
        _write(result_path, {
            "pipeline": "vision", "engine": engine, "run_id": root.name,
            "plan": plan, "runtime_snapshot": runtime.snapshot(),
            "converter": provenance, "candidates": candidates, "published": False,
            "baseline_failed": baseline_failed, "baseline_failures": baseline_failures,
            **({"status": "rejected"} if not any(
                item["status"] == "passed" for item in candidates) else {}),
        })
    result = {
        "pipeline": "vision",
        "engine": engine,
        "run_id": root.name,
        "plan": plan,
        "runtime_snapshot": runtime.snapshot(),
        "converter": provenance,
        "candidates": candidates,
        "published": False,
        "baseline_failed": baseline_failed,
        "baseline_failures": baseline_failures,
    }
    if not any(item["status"] == "passed" for item in candidates):
        result["status"] = "rejected"
    _write(result_path, result)
    if publish and any(item["status"] == "passed" for item in candidates):
        result["publication"] = publish_bundle(root, plan, candidates, provenance=provenance)
        result["published"] = True
        _write(result_path, result)
    return result


def _timed_run(command, log_path, *, timeout=3600):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    time_output = Path(log_path).with_suffix(".time")
    timed = ["/usr/bin/time", "-v", "-o", str(time_output), *map(str, command)]
    if not Path("/usr/bin/time").is_file():
        timed = list(map(str, command))
    started = time.monotonic()
    completed = subprocess.run(timed, capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - started
    Path(log_path).write_text(
        f"argv: {json.dumps(list(map(str, command)))}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n",
        encoding="utf-8")
    if completed.returncode:
        raise DetectionValidationError(
            f"command failed ({completed.returncode}); log: {log_path}: "
            f"{completed.stderr.strip()[-400:]}")
    peak_rss = None
    if time_output.is_file():
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                          time_output.read_text(encoding="utf-8", errors="replace"))
        if match:
            peak_rss = int(match.group(1))
    return {"wall_seconds": elapsed, "peak_rss_kib": peak_rss,
            "argv": list(map(str, command)), "log": str(log_path)}


def _normalize_reference(record):
    boxes = record.get("boxes") or []
    scores = record.get("scores") or []
    classes = record.get("class_ids") or []
    detections = []
    for index, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        item = {"box": [float(value) for value in box], "score": float(score),
                "class_id": int(class_id)}
        if index < len(record.get("mask_paths", [])):
            item["mask"] = record["mask_paths"][index]
        if index < len(record.get("keypoints", [])):
            item["keypoints"] = record["keypoints"][index]
            item["keypoint_scores"] = record.get("keypoint_scores", [])[index]
        detections.append(item)
    return detections


def _normalize_native(path, engine, task, masks_dir=None):
    if task == "image-classification":
        raw = Path(path).read_bytes()
        if raw[:8] != b"YCLS0001" or len(raw) < 12:
            raise DetectionValidationError(f"invalid YOLO classification payload: {path}")
        count = struct.unpack_from("<i", raw, 8)[0]
        if count < 1 or len(raw) != 12 + count * 4:
            raise DetectionValidationError(f"invalid YOLO classification length: {path}")
        values = list(struct.unpack_from(f"<{count}f", raw, 12))
        order = sorted(range(count), key=lambda index: (-values[index], index))
        return {"probabilities": values, "top1": order[0], "top5": order[:5]}
    output = json.loads(Path(path).read_text(encoding="utf-8"))
    predictions = []
    raw_detections = output.get("detections", [])
    for index, item in enumerate(raw_detections):
        if "xyxy" in item:
            box, score, class_id = item["xyxy"], item["conf"], item["cls"]
        else:
            box, score, class_id = item["bbox"], item["score"], item["class_id"]
        prediction = {"box": [float(value) for value in box], "score": float(score),
                      "class_id": int(class_id)}
        if "kpts" in item:
            values = item["kpts"]
            prediction["keypoints"] = [values[pos:pos + 2] for pos in range(0, len(values), 3)]
            prediction["keypoint_scores"] = [values[pos + 2] for pos in range(0, len(values), 3)]
        if masks_dir:
            matches = sorted(Path(masks_dir).glob(f"det_{index:03d}_*.png"))
            if matches:
                prediction["mask"] = str(matches[0])
        predictions.append(prediction)
    return predictions


def _mask_iou(left, right):
    import numpy as np
    from PIL import Image
    def read(value):
        if isinstance(value, str) and value.endswith(".npy"):
            return np.load(value).astype(bool)
        if isinstance(value, str):
            return np.asarray(Image.open(value).convert("L")) > 0
        return np.asarray(value).astype(bool)
    first, second = read(left), read(right)
    if first.shape != second.shape:
        second = np.asarray(Image.fromarray(second.astype("uint8")).resize(
            (first.shape[1], first.shape[0]), Image.Resampling.NEAREST)).astype(bool)
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 1.0


def _keypoint_error(pairs):
    errors = []
    for reference, native, _ in pairs:
        for ref_point, native_point, ref_score, native_score in zip(
                reference.get("keypoints", []), native.get("keypoints", []),
                reference.get("keypoint_scores", []), native.get("keypoint_scores", [])):
            if float(ref_score) >= 0.5 and float(native_score) >= 0.5:
                errors.append(((float(ref_point[0]) - float(native_point[0])) ** 2 +
                               (float(ref_point[1]) - float(native_point[1])) ** 2) ** 0.5)
    return sum(errors) / len(errors) if errors else float("inf")


def _evaluate_candidate(plan, family, model_path, binary, root, baseline, python):
    images_path = root / "images.json"
    manifest = json.loads(images_path.read_text(encoding="utf-8"))
    reference_path = root / "reference.json"
    reference_masks = root / "reference-masks"
    reference_timing = None
    if not reference_path.is_file():
        source_info = json.loads((root / "source.json").read_text(encoding="utf-8"))
        reference_script = Path(__file__).with_name("detection_reference.py")
        reference_timing = _timed_run(
            [python, reference_script, "--engine", plan["engine"], "--task", plan["task"],
             "--weights", source_info["path"],
             "--images", images_path, "--output", reference_path,
             "--mask-dir", reference_masks],
            root / "logs" / "reference.log")
    reference_records = json.loads(reference_path.read_text(encoding="utf-8"))["images"]
    reference_by_id = {item["image_id"]: item for item in reference_records}
    native_by_id = {}
    timings = []
    for image in manifest["images"]:
        image_id = image["id"]
        output = root / "native" / model_path.stem / f"{image_id}.bin"
        masks_dir = (root / "native" / model_path.stem / f"{image_id}-masks"
                     if plan["task"] == "image-segmentation" else None)
        output.parent.mkdir(parents=True, exist_ok=True)
        command = native_command(binary, family, model_path, image["local_path"], output,
                                 masks=masks_dir)
        timings.append(_timed_run(command, root / "logs" / model_path.stem /
                                  f"{image_id}.log"))
        native_by_id[image_id] = _normalize_native(
            output, plan["engine"], plan["task"], masks_dir)
    if plan["task"] == "image-classification":
        ref_records = [item["probabilities"] for item in reference_records]
        native_records = [native_by_id[item["id"]]["probabilities"] for item in manifest["images"]]
        top1, overlap, error = [], [], []
        for ref, native in zip(ref_records, native_records):
            ref_order = sorted(range(len(ref)), key=lambda index: (-ref[index], index))
            native_order = sorted(range(len(native)), key=lambda index: (-native[index], index))
            top1.append(int(ref_order[0] == native_order[0]))
            overlap.append(len(set(ref_order[:5]) & set(native_order[:5])) / 5)
            error.append(sum(abs(float(a) - float(b)) for a, b in zip(ref, native)) / len(ref))
        metrics = {"top1_agreement": sum(top1) / len(top1),
                   "mean_top5_overlap": sum(overlap) / len(overlap),
                   "probability_mae": sum(error) / len(error),
                   "reference_run": reference_timing, "native_runs": timings}
        if baseline is not None:
            f32_top1, f32_overlap = [], []
            f32_error = []
            for image in manifest["images"]:
                f32_path = root / "native" / "f32" / f"{image['id']}.bin"
                f32 = _normalize_native(f32_path, plan["engine"], plan["task"])
                native = native_by_id[image["id"]]
                f32_top1.append(int(native["top1"] == f32["top1"]))
                f32_overlap.append(len(set(native["top5"]) & set(f32["top5"])) / 5)
                f32_error.append(sum(abs(float(a) - float(b)) for a, b in zip(
                    native["probabilities"], f32["probabilities"])) / len(f32["probabilities"]))
            metrics.update(f32_top1_agreement=sum(f32_top1) / len(f32_top1),
                           f32_mean_top5_overlap=sum(f32_overlap) / len(f32_overlap),
                           f32_probability_mae=sum(f32_error) / len(f32_error),
                           f32=baseline)
        gate = detection.classification_gate(
            metrics, baseline_passed=baseline.get("passed") if baseline is not None else None)
        metrics["gate_failures"] = gate["failures"]
        metrics["passed"] = gate["passed"]
        return metrics

    per_image = []
    keypoint_errors = []
    mask_ious = []
    for image in manifest["images"]:
        ref = _normalize_reference(reference_by_id[image["id"]])
        native = native_by_id[image["id"]]
        per_image.append((ref, native))
        matches = detection.match_detections(ref, native)["matches"]
        if plan["task"] == "keypoint-detection":
            keypoint_errors.extend(
                [((float(rp[0]) - float(np[0])) ** 2 +
                  (float(rp[1]) - float(np[1])) ** 2) ** 0.5
                 for r, n, _ in matches
                 for rp, np, rs, ns in zip(
                     r.get("keypoints", []), n.get("keypoints", []),
                     r.get("keypoint_scores", []), n.get("keypoint_scores", []))
                 if float(rs) >= 0.5 and float(ns) >= 0.5])
        if plan["task"] == "image-segmentation":
            mask_ious.extend([
                _mask_iou(r["mask"], n["mask"]) if "mask" in r and "mask" in n else 0.0
                for r, n, _ in matches])
    d25 = detection.aggregate_d25(per_image)
    annotations_path = FIXTURES / "coco_val2017_annotations.json"
    native_map = _coco_map50(native_by_id, manifest, annotations_path, plan["task"])
    reference_map = _coco_map50(
        {item["image_id"]: _normalize_reference(item) for item in reference_records},
        manifest, annotations_path, plan["task"])
    metrics = {**d25, "mean_mask_iou": (
        sum(mask_ious) / len(mask_ious) if mask_ious else 0.0),
        "keypoint_error_px": (
            sum(keypoint_errors) / len(keypoint_errors) if keypoint_errors else float("inf")),
        "reference_run": reference_timing, "native_runs": timings,
        "coco": {"map_50": native_map}, "reference": {"coco": {"map_50": reference_map}},
    }
    if baseline is not None:
        metrics["f32"] = baseline
    if plan["task"] == "image-segmentation":
        metrics["mask"] = {
            "map_50": _coco_map50(native_by_id, manifest, annotations_path,
                                  plan["task"], segmentation=True),
            "reference": {"map_50": _coco_map50(
                {item["image_id"]: _normalize_reference(item) for item in reference_records},
                manifest, annotations_path, plan["task"], segmentation=True)},
        }
    if baseline is not None:
        f32_by_id = {}
        f32_masks_by_id = {}
        for image in manifest["images"]:
            image_id = image["id"]
            f32_path = root / "native" / "f32" / f"{image_id}.bin"
            f32_masks = root / "native" / "f32" / f"{image_id}-masks"
            f32_by_id[image_id] = _normalize_native(
                f32_path, plan["engine"], plan["task"], f32_masks)
        f32_pairs = [(f32_by_id[item["id"]], native_by_id[item["id"]])
                     for item in manifest["images"]]
        d25_vs_f32 = detection.aggregate_d25(f32_pairs)
        if plan["task"] == "image-segmentation":
            f32_mask_ious = []
            for image in manifest["images"]:
                matches = detection.match_detections(
                    f32_by_id[image["id"]], native_by_id[image["id"]])["matches"]
                f32_mask_ious.extend([
                    _mask_iou(left["mask"], right["mask"])
                    if "mask" in left and "mask" in right else 0.0
                    for left, right, _ in matches])
            d25_vs_f32["mean_mask_iou"] = (
                sum(f32_mask_ious) / len(f32_mask_ious) if f32_mask_ious else 0.0)
        if plan["task"] == "keypoint-detection":
            d25_vs_f32["keypoint_error_px"] = _keypoint_error(
                [match for image in manifest["images"]
                 for match in detection.match_detections(
                     f32_by_id[image["id"]], native_by_id[image["id"]])["matches"]])
        metrics["d25_vs_f32"] = d25_vs_f32
        metrics["f32"] = baseline
    gate = detection.detection_gate(
        metrics, baseline=baseline is None,
        baseline_passed=baseline.get("passed") if baseline is not None else None,
        segmentation=plan["task"] == "image-segmentation",
        pose=plan["task"] == "keypoint-detection")
    metrics["gate_failures"] = gate["failures"]
    metrics["passed"] = gate["passed"]
    return metrics


def _coco_map50(predictions, manifest, annotations_path, task, *, segmentation=False):
    import contextlib
    import io
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils
    from PIL import Image

    coco_gt = COCO(str(annotations_path))
    category_ids = {item["id"] for item in coco_gt.dataset["categories"]}
    rows = []
    for image in manifest["images"]:
        for item in predictions.get(image["id"], []):
            class_id = 1 if task == "keypoint-detection" else int(item["class_id"])
            if class_id not in category_ids:
                continue
            x1, y1, x2, y2 = item["box"]
            row = {"image_id": image["id"], "category_id": class_id,
                   "bbox": [x1, y1, x2 - x1, y2 - y1], "score": item["score"]}
            if segmentation:
                if "mask" not in item:
                    continue
                if item["mask"].endswith(".npy"):
                    mask = np.load(item["mask"]).astype(np.uint8)
                else:
                    mask = (np.asarray(Image.open(item["mask"]).convert("L")) > 0).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("ascii")
                row["segmentation"] = rle
            rows.append(row)
    if not rows:
        return 0.0
    result = coco_gt.loadRes(rows)
    evaluator = COCOeval(coco_gt, result, "segm" if segmentation else "bbox")
    evaluator.params.imgIds = [item["id"] for item in manifest["images"]]
    evaluator.params.maxDets = [1, 10, 300]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return float(evaluator.stats[1])


def publish_bundle(root, plan, candidates, api=None, provenance=None):
    from huggingface_hub import CommitOperationAdd, HfApi
    api = api or HfApi(token=config.TOKEN)
    passing = [item for item in candidates if item["status"] == "passed"]
    if not passing:
        raise DetectionValidationError("no passing precision is publishable")
    card = _runtime_card(plan, provenance, detection.gguf_labels(passing[0]["path"]))
    target = plan["target_repo"]
    artifact_root = Path(root) / "publication"
    operations = []
    manifest_candidates = []
    for item in passing:
        repo_path = f"{item['precision']}/{Path(item['path']).name}"
        local_path = artifact_root / repo_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["path"], local_path)
        operations.append(CommitOperationAdd(path_in_repo=repo_path,
                                             path_or_fileobj=str(local_path)))
        manifest_candidates.append({**item, "path": repo_path})
    card_path = Path(root) / "README.md"
    card_path.write_text("# AgentQuantix detection bundle\n\n" +
                         json.dumps(card, indent=2) + "\n", encoding="utf-8")
    operations.append(CommitOperationAdd(path_in_repo="README.md",
                                         path_or_fileobj=str(card_path)))
    manifest = {"schema_version": 1, "pipeline": "vision", "engine": plan["engine"],
                "task": plan["task"], "family": plan["family"], "target_repo": target,
                "card": card, "candidates": manifest_candidates}
    manifest_path = artifact_root / "manifest.json"
    _write(manifest_path, manifest)
    operations.append(CommitOperationAdd(path_in_repo="manifest.json",
                                         path_or_fileobj=str(manifest_path)))
    api.create_repo(target, repo_type="model", exist_ok=True)
    api.create_commit(repo_id=target, repo_type="model", operations=operations,
                      commit_message="Add accepted detection precision bundle")
    return {"target_repo": target, "precisions": [item["precision"] for item in passing]}


def publish_run(run_id, precision, root=None):
    if root is None:
        record_root = config.TEMP_DIR / "aqx-detection-runs"
        matches = list(record_root.glob(f"*-{run_id}"))
        if len(matches) != 1:
            raise DetectionValidationError(f"cannot resolve detection run {run_id}")
        root = matches[0]
    root = Path(root)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    selected = next((item for item in result["candidates"]
                     if item["precision"].casefold() == precision.casefold()), None)
    if not selected or selected["status"] != "passed":
        raise DetectionValidationError(f"{precision} is not an accepted detection precision")
    publication = publish_bundle(root, result["plan"], result["candidates"],
                                 provenance=result.get("converter"))
    result.update(published=True, publication=publication)
    _write(root / "result.json", result)
    return {"run_id": run_id, **publication, "published": True}


def verify_manifest(manifest, root):
    root = Path(root)
    problems = []
    for item in manifest.get("candidates", []):
        path = root / item["path"]
        if not path.is_file():
            problems.append(f"missing candidate: {item['path']}")
        elif _hash(path) != item["sha256"]:
            problems.append(f"hash mismatch: {item['path']}")
    return problems
