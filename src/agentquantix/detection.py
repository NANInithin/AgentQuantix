import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from . import vision
from .package_data import package_path


RUNTIME_REVISIONS = {
    "ultralytics-ggml": "59792c1d1b10a87172c1fdf6eceaa19262c1f11e",
    "rf-detr.cpp": "98d0f381b832ef08a608b65c7dd78db066ed8b9a",
}

EVALUATION_VERSION = 2

ULTRALYTICS_PRECISIONS = ("f32", "f16", "q8_0")
RFDETR_PRECISIONS = ("f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0")


def precision_types(engine):
    if engine == "ultralytics-ggml":
        return ULTRALYTICS_PRECISIONS
    if engine == "rf-detr.cpp":
        return RFDETR_PRECISIONS
    raise ValueError(f"Unsupported detection engine: {engine}")


def family_target(family):
    return f"{family.get('engine', 'vision.cpp')}:{family['family']}"


def source_asset(family):
    asset = family.get("source")
    if not isinstance(asset, dict) or asset.get("kind") != "url":
        raise ValueError(f"{family['family']} must declare a URL source")
    required = ("url", "sha256", "size", "licence")
    if any(not asset.get(key) for key in required):
        raise ValueError(f"{family['family']} has an incomplete URL source")
    if not asset["url"].startswith("https://"):
        raise ValueError(f"{family['family']} source URL must use HTTPS")
    if len(asset["sha256"]) != 64 or asset["size"] <= 0:
        raise ValueError(f"{family['family']} source integrity metadata is invalid")
    return asset


def family_for(model, task, family=None, engine=None):
    return vision.family_for(model, family=family, engine=engine, task=task)


def verify_source(path, asset):
    path = Path(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if size != asset["size"] or digest.hexdigest() != asset["sha256"]:
        raise ValueError(f"Source integrity check failed for {path.name}")
    return path


def download_source(family, cache_dir, *, opener=urllib.request.urlopen):
    asset = source_asset(family)
    filename = family.get("source_filename") or family["source_files"][0]
    destination = Path(cache_dir) / "detection-sources" / asset["sha256"] / filename
    if destination.is_file():
        return verify_source(destination, asset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        with opener(asset["url"], timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        verify_source(temporary, asset)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_coco_image_manifest():
    manifest_path = package_path("fixtures/detection/coco_val2017_images.json")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def download_coco_images(cache_dir, *, opener=urllib.request.urlopen):
    manifest = load_coco_image_manifest()
    downloaded = []
    for image in manifest["images"]:
        parsed = urllib.parse.urlparse(image["url"])
        if (parsed.hostname != "images.cocodataset.org" or
                parsed.scheme not in ("http", "https")):
            raise ValueError(f"untrusted COCO image URL: {image['url']}")
        destination = Path(cache_dir) / "detection-images" / f"{image['id']}.jpg"
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
            try:
                with opener(image["url"], timeout=60) as response, temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                verify_source(temporary, {
                    "size": image["size"], "sha256": image["sha256"]})
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        verify_source(destination, {"size": image["size"], "sha256": image["sha256"]})
        downloaded.append({**image, "local_path": str(destination)})
    manifest["images"] = downloaded
    return manifest


def guard_runtime(engine, revision):
    expected = RUNTIME_REVISIONS[engine]
    if revision != expected:
        command = f"aqx runtime install {engine} --persistent --revision {expected}"
        raise ValueError(f"{engine} revision mismatch; install the pinned runtime with: {command}")


def requirements(engine):
    packages = [
        "torch==2.6.0+cpu",
        "torchvision==0.21.0+cpu",
        "transformers==5.22.0",
        "numpy==1.26.4",
        "Pillow==10.4.0",
        "pycocotools==2.0.11",
        "gguf==0.19.0",
        "safetensors==0.8.0",
        "tokenizers==0.23.3",
    ]
    if engine == "ultralytics-ggml":
        packages.append("ultralytics==8.4.121")
    elif engine == "rf-detr.cpp":
        packages.append("rfdetr==1.9.0")
    else:
        raise ValueError(f"Unsupported detection engine: {engine}")
    return tuple(packages)


def _box_iou(left, right):
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = width * height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def match_detections(reference, native, *, score_threshold=0.25, iou_threshold=0.5):
    refs = [item for item in reference if item["score"] >= score_threshold]
    natives = [item for item in native if item["score"] >= score_threshold]
    pairs = []
    for ref_index, ref in enumerate(refs):
        for native_index, candidate in enumerate(natives):
            pairs.append((_box_iou(ref["box"], candidate["box"]), ref_index, native_index))
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    used_refs = set()
    used_natives = set()
    matches = []
    for iou, ref_index, native_index in pairs:
        if iou < iou_threshold:
            break
        if ref_index not in used_refs and native_index not in used_natives:
            used_refs.add(ref_index)
            used_natives.add(native_index)
            matches.append((refs[ref_index], natives[native_index], iou))
    matched = len(matches)
    return {
        "reference_count": len(refs),
        "native_count": len(natives),
        "matched": matched,
        "ref_recall": matched / len(refs) if refs else 1.0,
        "native_precision": matched / len(natives) if natives else 1.0,
        "class_agreement": (
            sum(ref["class_id"] == native["class_id"] for ref, native, _ in matches) / matched
            if matched else (1.0 if not refs and not natives else 0.0)
        ),
        "mean_box_iou": (
            sum(iou for _, _, iou in matches) / matched
            if matched else (1.0 if not refs and not natives else 0.0)
        ),
        "matches": matches,
    }


def aggregate_d25(image_predictions):
    totals = {"reference_count": 0, "native_count": 0, "matched": 0,
              "class_matches": 0, "iou_sum": 0.0}
    for reference, native in image_predictions:
        result = match_detections(reference, native)
        totals["reference_count"] += result["reference_count"]
        totals["native_count"] += result["native_count"]
        totals["matched"] += result["matched"]
        totals["class_matches"] += sum(
            ref["class_id"] == candidate["class_id"]
            for ref, candidate, _ in result["matches"])
        totals["iou_sum"] += sum(iou for _, _, iou in result["matches"])
    matched = totals["matched"]
    refs, natives = totals["reference_count"], totals["native_count"]
    return {
        "ref_recall": matched / refs if refs else 1.0,
        "native_precision": matched / natives if natives else 1.0,
        "class_agreement": totals["class_matches"] / matched if matched else (
            1.0 if not refs and not natives else 0.0),
        "mean_box_iou": totals["iou_sum"] / matched if matched else (
            1.0 if not refs and not natives else 0.0),
        "reference_count": refs,
        "native_count": natives,
        "matched": matched,
    }


def coco_map50(metrics):
    if isinstance(metrics, dict) and "coco" in metrics:
        metrics = metrics["coco"]
    for key in ("map_50", "mAP50", "map50"):
        if key in metrics:
            return float(metrics[key])
    raise ValueError("COCO metrics do not contain mAP50")


def detection_gate(metrics, *, baseline, baseline_passed=None,
                   segmentation=False, pose=False):
    bars = {
        "ref_recall": 0.95,
        "native_precision": 0.95,
        "class_agreement": 0.98,
        "mean_box_iou": 0.90,
    }
    if baseline:
        failures = [f"{name}<{threshold}" for name, threshold in bars.items()
                    if metrics.get(name, 0.0) < threshold]
        if abs(coco_map50(metrics) - coco_map50(metrics["reference"])) > 0.02 + 1e-12:
            failures.append("mAP50 delta>0.02")
        if segmentation:
            if metrics.get("mean_mask_iou", 0.0) < 0.85:
                failures.append("mean_mask_iou<0.85")
            if abs(coco_map50(metrics["mask"]) -
                   coco_map50(metrics["mask"]["reference"])) > 0.02 + 1e-12:
                failures.append("mask mAP50 delta>0.02")
        if pose and metrics.get("keypoint_error_px", float("inf")) > 2.0:
            failures.append("keypoint_error_px>2")
    else:
        failures = []
        if baseline_passed is not True:
            failures.append("baseline F32 failed reference parity")
        f32 = metrics["f32"]
        comparison = metrics.get("d25_vs_f32", metrics)
        for field in ("ref_recall", "native_precision", "class_agreement", "mean_box_iou"):
            if comparison.get(field, 0.0) < bars[field]:
                failures.append(f"{field}<{bars[field]}")
        if segmentation and comparison.get("mean_mask_iou", 0.0) < 0.85:
            failures.append("mean_mask_iou<0.85")
        if pose and comparison.get("keypoint_error_px", float("inf")) > 2.0:
            failures.append("keypoint_error_px>2")
        if abs(coco_map50(metrics) - coco_map50(f32)) > 0.03 + 1e-12:
            failures.append("mAP50 delta from F32>0.03")
        if segmentation and abs(coco_map50(metrics["mask"]) -
                                coco_map50(f32["mask"])) > 0.03 + 1e-12:
            failures.append("mask mAP50 delta from F32>0.03")
    return {"passed": not failures, "failures": failures}


def classification_gate(metrics, *, baseline_passed=None):
    failures = []
    if "f32" in metrics:
        if baseline_passed is not True:
            failures.append("baseline F32 failed reference parity")
        if metrics.get("f32_top1_agreement", 0.0) < 0.95:
            failures.append("f32_top1_agreement<0.95")
        if metrics.get("f32_mean_top5_overlap", 0.0) < 0.90:
            failures.append("f32_mean_top5_overlap<0.90")
    else:
        if metrics.get("top1_agreement", 0.0) < 0.95:
            failures.append("top1_agreement<0.95")
        if metrics.get("mean_top5_overlap", 0.0) < 0.90:
            failures.append("mean_top5_overlap<0.90")
    return {"passed": not failures, "failures": failures}


def gguf_labels(path):
    from gguf import GGUFReader
    reader = GGUFReader(str(path))
    for key in ("yolo.class_names", "rfdetr.class_names"):
        field = reader.fields.get(key)
        if field:
            return [bytes(field.parts[index]).decode("utf-8")
                    for index in field.data]
    return []
