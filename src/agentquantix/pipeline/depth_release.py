"""Pinned depth-anything.cpp conversion, evaluation and atomic publication."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from .. import config, detection, vision
from ..runtimes.spec import RUNTIMES

RUNTIME_REVISION = "14f7461d1f704761a038ac9f50dbde8fdb7275e2"
DA2_REFERENCE_REVISION = "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
DA3_REFERENCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_REFERENCE_VERSION = "0.0.0"
PRECISIONS = ("F32", "F16", "Q8_0", "Q4_K")
REQUIREMENTS = (
    "torch==2.6.0+cpu", "torchvision==0.21.0+cpu",
    "transformers==5.22.0", "numpy==1.26.4", "Pillow==10.4.0",
    "safetensors==0.8.0", "gguf==0.10.0", "opencv-python==4.11.0.86",
    "e3nn==0.5.1", "omegaconf==2.3.0", "sentencepiece==0.2.2",
    "einops==0.8.2", "imageio==2.38.1", "addict==2.4.0",
    "evo==1.38.1",
)


class DepthValidationError(RuntimeError):
    pass


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guard_runtime(revision):
    if revision != RUNTIME_REVISION:
        raise DepthValidationError(
            "depth-anything.cpp revision mismatch; install the pinned runtime with: "
            f"aqx runtime install depth-anything.cpp --persistent --revision {RUNTIME_REVISION}")


def _source_url(source):
    repo = source["repo"]
    revision = source["revision"]
    filename = source["weight_file"]
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def verify_source(path, source):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != source["size"] or \
            _hash(path) != source["sha256"]:
        raise DepthValidationError(f"source integrity check failed for {path}")
    return path


def plan_release(model, *, task, family=None, target_repo=None, quants=None,
                 runtime_revision=None):
    matched = vision.family_for(
        model, family=family, engine="depth-anything.cpp", task=task)
    if not matched:
        raise DepthValidationError(f"no depth-anything.cpp family matches {model} for {task}")
    if runtime_revision:
        guard_runtime(runtime_revision)
    requested = tuple(str(value).upper() for value in (quants or PRECISIONS))
    if requested != PRECISIONS:
        raise DepthValidationError("all four pinned depth precisions are evaluated")
    source = dict(matched["source"])
    source["url"] = _source_url(source)
    return {
        "pipeline": "vision", "engine": "depth-anything.cpp",
        "task": task, "model": source["repo"], "family": matched["family"],
        "family_target": f"depth-anything.cpp:{matched['family']}",
        "source": source, "source_files": matched["source_files"],
        "source_license": source["licence"], "engine_license": "MIT",
        "runtime_revision": RUNTIME_REVISION,
        "runtime_repository": RUNTIMES["depth-anything.cpp"].repository,
        "converter": ("scripts/convert_da2_to_gguf.py"
                      if matched["family"] == "da2-metric-hypersim-small"
                      else "scripts/convert_da3_to_gguf.py"),
        "quants": list(PRECISIONS),
        "precision_producers": {
            "F32": "converter", "F16": "native-quantizer",
            "Q8_0": "native-quantizer", "Q4_K": "native-quantizer",
        },
        "target_repo": target_repo or f"{config.namespace()}/{matched['family']}-GGUF",
        "status": "ready", "blockers": [],
        "quality_profile": "depth-0.5.8-coco-license-4-7",
        "evaluation_version": 1,
    }


def converter_command(python, engine_source, reference_source, family, source, output):
    return [str(python), str(Path(__file__).with_name("depth_converter.py")),
            str(engine_source), str(reference_source), family, str(source), str(output)]


def quantize_command(binary, source, output, precision):
    if precision not in PRECISIONS[1:]:
        raise DepthValidationError(f"unsupported native quantizer precision: {precision}")
    return [str(binary), "quantize", str(source), str(output), precision.casefold()]


def native_command(binary, family, model, image, output, pose=None):
    command = [str(binary), "depth", "--model", str(model), "--input", str(image),
               "--pfm", str(output), "--no-invert", "--threads", "4"]
    if family == "da3-small":
        command.extend(["--pose", str(pose or Path(output).with_name(
            Path(output).stem + "-pose.json"))])
    return command


def work_dir(family, source_sha):
    return config.TEMP_DIR / "aqx-vision-runs" / f"depth-anything.cpp--{family}-{source_sha[:12]}"


def _download_source(source, cache_dir):
    from huggingface_hub import snapshot_download

    local_dir = Path(cache_dir) / "depth-sources" / source["revision"]
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=source["repo"], revision=source["revision"],
                      allow_patterns=[source["weight_file"], *(
                          ["config.json"] if source["weight_file"] == "model.safetensors" else [])],
                      local_dir=str(local_dir))
    weight = local_dir / source["weight_file"]
    verify_source(weight, source)
    return local_dir, weight


def _checkout_reference(root, family):
    if family == "da2-metric-hypersim-small":
        repo = "https://github.com/DepthAnything/Depth-Anything-V2.git"
        revision = DA2_REFERENCE_REVISION
        destination = Path(root) / "reference-sources" / "Depth-Anything-V2"
    else:
        repo = "https://github.com/ByteDance-Seed/Depth-Anything-3.git"
        revision = DA3_REFERENCE_REVISION
        destination = Path(root) / "reference-sources" / "Depth-Anything-3"
    if not (destination / ".git").is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        repo, str(destination)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1",
                    "origin", revision], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision],
                   check=True, capture_output=True, text=True)
    actual = subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise DepthValidationError(f"reference source revision mismatch: {actual}")
    return destination


def _timed_run(command, log_path, *, timeout=3600):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    time_path = log_path.with_suffix(".time")
    timed = ["/usr/bin/time", "-v", "-o", str(time_path), *map(str, command)]
    if not Path("/usr/bin/time").is_file():
        timed = list(map(str, command))
    started = time.monotonic()
    completed = subprocess.run(timed, capture_output=True, text=True, timeout=timeout,
                               check=False)
    wall = time.monotonic() - started
    log_path.write_text(f"argv: {json.dumps(list(map(str, command)))}\n"
                        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n",
                        encoding="utf-8")
    if completed.returncode:
        raise DepthValidationError(
            f"command failed ({completed.returncode}); log: {log_path}: "
            f"{completed.stderr.strip()[-400:]}")
    rss = None
    if time_path.is_file():
        match = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)",
                          time_path.read_text(encoding="utf-8"))
        rss = int(match.group(1)) if match else None
    return {"wall_seconds": wall, "peak_rss_kib": rss, "argv": list(map(str, command))}


def _requirements_provenance(python):
    packages = subprocess.run([str(python), "-m", "pip", "freeze"],
                              check=True, capture_output=True, text=True).stdout
    return {
        "python": str(python),
        "python_version": subprocess.run(
            [str(python), "-c", "import sys; print(sys.version.split()[0])"],
            check=True, capture_output=True, text=True).stdout.strip(),
        "requested_packages": list(REQUIREMENTS),
        "packages": sorted(packages.splitlines()),
        "da3_reference": {"package": "depth-anything-3", "version": DA3_REFERENCE_VERSION,
                          "revision": DA3_REFERENCE_REVISION},
        "da2_reference_revision": DA2_REFERENCE_REVISION,
        "precision_producers": dict(zip(PRECISIONS, ("converter",) +
                                        ("native-quantizer",) * 3, strict=True)),
    }


def _read_pfm(path):
    import numpy as np
    with Path(path).open("rb") as stream:
        if stream.readline().decode("ascii").strip() != "Pf":
            raise DepthValidationError(f"invalid PFM output: {path}")
        width, height = map(int, stream.readline().decode("ascii").split())
        scale = float(stream.readline().decode("ascii").strip())
        dtype = "<f4" if scale < 0 else ">f4"
        result = np.fromfile(stream, dtype=dtype, count=width * height)
    if result.size != width * height:
        raise DepthValidationError(f"truncated PFM output: {path}")
    return np.flipud(result.reshape(height, width)).astype(np.float32) * abs(scale)


def _resize(array, shape):
    import numpy as np
    array = np.asarray(array, dtype=np.float32)
    if tuple(array.shape) == tuple(shape):
        return array
    from PIL import Image
    return np.asarray(Image.fromarray(array, mode="F").resize(
        (int(shape[1]), int(shape[0])), Image.Resampling.BILINEAR), dtype=np.float32)


def _metric_comparison(reference, estimate):
    import numpy as np
    ref = np.asarray(reference, dtype=np.float64).ravel()
    est = np.asarray(estimate, dtype=np.float64).ravel()
    valid = np.isfinite(ref) & np.isfinite(est) & (ref > 0)
    ref, est = ref[valid], est[valid]
    median = float(np.median(ref)) if ref.size else 0.0
    error = np.abs(est - ref)
    mae = float(np.median(error)) if error.size else float("inf")
    return {
        "median_absolute_error_m": mae,
        "mean_absrel": float(np.mean(error / np.abs(ref))) if ref.size else float("inf"),
        "reference_median_depth_m": median,
        "median_error_percent_of_reference_median": 100.0 * mae / median if median else float("inf"),
        "valid_pixels": int(valid.sum()),
    }


def _camera_comparison(reference, estimate):
    import numpy as np
    rotations, translations, intrinsics = [], [], []
    for key, ref in reference.items():
        est = estimate[key]
        ref_ext = np.asarray(ref["extrinsics"], dtype=np.float64)
        est_ext = np.asarray(est["extrinsics"], dtype=np.float64)
        ref_intr = np.asarray(ref["intrinsics"], dtype=np.float64)
        est_intr = np.asarray(est["intrinsics"], dtype=np.float64)
        relative = ref_ext[:3, :3].T @ est_ext[:3, :3]
        cosine = np.clip((np.trace(relative) - 1) / 2, -1, 1)
        rotations.append(float(np.degrees(np.arccos(cosine))))
        translations.append(float(np.linalg.norm(ref_ext[:3, 3] - est_ext[:3, 3])))
        valid = np.abs(ref_intr) > np.finfo(np.float64).eps
        relative = np.abs(ref_intr[valid] - est_intr[valid]) / np.abs(ref_intr[valid])
        intrinsics.append(float(np.mean(relative)) if relative.size else float("inf"))
    return {
        "rotation_mean_error_degrees": float(np.mean(rotations)),
        "translation_mean_l2": float(np.mean(translations)),
        "intrinsics_mean_relative_error": float(np.mean(intrinsics)),
    }


def gate_quality(family, quality, *, is_baseline, baseline_passed=True):
    failures = []
    source_key = "vs_reference" if is_baseline else "vs_f32"
    depth = quality.get(source_key, {})
    if family == "da2-metric-hypersim-small":
        metric_key = "metric_vs_reference" if is_baseline else "metric_vs_f32"
        metric = quality.get(metric_key, {})
        if depth.get("spearman", 0) < 0.99:
            failures.append("spearman<0.99")
        if metric.get("mean_absrel", float("inf")) > 0.05:
            failures.append("absrel>0.05")
        if metric.get("median_error_percent_of_reference_median", float("inf")) > 5:
            failures.append("median_abs_error>5%_reference_median")
    else:
        if depth.get("spearman", 0) < 0.99:
            failures.append("spearman<0.99")
        if depth.get("scale_shift_absrel", float("inf")) > 0.05:
            failures.append("scale_shift_absrel>0.05")
        camera_key = "camera_vs_reference" if is_baseline else "camera_vs_f32"
        camera = quality.get(camera_key, {})
        if camera.get("rotation_mean_error_degrees", float("inf")) > 1.0:
            failures.append("rotation>1.0_degrees")
        if camera.get("translation_mean_l2", float("inf")) > 1e-3:
            failures.append("translation_l2>1e-3")
        if camera.get("intrinsics_mean_relative_error", float("inf")) > 0.01:
            failures.append("intrinsics_relative_error>0.01")
    if not is_baseline and not baseline_passed:
        failures.append("baseline F32 failed reference parity")
    return {"passed": not failures, "gate_failures": failures}


def _load_reference(path):
    import numpy as np
    npz = np.load(path)
    metadata = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    return npz, metadata


def _evaluate(plan, family, precision, root, images, reference_npz, metadata, baseline=None):
    import numpy as np
    reference_depths, native_depths, base_depths = [], [], []
    reference_pose, native_pose, base_pose = {}, {}, {}
    frame_key = "model_shape" if family == "da3-small" else "source_shape"
    output_dir = Path(root) / "outputs" / precision.casefold()
    for sample in images:
        key = str(sample["id"])
        shape = metadata[key][frame_key]
        ref = _resize(reference_npz[f"depth_{key}"], shape)
        pred = _resize(_read_pfm(output_dir / f"{key}.pfm"), shape)
        reference_depths.append(ref.ravel())
        native_depths.append(pred.ravel())
        if baseline:
            base_depths.append(_resize(_read_pfm(
                Path(root) / "outputs" / "f32" / f"{key}.pfm"), shape).ravel())
        if family == "da3-small":
            reference_pose[key] = {
                name: reference_npz[f"{name}_{key}"] for name in ("extrinsics", "intrinsics")}
            native_pose[key] = json.loads(
                (output_dir / f"{key}-pose.json").read_text(encoding="utf-8"))
            if baseline:
                base_pose[key] = json.loads(
                    (Path(root) / "outputs" / "f32" / f"{key}-pose.json").read_text(
                        encoding="utf-8"))
    ref = np.concatenate(reference_depths)
    pred = np.concatenate(native_depths)
    quality = {}
    if baseline:
        base = np.concatenate(base_depths)
        quality["vs_f32"] = vision.depth_comparison(pred, base)
        if family == "da2-metric-hypersim-small":
            quality["metric_vs_f32"] = _metric_comparison(base, pred)
        else:
            quality["camera_vs_f32"] = _camera_comparison(base_pose, native_pose)
    else:
        quality["vs_reference"] = vision.depth_comparison(pred, ref)
        if family == "da2-metric-hypersim-small":
            quality["metric_vs_reference"] = _metric_comparison(ref, pred)
        else:
            quality["camera_vs_reference"] = _camera_comparison(reference_pose, native_pose)
    quality.update(gate_quality(family, quality, is_baseline=not baseline,
                                baseline_passed=baseline.get("passed", True) if baseline else True))
    return quality


def _avx_choice(runtime):
    snapshot = runtime.snapshot()
    fingerprint = (snapshot.get("build_fingerprint") or
                   snapshot.get("provenance", {}).get("build_fingerprint", {}))
    flags = fingerprint.get("flags", [])
    return next((flag.split("=", 1)[1] for flag in flags
                 if flag.startswith("-DDA_HAS_AVX512F=")), "unknown")


def run_release(plan, *, user_approved, publish=False, runtime=None,
                download_source=None, evaluate=None, convert=None, quantize=None):
    if user_approved is not True:
        raise DepthValidationError("start_release requires approval of the exact depth plan")
    if plan.get("status") != "ready" or plan.get("blockers"):
        raise DepthValidationError("depth plan is blocked: " + "; ".join(plan.get("blockers", [])))
    guard_runtime(plan["runtime_revision"])
    if runtime is None:
        from ..runtime_control import ensure_runtime
        runtime = ensure_runtime(
            "depth-anything.cpp",
            revision={"revision": RUNTIME_REVISION, "ref": RUNTIME_REVISION,
                      "channel": "pinned", "repository": plan["runtime_repository"]},
            required_targets=("da3-cli",))
    guard_runtime(runtime.revision)
    family = vision.family_for(plan["model"], family=plan["family"],
                               task="depth-estimation", engine="depth-anything.cpp")
    if not family:
        raise DepthValidationError("saved depth family no longer matches the catalog")
    root = work_dir(plan["family"], plan["source"]["sha256"])
    root.mkdir(parents=True, exist_ok=True)
    run_path, result_path = root / "run.json", root / "result.json"
    progress = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    result_saved = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    saved = result_saved if result_saved.get("candidates") else progress
    old = saved.get("plan") or {}
    runtime_snapshot = runtime.snapshot()
    same_run = all(old.get(key) == plan.get(key) for key in (
        "model", "task", "family", "source", "quants", "runtime_revision")) and (
            saved.get("runtime_snapshot") or {}).get("identity") == runtime_snapshot.get("identity")
    candidates = []
    if same_run:
        candidates = [
            item for item in saved.get("candidates", [])
            if item.get("status") != "conversion-or-evaluation-failed"
            and Path(item.get("path", "")).is_file()
            and item.get("sha256") == _hash(item["path"])]
    if (result_saved and same_run and all(
            item.casefold() in {candidate["precision"].casefold() for candidate in candidates}
            for item in plan["quants"])):
        if publish and not result_saved.get("published"):
            result_saved["publication"] = publish_bundle(
                root, old, result_saved["candidates"], result_saved["runtime_snapshot"],
                result_saved["converter"])
            result_saved["published"] = True
            _write(result_path, result_saved)
        return result_saved
    _write(run_path, {"plan": plan, "created": progress.get("created") or time.time(),
                      "runtime_snapshot": runtime_snapshot, "candidates": candidates})
    if download_source:
        source_dir, checkpoint = download_source(plan["source"])
    else:
        source_dir, checkpoint = _download_source(plan["source"], config.TEMP_DIR)
    verify_source(checkpoint, plan["source"])
    reference_source = _checkout_reference(root, plan["family"])
    python = vision.converter_venv(runtime.source, REQUIREMENTS)
    provenance = _requirements_provenance(python)
    provenance["source"] = {"repo": plan["source"]["repo"],
                            "revision": plan["source"]["revision"],
                            "weight_file": plan["source"]["weight_file"],
                            "sha256": plan["source"]["sha256"],
                            "size": plan["source"]["size"]}
    images_manifest = detection.download_coco_images(config.TEMP_DIR)
    _write(root / "images.json", images_manifest)
    reference_path = root / "reference.npz"
    reference_timing_path = root / "reference-timing.json"
    if not reference_path.is_file():
        command = [python, str(Path(__file__).with_name("depth_reference.py")),
                   plan["family"], str(runtime.source), str(reference_source),
                   str(checkpoint if plan["family"] == "da2-metric-hypersim-small"
                       else source_dir),
                   str(root / "images.json"), str(reference_path)]
        reference_timing = _timed_run(command, root / "logs" / "reference.log", timeout=7200)
        _write(reference_timing_path, reference_timing)
    elif reference_timing_path.is_file():
        reference_timing = json.loads(reference_timing_path.read_text(encoding="utf-8"))
    else:
        reference_timing = {"reused": True}
    reference_npz, metadata = _load_reference(reference_path)
    binary = runtime.binaries["da3-cli"]
    f32_path = root / "models" / "f32.gguf"
    baseline_quality = next((item.get("quality") for item in candidates
                             if item["precision"].casefold() == "f32"), None)
    baseline_failed = bool(baseline_quality is not None and not baseline_quality.get("passed"))
    baseline_failures = list(baseline_quality.get("gate_failures", [])) if baseline_quality else []
    producer = plan["precision_producers"]
    for precision in PRECISIONS:
        if any(item["precision"].casefold() == precision.casefold() for item in candidates):
            continue
        model_path = root / "models" / f"{precision.casefold()}.gguf"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not model_path.is_file():
                if precision == "F32":
                    command = converter_command(python, runtime.source, reference_source,
                                                plan["family"],
                                                checkpoint if plan["family"] ==
                                                "da2-metric-hypersim-small" else source_dir,
                                                f32_path)
                    if convert:
                        convert(command)
                        conversion = {"argv": command, "wall_seconds": None,
                                      "peak_rss_kib": None}
                    else:
                        conversion = _timed_run(command, root / "logs" / "convert-f32.log",
                                                timeout=7200)
                else:
                    command = quantize_command(binary, f32_path, model_path, precision)
                    if quantize:
                        quantize(command)
                        conversion = {"argv": command, "wall_seconds": None,
                                      "peak_rss_kib": None}
                    else:
                        conversion = _timed_run(
                            command, root / "logs" / f"quantize-{precision.casefold()}.log",
                            timeout=7200)
            else:
                conversion = {"reused": True, "wall_seconds": 0.0, "peak_rss_kib": None}
            if not model_path.is_file() or model_path.stat().st_size == 0:
                raise DepthValidationError("precision producer did not create a GGUF")
            output_dir = root / "outputs" / precision.casefold()
            output_dir.mkdir(parents=True, exist_ok=True)
            native_runs = []
            if not all((output_dir / f"{sample['id']}.pfm").is_file()
                       for sample in images_manifest["images"]):
                for sample in images_manifest["images"]:
                    output = output_dir / f"{sample['id']}.pfm"
                    pose = output_dir / f"{sample['id']}-pose.json"
                    command = native_command(binary, plan["family"], model_path,
                                             sample["local_path"], output, pose)
                    native_runs.append(_timed_run(
                        command, root / "logs" /
                        f"native-{precision.casefold()}-{sample['id']}.log", timeout=3600))
            else:
                native_runs = [{"reused": True} for _ in images_manifest["images"]]
            quality = (evaluate or _evaluate)(
                plan, plan["family"], precision, root, images_manifest["images"],
                reference_npz, metadata, baseline_quality if precision != "F32" else None)
            quality["conversion_run"] = conversion
            quality["produced_by"] = producer[precision]
            quality["native_runs"] = native_runs
            if precision == "F32":
                baseline_quality = quality
                baseline_failed = not bool(quality["passed"])
                baseline_failures = list(quality["gate_failures"])
            status = "passed" if quality["passed"] and not baseline_failed else "quality-rejected"
            candidates.append({
                "precision": precision, "produced_by": producer[precision],
                "path": str(model_path), "sha256": _hash(model_path),
                "bytes": model_path.stat().st_size, "quality": quality, "status": status,
            })
        except Exception as error:  # noqa: BLE001
            candidates.append({"precision": precision, "produced_by": producer[precision],
                               "path": str(model_path),
                               "status": "quality-rejected" if baseline_failed else
                               "conversion-or-evaluation-failed", "error": str(error)})
            if precision == "F32" and baseline_quality is None:
                baseline_failed = True
                baseline_failures = ["F32 baseline conversion or evaluation failed"]
        if baseline_failed:
            for item in candidates:
                item["status"] = "quality-rejected"
        plan["baseline_failed"] = baseline_failed
        plan["baseline_failures"] = baseline_failures
        _write(root / "run.json", {
            "plan": plan, "baseline_failed": baseline_failed,
            "baseline_failures": baseline_failures, "runtime_snapshot": runtime_snapshot,
            "created": progress.get("created") or time.time(), "candidates": candidates,
        })
    result = {
        "pipeline": "vision", "engine": "depth-anything.cpp",
        "run_id": root.name, "plan": plan, "runtime_snapshot": runtime_snapshot,
        "converter": provenance, "reference_timing": reference_timing,
        "evaluation_images": images_manifest["images"],
        "avx512f_build_flag": _avx_choice(runtime), "candidates": candidates,
        "baseline_failed": baseline_failed, "baseline_failures": baseline_failures,
        "published": False,
        "status": "passed" if any(item["status"] == "passed" for item in candidates)
        else "rejected",
    }
    if not any(item["status"] == "passed" for item in candidates):
        result["status"] = "rejected"
    _write(result_path, result)
    if publish and any(item["status"] == "passed" for item in candidates):
        result["publication"] = publish_bundle(root, plan, candidates, runtime_snapshot, provenance)
        result["published"] = True
        _write(result_path, result)
    return result


def runtime_card(plan, runtime_snapshot, provenance):
    fingerprint = (runtime_snapshot.get("build_fingerprint") or
                   runtime_snapshot.get("provenance", {}).get("build_fingerprint", {}))
    flags = fingerprint.get("flags", [])
    avx = next((flag.split("=", 1)[1] for flag in flags
                if flag.startswith("-DDA_HAS_AVX512F=")), "unknown")
    semantics = {
        "da2-metric-hypersim-small":
            "metric depth in metres, indoor Hypersim training, max 20 m",
        "da3-small": "relative depth + camera pose/intrinsics",
    }[plan["family"]]
    return {
        "task": "depth-estimation", "family": plan["family"],
        "semantics": semantics, "weights_license": plan["source_license"],
        "engine_license": "MIT", "source_url": plan["source"]["url"],
        "source_revision": plan["source"]["revision"],
        "source_sha256": plan["source"]["sha256"],
        "evaluation": "evaluated on 10 COCO val2017 images; small sample",
        "comparison": (
            "DA2 metric: unaligned metres, Spearman >= 0.99, AbsRel <= 0.05, "
            "median absolute error <= 5% of reference median; lower precisions vs F32."
            if plan["family"] == "da2-metric-hypersim-small" else
            "DA3 model frame: Spearman >= 0.99, scale-shift AbsRel <= 0.05; "
            "camera rotation <= 1 degree, translation L2 <= 1e-3, intrinsics relative error <= 0.01."
        ),
        "precision_producers": plan["precision_producers"],
        "avx512f_build_flag": avx,
        "confidence_output": "not evaluated" if plan["family"] == "da3-small" else "not applicable",
        "reference_provenance": provenance,
        "runtime": runtime_snapshot,
    }


def publish_bundle(root, plan, candidates, runtime_snapshot, provenance, api=None):
    from huggingface_hub import CommitOperationAdd, HfApi
    api = api or HfApi(token=config.TOKEN)
    passing = [item for item in candidates if item.get("status") == "passed"]
    if not passing:
        raise DepthValidationError("no passing precision is publishable")
    card = runtime_card(plan, runtime_snapshot, provenance)
    artifact_root = Path(root) / "publication"
    operations, members = [], []
    for item in passing:
        repo_path = f"{item['precision']}/{Path(item['path']).name}"
        local = artifact_root / repo_path
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["path"], local)
        operations.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local)))
        members.append({**item, "path": repo_path})
    readme = Path(root) / "README.md"
    readme.write_text("# AgentQuantix depth bundle\n\n" +
                      json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme)))
    manifest = {"schema_version": 1, "pipeline": "vision", "engine": plan["engine"],
                "task": plan["task"], "family": plan["family"], "target_repo": plan["target_repo"],
                "card": card, "candidates": members}
    manifest_path = artifact_root / "manifest.json"
    _write(manifest_path, manifest)
    operations.append(CommitOperationAdd(path_in_repo="manifest.json",
                                         path_or_fileobj=str(manifest_path)))
    api.create_repo(plan["target_repo"], repo_type="model", exist_ok=True)
    api.create_commit(repo_id=plan["target_repo"], repo_type="model",
                      operations=operations, commit_message="Add accepted depth precision bundle")
    return {"target_repo": plan["target_repo"],
            "precisions": [item["precision"] for item in passing]}


def publish_run(run_id, precision, root=None):
    if root is None:
        root = config.TEMP_DIR / "aqx-vision-runs" / run_id
    root = Path(root)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    selected = next((item for item in result["candidates"]
                     if item["precision"].casefold() == precision.casefold()), None)
    if not selected or selected["status"] != "passed":
        raise DepthValidationError(f"{precision} is not an accepted depth precision")
    result["publication"] = publish_bundle(
        root, result["plan"], result["candidates"], result["runtime_snapshot"],
        result["converter"])
    result["published"] = True
    _write(root / "result.json", result)
    return {"run_id": run_id, **result["publication"], "published": True}


def verify_manifest(manifest, root):
    problems = []
    for item in manifest.get("candidates", []):
        path = Path(root) / item["path"]
        if not path.is_file():
            problems.append(f"missing candidate: {item['path']}")
        elif _hash(path) != item.get("sha256"):
            problems.append(f"hash mismatch: {item['path']}")
    return problems
