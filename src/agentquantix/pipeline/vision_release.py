"""Pinned vision.cpp conversion, native quality gates and atomic bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from .. import config, vision

FIXTURES = Path(__file__).parents[1] / "fixtures" / "vision"
REVISION = vision.VISIONCPP_REVISION
REQUIREMENTS = ("numpy==2.2.6", "Pillow==12.3.0", "torch==2.11.0+cpu",
                "transformers==4.57.6", "safetensors==0.8.0", "gguf==0.19.0")


class VisionValidationError(RuntimeError):
    pass


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(
        value, indent=2, sort_keys=True,
        default=lambda item: item.item() if hasattr(item, "item") else str(item)),
        encoding="utf-8")
    pending.replace(path)


def work_dir(repo_id):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", repo_id).strip("-")
    return config.TEMP_DIR / "aqx-vision-runs" / name


def runtime_guard(revision):
    if revision != REVISION:
        raise VisionValidationError(
            f"vision.cpp runtime {revision} does not match the pinned family catalog "
            f"{vision.VISIONCPP_TAG}; install it with: aqx runtime install vision.cpp "
            f"--persistent --revision {REVISION}")


def plan_release(model, *, task, family=None, target_repo=None, quants=None,
                 metadata=None, runtime_revision=None):
    matched = vision.family_for(model, metadata or {}, family=family)
    if not matched or matched["task"] != task:
        raise VisionValidationError(f"no vision.cpp family matches {model} for {task}")
    if runtime_revision:
        runtime_guard(runtime_revision)
    from huggingface_hub import HfApi
    info = HfApi(token=config.TOKEN).model_info(model, files_metadata=True)
    revision = info.sha
    files = [item.rfilename for item in info.siblings]
    weights = [name for name in files if any(Path(name).match(pattern)
                                              for pattern in matched["source_files"])]
    if not weights:
        raise VisionValidationError(f"{model} has no supported source weights")
    license_name = getattr(info, "card_data", None)
    license_name = getattr(license_name, "license", None) if license_name else None
    if model.casefold() == "dhkim2810/mobilesam":
        license_note = "HF metadata MIT; upstream code Apache-2.0"
    else:
        license_note = matched["licence_notes"]
    return {"pipeline": "vision", "engine": "vision.cpp", "task": task,
            "model": model, "family": matched["family"], "source_revision": revision,
            "source_files": weights, "source_license": license_name or license_note,
            "license_note": license_note,
            "target_repo": target_repo or f"{config.namespace()}/{model.rsplit('/', 1)[-1]}-GGUF",
            "quants": [str(item).upper() for item in
                       (quants or ["F32", "F16"])],
            "runtime_revision": REVISION, "converter": "scripts/convert.py",
            "status": "ready", "blockers": [],
            "quality_profile": "vision-0.5.6-fixed-fixtures"}


def cli_command(binary, family, model, image, output, prompt=None):
    args = [str(binary), family["cli_command"], "--backend", "cpu",
            "--model", str(model), "--input", str(image), "--output", str(output)]
    if prompt is not None:
        args.extend(["-p", *map(str, prompt)])
    return args


def _image(path):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"))


def _mask(path):
    return _image(path) > 127


def _depth_image(path):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path), dtype=np.float32)


def _run(command, *, cwd=None):
    start = time.monotonic()
    result = subprocess.run([str(value) for value in command], cwd=cwd,
                            capture_output=True, text=True, errors="replace",
                            check=False)
    if result.returncode:
        raise VisionValidationError(
            f"native vision command failed ({result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}")
    return time.monotonic() - start


def _evaluate(task, family, model, binary, root, baseline=None):
    result = {"outputs": [], "elapsed_seconds": 0.0}
    if task == "image-segmentation":
        from ..vision import boundary_fscore, iou
        image = FIXTURES / "segmentation" / "image.png"
        truth = _mask(FIXTURES / "segmentation" / "mask.png")
        output = root / "segmentation.png"
        result["elapsed_seconds"] += _run(cli_command(binary, family, model, image, output))
        prediction = _mask(output)
        result.update(output_path=str(output), iou=iou(prediction, truth),
                      boundary_fscore=boundary_fscore(prediction, truth))
        result["outputs"].append(prediction)
    elif task == "mask-generation":
        from ..vision import iou
        image = FIXTURES / "segmentation" / "image.png"
        truth = _mask(FIXTURES / "segmentation" / "mask.png")
        for name, prompt in (("point", (160, 105)), ("box", (77, 40, 244, 214))):
            output = root / f"{name}.png"
            result["elapsed_seconds"] += _run(
                cli_command(binary, family, model, image, output, prompt))
            prediction = _mask(output)
            result[f"{name}_iou"] = iou(prediction, truth)
            result[f"{name}_output_path"] = str(output)
            result["outputs"].append(prediction)
    else:
        from ..vision import depth_baseline_passes, depth_comparison
        scenes = sorted((FIXTURES / "depth").glob("*.png"))
        scenes = [path for path in scenes if path.name != "two-plane.png"]
        outputs = []
        for scene in scenes:
            output = root / f"depth-{scene.stem}.png"
            result["elapsed_seconds"] += _run(cli_command(binary, family, model, scene, output))
            outputs.append(_depth_image(output))
            result["outputs"].append(outputs[-1])
            result.setdefault("output_paths", []).append(str(output))
        if baseline is None:
            plane_input = FIXTURES / "depth" / "two-plane.png"
            plane_output = root / "depth-two-plane.png"
            result["elapsed_seconds"] += _run(
                cli_command(binary, family, model, plane_input, plane_output))
            plane = _depth_image(plane_output)
            result["two_plane_output_path"] = str(plane_output)
            result["two_plane_near_mean"] = float(plane[plane.shape[0] // 2:].mean())
            result["two_plane_far_mean"] = float(plane[:plane.shape[0] // 2].mean())
            result["baseline_valid"] = depth_baseline_passes(outputs, plane)
        else:
            result["depth_comparisons"] = [
                depth_comparison(candidate, base)
                for candidate, base in zip(outputs, baseline["outputs"])]
    return result


def gate(task, quality, baseline=None, *, is_baseline=False):
    if task == "image-segmentation":
        if is_baseline:
            return quality["iou"] >= 0.90 and quality["boundary_fscore"] >= 0.80
        return (quality["iou"] >= baseline["iou"] - 0.01
                and quality["pixel_agreement"] >= 0.99)
    if task == "mask-generation":
        if is_baseline:
            return quality["point_iou"] >= 0.90 and quality["box_iou"] >= 0.90
        return (quality["point_iou"] >= baseline["point_iou"] - 0.01
                and quality["box_iou"] >= baseline["box_iou"] - 0.01
                and quality["pixel_agreement"] >= 0.99)
    if is_baseline:
        return bool(quality["baseline_valid"])
    return all(item["spearman"] >= 0.999 and item["scale_shift_absrel"] <= 0.01
               for item in quality["depth_comparisons"])


def _requirements_provenance(python):
    command = [python, "-m", "pip", "freeze"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    version = subprocess.run(
        [python, "-c", "import sys; print(sys.version)"],
        capture_output=True, text=True, check=True).stdout.strip()
    return {"python": python, "python_version": version,
            "packages": sorted(result.stdout.splitlines())}


def run_release(plan, *, user_approved, publish=False, runtime=None,
                download_source=None, convert=None, evaluate=None):
    if user_approved is not True:
        raise VisionValidationError("start_release requires approval of the exact vision plan")
    if plan.get("blockers") or plan.get("status") != "ready":
        raise VisionValidationError("vision plan is blocked: " + "; ".join(plan.get("blockers", [])))
    if runtime is None:
        from ..runtime_control import ensure_runtime
        runtime = ensure_runtime("vision.cpp", revision=REVISION)
    runtime_guard(runtime.revision)
    root = work_dir(plan["model"])
    root.mkdir(parents=True, exist_ok=True)
    result_path = root / "result.json"
    if result_path.is_file():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        old = saved.get("plan") or {}
        same_run = all(old.get(key) == plan.get(key) for key in (
            "model", "task", "family", "source_revision", "quants", "runtime_revision"))
        if same_run and all(Path(item["path"]).is_file() for item in saved.get("candidates", [])):
            if publish and not saved.get("published"):
                saved["publication"] = publish_bundle(
                    root, old, saved["candidates"], converter_provenance=saved.get("converter"))
                saved["published"] = True
                _write(result_path, saved)
            return saved
    _write(root / "run.json", {"created": time.time(), "plan": plan})
    source = root / "source"
    if download_source is None:
        from huggingface_hub import snapshot_download
        download_source = snapshot_download
    source = Path(download_source(plan["model"], revision=plan["source_revision"],
                                 local_dir=source, allow_patterns=plan["source_files"]))
    source_paths = [source / item for item in plan["source_files"] if (source / item).is_file()]
    if len(source_paths) != 1:
        raise VisionValidationError(
            "vision.cpp convert.py requires exactly one source weight file; "
            f"found {len(source_paths)}")
    source_weight = source_paths[0]
    python = vision.converter_venv(runtime.source, REQUIREMENTS)
    package_provenance = _requirements_provenance(python)
    policy = vision.precision_policy(vision.source_dtypes(
        source, [str(path.relative_to(source)) for path in source_paths], python=python))
    precisions = [item for item in policy["precisions"] if item in plan["quants"]]
    if not precisions:
        raise VisionValidationError("requested precisions do not match source dtype policy")
    family = vision.family_for(plan["model"], {}, family=plan["family"])
    converter = Path(runtime.source) / "scripts" / "convert.py"
    if any(path.suffix == ".pth" for path in source_paths):
        if "weights_only=True" not in converter.read_text(encoding="utf-8"):
            raise VisionValidationError(".pth conversion requires convert.py weights_only=True")
    binary = runtime.binaries["vision-cli"]
    import numpy as np
    results, baseline = [], None
    for precision in precisions:
        candidate = root / "models" / f"{precision}.gguf"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        command = vision.converter_command(python, converter, source_weight, candidate,
                                           family, precision)
        if not candidate.is_file() or candidate.stat().st_size == 0:
            if convert:
                convert(command)
            else:
                _run(command)
        start = time.monotonic()
        quality = (evaluate or _evaluate)(plan["task"], family, candidate, binary, root, baseline)
        quality.setdefault("elapsed_seconds", time.monotonic() - start)
        quality["source_dtype"] = policy["source_dtype"]
        quality["precision_reason"] = policy["precision_reason"]
        if baseline is not None:
            if plan["task"] in ("image-segmentation", "mask-generation"):
                quality["pixel_agreement"] = float(np.mean([
                    np.mean(a == b) for a, b in zip(quality["outputs"], baseline["outputs"])]))
        passed = gate(plan["task"], quality, baseline, is_baseline=baseline is None)
        if baseline is None:
            baseline = quality
        quality_record = {key: value for key, value in quality.items() if key != "outputs"}
        results.append({"precision": precision, "path": str(candidate), "sha256": _hash(candidate),
                        "bytes": candidate.stat().st_size, "quality": quality_record,
                        "status": "passed" if passed else "quality-rejected"})
        _write(root / "quality.json", {item["precision"]: item["quality"] for item in results})
    release_plan = {**plan, **policy, "converter_provenance": package_provenance}
    result = {"pipeline": "vision", "run_id": root.name, "plan": release_plan,
              "runtime_snapshot": runtime.snapshot(), "converter": package_provenance,
              "candidates": results, "published": False}
    _write(root / "result.json", result)
    if publish and any(item["status"] == "passed" for item in results):
        publish_bundle(root, release_plan, results, converter_provenance=package_provenance)
        result["published"] = True
        _write(root / "result.json", result)
    return result


def publish_bundle(root, plan, candidates, api=None, converter_provenance=None):
    from huggingface_hub import CommitOperationAdd, HfApi
    api = api or HfApi(token=config.TOKEN)
    passing = [item for item in candidates if item["status"] == "passed"]
    if not passing:
        raise VisionValidationError("no passing vision precision is publishable")
    cards = {"task": plan["task"], "family": plan["family"],
             "license": plan["source_license"], "license_note": plan["license_note"],
             "source_dtype": plan.get("source_dtype", "recorded at conversion"),
             "runtime_revision": plan["runtime_revision"],
             "converter": "vision.cpp scripts/convert.py",
             "converter_provenance": converter_provenance or plan.get("converter_provenance")}
    if plan["task"] == "depth-estimation":
        cards["limitation"] = "relative depth only; no metric-depth claim"
    target = plan["target_repo"]
    operations = []
    manifest_candidates = []
    artifact_root = root / "publication"
    for item in passing:
        repo_path = f"{item['precision']}/{Path(item['path']).name}"
        local_path = artifact_root / repo_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["path"], local_path)
        operations.append(CommitOperationAdd(
            path_in_repo=repo_path,
            path_or_fileobj=str(local_path)))
        manifest_candidates.append({**item, "path": repo_path})
    root = Path(root)
    card_path = root / "README.md"
    card_path.write_text("# AgentQuantix vision.cpp bundle\n\n" +
                         json.dumps(cards, indent=2) + "\n", encoding="utf-8")
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path)))
    manifest = {"schema_version": 1, "pipeline": "vision", "engine": "vision.cpp",
                "task": plan["task"], "family": plan["family"], "target_repo": target,
                "candidates": manifest_candidates, "card": cards}
    manifest_path = root / "publication" / "manifest.json"
    _write(manifest_path, manifest)
    operations.append(CommitOperationAdd(path_in_repo="manifest.json",
                                         path_or_fileobj=str(manifest_path)))
    api.create_repo(target, repo_type="model", exist_ok=True)
    api.create_commit(repo_id=target, repo_type="model", operations=operations,
                       commit_message="Add accepted vision.cpp precision bundle")
    return {"target_repo": target, "precisions": [item["precision"] for item in passing]}


def publish_run(run_id, precision, root=None):
    root = Path(root or config.TEMP_DIR / "aqx-vision-runs" / run_id)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    selected = next((item for item in result["candidates"]
                     if item["precision"].casefold() == precision.casefold()), None)
    if not selected or selected["status"] != "passed":
        raise VisionValidationError(f"{precision} is not an accepted vision precision in {run_id}")
    published = publish_bundle(root, result["plan"], result["candidates"],
                               converter_provenance=result.get("converter"))
    result["published"] = True
    result["publication"] = published
    _write(root / "result.json", result)
    return {"run_id": run_id, **published, "published": True}


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
