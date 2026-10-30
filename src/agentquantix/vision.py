"""vision.cpp family contracts, source precision and converter environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

from .package_data import package_path


CATALOG_PATH = package_path("fixtures/vision_families.json")
VISIONCPP_REVISION = "b766b0acc30f6706f1bfe46769635e361172ce96"
VISIONCPP_TAG = "v0.3.0"


def families():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def family_for(repo_id, config=None, family=None, *, engine=None, task=None):
    catalog = [item for item in families()
               if (engine is None or item.get("engine", "vision.cpp") == engine)
               and (task is None or item["task"] == task)]
    if family:
        requested = str(family).casefold()
        if ":" in requested:
            target_engine, requested = requested.split(":", 1)
            catalog = [item for item in catalog
                       if item.get("engine", "vision.cpp").casefold() == target_engine]
        return next((item for item in catalog
                     if item["family"].casefold() == requested), None)
    repo = (repo_id or "").casefold()
    if ":" in repo:
        target_engine, target_family = repo.split(":", 1)
        targeted = next((item for item in catalog
                         if item.get("engine", "vision.cpp").casefold() == target_engine
                         and item["family"].casefold() == target_family), None)
        if targeted:
            return targeted
    match = next((item for item in catalog
                  if any(repo == value.casefold() for value in item["repos"])), None)
    if match:
        return match
    config = config or {}
    model_type = str(config.get("model_type") or "").casefold()
    architectures = {str(value).casefold()
                     for value in config.get("architectures", [])}
    depth_type = str(config.get("depth_estimation_type") or "").casefold()
    return next((item for item in catalog
                 if ((model_type and model_type in {value.casefold()
                                                    for value in item["model_types"]})
                     or bool(architectures & {value.casefold()
                                              for value in item["architectures"]}))
                 and (item.get("depth_estimation_type") != "metric" or depth_type == "metric")
                 and (item.get("depth_estimation_type") != "relative" or depth_type != "metric")), None)


def _tensor_dtypes(safetensors_path):
    with Path(safetensors_path).open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(length))
    return {item.get("dtype") for name, item in header.items()
            if name != "__metadata__" and isinstance(item, dict)}


def source_dtypes(source_dir, source_files=None, *, python=None, runner=subprocess.run):
    root = Path(source_dir)
    files = [root / name for name in (source_files or [])
             if (root / name).is_file()]
    safe = [path for path in files if path.suffix == ".safetensors"]
    found = set().union(*(_tensor_dtypes(path) for path in safe)) if safe else set()
    weights = [path for path in files if path.suffix in (".pt", ".pth")]
    if not safe and not weights:
        raise ValueError("no source weights found to determine floating tensor dtype")
    if weights:
        if python is None:
            raise ValueError("reading .pt/.pth source dtypes requires the converter environment")
        script = (
            "import collections.abc,json,sys,torch\n"
            "def walk(value):\n"
            " if isinstance(value,torch.Tensor): yield str(value.dtype).removeprefix('torch.')\n"
            " elif isinstance(value,collections.abc.Mapping):\n"
            "  for item in value.values(): yield from walk(item)\n"
            " elif isinstance(value,(list,tuple)):\n"
            "  for item in value: yield from walk(item)\n"
            "found=set()\n"
            "for path in sys.argv[1:]:\n"
            " found.update(walk(torch.load(path,map_location='cpu',weights_only=True)))\n"
            "print(json.dumps(sorted(found)))\n"
        )
        result = runner([str(python), "-c", script, *map(str, weights)],
                        capture_output=True, text=True, timeout=300, check=True)
        found.update(json.loads(result.stdout))
    return found


def precision_policy(dtypes):
    floating = {str(dtype).removeprefix("torch.")
                for dtype in dtypes
                if str(dtype).removeprefix("torch.") in
                {"float16", "float32", "float64", "bfloat16", "F16", "F32", "BF16", "F64"}}
    is_f32 = bool(floating) and floating <= {"float32", "F32"}
    if is_f32:
        return {"precisions": ["F32", "F16"], "source_dtype": "F32",
                "precision_reason": "source floating tensors are F32"}
    if floating:
        return {"precisions": ["F16"], "source_dtype": "F16",
                "precision_reason": "source floating tensors are not all F32; F32 promotion is disabled"}
    raise ValueError("source weights contain no supported floating-point tensors")


def resolve_python(environ=None, which=shutil.which, run=subprocess.run):
    env = os.environ if environ is None else environ
    candidates = []
    if env.get("AQX_VISION_PYTHON"):
        candidates.append(env["AQX_VISION_PYTHON"])
    for name in ("python3.13", "python3.12"):
        if path := which(name):
            candidates.append(path)
    for candidate in candidates:
        try:
            result = run([candidate, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                         capture_output=True, text=True, timeout=10, check=True)
            version = tuple(int(part) for part in result.stdout.strip().split("."))
            if version >= (3, 12):
                return str(Path(candidate).resolve())
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
    uv = which("uv")
    if uv:
        try:
            result = run([uv, "python", "find", ">=3.12"], capture_output=True,
                         text=True, timeout=30, check=True)
            candidate = result.stdout.strip().splitlines()[0]
            if candidate:
                return str(Path(candidate).resolve())
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    raise RuntimeError(
        "vision.cpp conversion needs Python >=3.12. Set AQX_VISION_PYTHON, "
        "install python3.12/python3.13, or install uv and run `uv python install 3.12`.")


def converter_venv(runtime, requirements, runner=subprocess.run):
    tool_dir = Path(runtime).resolve() / "build" / "tools"
    venv = tool_dir / "vision-converter"
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        tool_dir.mkdir(parents=True, exist_ok=True)
        interpreter = resolve_python()
        runner([interpreter, "-m", "venv", str(venv)], check=True)
    if requirements:
        args = [str(python), "-m", "pip", "install", "--disable-pip-version-check",
                "--extra-index-url", "https://download.pytorch.org/whl/cpu", *requirements]
        runner(args, check=True)
    return str(python)


def converter_command(python, converter, source, output, family, precision):
    args = [str(python), str(converter), family["converter_arch"], str(source),
            "--output", str(output)]
    if precision == "F16":
        args.extend(["--quantize", "f16"])
    return args


def iou(prediction, truth):
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    union = int((prediction | truth).sum())
    return float((prediction & truth).sum() / union) if union else 1.0


def _edge(mask):
    import numpy as np
    mask = mask.astype(bool)
    inner = mask.copy()
    height, width = mask.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            shifted = np.zeros_like(mask)
            ys = slice(max(0, dy), min(height, height + dy))
            xs = slice(max(0, dx), min(width, width + dx))
            src_y = slice(max(0, -dy), min(height, height - dy))
            src_x = slice(max(0, -dx), min(width, width - dx))
            shifted[ys, xs] = mask[src_y, src_x]
            inner &= shifted
    return mask & ~inner


def boundary_fscore(prediction, truth, tolerance=2):
    left, right = _edge(prediction), _edge(truth)
    if not left.any() or not right.any():
        return float(not left.any() and not right.any())
    import numpy as np

    def dilate(mask):
        result = mask.copy()
        for dy in range(-tolerance, tolerance + 1):
            for dx in range(-tolerance, tolerance + 1):
                if dx * dx + dy * dy > tolerance * tolerance:
                    continue
                height, width = mask.shape
                shifted = np.zeros_like(mask)
                ys = slice(max(0, dy), min(height, height + dy))
                xs = slice(max(0, dx), min(width, width + dx))
                src_y = slice(max(0, -dy), min(height, height - dy))
                src_x = slice(max(0, -dx), min(width, width - dx))
                shifted[ys, xs] = mask[src_y, src_x]
                result |= shifted
        return result

    def matched(a, b):
        return float((a & dilate(b)).sum() / a.sum())

    precision = matched(left, right)
    recall = matched(right, left)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def pixel_agreement(candidate, baseline):
    return float((candidate == baseline).mean())


def spearman(left, right):
    import numpy as np

    def ranks(values):
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(order), dtype=np.float64)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            result[order[start:end]] = (start + end - 1) / 2
            start = end
        return result

    a, b = ranks(np.asarray(left).ravel()), ranks(np.asarray(right).ravel())
    if a.size != b.size or not a.size:
        raise ValueError("Spearman inputs must have equal nonzero size")
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def depth_comparison(candidate, baseline):
    import numpy as np

    candidate, baseline = np.asarray(candidate, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if candidate.shape != baseline.shape:
        from PIL import Image
        candidate = np.asarray(Image.fromarray(candidate.astype(np.float32), mode="F").resize(
            (baseline.shape[1], baseline.shape[0]), Image.Resampling.BILINEAR))
    valid = np.isfinite(candidate) & np.isfinite(baseline)
    threshold = float(np.percentile(baseline[valid], 5))
    valid &= baseline > threshold
    x, y = candidate[valid], baseline[valid]
    scale, shift = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    aligned = scale * x + shift
    denom = np.maximum(np.abs(y), np.finfo(np.float64).eps)
    return {"spearman": spearman(candidate[valid], baseline[valid]),
            "scale_shift_absrel": float(np.mean(np.abs(aligned - y) / denom)),
            "scale": float(scale), "shift": float(shift),
            "valid_pixels": int(valid.sum()), "baseline_p05": threshold}


def depth_baseline_passes(outputs, two_plane):
    import numpy as np
    values = [np.asarray(value) for value in outputs]
    if not values or any(not np.isfinite(value).all() or
                         float(value.max()) == float(value.min()) for value in values):
        return False
    two_plane = np.asarray(two_plane)
    return float(two_plane[two_plane.shape[0] // 2:].mean()) > float(
        two_plane[:two_plane.shape[0] // 2].mean())
