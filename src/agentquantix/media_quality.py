"""Mandatory structural checks and explicitly pinned optional media evaluators."""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess

from . import media


def _dependencies():
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise media.MediaValidationError("install agentquantix[media] for structural media validation") from error
    return np, Image


def inspect_image(path, settings):
    np, Image = _dependencies()
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != (settings["width"], settings["height"]):
                raise ValueError("wrong image dimensions")
            if image.mode not in ("RGB", "RGBA", "L", "I;16"):
                raise ValueError(f"unexpected image channels/bit depth: {image.mode}")
            if image.mode == "RGBA" and image.getextrema()[-1][1] == 0:
                raise ValueError("fully transparent output")
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
            if not np.isfinite(pixels).all() or float(pixels.std()) < 1 or float(np.ptp(pixels)) < 3:
                raise ValueError("blank or uniformly colored output")
            return {"width": image.width, "height": image.height, "channels": len(image.getbands()),
                    "mode": image.mode, "pixel_std": float(pixels.std()), "dynamic_range": float(np.ptp(pixels)),
                    "sha256": media.file_hash(path)}
    except (OSError, ValueError) as error:
        raise media.MediaValidationError(f"invalid image: {error}") from error


def _native(command):
    result = subprocess.run([str(c) for c in command], capture_output=True, timeout=300)
    if result.returncode:
        raise media.MediaValidationError("media decoding failed: " + result.stderr.decode(errors="replace")[-2000:])
    return result.stdout


def inspect_video(path, settings, *, require_audio=False, sample_directory=None):
    np, Image = _dependencies()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise media.MediaValidationError("video validation requires ffmpeg and ffprobe on PATH")
    try:
        facts = json.loads(_native([ffprobe, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", path]))
        stream = next(s for s in facts["streams"] if s["codec_type"] == "video")
        width, height = stream["width"], stream["height"]
        fps = float(Fraction(stream.get("avg_frame_rate") or stream["r_frame_rate"]))
        count = int(stream.get("nb_read_frames") or stream["nb_frames"])
        if (width, height) != (settings["width"], settings["height"]):
            raise ValueError("wrong video dimensions")
        if count != settings["frames"] or abs(fps - settings["fps"]) > 0.01:
            raise ValueError("wrong video frame count or FPS")
        expected_bytes = count * width * height * 3
        if expected_bytes > 512 * 1024 * 1024:
            raise ValueError("fixture decode exceeds 512 MiB; use a smaller acceptance fixture")
        raw = _native([ffmpeg, "-v", "error", "-i", path, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        if len(raw) != expected_bytes:
            raise ValueError("not all expected video frames decoded")
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3)
        std = frames.reshape(count, -1).std(axis=1)
        if (std < 1).any():
            raise ValueError("blank or black video frame")
        changes = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
        if count > 1 and float(changes.max()) < 0.1:
            raise ValueError("frozen video")
        if count > 2 and (changes > 150).any():
            raise ValueError("gross flash or temporal discontinuity")
        if sample_directory:
            destination = Path(sample_directory)
            destination.mkdir(parents=True, exist_ok=True)
            for index in sorted({0, count // 2, count - 1}):
                Image.fromarray(frames[index]).save(destination / f"frame-{index:06d}.png")
        result = {"width": width, "height": height, "frames": count, "fps": fps,
                  "duration_seconds": count / fps, "max_frame_change": float(changes.max()) if len(changes) else 0,
                  "sha256": media.file_hash(path)}
        if require_audio:
            audio = next((s for s in facts["streams"] if s["codec_type"] == "audio"), None)
            if not audio or audio["channels"] != 2:
                raise ValueError("required stereo audio stream missing")
            rate = int(audio["sample_rate"])
            if rate != settings.get("audio_sample_rate", 32000):
                raise ValueError("wrong audio sample rate")
            audio_raw = _native([ffmpeg, "-v", "error", "-i", path, "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", "-"])
            samples = np.frombuffer(audio_raw, dtype="<f4")
            if not samples.size or not np.isfinite(samples).all():
                raise ValueError("empty or invalid decoded audio")
            duration = samples.size / 2 / rate
            if abs(duration - result["duration_seconds"]) > max(0.1, 1 / fps):
                raise ValueError("audio/video duration mismatch")
            silence, clipping = float((np.abs(samples) < 0.001).mean()), float((np.abs(samples) >= 0.999).mean())
            if silence > 0.98 or clipping > 0.02:
                raise ValueError("silent or clipped audio")
            result["audio"] = {"channels": 2, "sample_rate": rate, "duration_seconds": duration,
                               "silence_ratio": silence, "clipping_ratio": clipping}
        return result
    except (KeyError, StopIteration, ValueError, ZeroDivisionError, OSError) as error:
        raise media.MediaValidationError(f"invalid video/audio container: {error}") from error


def _evaluator(profile, output, baseline, recipe, root):
    evaluator = profile.get("evaluator", {})
    command = evaluator.get("command", [])
    if (not command or not evaluator.get("sha256") or not evaluator.get("model_revision")
            or not evaluator.get("name")):
        raise media.MediaValidationError("perceptual/prompt evaluator is unavailable or not pinned")
    executable = Path(shutil.which(command[0]) or command[0])
    if not executable.is_file() or media.file_hash(executable) != evaluator["sha256"]:
        raise media.MediaValidationError("quality evaluator executable changed")
    for argument in command[1:]:
        if Path(argument).is_file() and evaluator.get("artifacts", {}).get(argument) != media.file_hash(argument):
            raise media.MediaValidationError(f"quality evaluator artifact is not pinned: {argument}")
    request = {"candidate": str(Path(output).resolve()), "baseline": str(Path(baseline or output).resolve()),
               "repeat": str(Path(recipe["repeat_output"]).resolve()) if recipe.get("repeat_output") else None,
               "prompt": recipe["prompt"], "conditioning": recipe.get("conditioning"),
               "model_revision": evaluator["model_revision"], "settings": recipe["settings"]}
    request_path = Path(root) / "evaluation-request.json"
    media.atomic_json(request_path, request)
    result = subprocess.run([*command, str(request_path)], capture_output=True, text=True,
                            timeout=evaluator.get("timeout_seconds", 600))
    if result.returncode:
        raise media.MediaValidationError("pinned quality evaluator failed: " + result.stderr[-2000:])
    scores = json.loads(result.stdout)
    if scores.get("model_revision") != evaluator["model_revision"]:
        raise media.MediaValidationError("quality evaluator returned a different model revision")
    for value in scores.get("metrics", {}).values():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise media.MediaValidationError("quality evaluator returned non-finite metrics")
    return {"name": evaluator["name"], "sha256": evaluator["sha256"], **scores}


def assess(output, recipe, baseline, profile, root):
    facts = (inspect_video(output, recipe["settings"], require_audio=recipe.get("requires_audio", "audio_vae" in recipe["components"]),
                           sample_directory=Path(root) / "review-frames")
             if recipe["mode"] == "vid_gen" else inspect_image(output, recipe["settings"]))
    result = {"structural_passed": True, "passed": False, "structure": facts,
              "baseline_sha256": media.file_hash(baseline) if baseline else facts["sha256"],
              "unavailable_metrics": [], "evaluator": None, "rejected_metrics": []}
    required = ["perceptual_similarity", "prompt_alignment", "determinism"]
    if recipe.get("conditioning"):
        required.append("conditioning_preservation")
    if recipe["mode"] == "vid_gen":
        required.append("temporal_consistency")
    try:
        if not profile.get("calibration_evidence"):
            raise media.MediaValidationError("family/task quality thresholds have not been calibrated")
        evaluation = _evaluator(profile, output, baseline, recipe, root)
        result["evaluator"] = evaluation
        for metric in required:
            minimum = profile.get("thresholds", {}).get(metric)
            measured = evaluation.get("metrics", {}).get(metric)
            if minimum is None or measured is None:
                result["unavailable_metrics"].append(metric)
            elif measured < minimum:
                result["rejected_metrics"].append({"metric": metric, "value": measured, "minimum": minimum})
        if not recipe.get("repeat_output"):
            result["unavailable_metrics"].append("determinism")
        if baseline is None and not recipe.get("is_baseline"):
            result["unavailable_metrics"].append("high_precision_baseline")
        result["passed"] = not (result["unavailable_metrics"] or result["rejected_metrics"])
    except (media.MediaValidationError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        result["unavailable_metrics"] = required
        result["evaluation_error"] = str(error)
    return result
