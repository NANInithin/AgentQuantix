"""Run explicit local native acceptance recipes and write auditable evidence.

No downloading or publishing. Recipes name exact local models/binaries and
hashes. Native acceptance is separate from mocked/offline unit tests.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentquantix import __version__, media, media_quality, voice  # noqa: E402

TEXT_VOICE_TRACKS = {"text", "llama-tts", "audio.cpp-tts", "audio.cpp-asr", "whisper-asr"}
MEDIA_TRACKS = {"media-single-image", "media-multi-image", "media-edit", "media-video", "media-minimax-h3"}


def run_suite(spec, output, require_media=False):
    entries = spec.get("recipes", [])
    required = TEXT_VOICE_TRACKS | (MEDIA_TRACKS if require_media else set())
    report = {"version": __version__, "platform": platform.platform(), "at": time.time(),
              "required_tracks": sorted(required), "results": [], "complete": False}
    for recipe in entries:
        began = time.monotonic()
        record = {"track": recipe["track"], "name": recipe["name"], "passed": False}
        try:
            if not recipe.get("files"):
                raise ValueError("native recipe must pin binaries and model files by SHA-256")
            for name, expected in recipe["files"].items():
                if media.file_hash(name) != expected:
                    raise ValueError(f"native acceptance input changed: {name}")
            for command in recipe["commands"]:
                if command[0] not in recipe["files"]:
                    raise ValueError("each native executable must be pinned in files")
                result = subprocess.run(command, cwd=recipe.get("cwd"), capture_output=True, text=True,
                                        errors="replace", timeout=recipe.get("timeout_seconds", 1800))
                record.setdefault("commands", []).append({"argv": command, "returncode": result.returncode,
                                                            "stdout": result.stdout, "stderr": result.stderr})
                if result.returncode:
                    raise ValueError(f"native command exited {result.returncode}")
            artifact = Path(recipe["output"])
            if not artifact.is_file() or not artifact.stat().st_size:
                raise ValueError("native recipe produced no artifact")
            if recipe["track"] in {"llama-tts", "audio.cpp-tts"}:
                record["quality"] = voice.inspect_wav(artifact)
                if record["quality"].get("silence_ratio", 0) > 0.98 or record["quality"].get("clipping_ratio", 0) > 0.02:
                    raise ValueError("native TTS failed silence/clipping gate")
            elif recipe["track"] in {"audio.cpp-asr", "whisper-asr"}:
                wer = voice.word_error_rate(recipe["expected_transcript"], artifact.read_text(encoding="utf-8"))
                record["wer"] = wer
                if wer > recipe.get("max_wer", 0.35):
                    raise ValueError("native ASR failed transcript gate")
            elif recipe["track"].startswith("media-"):
                quality = media_quality.assess(artifact, recipe["runtime_recipe"], recipe.get("baseline"),
                                              recipe["quality_profile"], Path(output).parent / recipe["name"])
                record["quality"] = quality
                if not quality["passed"]:
                    raise ValueError("native media quality gates failed")
            elif recipe["track"] == "text":
                if recipe.get("expected_text") not in artifact.read_text(encoding="utf-8"):
                    raise ValueError("native text inference did not produce the expected response")
            record["passed"] = True
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            record["error"] = str(error)
        record["duration_seconds"] = time.monotonic() - began
        report["results"].append(record)
    covered = {r["track"] for r in report["results"] if r["passed"]}
    report["missing_tracks"] = sorted(required - covered)
    report["complete"] = not report["missing_tracks"] and all(r["passed"] for r in report["results"])
    media.atomic_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-media", action="store_true")
    args = parser.parse_args()
    result = run_suite(json.loads(args.spec.read_text(encoding="utf-8")), args.output, args.require_media)
    print(json.dumps({"complete": result["complete"], "missing_tracks": result["missing_tracks"]}))
    sys.exit(0 if result["complete"] else 1)
