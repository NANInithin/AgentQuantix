"""Generate authored image and video fixtures for multimodal release tests."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


def _font_path():
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
    )
    return next((path for path in candidates if path.is_file()), Path("DejaVuSans.ttf"))


def _run(args):
    print("$ " + " ".join(map(str, args)), flush=True)
    subprocess.run([str(value) for value in args], check=True)


def _document(output):
    image = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(_font_path()), 44)
    for y, text in zip((62, 164, 266),
                       ("INVOICE 4821", "Total: 37.50 EUR", "Due 2026-11-30")):
        draw.text((56, y), text, font=font, fill="black")
    image.save(output / "document.png", optimize=False)


def _video(output):
    extracted = output / "video-frames"
    extracted.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aqx-multimodal-frames-") as temporary:
        source = Path(temporary)
        for index in range(16):
            image = Image.new("RGB", (320, 240), "white")
            draw = ImageDraw.Draw(image)
            if index < 8:
                draw.ellipse((112, 72, 208, 168), fill=(220, 20, 60))
            else:
                draw.rectangle((112, 72, 208, 168), fill=(30, 90, 220))
            image.save(source / f"frame-{index:02d}.png", optimize=False)
        video = output / "red-circle-then-blue-square.mp4"
        _run(["ffmpeg", "-hide_banner", "-y", "-framerate", "4", "-start_number", "0",
              "-i", source / "frame-%02d.png", "-frames:v", "16", "-c:v", "libx264",
              "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", "-an",
              "-r", "4", video])
    _run(["ffmpeg", "-hide_banner", "-y", "-i", output / "red-circle-then-blue-square.mp4",
          "-vf", "fps=4", "-vsync", "0", "-start_number", "0",
          extracted / "frame-%02d.png"])


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _document(output)
    _video(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "src/agentquantix/fixtures/multimodal")
    generate(parser.parse_args().output)
