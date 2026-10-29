"""Generate stable, authored shape fixtures for CLIP retrieval tests."""
from pathlib import Path
import argparse

from PIL import Image, ImageDraw


def generate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    size, center, radius = 224, (112, 112), 64
    cases = {
        "red-circle.png": ("red", "circle"),
        "blue-square.png": ("blue", "square"),
        "green-triangle.png": ("green", "triangle"),
        "yellow-star.png": ("yellow", "star"),
        "black-stripes.png": ("black", "stripes"),
    }
    for filename, (color, shape) in cases.items():
        image = Image.new("RGB", (size, size), "white")
        draw = ImageDraw.Draw(image)
        if shape == "circle":
            draw.ellipse((center[0] - radius, center[1] - radius,
                          center[0] + radius, center[1] + radius), fill=color)
        elif shape == "square":
            draw.rectangle((48, 48, 176, 176), fill=color)
        elif shape == "triangle":
            draw.polygon(((112, 42), (42, 176), (182, 176)), fill=color)
        elif shape == "star":
            import math
            points = [(112 + (70 if i % 2 == 0 else 30) * math.cos(i * math.pi / 5 - math.pi / 2),
                       112 + (70 if i % 2 == 0 else 30) * math.sin(i * math.pi / 5 - math.pi / 2))
                      for i in range(10)]
            draw.polygon(points, fill=color)
        else:
            for x in range(32, 224, 48):
                draw.rectangle((x, 0, x + 18, 223), fill=color)
        image.save(output / filename, format="PNG", optimize=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        "src/agentquantix/fixtures/retrieval")
    generate(parser.parse_args().output)
