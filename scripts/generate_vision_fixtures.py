from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = (Path(__file__).resolve().parents[1] / "src" / "agentquantix"
        / "fixtures" / "vision")


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def textured_background(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:height, :width]
    noise = rng.integers(-10, 11, size=(height, width, 1))
    rgb = np.stack(
        (
            68 + x * 0.16 + y * 0.04,
            104 + y * 0.10 - x * 0.03,
            139 + x * 0.05 - y * 0.07,
        ),
        axis=-1,
    )
    rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(image)
    for x0 in range(12, width, 28):
        draw.line((x0, 0, x0, height), fill=(82, 115, 145), width=1)
    for y0 in range(9, height, 24):
        draw.line((0, y0, width, y0), fill=(75, 118, 142), width=1)
    return np.asarray(image).copy()


def make_segmentation() -> tuple[Image.Image, Image.Image]:
    width, height = 320, 240
    image = Image.fromarray(textured_background(width, height, 20261101), "RGB")
    mask = Image.new("L", (width, height), 0)
    draw_image = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.ellipse((80, 42, 240, 168), fill=255)
    draw_mask.rectangle((139, 132, 181, 210), fill=255)
    draw_image.ellipse((80, 42, 240, 168), fill=(225, 66, 32), outline=(250, 208, 85), width=3)
    draw_image.rectangle((139, 132, 181, 210), fill=(230, 105, 38), outline=(250, 208, 85), width=2)
    draw_image.ellipse((126, 82, 194, 122), fill=(248, 178, 42), outline=(255, 224, 112), width=2)
    return image, mask


def make_depth_scene(seed: int, variant: int) -> Image.Image:
    width, height = 512, 384
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:height, :width]
    sky = np.stack(
        (
            105 + x * 0.05 - y * 0.04,
            148 + x * 0.02 - y * 0.02,
            188 + x * 0.01,
        ),
        axis=-1,
    )
    ground = np.stack(
        (
            50 + x * 0.035 + y * 0.025,
            94 + x * 0.02 + y * 0.055,
            46 + y * 0.03,
        ),
        axis=-1,
    )
    split = int(165 + variant * 19)
    rgb = np.where((y < split)[..., None], sky, ground)
    rgb += rng.normal(0, 3.5, size=rgb.shape)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [(0, split + 30), (90, split - 36), (204, split + 12), (348, split - 54), (512, split + 28), (512, 260), (0, 260)],
        fill=(62 + variant * 5, 103, 77 + variant * 4),
    )
    draw.rectangle((38 + variant * 15, 210, 140 + variant * 15, 362), fill=(150, 104, 65), outline=(78, 54, 38), width=4)
    draw.polygon(
        [(20 + variant * 15, 215), (88 + variant * 15, 158), (158 + variant * 15, 215)],
        fill=(112, 61, 47),
    )
    draw.rectangle((67 + variant * 15, 260, 103 + variant * 15, 310), fill=(71, 115, 151))
    draw.ellipse((330 - variant * 17, 205, 424 - variant * 17, 302), fill=(40, 115 + variant * 8, 61), outline=(22, 74, 40), width=3)
    for offset in range(0, 512, 40):
        draw.line((offset, 320, offset - 50, 382), fill=(87, 123, 57), width=2)
    return image


def make_two_plane() -> Image.Image:
    width, height = 384, 256
    rng = np.random.default_rng(20261102)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    top_noise = rng.integers(-7, 8, size=(height // 2, width, 1))
    bottom_noise = rng.integers(-12, 13, size=(height // 2, width, 1))
    rgb[: height // 2] = np.clip(np.array([92, 147, 199]) + top_noise, 0, 255)
    rgb[height // 2 :] = np.clip(np.array([55, 111, 58]) + bottom_noise, 0, 255)
    image = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height // 2 - 1), outline=(242, 229, 190), width=3)
    draw.rectangle((0, height // 2, width - 1, height - 1), outline=(36, 75, 39), width=3)
    return image


def make_inpainting() -> tuple[Image.Image, Image.Image]:
    size = 512
    y, x = np.mgrid[:size, :size]
    rgb = np.stack(
        (
            36 + 0.28 * x + 0.11 * y,
            54 + 0.12 * x + 0.24 * y,
            172 - 0.16 * x + 0.08 * y,
        ),
        axis=-1,
    )
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(image)
    for k in range(24, size, 32):
        draw.line((0, k, size, k + 28), fill=(238, 205, 80), width=3)
        draw.line((k, 0, k - 24, size), fill=(70, 211, 191), width=2)
    draw.ellipse((72, 75, 168, 171), fill=(241, 72, 48), outline=(255, 255, 255), width=4)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rectangle((190, 158, 328, 350), fill=255)
    return image, mask


def make_super_resolution() -> tuple[Image.Image, Image.Image]:
    size = 256
    y, x = np.mgrid[:size, :size]
    rgb = np.stack(
        (
            95 + 55 * np.sin(x / 19) + 0.18 * y,
            100 + 62 * np.cos(y / 23) + 0.12 * x,
            110 + 48 * np.sin((x + y) / 31),
        ),
        axis=-1,
    )
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    draw.polygon([(24, 216), (81, 56), (139, 216)], fill=(42, 108, 55), outline=(239, 221, 130))
    draw.polygon([(124, 216), (182, 34), (238, 216)], fill=(58, 121, 68), outline=(248, 230, 152))
    draw.rectangle((23, 18, 80, 47), fill=(213, 173, 80), outline=(60, 46, 27), width=2)
    draw.ellipse((150, 148, 214, 211), fill=(205, 73, 58), outline=(255, 218, 103), width=3)
    for k in range(0, size, 8):
        draw.line((0, k, size, k), fill=(255, 255, 255), width=1)
    low = image.resize((size // 4, size // 4), Image.Resampling.BICUBIC)
    return image, low


def main() -> None:
    items: dict[str, Image.Image] = {}
    seg_image, seg_mask = make_segmentation()
    items["segmentation/image.png"] = seg_image
    items["segmentation/mask.png"] = seg_mask
    for i in range(1, 4):
        items[f"depth/scene-{i}.png"] = make_depth_scene(20261110 + i, i)
    items["depth/two-plane.png"] = make_two_plane()
    inp_image, inp_mask = make_inpainting()
    items["inpainting/image.png"] = inp_image
    items["inpainting/mask.png"] = inp_mask
    sr_image, sr_low = make_super_resolution()
    items["super-resolution/reference.png"] = sr_image
    items["super-resolution/input-x4.png"] = sr_low

    manifest = {
        "seed": 20261101,
        "fixtures": [],
        "prompts": {"segmentation_point_xy": [160, 105], "segmentation_box_xyxy": [77, 40, 244, 214]},
        "mask_convention": "MI-GAN input mask is 255 for the rectangular hole and 0 for known pixels; vision-cli inverts it internally",
        "depth_two_plane": "upper half is farther sky; lower half is nearer ground",
    }
    for relative, image in sorted(items.items()):
        path = ROOT / relative
        save_image(path, image)
        raw = path.read_bytes()
        manifest["fixtures"].append(
            {
                "path": relative,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
