"""Run the pinned PyTorch references for depth-anything.cpp."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch


def _engine_scripts(source):
    scripts = Path(source) / "scripts"
    package = types.ModuleType("scripts")
    package.__path__ = [str(scripts)]
    package.__package__ = "scripts"
    package.__file__ = str(scripts)
    sys.modules["scripts"] = package
    sys.path.insert(0, str(source))


def run(family, engine_source, reference_source, checkpoint, images, output):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    _engine_scripts(engine_source)
    arrays = {}
    metadata = {}
    if family == "da2-metric-hypersim-small":
        os.environ["DA2_SRC"] = str(reference_source)
        from scripts.da2_reference import load_da2_model

        model = load_da2_model("vits", str(checkpoint), 20.0)
        for sample in images:
            image = cv2.imread(sample["local_path"], cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read image {sample['local_path']}")
            depth = model.infer_image(image, input_size=518).astype(np.float32)
            arrays[f"depth_{sample['id']}"] = depth
            metadata[str(sample["id"])] = {
                "source_shape": list(image.shape[:2]),
                "output_shape": list(depth.shape),
            }
    elif family == "da3-small":
        sys.path.insert(0, str(Path(reference_source) / "src"))
        __import__("e3nn.o3")
        from torchvision import transforms

        from scripts.da3_reference import load_model

        model, net = load_model(str(checkpoint))
        model.to(device="cpu")
        net = net.to(device="cpu").eval().float()
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def normalize_image(image):
            rgb = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)
            return normalize(tensor)

        model.input_processor._normalize_image = normalize_image
        for sample in images:
            processed, _, _ = model.input_processor(
                [sample["local_path"]], process_res=504,
                process_res_method="upper_bound_resize", num_workers=1,
                print_progress=False, sequential=True)
            with torch.inference_mode():
                result = net(processed[None].float())
            key = str(sample["id"])
            arrays[f"depth_{key}"] = np.squeeze(
                result["depth"].detach().cpu().numpy()).astype(np.float32)
            details = {
                "source_shape": list(cv2.imread(sample["local_path"]).shape[:2]),
                "model_shape": list(processed.shape[-2:]),
            }
            for name in ("extrinsics", "intrinsics"):
                value = result.get(name)
                if value is not None:
                    arrays[f"{name}_{key}"] = np.squeeze(
                        value.detach().cpu().numpy()).astype(np.float32)
                    details[f"{name}_shape"] = list(arrays[f"{name}_{key}"].shape)
            metadata[key] = details
    else:
        raise ValueError(f"unsupported depth reference family: {family}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: depth_reference FAMILY ENGINE_SOURCE REFERENCE_SOURCE "
            "CHECKPOINT IMAGES_JSON OUTPUT_NPZ")
    family, engine_source, reference_source, checkpoint, images_path, output = sys.argv[1:]
    images = json.loads(Path(images_path).read_text(encoding="utf-8"))["images"]
    run(family, engine_source, reference_source, checkpoint, images, output)


if __name__ == "__main__":
    main()
