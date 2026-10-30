import argparse
import json
from pathlib import Path


def _array(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return value.tolist() if hasattr(value, "tolist") else value


def _mask_paths(value, image_id, directory):
    if value is None:
        return []
    import numpy as np
    masks = np.asarray(_array(value))
    if masks.ndim == 2:
        masks = masks[None, ...]
    paths = []
    directory.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(masks):
        path = directory / f"{image_id}-{index:03d}.npy"
        np.save(path, np.asarray(mask >= 0.5, dtype=np.uint8))
        paths.append(str(path))
    return paths


def _load_model(engine, task, weights):
    if engine == "ultralytics-ggml":
        from ultralytics import YOLO
        return YOLO(str(weights))
    if task == "image-segmentation":
        from rfdetr import RFDETRSegNano
        return RFDETRSegNano(pretrain_weights=str(weights))
    from rfdetr import RFDETRNano
    return RFDETRNano(pretrain_weights=str(weights))


def _yolo(model, task, image):
    if task == "image-classification":
        results = model.predict(str(image), imgsz=224, device="cpu", batch=1,
                                verbose=False)
    else:
        results = model.predict(str(image), imgsz=640, conf=0.001, iou=0.7,
                                max_det=300, device="cpu", verbose=False)
    result = results[0]
    if task == "image-classification":
        probabilities = _array(result.probs.data)
        order = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
        return {"probabilities": probabilities, "top5": order[:5], "top1": order[0]}
    boxes = result.boxes
    record = {
        "boxes": _array(boxes.xyxy) if boxes is not None else [],
        "scores": _array(boxes.conf) if boxes is not None else [],
        "class_ids": _array(boxes.cls) if boxes is not None else [],
        "class_names": model.names,
    }
    if task == "keypoint-detection" and result.keypoints is not None:
        record["keypoints"] = _array(result.keypoints.xy)
        record["keypoint_scores"] = _array(result.keypoints.conf)
    if task == "image-segmentation" and result.masks is not None:
        record["masks"] = _array(result.masks.data)
    return record


def _rfdetr(model, task, image):
    from PIL import Image
    result = model.predict(Image.open(image).convert("RGB"), threshold=0.001)
    return {
        "boxes": _array(getattr(result, "xyxy", [])),
        "scores": _array(getattr(result, "confidence", [])),
        "class_ids": _array(getattr(result, "class_id", [])),
        "class_names": getattr(result, "class_names", None),
        "masks": _array(getattr(result, "mask", None)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.images.read_text(encoding="utf-8"))
    model = _load_model(args.engine, args.task, args.weights)
    records = []
    for item in manifest["images"]:
        image = Path(item["local_path"])
        prediction = (_yolo(model, args.task, image)
                      if args.engine == "ultralytics-ggml"
                      else _rfdetr(model, args.task, image))
        if prediction.get("masks") is not None:
            prediction["mask_paths"] = _mask_paths(
                prediction.pop("masks"), item["id"], args.mask_dir)
        records.append({"image_id": item["id"], **prediction})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"task": args.task, "images": records},
                                      separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
