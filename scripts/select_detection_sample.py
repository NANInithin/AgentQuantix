import argparse
import json
from pathlib import Path


IMAGE_IDS = (785, 872, 885, 1000, 1268, 2473, 5001, 274219, 427077, 560880)
ALLOWED_LICENSES = {4, 7, 8}


def select(instances, keypoints, manifest):
    ids = set(IMAGE_IDS)
    instance_images = {item["id"]: item for item in instances["images"]}
    keypoint_images = {item["id"]: item for item in keypoints["images"]}
    records = {item["id"]: item for item in manifest["images"]}
    images = []
    for image_id in IMAGE_IDS:
        item = instance_images[image_id]
        if item["license"] not in ALLOWED_LICENSES:
            raise ValueError(f"COCO image {image_id} has an unapproved license")
        if not any(annotation["image_id"] == image_id
                   for annotation in instances["annotations"]):
            raise ValueError(f"COCO image {image_id} has no instance annotations")
        if not any(annotation["image_id"] == image_id and annotation["category_id"] == 1
                   and any(annotation["keypoints"][2::3])
                   for annotation in keypoints["annotations"]):
            raise ValueError(f"COCO image {image_id} has no visible person keypoints")
        record = records[image_id]
        images.append({
            **item,
            "license_id": item["license"],
            "license_name": record["license"],
            "license_url": record["license_url"],
            "url": record["url"],
            "sha256": record["sha256"],
            "size": record["size"],
        })
    return {
        "info": instances.get("info", {}),
        "annotation_attribution": "COCO annotations are licensed under CC BY 4.0.",
        "licenses": [item for item in instances["licenses"]
                     if item["id"] in ALLOWED_LICENSES],
        "images": images,
        "categories": instances["categories"],
        "annotations": [item for item in instances["annotations"]
                        if item["image_id"] in ids],
        "keypoint_categories": keypoints["categories"],
        "keypoint_annotations": [item for item in keypoints["annotations"]
                                 if item["image_id"] in ids],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--keypoints", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=(
        Path(__file__).parents[1] / "src/agentquantix/fixtures/detection/"
        "coco_val2017_images.json"))
    parser.add_argument("--output", type=Path, default=(
        Path(__file__).parents[1] / "src/agentquantix/fixtures/detection/"
        "coco_val2017_annotations.json"))
    args = parser.parse_args()
    instances = json.loads(args.instances.read_text(encoding="utf-8"))
    keypoints = json.loads(args.keypoints.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = select(instances, keypoints, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, separators=(",", ":")) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
