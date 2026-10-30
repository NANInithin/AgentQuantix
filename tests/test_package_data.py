import ast
from pathlib import Path

import agentquantix
from agentquantix import detection


def test_package_data_loader_paths_exist():
    source_root = Path(agentquantix.__file__).resolve().parent
    package_sources = source_root.rglob("*.py")
    references = set()
    for source in package_sources:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "package_path" or not node.args:
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                references.add(argument.value)

    assert references
    assert all((source_root / relative).exists() for relative in references)
    assert "fixtures/vision_families.json" in references
    assert "fixtures/detection/coco_val2017_images.json" in references


def test_detection_manifest_is_packaged_and_loadable():
    manifest = detection.load_coco_image_manifest()

    assert len(manifest["images"]) == 10
    assert all(image["license_id"] in {4, 7} for image in manifest["images"])
