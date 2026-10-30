import hashlib
from pathlib import Path

import agentquantix
from agentquantix import detection, vision
from agentquantix.actions import route_named


source_root = Path(__file__).resolve().parents[1] / "src" / "agentquantix" / "fixtures"
installed_root = Path(agentquantix.__file__).resolve().parent / "fixtures"
expected = {path.relative_to(source_root) for path in source_root.rglob("*") if path.is_file()}
installed = {path.relative_to(installed_root) for path in installed_root.rglob("*") if path.is_file()}
assert expected == installed, {
    "missing": sorted(str(path) for path in expected - installed),
    "unexpected": sorted(str(path) for path in installed - expected),
}

for relative in expected:
    source = source_root / relative
    installed_file = installed_root / relative
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(
        installed_file.read_bytes()).digest(), relative

repo = "depth-anything/Depth-Anything-V2-Metric-Hypersim-Small"
route = route_named(repo)
assert route.engine == "depth-anything.cpp", route
assert vision.families()
assert len(detection.load_coco_image_manifest()["images"]) == 10
assert not any(path.suffix.lower() in {".jpg", ".jpeg"} for path in
               (installed_root / "detection").rglob("*"))
