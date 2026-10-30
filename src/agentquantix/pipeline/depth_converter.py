"""Invoke the pinned depth-anything.cpp converters with their reference sources."""
from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path


def main():
    if len(sys.argv) < 6:
        raise SystemExit(
            "usage: depth_converter ENGINE_SOURCE REFERENCE_SOURCE FAMILY SOURCE OUTPUT")
    engine_source, reference_source, family, source, output = sys.argv[1:6]
    engine_source, reference_source = Path(engine_source), Path(reference_source)
    scripts = engine_source / "scripts"
    package = types.ModuleType("scripts")
    package.__path__ = [str(scripts)]
    package.__package__ = "scripts"
    package.__file__ = str(scripts)
    sys.modules["scripts"] = package
    sys.path.insert(0, str(engine_source))
    if family == "da2-metric-hypersim-small":
        os.environ["DA2_SRC"] = str(reference_source)
        converter = scripts / "convert_da2_to_gguf.py"
        sys.argv = [str(converter), "--encoder", "vits", "--ckpt", source,
                    "--output", output,
                    "--name", "Depth-Anything-V2-Metric-Hypersim-Small",
                    "--max-depth", "20"]
    elif family == "da3-small":
        sys.path.insert(0, str(reference_source / "src"))
        __import__("e3nn.o3")

        converter = scripts / "convert_da3_to_gguf.py"
        sys.argv = [str(converter), "--model", source, "--output", output]
    else:
        raise ValueError(f"unsupported converter family: {family}")
    runpy.run_path(str(converter), run_name="__main__")


if __name__ == "__main__":
    main()
