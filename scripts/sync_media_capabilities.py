"""Regenerate or check the evidence-bearing catalog from the exact engine pin."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentquantix.media_catalog import CATALOG_PATH, sync_catalog  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--output", type=Path, default=CATALOG_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = sync_catalog(args.checkout, args.output, args.check)
    print(f"{len(result['capabilities'])} loader families at {result['revision']}")
