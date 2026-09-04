"""Turning a bare machine into one that can quantize.

`aqx bootstrap` is the answer to "fresh VM, nothing installed". It checks what
is missing, tells you exactly how to install the parts it cannot install
itself, then clones and builds llama.cpp.

The division is deliberate. This script will happily spend twenty minutes
compiling llama.cpp, but it will NOT install system packages behind your back —
no silent apt-get, no winget, no curl-pipe-sudo. Those need a package manager,
usually root, and a choice about CUDA that depends on the machine. So they are
printed as commands you can read before running.

What it does do:
  * report every prerequisite, present or missing, with the reason it matters
  * clone llama.cpp (shallow) into the work root
  * configure and build llama-quantize and llama-imatrix, with CUDA detected
    rather than assumed
  * verify the binaries actually run afterwards
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from . import config, sysprobe, transfer
from .pipeline import build as build_mod, source as source_mod


# =====================================================
# PREREQUISITES
# =====================================================
def _python_package(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def check():
    """Every prerequisite, with whether it is present and why it is needed.

    Returns a list of dicts. `required` distinguishes "cannot work at all"
    from "will work, but worse or slower".
    """
    cuda = build_mod.has_cuda_toolkit()
    gpus = sysprobe._gpu()
    _missing_converter_deps = source_mod.converter_missing()

    items = [
        {"name": "git", "ok": bool(build_mod.find_tool("git")), "required": True,
         "why": "clones llama.cpp, and any fork needed for a new architecture",
         "install": {"linux": "sudo apt-get install -y git",
                     "win32": "winget install --id Git.Git"}},
        {"name": "cmake", "ok": bool(build_mod.find_tool("cmake")), "required": True,
         "why": "configures the llama.cpp build",
         "install": {"linux": "sudo apt-get install -y cmake",
                     "win32": "winget install --id Kitware.CMake"}},
        {"name": "C++ compiler", "ok": _have_compiler(), "required": True,
         "why": "compiles llama-quantize and llama-imatrix",
         "install": {"linux": "sudo apt-get install -y build-essential",
                     "win32": "winget install --id Microsoft.VisualStudio."
                              "2022.BuildTools  (with the C++ workload)"}},
        {"name": "ninja", "ok": bool(build_mod.find_tool("ninja")), "required": False,
         "why": "parallelises the build per FILE instead of per project - "
                "several times faster on ggml's ~100 CUDA sources",
         "install": {"linux": "sudo apt-get install -y ninja-build",
                     "win32": "winget install --id Ninja-build.Ninja"}},
        {"name": "CUDA toolkit", "ok": cuda, "required": False,
         "why": ("GPU offload for the imatrix pass; without it that step runs "
                 "on CPU and takes considerably longer"
                 + ("" if gpus else " (no NVIDIA GPU detected here anyway)")),
         "install": {"linux": "see developer.nvidia.com/cuda-downloads",
                     "win32": "see developer.nvidia.com/cuda-downloads"}},
        {"name": "huggingface_hub", "ok": _python_package("huggingface_hub"),
         "required": True, "why": "every Hub download and upload",
         "install": {"*": "pip install huggingface_hub"}},
        {"name": "datasets", "ok": _python_package("datasets"),
         "required": True, "why": "fetches the wikitext calibration text",
         "install": {"*": "pip install datasets"}},
        {"name": "gguf", "ok": _python_package("gguf"), "required": True,
         "why": "reads GGUF headers to find imatrix coverage gaps",
         "install": {"*": "pip install gguf"}},
        {"name": "converter deps",
         "ok": not _missing_converter_deps, "required": False,
         # Naming the ones actually missing beats naming the set: three of the
         # four installed and one not is the case that produced a run which
         # downloaded the weights and then died in set_vocab().
         "why": (f"missing {', '.join(_missing_converter_deps)} - "
                 "convert_hf_to_gguf.py needs them; without them only models "
                 "whose publisher already ships a GGUF can be built"
                 if _missing_converter_deps else
                 "torch, transformers, sentencepiece, protobuf - "
                 "convert_hf_to_gguf.py needs all four"),
         "install": {"*": 'uv tool install --force --reinstall '
                          '"agentquantix[full] @ '
                          'git+https://github.com/NANInithin/AgentQuantix"'
                          '   (~1 GB)'}},
        {"name": "hf_xet", "ok": transfer.installed(), "required": False,
         "why": f"downloads over {transfer.HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB "
                "are impossible without it",
         "install": {"*": "pip install hf_xet"}},
        {"name": "HF token", "ok": bool(config.TOKEN), "required": True,
         "why": "uploading needs a token with write permission",
         "install": {"linux": "export HF_TOKEN=hf_...   (or: hf auth login)",
                     "win32": "$env:HF_TOKEN = 'hf_...'   (or: hf auth login)"}},
    ]
    return items


def _have_compiler():
    if sys.platform == "win32":
        return build_mod.msvc_env_bat() is not None or bool(shutil.which("cl"))
    return bool(shutil.which("c++") or shutil.which("g++")
                or shutil.which("clang++"))


def _install_hint(item):
    table = item["install"]
    return table.get("*") or table.get(sys.platform) or table.get("linux", "")


def report(items):
    """Print the prerequisite table. Returns (missing_required, missing_other)."""
    missing_required, missing_other = [], []
    for item in items:
        mark = "ok  " if item["ok"] else ("MISS" if item["required"] else "----")
        print(f"  [{mark}] {item['name']:<16} {item['why']}")
        if not item["ok"]:
            (missing_required if item["required"] else missing_other).append(item)
    return missing_required, missing_other


# =====================================================
# THE COMMAND
# =====================================================
def run(build=True, clone=True):
    """Check prerequisites, then clone and build llama.cpp. Returns an exit code."""
    print(f"AgentQuantix bootstrap")
    print(f"work root: {config.WORK_ROOT}")
    print(f"platform:  {sys.platform}\n")

    items = check()
    missing_required, missing_other = report(items)

    if missing_required:
        print("\nMissing prerequisites that must be installed first:\n")
        for item in missing_required:
            print(f"  {item['name']}")
            print(f"    {_install_hint(item)}")
        print("\nInstall those, then run `aqx bootstrap` again. Nothing was "
              "changed.")
        return 1

    if missing_other:
        print("\nOptional, and worth having:\n")
        for item in missing_other:
            print(f"  {item['name']:<16} {_install_hint(item)}")

    if not build:
        print("\n--check-only: stopping before cloning or building.")
        return 0

    # Directories first: a clone into a path whose parent does not exist fails
    # with a confusing git error rather than a clear one.
    for directory in (config.WORK_ROOT, config.TEMP_DIR, config.STATE_DIR,
                      config.REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    cuda = build_mod.has_cuda_toolkit()
    print(f"\nBuilding llama.cpp with CUDA {'ON' if cuda else 'OFF'}"
          f"{'' if cuda else ' (no nvcc found - CPU build)'}. "
          "This takes a few minutes.\n")

    try:
        llama_dir, quantize, imatrix = build_mod.ensure_upstream(clone=clone)
    except Exception as e:
        print(f"\nBootstrap failed: {type(e).__name__}: {e}")
        return 1

    # Prove the binaries run, rather than trusting that they exist. A build
    # that produced an unloadable binary (a missing CUDA runtime, most often)
    # would otherwise be discovered hours into the first real job.
    print("\nVerifying the binaries...")
    for label, binary in (("llama-quantize", quantize),
                          ("llama-imatrix", imatrix)):
        try:
            done = subprocess.run([str(binary), "--help"], capture_output=True,
                                  text=True, timeout=120)
            output = (done.stdout + done.stderr).strip()
            print(f"  {label}: {'runs' if output else 'produced no output'}")
        except Exception as e:
            print(f"  {label}: FAILED to run - {type(e).__name__}: {e}")
            return 1

    print(f"\nllama.cpp ready at {llama_dir}")
    print(f"\n{sysprobe.summary(sysprobe.probe(measure_disk=False))}")
    print("\nReady. Next: `aqx research`.")
    return 0
