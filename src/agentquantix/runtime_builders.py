"""Native build contracts for independently versioned runtime installations.

This layer never clones, switches, activates, or deletes a checkout. The runtime
manager supplies an approved immutable source snapshot and a distinct build
directory. Existing installations can be inspected without modifying their files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from .pipeline import build as native_build
from .runtimes.spec import RuntimeErrorBase


class RuntimeBuildError(RuntimeErrorBase):
    """A candidate did not satisfy its native build or startup contract."""


_TARGETS = {
    "llama.cpp": ("llama-quantize", "llama-imatrix"),
    "clip.cpp": ("extract", "zsl", "quantize"),
    "audio.cpp": ("audiocpp_cli", "audiocpp_gguf"),
    "whisper.cpp": ("whisper-cli", "whisper-quantize"),
    "stable-diffusion.cpp": ("sd-cli",),
}
_NONFINITE = re.compile(
    r"(?<![\w/\\.\-])(?:nan|[+-]?inf(?:inity)?)(?![\w/\\.\-])"
    r"|\b(?:assertion failed|assert failed)\b", re.I)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _tool_stamp(name):
    executable = native_build.find_tool(name)
    if not executable:
        return None
    result = {"path": executable}
    if Path(executable).is_file():
        result["sha256"] = _hash(executable)
    return result


def fingerprint(engine, compute_backend="auto", required_targets=None):
    """Return the effective native configuration used to key a candidate build.

    Initial text/audio/Whisper adapters expose the CPU/CUDA build modes already
    used by these pipelines. Media retains its explicit native backend flags.
    A detected GPU driver alone is not evidence of an installed CUDA compiler.
    """
    if engine not in _TARGETS:
        raise RuntimeBuildError(f"no native build adapter for {engine}")
    backend = compute_backend
    if backend == "auto":
        backend = ("cpu" if engine == "clip.cpp" else
                   "cuda" if native_build.has_cuda_toolkit() else "cpu")
    if engine == "clip.cpp" and backend != "cpu":
        raise RuntimeBuildError("clip.cpp CUDA is not verified")
    if engine != "stable-diffusion.cpp" and backend not in ("cpu", "cuda"):
        raise RuntimeBuildError(f"{engine}: compute backend not yet verified: {backend}")
    if backend == "cuda" and not native_build.has_cuda_toolkit():
        raise RuntimeBuildError("CUDA toolkit is not available for this build")
    if engine == "stable-diffusion.cpp":
        from .pipeline.media_release import cmake_flags
        from . import sysprobe
        if not sysprobe.compute_backends().get(backend):
            raise RuntimeBuildError(f"compute toolchain not detected: {backend}")
        flags = cmake_flags(backend)
    elif engine == "clip.cpp":
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DCLIP_NATIVE=OFF", "-DGGML_CUBLAS=OFF"]
        cmake = native_build.find_tool("cmake")
        if cmake:
            try:
                version = subprocess.run([cmake, "--version"], capture_output=True,
                                         text=True, timeout=10).stdout.splitlines()[0]
                match = re.search(r"version\s+(\d+)", version)
                if match and int(match.group(1)) >= 4:
                    flags.append("-DCMAKE_POLICY_VERSION_MINIMUM=3.5")
            except (OSError, subprocess.SubprocessError, IndexError):
                pass
    elif engine == "audio.cpp":
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DAUDIOCPP_DEPLOYMENT_BUILD=ON",
                 "-DAUDIOCPP_MODEL_SET=full",
                 f"-DENGINE_ENABLE_CUDA={'ON' if backend == 'cuda' else 'OFF'}"]
    else:
        flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
                 f"-DGGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}"]
    if backend == "cuda" and not any(flag.startswith("-DCMAKE_CUDA_ARCHITECTURES=") for flag in flags):
        if architecture := native_build.cuda_arch_from_gpu():
            flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={architecture}")
    targets = list(dict.fromkeys([*_TARGETS[engine], *(required_targets or [])]))
    if any(not re.fullmatch(r"[A-Za-z0-9_.+-]+", target) for target in targets):
        raise RuntimeBuildError("invalid native build target")
    vcvars = native_build.msvc_env_bat() if sys.platform == "win32" else None
    generator = os.getenv("CMAKE_GENERATOR")
    if not generator and native_build.find_tool("ninja") and (sys.platform != "win32" or vcvars):
        generator = "Ninja"
    compiler = _tool_stamp(os.getenv("CXX") or "c++") or _tool_stamp("cl")
    if compiler is None and vcvars:
        compiler = {"environment_script": str(vcvars), "sha256": _hash(vcvars)}
        candidates = sorted((Path(vcvars).parents[2] / "Tools/MSVC").glob("*/bin/Hostx64/x64/cl.exe"))
        if candidates:
            compiler["native_compiler"] = str(candidates[-1])
            compiler["native_compiler_sha256"] = _hash(candidates[-1])
    return {"adapter_schema": 1, "engine": engine, "platform": platform.platform(),
            "machine": platform.machine(), "compute_backend": backend,
            "flags": flags, "targets": targets, "generator": generator,
            "compiler": compiler, "cmake": _tool_stamp("cmake"),
            "ninja": _tool_stamp("ninja") if generator == "Ninja" else None,
            "vcvars": str(vcvars) if vcvars else None}


def _execute(command, log_path=None, *, cwd=None, timeout=3600, help_probe=False):
    command = [str(argument) for argument in command]
    started = time.monotonic()
    record = {"argv": command, "cwd": str(cwd) if cwd else None}
    try:
        result = subprocess.run(command, cwd=cwd, env=native_build.build_env(),
                                capture_output=True, text=True, errors="replace",
                                stdin=subprocess.DEVNULL, timeout=timeout)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        record.update(returncode=None, stdout=str(getattr(error, "stdout", "") or ""),
                      stderr=str(getattr(error, "stderr", "") or error))
        raise RuntimeBuildError(f"native command could not execute: {error}; log: {log_path}") from error
    finally:
        record["duration_seconds"] = time.monotonic() - started
        if log_path is not None:
            _atomic_json(log_path, record)
    diagnostics = record["stdout"] + "\n" + record["stderr"]
    usage_exit = (help_probe and result.returncode == 1
                  and re.search(r"\b(?:usage|options|help)\b", diagnostics, re.I))
    if (result.returncode and not usage_exit) or _NONFINITE.search(diagnostics):
        tail = next((line for line in reversed(diagnostics.splitlines()) if line.strip()), "no diagnostics")
        raise RuntimeBuildError(f"native command failed ({result.returncode}): {tail}; log: {log_path}")
    return record


def _git(source, *arguments):
    result = _execute(["git", "-C", source, *arguments], timeout=60)
    return result["stdout"].strip()


def _find_binary(build_dir, name):
    roots = [Path(build_dir) / "bin", Path(build_dir)]
    for root in roots:
        for candidate in (root / f"{name}.exe", root / "Release" / f"{name}.exe",
                          root / name, root / "Release" / name):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _binary_contract(engine, source, targets):
    contract = {name: name for name in targets}
    if engine == "whisper.cpp":
        cmake_file = Path(source) / "examples/quantize/CMakeLists.txt"
        text = cmake_file.read_text(encoding="utf-8") if cmake_file.is_file() else ""
        if "whisper-quantize" not in text:
            contract["whisper-quantize"] = "quantize"
    return contract


def _resources(engine, source, build_dir):
    paths = {
        "llama.cpp": ("convert_hf_to_gguf.py",),
        "clip.cpp": ("models/convert_hf_to_gguf.py",),
        "whisper.cpp": ("models/convert-h5-to-ggml.py", "models/convert-pt-to-ggml.py"),
    }.get(engine, ())
    resources = {name: {"path": str(Path(source) / name), "sha256": _hash(Path(source) / name)}
                 for name in paths if (Path(source) / name).is_file()}
    if engine == "audio.cpp":
        root = Path(source) / "model_specs"
        for path in sorted(root.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not data.get("family"):
                raise RuntimeBuildError(f"audio.cpp: invalid model spec {path}")
            resources[path.relative_to(source).as_posix()] = {"path": str(path), "sha256": _hash(path)}
        if not resources:
            raise RuntimeBuildError("audio.cpp build lacks its model_specs contract")
    # GGML backends can be dynamically loaded after process startup. Binding
    # only the CLI hash would let a changed CUDA/CPU backend bypass provenance.
    build_dir = Path(build_dir)
    for directory in (build_dir / "bin", build_dir / "bin/Release", build_dir / "src"):
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file() and (path.suffix.lower() in (".dll", ".dylib") or ".so" in path.name):
                    name = "native/" + path.relative_to(build_dir).as_posix()
                    resources[name] = {"path": str(path), "sha256": _hash(path)}
    return resources


def _result(engine, source, build_dir, binaries, options, *, adopted=False, logs=None):
    probes = {}
    for name, binary in binaries.items():
        probes[name] = _execute([binary, "--help"],
                               Path(logs) / f"probe-{name}.json" if logs else None,
                               timeout=60, help_probe=True)
    resources = _resources(engine, source, build_dir)
    revision = _git(source, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise RuntimeBuildError("native source has no immutable Git revision")
    dirty = _git(source, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeBuildError("native source contains tracked modifications; leaving it untouched")
    provenance = {"engine": engine, "revision": revision, "adopted": adopted,
                  "build_fingerprint": options,
                  "binary_sha256": {name: _hash(path) for name, path in binaries.items()},
                  "submodules": _git(source, "submodule", "status", "--recursive").splitlines(),
                  "probes": probes}
    cache = Path(build_dir) / "CMakeCache.txt"
    if cache.is_file():
        provenance["cmake_cache_sha256"] = _hash(cache)
    return {"source": str(source), "build_dir": str(build_dir),
            "binaries": {name: str(path) for name, path in binaries.items()},
            "resources": resources, "provenance": provenance}


def _batch_quote(value):
    # CMake argv are generated here, but workspace paths are user-controlled.
    # Quote every argument, preserve literal percent signs in a batch file,
    # and reject quote/newline injection rather than constructing shell code.
    value = str(value)
    if any(character in value for character in ('"', '\r', '\n')):
        raise RuntimeBuildError("unsupported quote or newline in Windows build argument")
    return '"' + value.replace("%", "%%") + '"'


def build_runtime(engine, source, build_dir, options=None):
    """Build one candidate in its explicit directory and return its contract.

    The caller must supply a new managed source snapshot, not an adopted live
    installation. Clone/revision selection and activation belong to the manager.
    """
    if engine not in _TARGETS:
        raise RuntimeBuildError(f"no native build adapter for {engine}")
    source, build_dir = Path(source).resolve(), Path(build_dir).resolve()
    if not (source / "CMakeLists.txt").is_file():
        raise RuntimeBuildError(f"{engine}: source lacks CMakeLists.txt")
    if source == build_dir:
        raise RuntimeBuildError("native build directory must differ from its source directory")
    supplied = options or {}
    effective = (dict(supplied) if "flags" in supplied and "targets" in supplied else
                 fingerprint(engine, supplied.get("compute_backend", "auto"),
                             supplied.get("required_targets")))
    if effective.get("engine") != engine:
        raise RuntimeBuildError("build fingerprint belongs to a different engine")
    build_dir.mkdir(parents=True, exist_ok=True)
    logs = build_dir / "aqx-logs"
    submodules = (["ggml", "thirdparty/libwebp", "thirdparty/libwebm"]
                  if engine == "stable-diffusion.cpp" else
                  ["ggml"] if engine == "clip.cpp" else [])
    _execute(["git", "-c", "url.https://github.com/.insteadOf=git@github.com:",
              "-C", source, "submodule", "update", "--init", "--recursive", *submodules],
             logs / "submodules.json", timeout=900)
    contract = _binary_contract(engine, source, effective["targets"])
    configure = ["cmake", "-S", source, "-B", build_dir]
    if effective.get("generator"):
        configure += ["-G", effective["generator"]]
    configure += effective["flags"]
    compile_command = ["cmake", "--build", build_dir, "--config", "Release",
                       "--target", *dict.fromkeys(contract.values()),
                       "--parallel", native_build.build_jobs()]
    if sys.platform == "win32" and effective.get("vcvars"):
        script = build_dir / "aqx-build.bat"
        script.write_text("@echo off\r\nsetlocal DisableDelayedExpansion\r\n"
                          f"call {_batch_quote(effective['vcvars'])} >nul || exit /b 1\r\n"
                          + " ".join(map(_batch_quote, configure)) + " || exit /b 1\r\n"
                          + " ".join(map(_batch_quote, compile_command)) + " || exit /b 1\r\n",
                          encoding="utf-8")
        # Use a fixed basename and native cwd instead of passing a user path
        # through cmd.exe's additional command-line parsing/expansion layer.
        _execute(["cmd.exe", "/d", "/v:off", "/c", ".\\" + script.name],
                 logs / "build.json", cwd=build_dir, timeout=7200)
    else:
        _execute(configure, logs / "configure.json")
        _execute(compile_command, logs / "build.json", timeout=7200)
    binaries = {role: _find_binary(build_dir, target) for role, target in contract.items()}
    if missing := [role for role, path in binaries.items() if path is None]:
        raise RuntimeBuildError(f"{engine}: native build produced no {', '.join(missing)}")
    result = _result(engine, source, build_dir, binaries, effective, logs=logs)
    if engine == "stable-diffusion.cpp":
        from .media_catalog import generate_catalog
        try:
            catalog = generate_catalog(source, expected_revision=result["provenance"]["revision"])
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            raise RuntimeBuildError(f"media capability extraction failed: {error}") from error
        path = build_dir / "capabilities.json"
        _atomic_json(path, catalog)
        result["resources"]["capability_catalog"] = {"path": str(path), "sha256": _hash(path)}
    _atomic_json(build_dir / "aqx-build.json", result)
    return result


def probe_existing(engine, source):
    """Inspect an existing original without rebuilding or changing its files.

    Return None if required binaries are absent. Invalid/mutated installations
    raise a diagnostic rather than being silently rebuilt inside user-owned code.
    """
    if engine not in _TARGETS:
        raise RuntimeBuildError(f"no native build adapter for {engine}")
    source = Path(source).resolve()
    contract = _binary_contract(engine, source, _TARGETS[engine])
    for build_dir in (source / "build", source.parent / "build"):
        binaries = {role: _find_binary(build_dir, target) for role, target in contract.items()}
        if all(binaries.values()):
            optional_tts = _find_binary(build_dir, "llama-tts") if engine == "llama.cpp" else None
            if optional_tts:
                binaries["llama-tts"] = optional_tts
            result = _result(engine, source, build_dir, binaries, {}, adopted=True)
            if engine == "stable-diffusion.cpp":
                from .media_catalog import CATALOG_PATH, ENGINE_REVISION
                revision = result["provenance"]["revision"]
                catalog_path = build_dir / "capabilities.json"
                if not catalog_path.is_file():
                    if revision != ENGINE_REVISION:
                        # A current binary does not prove compatibility with
                        # the shipped pinned catalog. Let the manager build a
                        # managed candidate with its own extracted contract.
                        return None
                    catalog_path = CATALOG_PATH
                try:
                    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as error:
                    raise RuntimeBuildError(f"invalid adopted media catalog: {error}") from error
                if catalog.get("revision") != revision:
                    raise RuntimeBuildError("adopted media catalog does not describe its source revision")
                result["resources"]["capability_catalog"] = {
                    "path": str(catalog_path), "sha256": _hash(catalog_path)}
            return result
    return None
