"""End-to-end TTS and ASR release pipelines.

The publication unit is a verified :class:`VoiceBundle`.  TTS uses the cached
llama.cpp or audio.cpp checkout; ASR has a separate whisper.cpp checkout,
build, converter, model format, quantizer, and acceptance corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave

from .. import archsupport, config, voice
from . import build as build_mod, sanity


AUDIOCPP_REVISION = "v0.8.0"
AUDIOCPP_REVISION_SHA = "4af143229384fb6da3f373dc87de145ae954609b"
TTS_QUALITY_CACHE_SCHEMA = 1


@dataclass
class VoiceReleaseOptions:
    quants: list[str] = field(default_factory=list)
    language: str | None = None
    speaker: Path | None = None
    publish: bool = True
    max_roundtrip_wer: float = 0.35
    max_wer_regression: float = 0.05
    runtime_timeout: int = 900
    pocket_language: str = "english"
    human_review: Path | None = None
    family: str | None = None


def _slug(repo_id: str) -> str:
    return (repo_id.replace("/", "--").replace("\\", "--")
            .replace(":", "--"))


def work_dir(repo_id: str) -> Path:
    return config.TEMP_DIR / "voice" / _slug(repo_id)


def _with_recorded_reviews(repo_id: str,
                           options: VoiceReleaseOptions) -> VoiceReleaseOptions:
    """Use reviews written by ``aqx review`` / ``record_review`` unless overridden."""
    if options.human_review is not None:
        return options
    reviews = work_dir(repo_id) / "reviews"
    if reviews.is_dir() and any(reviews.glob("*.json")):
        return replace(options, human_review=reviews)
    return options


def _git_revision(directory: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=20,
        ).stdout.strip()
    except Exception:
        return None


def _audiocpp_runtime_revision(audio_dir: Path | None = None) -> str:
    revision = voice.audio_cpp_catalog()["revision"]
    if revision is not None:
        return str(revision)
    if audio_dir is not None and (revision := _git_revision(audio_dir)) is not None:
        return revision
    return f"audio.cpp-{AUDIOCPP_REVISION}"


def ensure_llama_tts() -> tuple[Path, Path, Path]:
    """Return the llama.cpp checkout, llama-tts, and llama-quantize."""
    from .. import runtime_control
    if handle := runtime_control.ensure_runtime("llama.cpp", required_targets=("llama-tts",)):
        runtime, quantizer = runtime_control.require_binaries(handle, "llama-tts", "llama-quantize")
        return handle.source, runtime, quantizer
    llama_dir, quantize, _ = build_mod.ensure_tools()
    runtime = build_mod.find_binary(llama_dir, "llama-tts")
    if runtime is None:
        build_mod.run(["cmake", "--build", llama_dir / "build", "--config",
                       "Release", "--target", "llama-tts", "-j",
                       build_mod.build_jobs()])
        runtime = build_mod.find_binary(llama_dir, "llama-tts")
    if runtime is None:
        raise RuntimeError("llama.cpp build completed without llama-tts")
    return llama_dir, runtime, quantize


def _audio_cpp_defines() -> list[str]:
    cuda = build_mod.has_cuda_toolkit()
    defines = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DAUDIOCPP_DEPLOYMENT_BUILD=ON",
        # The runtime catalog is the capability boundary. Building a custom
        # two-family subset would make AgentQuantix itself the allowlist.
        "-DAUDIOCPP_MODEL_SET=full",
        f"-DENGINE_ENABLE_CUDA={'ON' if cuda else 'OFF'}",
    ]
    if cuda and (arch := build_mod.cuda_arch_from_gpu()):
        defines.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
    return defines


def ensure_audiocpp_tools() -> tuple[Path, Path, Path]:
    """Build/cache the pinned audio.cpp CLI and standalone GGUF converter."""
    from .. import runtime_control
    if handle := runtime_control.ensure_runtime("audio.cpp"):
        runtime, converter = runtime_control.require_binaries(handle, "audiocpp_cli", "audiocpp_gguf")
        return handle.source, runtime, converter
    directory = config.UPSTREAM_AUDIOCPP
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        build_mod.run([
            "git", "-c", "url.https://github.com/.insteadOf=git@github.com:",
            "clone", "--depth", "1", "--branch", AUDIOCPP_REVISION,
            "--recursive", "https://github.com/0xShug0/audio.cpp.git", directory,
        ])
    runtime = build_mod.find_binary(directory, "audiocpp_cli")
    converter = build_mod.find_binary(directory, "audiocpp_gguf")
    if runtime is None or converter is None:
        build_mod.run([
            "git", "-c", "url.https://github.com/.insteadOf=git@github.com:",
            "-C", directory, "submodule", "update", "--init", "--recursive",
        ])
        build_mod.run(["cmake", "-S", directory, "-B", directory / "build",
                       *_audio_cpp_defines()])
        build_mod.run([
            "cmake", "--build", directory / "build", "--config", "Release",
            "--target", "audiocpp_cli", "audiocpp_gguf", "-j",
            build_mod.build_jobs(),
        ])
        runtime = build_mod.find_binary(directory, "audiocpp_cli")
        converter = build_mod.find_binary(directory, "audiocpp_gguf")
    if runtime is None or converter is None:
        raise RuntimeError(
            "audio.cpp build did not produce audiocpp_cli and audiocpp_gguf")
    return directory, runtime, converter


def _cmake_build(directory: Path, targets: tuple[str, ...]) -> None:
    defines = build_mod.build_settings()
    build_mod.run(["cmake", "-S", directory, "-B", directory / "build",
                   *defines])
    for target in targets:
        build_mod.run(["cmake", "--build", directory / "build", "--config",
                       "Release", "--target", target, "-j",
                       build_mod.build_jobs()])


def ensure_whisper_runtime() -> tuple[Path, Path]:
    """Build/cache only whisper-cli, which is all TTS scoring requires."""
    from .. import runtime_control
    if handle := runtime_control.ensure_runtime("whisper.cpp"):
        runtime, = runtime_control.require_binaries(handle, "whisper-cli")
        return handle.source, runtime
    directory = config.UPSTREAM_WHISPER
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        build_mod.run(["git", "clone", "https://github.com/ggml-org/whisper.cpp.git",
                       directory])
    runtime = build_mod.find_binary(directory, "whisper-cli")
    if runtime is None:
        _cmake_build(directory, ("whisper-cli",))
        runtime = build_mod.find_binary(directory, "whisper-cli")
    if runtime is None:
        raise RuntimeError("whisper.cpp build did not produce whisper-cli")
    return directory, runtime


def _whisper_quantizer_target(directory: Path) -> str:
    """Return the CMake target used by this whisper.cpp checkout."""
    cmake_file = directory / "examples" / "quantize" / "CMakeLists.txt"
    try:
        content = cmake_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        content = ""
    return "whisper-quantize" if "whisper-quantize" in content else "quantize"


def _find_whisper_quantizer(directory: Path) -> Path | None:
    # Current whisper.cpp calls both the target and executable
    # ``whisper-quantize``; older releases emitted ``quantize``.
    return (build_mod.find_binary(directory, "whisper-quantize")
            or build_mod.find_binary(directory, "quantize"))


def ensure_whisper_tools() -> tuple[Path, Path, Path]:
    """Return whisper-cli plus the version-appropriate ASR quantizer."""
    from .. import runtime_control
    if handle := runtime_control.ensure_runtime("whisper.cpp"):
        runtime, quantizer = runtime_control.require_binaries(handle, "whisper-cli", "whisper-quantize")
        return handle.source, runtime, quantizer
    directory, runtime = ensure_whisper_runtime()
    quantize = _find_whisper_quantizer(directory)
    if quantize is None:
        _cmake_build(directory, (_whisper_quantizer_target(directory),))
        quantize = _find_whisper_quantizer(directory)
    if quantize is None:
        raise RuntimeError(
            "whisper.cpp build did not produce whisper-quantize or quantize")
    return directory, runtime, quantize


def _snapshot(repo_id: str, destination: Path) -> tuple[Path, str | None]:
    from huggingface_hub import HfApi, snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    revision = HfApi(token=config.TOKEN).model_info(repo_id).sha
    snapshot_download(repo_id=repo_id, revision=revision,
                      local_dir=str(destination), token=config.TOKEN)
    return destination, revision


def source_metadata(repo_id: str, api=None, family: str | None = None) -> dict:
    """Verify that a planned voice source exists and is readable now."""
    backend = voice.backend_for(repo_id, family=family)
    if backend is None:
        allowed, reason, _ = voice.execution_gate(repo_id, family=family)
        assert not allowed
        raise voice.VoiceValidationError(reason)
    if (repo_id.casefold().startswith("audio.cpp:")
            or any(repo_id.casefold() == item.casefold()
                   for item in backend.package_ids)
            or _audiocpp_package_reference(repo_id, backend)):
        spec = voice.audio_cpp_catalog()["registry"].get(backend.family, {})
        return {
            "repo_id": repo_id,
            "revision": _audiocpp_runtime_revision(),
            "source_bytes": None,
            "gated": False,
            "private": False,
            "license": (_audiocpp_source_license(repo_id, backend, spec)
                        if backend.track in (voice.SEP, voice.VC, voice.ALIGN)
                        else None),
            "package_ids": list(backend.package_ids),
        }
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=config.TOKEN)
    try:
        info = api.model_info(repo_id, files_metadata=True, token=config.TOKEN)
    except Exception as error:
        detail = str(error).splitlines()[0] if str(error) else type(error).__name__
        raise voice.VoiceValidationError(
            f"registered voice source is not accessible: {repo_id}. "
            f"Check the exact repo id and gated/private access. {detail}") from error
    total = 0
    source_files = []
    for sibling in getattr(info, "siblings", ()) or ():
        name = (getattr(sibling, "rfilename", None)
                or getattr(sibling, "path", None))
        if name:
            source_files.append(str(name).replace("\\", "/"))
        lfs = getattr(sibling, "lfs", None) or {}
        total += (getattr(sibling, "size", None)
                  or (lfs.get("size") if isinstance(lfs, dict) else 0) or 0)
    card_data = getattr(info, "card_data", None)
    license_name = (getattr(card_data, "license", None)
                    or (card_data.get("license")
                        if isinstance(card_data, dict) else None))
    if str(license_name or "").casefold() in {"", "other", "unknown", "n/a"}:
        license_name = None
    return {
        "repo_id": repo_id,
        "revision": getattr(info, "sha", None),
        "source_bytes": total or None,
        "source_files": sorted(source_files) if source_files else None,
        "gated": bool(getattr(info, "gated", False)),
        "private": bool(getattr(info, "private", False)),
        "license": license_name,
    }


def _audiocpp_source_license(repo_id: str, backend: voice.VoiceBackend,
                             spec: dict | None = None) -> str | None:
    spec = spec or voice.audio_cpp_catalog()["registry"].get(backend.family, {})
    packages = spec.get("packages", [])
    for key in ("license", "license_name"):
        if value := spec.get(key):
            if str(value).casefold() not in {"other", "unknown", "n/a"}:
                return str(value)
    for package in packages:
        for key in ("license", "license_name"):
            if value := package.get(key):
                if str(value).casefold() not in {"other", "unknown", "n/a"}:
                    return str(value)
    defaults = spec.get("package_defaults", {}).get("download", {})
    package_repo = None
    package_revision = "main"
    for package in packages:
        if package.get("format") != "gguf":
            continue
        download = {**defaults, **package.get("download", {})}
        candidate = str(download.get("repo", ""))
        if candidate and (repo_id.startswith("audio.cpp:")
                          or candidate.casefold() == repo_id.casefold()):
            package_repo = candidate
            package_revision = str(download.get("revision", "main"))
            break
    if not package_repo:
        return None
    try:
        from huggingface_hub import hf_hub_download
        readme = Path(hf_hub_download(
            repo_id=package_repo, filename="README.md",
            revision=package_revision, token=config.TOKEN))
        rows = readme.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    family = backend.family.casefold()
    for row in rows:
        if not row.lstrip().startswith("|") or family not in row.casefold():
            continue
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        license_cell = cells[-1]
        linked = re.search(r"\[([^\]]+)\]\([^)]+\)", license_cell)
        value = linked.group(1) if linked else license_cell.strip("` ")
        canonical = re.search(
            r"(?i)\b(Apache-2\.0|MIT|BSD-2-Clause|BSD-3-Clause|"
            r"CC-BY-4\.0|CC-BY-3\.0|CC0-1\.0)\b", value)
        if canonical:
            value = canonical.group(1)
        if value and value.casefold() not in {"-", "unknown", "n/a"}:
            return value
    return None


def audiocpp_source_plan(repo_id: str, backend: voice.VoiceBackend,
                         source_files: list[str] | None = None) -> list[dict]:
    """Describe tensor inputs separately from published companions.

    A source tensor can be embedded into the resulting standalone GGUF and is
    therefore not a bundle companion.  Plans still need to disclose it before
    a multi-gigabyte download begins.  When the official repo carries a
    same-named ``.pth`` instead of the SafeTensors input, record the pinned
    audio.cpp preparation package that will adapt it.
    """
    if backend.backend != "audio.cpp" or _audiocpp_package_reference(
            repo_id, backend):
        return []
    spec = voice.audio_cpp_catalog()["registry"].get(backend.family, {})
    contract = next((item for item in spec.get("sources", [])
                     if item.get("format") == "safetensors"), None)
    if contract is None:
        return []
    known = ({str(path).replace("\\", "/") for path in source_files}
             if source_files is not None else None)
    roots = contract.get("roots", {})
    result = []
    for namespace, value in contract.get("tensors", {}).items():
        reference = value.get("source") if isinstance(value, dict) else value
        if not isinstance(reference, str) or ":" not in reference:
            continue
        root_name, relative = reference.split(":", 1)
        root = str(roots.get(root_name, "")).replace("\\", "/")
        root = "" if root == "." else root.rstrip("/")
        required = "/".join(part for part in (root, relative) if part)
        available = required if known is not None and required in known else None
        preparation = None
        alternatives = []
        if required.casefold().endswith(".safetensors"):
            pytorch_path = required[:-len(".safetensors")] + ".pth"
            alternatives.append(pytorch_path)
            if known is not None and not available and pytorch_path in known:
                available = pytorch_path
                package_id = f"{backend.family}_{Path(required).stem}"
                preparation = {
                    "tool": "audio.cpp model_manager_deprecated.py",
                    "package": package_id,
                    "output": required,
                }
        status = ("unknown" if known is None else
                  "ready" if available == required else
                  "needs_preparation" if available else "missing")
        result.append({
            "namespace": str(namespace),
            "required_path": required,
            "accepted_source_paths": [required, *alternatives],
            "available_path": available,
            "status": status,
            "preparation": preparation,
        })
    return result


def find_backend_forks(repo_id: str) -> list[dict]:
    """Run the same publisher-fork/upstream-PR hunt used by text models."""
    from .. import hub

    try:
        candidate = hub.one(repo_id)
        hub.enrich(candidate, check_ggufs=False, want_readme=False)
    except Exception:
        # A Hub lookup error must not hide the normal unsupported-family
        # explanation. The repo name still gives the hunt useful needles.
        candidate = hub.Candidate(repo_id=repo_id, rank=0)
    return archsupport.find_voice_forks(candidate)


def _converter_requirements() -> list[str]:
    missing = []
    for package in ("torch", "transformers", "safetensors"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing


def _run(command: list, label: str) -> None:
    try:
        build_mod.run_verbose(command, label=label)
    except build_mod.CommandFailed as error:
        raise voice.VoiceValidationError(str(error)) from error


def convert_tts_source(repo_id: str, options: VoiceReleaseOptions
                       ) -> tuple[voice.VoiceBackend, Path, Path, str | None,
                                  Path, Path, Path]:
    """Convert one TTS source to a BF16 primary and Q8_0 mmproj."""
    backend = voice.backend_for(repo_id, track=voice.TTS)
    if backend is None:
        raise voice.VoiceValidationError(f"no TTS backend for {repo_id}")
    if missing := _converter_requirements():
        raise RuntimeError("TTS conversion requires " + ", ".join(missing)
                           + "; install AgentQuantix[convert]")
    llama_dir, runtime, quantizer = ensure_llama_tts()
    root = work_dir(repo_id)
    source, revision = _snapshot(repo_id, root / "source")
    model_source = source
    if backend.family == "pocket-tts":
        model_source = source / "languages" / options.pocket_language
        if not model_source.is_dir():
            raise voice.VoiceValidationError(
                f"Pocket TTS language directory is missing: {model_source}")
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    base = models / f"{repo_id.split('/')[-1]}-BF16.gguf"
    mmproj = models / f"mmproj-{repo_id.split('/')[-1]}-Q8_0.gguf"
    converter = llama_dir / "convert_hf_to_gguf.py"
    if not base.exists():
        _run([sys.executable, converter, model_source, "--outfile", base,
              "--outtype", "bf16"], f"convert {backend.family} primary")
    if not mmproj.exists():
        _run([sys.executable, converter, model_source, "--outfile", mmproj,
              "--outtype", "q8_0", "--mmproj"],
             f"convert {backend.family} mmproj")
        prefixed = mmproj.with_name("mmproj-" + mmproj.name)
        if not mmproj.exists() and prefixed.exists():
            prefixed.replace(mmproj)
    if not base.is_file() or not mmproj.is_file():
        raise voice.VoiceValidationError(
            "llama.cpp conversion did not produce both primary and mmproj GGUFs")
    return backend, base, mmproj, revision, runtime, quantizer, llama_dir


def _tts_imatrix(base: Path, llama_dir: Path) -> tuple[Path, list[str]]:
    """Build an importance matrix from the licensed multilingual TTS prompts."""
    from . import imatrix as imatrix_mod

    from ..runtime_control import handle_for
    handle = handle_for("llama.cpp")
    executable = handle.binaries.get("llama-imatrix") if handle else build_mod.find_binary(llama_dir, "llama-imatrix")
    if executable is None:
        raise voice.VoiceValidationError(
            "llama.cpp build has no llama-imatrix for low-bit TTS quants")
    calibration = base.with_name(base.stem + "-voice-calibration.txt")
    prompts = [item["text"] for item in load_fixtures(voice.TTS)]
    calibration.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    output = base.with_name(base.stem + "-imatrix.dat")
    if (not output.exists()
            or output.stat().st_mtime < calibration.stat().st_mtime):
        try:
            _run([executable, "-m", base, "-f", calibration,
                  "-o", output, "-ngl", "0"], "llama TTS imatrix")
        except Exception:
            output.unlink(missing_ok=True)
            raise
    gap_args = []
    for layer in imatrix_mod.gap_layers(base, output):
        gap_args += ["--tensor-type",
                     rf"blk\.{layer}\.={config.GAP_FALLBACK_TYPE}"]
    return output, gap_args


def quantize_tts(base: Path, quantizer: Path, quant: str,
                 imatrix: Path | None = None,
                 gap_args: list[str] | None = None) -> Path:
    quant = quant.upper()
    if not quant or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
                        for character in quant):
        raise voice.VoiceValidationError(
            f"invalid llama.cpp TTS quant name: {quant!r}")
    if quant in config.IMATRIX_REQUIRED and imatrix is None:
        raise voice.VoiceValidationError(
            f"{quant} requires a TTS importance matrix")
    output = base.with_name(base.name.replace("-BF16.gguf", f"-{quant}.gguf"))
    if not output.exists():
        extra = []
        if imatrix is not None and (quant in config.IQ_QUANTS
                                   or quant in config.IMATRIX_GUIDED):
            extra = ["--imatrix", imatrix]
            if quant in config.GAP_AFFECTED:
                extra += gap_args or []
        _run([quantizer, *extra, base, output, quant],
             f"quantize TTS {quant}")
    return output


def _audiocpp_spec(audio_dir: Path, family: str) -> tuple[Path, dict]:
    path = audio_dir / "model_specs" / f"{family}.json"
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise voice.VoiceValidationError(
            f"audio.cpp catalog has no readable spec for {family}: {error}") from error
    if spec.get("family") != family:
        raise voice.VoiceValidationError(
            f"audio.cpp model spec family mismatch: expected {family}")
    return path, spec


def _audiocpp_tensor_inputs(source: Path, spec: dict, *,
                            match_gguf_namespaces: bool = False) -> list[str]:
    """Resolve the generic safetensors input contract from a model spec."""
    source_contract = next((item for item in spec.get("sources", [])
                            if item.get("format") == "safetensors"), None)
    if source_contract is None:
        raise voice.VoiceValidationError(
            f"audio.cpp {spec['family']} has no safetensors conversion contract; "
            "use one of the GGUF packages declared by that backend catalog")
    gguf_contract = next((item for item in spec.get("sources", [])
                          if item.get("format") == "gguf"), None)
    if match_gguf_namespaces and gguf_contract is None:
        raise voice.VoiceValidationError(
            f"audio.cpp {spec['family']} has no GGUF conversion contract")
    output_tensors = (gguf_contract.get("tensors", {})
                      if gguf_contract is not None else {})
    roots = source_contract.get("roots", {})
    inputs = []
    missing = []
    for namespace, value in source_contract.get("tensors", {}).items():
        reference = value.get("source") if isinstance(value, dict) else value
        if not isinstance(reference, str) or ":" not in reference:
            raise voice.VoiceValidationError(
                f"invalid audio.cpp tensor source for {namespace}: {reference!r}")
        root_name, relative = reference.split(":", 1)
        root_value = roots.get(root_name)
        if not isinstance(root_value, str) or root_value.startswith("$"):
            raise voice.VoiceValidationError(
                f"audio.cpp {spec['family']} has an unresolved source root "
                f"{root_name!r}")
        path = (source / root_value / relative).resolve()
        if not path.is_file():
            missing.append(f"{namespace}={path}")
        output_namespace = namespace
        if match_gguf_namespaces:
            output_tensor = output_tensors.get(namespace)
            if output_tensor is None:
                raise voice.VoiceValidationError(
                    f"audio.cpp {spec['family']} safetensors input "
                    f"{namespace!r} has no GGUF tensor mapping")
            output_namespace = (output_tensor.get("prefix", "")
                                if isinstance(output_tensor, dict) else "")
            if not isinstance(output_namespace, str):
                raise voice.VoiceValidationError(
                    f"audio.cpp {spec['family']} has an invalid GGUF prefix "
                    f"for {namespace!r}")
        inputs.append(
            f"{output_namespace}={path}" if output_namespace else str(path))
    if missing:
        raise voice.VoiceValidationError(
            f"{spec['family']} source is missing audio.cpp-declared tensor "
            "inputs: " + ", ".join(missing))
    if not inputs:
        raise voice.VoiceValidationError(
            f"audio.cpp {spec['family']} declares no safetensors inputs")
    return inputs


def _prepare_audiocpp_tensor_sources(
        source: Path, spec: dict, audio_dir: Path, *,
        match_gguf_namespaces: bool = False) -> list[str]:
    """Adapt source formats through audio.cpp's own pinned utilities.

    Some official repositories use a safe PyTorch weights-only checkpoint for
    one component even though ``audiocpp_gguf`` accepts SafeTensors inputs.
    audio.cpp ships narrowly scoped preparation packages for those cases.  The
    package id follows its catalog convention ``<family>_<component stem>``;
    deriving it from the model spec keeps AgentQuantix free of per-repository
    conversion code.

    VoxCPM2 is the first such route: OpenBMB publishes ``audiovae.pth`` and the
    audio.cpp utility ``voxcpm2_audiovae`` writes the required
    ``audiovae.safetensors`` without substituting a different VAE.
    """
    source_contract = next((item for item in spec.get("sources", [])
                            if item.get("format") == "safetensors"), None)
    if source_contract is None:
        return _audiocpp_tensor_inputs(
            source, spec, match_gguf_namespaces=match_gguf_namespaces)

    roots = source_contract.get("roots", {})
    for namespace, value in source_contract.get("tensors", {}).items():
        reference = value.get("source") if isinstance(value, dict) else value
        if not isinstance(reference, str) or ":" not in reference:
            continue
        root_name, relative = reference.split(":", 1)
        root_value = roots.get(root_name)
        if not isinstance(root_value, str) or root_value.startswith("$"):
            continue
        expected = (source / root_value / relative).resolve()
        if expected.is_file() or expected.suffix.casefold() != ".safetensors":
            continue
        pytorch_source = expected.with_suffix(".pth")
        if not pytorch_source.is_file():
            continue

        manager = audio_dir / "tools" / "model_manager_deprecated.py"
        if not manager.is_file():
            raise voice.VoiceValidationError(
                f"audio.cpp {spec['family']} needs {expected.name}, but the "
                f"official source provides {pytorch_source.name} and the "
                "pinned backend has no model preparation utility")
        package_id = f"{spec['family']}_{expected.stem}"
        _run([
            sys.executable, manager, "install", package_id,
            "--source-file", pytorch_source,
            "--output-file", expected,
            "--overwrite",
        ], f"prepare {spec['family']} {namespace} from {pytorch_source.name}")
        if not expected.is_file() or expected.stat().st_size < 1024:
            raise voice.VoiceValidationError(
                f"audio.cpp preparation package {package_id} did not produce "
                f"a valid {expected.name}")

    return _audiocpp_tensor_inputs(
        source, spec, match_gguf_namespaces=match_gguf_namespaces)


def convert_audiocpp_model(source: Path, backend: voice.VoiceBackend,
                           converter: Path, audio_dir: Path, output_dir: Path,
                           quant: str, *,
                           match_gguf_namespaces: bool = False) -> Path:
    """Create and inspect one self-contained audio.cpp-native GGUF."""
    quant = quant.upper()
    if quant not in voice.AUDIOCPP_QUANTS:
        raise voice.VoiceValidationError(
            "audio.cpp converter quants are "
            + ", ".join(voice.AUDIOCPP_QUANTS))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{backend.family}-{quant}.gguf"
    if not output.exists():
        command = [converter]
        spec_path, spec = _audiocpp_spec(audio_dir, backend.family)
        for tensor_input in _prepare_audiocpp_tensor_sources(
                source, spec, audio_dir,
                match_gguf_namespaces=match_gguf_namespaces):
            command += ["--input", tensor_input]
        command += [
            "--root", source,
            "--output", output,
            "--type", quant.casefold(),
            "--family", backend.family,
            "--model-spec", spec_path,
            "--overwrite",
        ]
        _run(command, f"convert {backend.family} {quant}")
    _run([converter, "--inspect", output], f"inspect {backend.family} {quant}")
    if not output.is_file() or output.stat().st_size < 1024:
        raise voice.VoiceValidationError(
            f"audio.cpp conversion produced no valid {quant} GGUF")
    return output


def _audiocpp_package_reference(repo_id: str,
                                backend: voice.VoiceBackend) -> bool:
    if (repo_id.casefold() == f"audio.cpp:{backend.family}".casefold()
            or any(repo_id.casefold() == package.casefold()
                   for package in backend.package_ids)):
        return True
    spec = voice.audio_cpp_catalog()["registry"].get(backend.family, {})
    defaults = spec.get("package_defaults", {}).get("download", {})
    for package in spec.get("packages", []):
        if package.get("format") != "gguf":
            continue
        download = {**defaults, **package.get("download", {})}
        package_repo = str(download.get("repo", ""))
        # The backend has already been resolved (with an explicit family when
        # this is a shared aggregate repo), so the matching package is safe.
        if package_repo and package_repo.casefold() == repo_id.casefold():
            return True
    return False


def _select_audiocpp_package(spec: dict, repo_id: str, quant: str) -> dict:
    packages = [package for package in spec.get("packages", [])
                if package.get("format") == "gguf"]
    exact = [package for package in packages
             if str(package.get("id", "")).casefold() == repo_id.casefold()]
    if exact:
        if _audio_cpp_precision(exact[0]) != quant.upper():
            raise voice.VoiceValidationError(
                f"package {repo_id} has precision {_audio_cpp_precision(exact[0])}, "
                f"not requested {quant.upper()}")
        return exact[0]
    candidates = [package for package in packages
                  if _audio_cpp_precision(package) == quant.upper()]
    recommended_id = spec.get("ui", {}).get("recommended_package")
    recommended = next((package for package in packages
                        if package.get("id") == recommended_id), None)
    if recommended is not None:
        same_variant = [package for package in candidates
                        if package.get("target_directory") ==
                        recommended.get("target_directory")]
        if len(same_variant) == 1:
            return same_variant[0]
    defaults = [package for package in candidates if package.get("default")]
    if len(defaults) == 1:
        return defaults[0]
    if len(candidates) == 1:
        return candidates[0]
    choices = ", ".join(str(package.get("id")) for package in candidates)
    raise voice.VoiceValidationError(
        f"audio.cpp family {spec['family']} has no unique {quant.upper()} "
        f"package; use one package id explicitly: {choices or 'none'}")


def _audio_cpp_precision(package: dict) -> str:
    return str(package.get("precision", "")).upper()


def _audiocpp_package_provenance(backend: voice.VoiceBackend,
                                 package_id: str) -> tuple[str | None,
                                                           str | None]:
    spec = voice.audio_cpp_catalog()["registry"].get(backend.family, {})
    package = next((item for item in spec.get("packages", [])
                    if item.get("id") == package_id), {})
    download = {**spec.get("package_defaults", {}).get("download", {}),
                **package.get("download", {})}
    return download.get("repo"), download.get("revision", "main")


def install_audiocpp_package(audio_dir: Path, backend: voice.VoiceBackend,
                             repo_id: str, models_root: Path,
                             quant: str) -> tuple[Path, tuple[Path, ...], str]:
    """Install exactly one package from audio.cpp's own model catalog."""
    _, spec = _audiocpp_spec(audio_dir, backend.family)
    package = _select_audiocpp_package(spec, repo_id, quant)
    manager = audio_dir / "tools" / "model_manager_v2.py"
    _run([sys.executable, manager, "install", package["id"],
          "--models-root", models_root],
         f"install audio.cpp package {package['id']}")
    target = models_root / package["target_directory"]
    strip_prefix = str(package.get("strip_prefix", "")).rstrip("/")
    paths = []
    for remote in package.get("files", []):
        relative = str(remote)
        if strip_prefix and relative.startswith(strip_prefix + "/"):
            relative = relative[len(strip_prefix) + 1:]
        path = target / relative
        if not path.is_file():
            raise voice.VoiceValidationError(
                f"audio.cpp package {package['id']} is missing {path}")
        paths.append(path)
    if not paths:
        raise voice.VoiceValidationError(
            f"audio.cpp package {package['id']} installed no files")
    ggufs = [path for path in paths if path.suffix.casefold() == ".gguf"]
    primary = ggufs[0] if ggufs else paths[0]
    runtime_model = target if len(paths) > 1 or len(ggufs) > 1 else primary
    return runtime_model, tuple(paths), str(package["id"])


def prepare_audiocpp_source(repo_id: str, track: str,
                            family: str | None = None):
    """Download a catalog-resolved source and return audio.cpp tools."""
    backend = voice.backend_for(repo_id, track=track, family=family)
    if backend is None or backend.backend != "audio.cpp":
        raise voice.VoiceValidationError(
            f"no audio.cpp {track.upper()} backend for {repo_id}")
    audio_dir, runtime, converter = ensure_audiocpp_tools()
    root = work_dir(repo_id)
    source, revision = _snapshot(repo_id, root / "source")
    return backend, source, revision, runtime, converter, audio_dir, root / "models"


def _standard_whisper_name(repo_id: str) -> str | None:
    if not repo_id.casefold().startswith("openai/whisper-"):
        return None
    return repo_id.split("/", 1)[1].removeprefix("whisper-")


def _ensure_openai_whisper_assets() -> Path:
    directory = config.VOICE_BACKENDS_DIR / "openai-whisper"
    from ..runtime_control import handle_for
    if handle := handle_for("whisper.cpp"):
        base = (config.WORK_ROOT / ".aqx-runtime-assets" if handle.manifest["storage_policy"] == "persistent"
                else config.TEMP_DIR / "aqx-runtime-assets")
        directory = base / "whisper.cpp" / handle.revision / "openai-whisper"
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        build_mod.run(["git", "clone", "https://github.com/openai/whisper.git",
                       directory])
    return directory


def prepare_whisper_source(repo_id: str) -> tuple[voice.VoiceBackend, Path,
                                                   str | None, Path, Path, Path]:
    """Download a standard model or convert a custom HF Whisper checkpoint."""
    backend = voice.backend_for(repo_id, track=voice.ASR)
    if backend is None:
        raise voice.VoiceValidationError(f"no Whisper backend for {repo_id}")
    whisper_dir, runtime, quantizer = ensure_whisper_tools()
    root = work_dir(repo_id)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    name = _standard_whisper_name(repo_id)
    revision = None
    if name:
        from huggingface_hub import HfApi, hf_hub_download
        filename = f"ggml-{name}.bin"
        revision = HfApi(token=config.TOKEN).model_info(
            "ggerganov/whisper.cpp").sha
        cached = Path(hf_hub_download(
            repo_id="ggerganov/whisper.cpp", filename=filename,
            revision=revision,
            token=config.TOKEN))
        base = models / filename
        if not base.exists():
            shutil.copy2(cached, base)
    else:
        if missing := _converter_requirements():
            raise RuntimeError("Whisper conversion requires " + ", ".join(missing))
        source, revision = _snapshot(repo_id, root / "source")
        assets = _ensure_openai_whisper_assets()
        base = models / f"ggml-{repo_id.split('/')[-1]}.bin"
        if not base.exists():
            _run([sys.executable, whisper_dir / "models" / "convert-h5-to-ggml.py",
                  source, assets, models], "convert Whisper checkpoint")
        candidates = list(models.glob("ggml-*.bin"))
        if not base.exists() and len(candidates) == 1:
            candidates[0].replace(base)
    if not base.is_file() or base.stat().st_size < 1024:
        raise voice.VoiceValidationError("Whisper conversion produced no valid model")
    return backend, base, revision, runtime, quantizer, whisper_dir


def quantize_whisper(base: Path, quantizer: Path, quant: str) -> Path:
    quant = quant.casefold()
    if quant not in voice.WHISPER_QUANTS:
        raise voice.VoiceValidationError(
            f"whisper.cpp quants are {', '.join(voice.WHISPER_QUANTS)}")
    output = base.with_name(base.stem + f"-{quant}.bin")
    if not output.exists():
        _run([quantizer, base, output, quant], f"quantize Whisper {quant}")
    return output


def load_fixtures(track: str, directory: Path | None = None) -> list[dict]:
    directory = directory or config.VOICE_FIXTURES_DIR
    path = directory / f"{track}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise voice.VoiceValidationError(f"cannot load voice fixtures {path}: {error}") \
            from error
    fixtures = data.get("fixtures") if isinstance(data, dict) else None
    if not isinstance(fixtures, list) or not fixtures:
        raise voice.VoiceValidationError(f"voice fixture pack is empty: {path}")
    return fixtures


def _tiny_whisper() -> tuple[Path, Path]:
    _, runtime = ensure_whisper_runtime()
    from huggingface_hub import hf_hub_download
    model = Path(hf_hub_download(
        repo_id="ggerganov/whisper.cpp", filename="ggml-tiny.bin",
        token=config.TOKEN))
    return runtime, model


def _path_stamp(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns}


def _tts_cache_inputs(bundle: voice.VoiceBundle, runtime: Path,
                      asr_runtime: Path, asr_model: Path,
                      fixtures: list[dict], options: VoiceReleaseOptions) -> dict:
    return {
        "schema": TTS_QUALITY_CACHE_SCHEMA,
        "backend": bundle.backend.id,
        "quant": bundle.quant,
        "members": [_path_stamp(member.path) for member in bundle.members],
        "runtime": _path_stamp(runtime),
        "asr_runtime": _path_stamp(asr_runtime),
        "asr_model": _path_stamp(asr_model),
        "speaker": _path_stamp(options.speaker),
        "language": options.language,
        "max_roundtrip_wer": options.max_roundtrip_wer,
        "fixtures": fixtures,
    }


def _aggregate_tts_quality(results: list[dict],
                           max_roundtrip_wer: float) -> dict:
    def average(key: str, digits: int):
        values = [result.get(key) for result in results
                  if isinstance(result.get(key), (int, float))]
        return round(sum(values) / len(values), digits) if values else None

    aggregate = {
        "fixtures": len(results),
        "intelligibility_metric": "WER for whitespace scripts; CER for CJK",
        "roundtrip_wer": average("roundtrip_wer", 6),
        "silence_ratio": round(max(r["silence_ratio"] for r in results), 6),
        "clipping_ratio": round(max(r["clipping_ratio"] for r in results), 6),
        "real_time_factor": average("real_time_factor", 4),
        "frames_per_second": average("frames_per_second", 2),
        "first_audio_seconds": average("first_output_seconds", 4),
        "minutes_per_audio_hour": average("minutes_per_audio_hour", 2),
    }
    aggregate["passed"] = voice.quality_passed(aggregate, max_roundtrip_wer)
    aggregate["results"] = results
    return aggregate


def _score_existing_tts_artifacts(bundle: voice.VoiceBundle,
                                  fixtures: list[dict],
                                  options: VoiceReleaseOptions,
                                  quality_dir: Path,
                                  review: dict | None = None) -> dict | None:
    """Recheck legacy fixture artifacts without rerunning expensive inference.

    This fallback is used only after an explicit human review exists. New runs
    write a provenance-aware quality cache and resume from that instead.
    """
    newest_member = max(member.path.stat().st_mtime_ns
                        for member in bundle.members)
    results = []
    for index, fixture in enumerate(fixtures):
        stem = f"{index:02d}-{fixture['id']}"
        output = quality_dir / f"{stem}.wav"
        asr_audio = quality_dir / f"{stem}-16khz.wav"
        prefix = quality_dir / f"{stem}-asr"
        transcript_path = Path(str(prefix) + ".txt")
        if not all(path.is_file()
                   for path in (output, asr_audio, transcript_path)):
            return None
        if output.stat().st_mtime_ns < newest_member:
            return None
        words = max(1, len(voice.normalize_transcript(fixture["text"])))
        generated = voice.validate_tts_audio(
            output, min_seconds=max(0.15, words * 0.04),
            max_seconds=min(180.0, max(4.0, words * 1.5)))
        if (bundle.backend.sample_rate
                and generated["sample_rate"] != bundle.backend.sample_rate):
            return None
        asr_facts = voice.inspect_wav(asr_audio)
        if asr_facts["sample_rate"] != 16_000 or asr_facts["sample_width"] != 2:
            return None
        transcript = voice.read_whisper_transcript(prefix)
        if not transcript:
            return None
        generated.update({
            "id": fixture["id"], "text": fixture["text"],
            "transcript": transcript,
            "roundtrip_wer": round(
                voice.word_error_rate(fixture["text"], transcript), 6),
            "real_time_factor": None, "frames_per_second": None,
            "first_output_seconds": None, "minutes_per_audio_hour": None,
            "resumed_from_artifacts": True,
        })
        results.append(generated)
    aggregate = _aggregate_tts_quality(results, options.max_roundtrip_wer)
    notes = str((review or {}).get("notes", ""))
    rtf = re.search(r"\bRTF\s+([0-9]+(?:\.[0-9]+)?)", notes,
                    flags=re.IGNORECASE)
    if rtf:
        aggregate["real_time_factor"] = float(rtf.group(1))
        aggregate["minutes_per_audio_hour"] = round(
            aggregate["real_time_factor"] * 60, 2)
        aggregate["performance_from_review"] = True
    aggregate["resumed_from_artifacts"] = True
    return aggregate


def score_tts(bundle: voice.VoiceBundle, runtime: Path, options: VoiceReleaseOptions,
              fixtures: list[dict] | None = None,
              asr_runtime: Path | None = None,
              asr_model: Path | None = None,
              existing_review: dict | None = None) -> dict:
    fixtures = fixtures or load_fixtures(voice.TTS)
    fixtures = [fixture for fixture in fixtures
                if bundle.backend.supports_language(fixture.get("language"))
                and (not fixture.get("speaker_conditioned") or options.speaker)]
    if not fixtures:
        raise voice.VoiceValidationError(
            "no TTS fixtures match this backend, language, and speaker setup")
    if asr_runtime is None or asr_model is None:
        asr_runtime, asr_model = _tiny_whisper()
    quality_dir = bundle.primary.path.parent / "quality" / (bundle.quant or "base")
    quality_dir.mkdir(parents=True, exist_ok=True)
    cache_path = quality_dir / "quality-cache.json"
    cache_inputs = _tts_cache_inputs(
        bundle, Path(runtime), Path(asr_runtime), Path(asr_model), fixtures, options)
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = None
    if (isinstance(cached, dict) and cached.get("inputs") == cache_inputs
            and isinstance(cached.get("quality"), dict)):
        return cached["quality"]
    if existing_review and existing_review.get("accepted"):
        existing = _score_existing_tts_artifacts(
            bundle, fixtures, options, quality_dir, existing_review)
        if existing is not None:
            return existing
    results = []
    for index, fixture in enumerate(fixtures):
        output = quality_dir / f"{index:02d}-{fixture['id']}.wav"
        output.unlink(missing_ok=True)
        generated = sanity.validate_tts_runtime(
            runtime, bundle, fixture["text"], output,
            language=fixture.get("language") or options.language,
            speaker=options.speaker, timeout=options.runtime_timeout)
        asr_audio = quality_dir / f"{index:02d}-{fixture['id']}-16khz.wav"
        asr_audio.unlink(missing_ok=True)
        voice.resample_pcm16_wav(output, asr_audio)
        prefix = quality_dir / f"{index:02d}-{fixture['id']}-asr"
        transcribed = sanity.validate_asr_runtime(
            asr_runtime, asr_model, asr_audio, fixture["text"], prefix,
            language=fixture.get("language"), timeout=options.runtime_timeout)
        generated.update({"id": fixture["id"], "text": fixture["text"],
                          "transcript": transcribed["transcript"],
                          "roundtrip_wer": transcribed["wer"]})
        results.append(generated)
    aggregate = _aggregate_tts_quality(results, options.max_roundtrip_wer)
    cache_path.write_text(json.dumps({"inputs": cache_inputs, "quality": aggregate},
                                     indent=2, sort_keys=True, default=str) + "\n",
                          encoding="utf-8")
    return aggregate


def score_asr(model: Path, runtime: Path, options: VoiceReleaseOptions,
              fixtures: list[dict] | None = None) -> dict:
    fixtures = fixtures or load_fixtures(voice.ASR)
    results = []
    for index, fixture in enumerate(fixtures):
        audio = config.VOICE_FIXTURES_DIR / fixture["audio"]
        prefix = model.parent / "quality" / model.stem / f"{index:02d}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        result = sanity.validate_asr_runtime(
            runtime, model, audio, fixture["transcript"], prefix,
            language=fixture.get("language"), vad=fixture.get("vad", False),
            timeout=options.runtime_timeout)
        result["id"] = fixture["id"]
        results.append(result)
    return {
        "fixtures": len(results),
        "wer": round(sum(result["wer"] for result in results) / len(results), 6),
        "real_time_factor": round(sum(result["real_time_factor"] for result in results)
                                  / len(results), 4),
        "audio_seconds_per_second": round(
            sum(result["audio_seconds_per_second"] for result in results)
            / len(results), 4),
        "results": results,
    }


def score_audiocpp_asr(backend: voice.VoiceBackend, model: Path,
                       runtime: Path, options: VoiceReleaseOptions,
                       fixtures: list[dict] | None = None) -> dict:
    fixtures = fixtures or load_fixtures(voice.ASR)
    results = []
    for index, fixture in enumerate(fixtures):
        if not backend.supports_language(fixture.get("language")):
            continue
        audio = config.VOICE_FIXTURES_DIR / fixture["audio"]
        output = model.parent / "quality" / model.stem / f"{index:02d}.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        result = voice.run_audiocpp_asr_smoke(
            runtime, backend, model, audio, fixture["transcript"], output,
            language=fixture.get("language"), timeout=options.runtime_timeout)
        result["id"] = fixture["id"]
        results.append(result)
    if not results:
        raise voice.VoiceValidationError(
            f"no ASR fixtures match {backend.family}'s declared languages")
    return {
        "fixtures": len(results),
        "wer": round(sum(result["wer"] for result in results) / len(results), 6),
        "real_time_factor": round(sum(result["real_time_factor"]
                                       for result in results) / len(results), 4),
        "audio_seconds_per_second": round(
            sum(result["audio_seconds_per_second"] for result in results)
            / len(results), 4),
        "results": results,
    }


def _mono_audio(path: Path, sample_rate: int) -> tuple[object, dict]:
    import numpy as np
    source_facts = voice.inspect_wav(path)
    source_rate = source_facts["sample_rate"]
    source = Path(path)
    with tempfile.TemporaryDirectory(prefix="aqx-audio-resample-") as temp_dir:
        if source_facts["sample_width"] != 2:
            with wave.open(str(source), "rb") as wav:
                raw = wav.readframes(wav.getnframes())
            pcm = np.asarray(
                list(voice._decode_samples(raw, source_facts["sample_width"])),
                dtype=np.float64)
            channels = source_facts["channels"]
            if channels > 1:
                pcm = pcm.reshape(-1, channels).mean(axis=1)
            converted = np.rint(np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
            source = Path(temp_dir) / "source-pcm16.wav"
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(source_rate)
                wav.writeframes(converted.tobytes())
        normalized = voice.resample_pcm16_wav(
            source, Path(temp_dir) / "resampled.wav", sample_rate)
        with wave.open(str(normalized), "rb") as wav:
            samples = (np.frombuffer(wav.readframes(wav.getnframes()),
                                     dtype="<i2").astype(np.float64)
                       / 32768.0)
    facts = {
        "sample_rate": sample_rate,
        "source_sample_rate": int(source_rate),
        "seconds": len(samples) / sample_rate,
        "rms": float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0,
        "peak": float(np.max(np.abs(samples))) if len(samples) else 0.0,
        "finite": bool(np.isfinite(samples).all()),
    }
    return samples, facts


def _fit_audio(samples, length: int):
    import numpy as np
    values = np.asarray(samples, dtype=np.float64)
    if len(values) > length:
        return values[:length]
    if len(values) < length:
        return np.pad(values, (0, length - len(values)))
    return values


def _si_sdr(reference, estimate) -> float:
    import numpy as np
    reference = np.asarray(reference, dtype=np.float64)
    estimate = _fit_audio(estimate, len(reference))
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    denominator = float(np.dot(reference, reference))
    if denominator <= 1e-20:
        raise voice.VoiceValidationError("SI-SDR reference is silent")
    target = float(np.dot(estimate, reference) / denominator) * reference
    noise = estimate - target
    ratio = float(np.dot(target, target)) / max(
        float(np.dot(noise, noise)), 1e-20)
    return 10.0 * math.log10(max(ratio, 1e-20))


def _separation_stems(out_dir: Path) -> tuple[Path, Path]:
    files = sorted(out_dir.rglob("*.wav"))
    vocal = next((path for path in files if "vocal" in path.name.casefold()), None)
    other = [path for path in files if path != vocal]
    accompaniment = next((path for path in other
                          if any(word in path.name.casefold()
                                 for word in ("instrumental", "accompaniment"))), None)
    if vocal is None:
        raise voice.VoiceValidationError(
            f"audio.cpp separation emitted no vocal stem in {out_dir}")
    if accompaniment is None and not other:
        raise voice.VoiceValidationError(
            f"audio.cpp separation emitted no accompaniment stems in {out_dir}")
    return vocal, accompaniment or other[0]


def score_audiocpp_separation(backend: voice.VoiceBackend, model: Path,
                              runtime: Path, options: VoiceReleaseOptions,
                              audio_dir: Path) -> dict:
    import numpy as np

    fixture_dir = config.VOICE_FIXTURES_DIR / "separation"
    mixture, mixture_facts = _mono_audio(
        fixture_dir / "mixture.wav", 44100)
    speech, _ = _mono_audio(fixture_dir / "speech.wav", 44100)
    accompaniment, _ = _mono_audio(
        fixture_dir / "accompaniment.wav", 44100)
    speech = _fit_audio(speech, len(mixture))
    accompaniment = _fit_audio(accompaniment, len(mixture))
    out_dir = model.parent / "quality" / model.stem / "stems"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = voice.audiocpp_sep_command(
        runtime, backend, model, fixture_dir / "mixture.wav", out_dir)
    execution = voice.run_checked(
        command, timeout=options.runtime_timeout, cwd=audio_dir)
    vocal_path, accompaniment_path = _separation_stems(out_dir)
    vocal, vocal_facts = _mono_audio(vocal_path, 44100)
    vocal = _fit_audio(vocal, len(speech))
    if accompaniment_path.name.casefold() in ("instrumental.wav", "accompaniment.wav"):
        inst, inst_facts = _mono_audio(accompaniment_path, 44100)
        accompaniment_estimate = _fit_audio(inst, len(accompaniment))
        accompaniment_paths = [str(accompaniment_path)]
    else:
        inst_paths = [path for path in sorted(out_dir.rglob("*.wav"))
                      if path != vocal_path]
        inst_facts = {}
        estimates = [_fit_audio(_mono_audio(path, 44100)[0], len(accompaniment))
                     for path in inst_paths]
        accompaniment_estimate = np.sum(estimates, axis=0)
        accompaniment_paths = [str(path) for path in inst_paths]
    vocal_score = _si_sdr(speech, vocal)
    accompaniment_score = _si_sdr(accompaniment, accompaniment_estimate)
    vocal_mixture = _si_sdr(speech, mixture)
    accompaniment_mixture = _si_sdr(accompaniment, mixture)
    result = {
        "audio_seconds": mixture_facts["seconds"],
        "vocal_sisdr_db": round(vocal_score, 6),
        "vocal_mixture_sisdr_db": round(vocal_mixture, 6),
        "vocal_improvement_db": round(vocal_score - vocal_mixture, 6),
        "accompaniment_sisdr_db": round(accompaniment_score, 6),
        "accompaniment_mixture_sisdr_db": round(accompaniment_mixture, 6),
        "accompaniment_improvement_db": round(
            accompaniment_score - accompaniment_mixture, 6),
        "output_paths": {
            "vocal": str(vocal_path),
            "accompaniment": accompaniment_paths,
        },
        "output_facts": {"vocal": vocal_facts, "accompaniment": inst_facts},
        "elapsed_seconds": execution["elapsed_seconds"],
        "command": command,
        "cwd": str(audio_dir),
    }
    return result


def score_audiocpp_vc(backend: voice.VoiceBackend, model: Path,
                      runtime: Path, options: VoiceReleaseOptions,
                      audio_dir: Path) -> dict:
    fixture_dir = config.VOICE_FIXTURES_DIR
    audio = fixture_dir / "jfk.wav"
    voice_ref = options.speaker or fixture_dir / "vc-reference.wav"
    output = model.parent / "quality" / model.stem / "converted.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    command = voice.audiocpp_vc_command(
        runtime, backend, model, audio, voice_ref, output)
    execution = voice.run_checked(
        command, timeout=options.runtime_timeout, cwd=audio_dir)
    _, facts = _mono_audio(output, 16000)
    source, source_facts = _mono_audio(audio, 16000)
    duration_delta = abs(facts["seconds"] - source_facts["seconds"])
    waveform_valid = (facts["finite"] and facts["rms"] > 1e-3
                      and duration_delta <= source_facts["seconds"] * 0.20)
    asr_audio = output.with_name("converted-16khz.wav")
    asr_result = None
    if waveform_valid:
        asr_runtime, asr_model = _tiny_whisper()
        voice.resample_pcm16_wav(output, asr_audio)
        fixture = load_fixtures(voice.ASR)[0]
        asr_result = sanity.validate_asr_runtime(
            asr_runtime, asr_model, asr_audio, fixture["transcript"],
            output.with_name("converted-asr"),
            language=fixture.get("language"),
            timeout=options.runtime_timeout)
    return {
        "finite": facts["finite"],
        "rms": round(facts["rms"], 6),
        "peak": round(facts["peak"], 6),
        "waveform_valid": waveform_valid,
        "sample_rate": facts["sample_rate"],
        "source_sample_rate": facts["source_sample_rate"],
        "duration_seconds": round(facts["seconds"], 6),
        "input_duration_seconds": round(source_facts["seconds"], 6),
        "duration_delta_seconds": round(duration_delta, 6),
        "roundtrip_wer": asr_result["wer"] if asr_result else None,
        "transcript": asr_result["transcript"] if asr_result else None,
        "output_path": str(output),
        "asr_audio_path": str(asr_audio) if asr_result else None,
        "elapsed_seconds": execution["elapsed_seconds"],
        "asr_elapsed_seconds": (
            asr_result["elapsed_seconds"] if asr_result else None),
        "command": command,
        "cwd": str(audio_dir),
    }


def _alignment_words(path: Path, sample_rate: int) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    words = data.get("words", data) if isinstance(data, dict) else data
    if not isinstance(words, list):
        raise voice.VoiceValidationError(
            "audio.cpp alignment output has no word list")
    result = []
    for item in words:
        if not isinstance(item, dict):
            raise voice.VoiceValidationError(
                "audio.cpp alignment output contains an invalid word")
        if "start_sample" in item:
            start = float(item["start_sample"]) / sample_rate
            end = float(item["end_sample"]) / sample_rate
        else:
            start = float(item.get("start", item.get("start_seconds")))
            end = float(item.get("end", item.get("end_seconds")))
        result.append({"word": str(item.get("word", "")),
                       "start": start, "end": end})
    return result


def score_audiocpp_align(backend: voice.VoiceBackend, model: Path,
                         runtime: Path, options: VoiceReleaseOptions,
                         audio_dir: Path) -> dict:
    fixture_dir = config.VOICE_FIXTURES_DIR
    fixture = load_fixtures(voice.ASR)[0]
    text = fixture["transcript"]
    audio = fixture_dir / fixture["audio"]
    words_out = model.parent / "quality" / model.stem / "words.json"
    words_out.parent.mkdir(parents=True, exist_ok=True)
    words_out.unlink(missing_ok=True)
    command = voice.audiocpp_align_command(
        runtime, backend, model, audio, text, fixture.get("language") or "en",
        words_out)
    execution = voice.run_checked(
        command, timeout=options.runtime_timeout, cwd=audio_dir)
    facts = voice.inspect_wav(audio)
    words = _alignment_words(words_out, facts["sample_rate"])
    expected_count = len(voice.normalize_transcript(text))
    starts = [word["start"] for word in words]
    ends = [word["end"] for word in words]
    valid = (len(words) == expected_count
             and all(0 <= start < end <= facts["seconds"]
                     for start, end in zip(starts, ends))
             and all(left <= right for left, right in zip(starts, starts[1:]))
             and all(left <= right for left, right in zip(ends, ends[1:])))
    return {
        "word_count": len(words),
        "expected_word_count": expected_count,
        "valid": valid,
        "duration_seconds": facts["seconds"],
        "words": words,
        "output_path": str(words_out),
        "elapsed_seconds": execution["elapsed_seconds"],
        "command": command,
        "cwd": str(audio_dir),
    }


def feasibility(bundle: voice.VoiceBundle, quality: dict | None = None,
                ram_gb: float | None = None, vram_gb: float | None = None) -> dict:
    quality = quality or {}
    total = sum(member.path.stat().st_size for member in bundle.members)
    return {
        "track": bundle.backend.track,
        "bundle_disk_bytes": total,
        "bundle_disk_gb": round(total / 1024 ** 3, 3),
        "ram_gb": ram_gb,
        "vram_gb": vram_gb,
        "real_time_factor": quality.get("real_time_factor"),
        "frames_per_second": quality.get("frames_per_second"),
        "audio_seconds_per_second": quality.get("audio_seconds_per_second"),
        "first_audio_seconds": quality.get("first_audio_seconds"),
        "minutes_per_audio_hour": quality.get("minutes_per_audio_hour"),
    }


def _remote_files(api, repo_id: str) -> dict[str, dict]:
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception:
        return {}
    result = {}
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        if isinstance(lfs, dict):
            digest = lfs.get("sha256") or lfs.get("oid")
            lfs_size = lfs.get("size")
        else:
            digest = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
            lfs_size = getattr(lfs, "size", None)
        if isinstance(digest, str):
            digest = digest.removeprefix("sha256:")
        result[sibling.rfilename] = {
            "bytes": getattr(sibling, "size", None) or lfs_size,
            "sha256": digest}
    return result


def publish_bundle(bundle: voice.VoiceBundle, target_repo: str,
                   license_name: str | None = None, api=None) -> dict:
    """Upload missing/mismatched members in one Hub commit, then verify."""
    from huggingface_hub import CommitOperationAdd, HfApi

    if problems := bundle.problems():
        raise voice.VoiceValidationError("refusing incomplete bundle: "
                                         + "; ".join(problems))
    api = api or HfApi(token=config.TOKEN)
    api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)
    if license_name is None and bundle.source_repo:
        try:
            source_info = api.model_info(bundle.source_repo)
            card_data = getattr(source_info, "card_data", None)
            license_name = (getattr(card_data, "license", None)
                            or (card_data.get("license")
                                if isinstance(card_data, dict) else None))
        except Exception:
            license_name = None
    stage = bundle.primary.path.parent / "publication" / (bundle.quant or "base")
    stage.mkdir(parents=True, exist_ok=True)
    manifest_path = voice.write_manifest(bundle, stage / "bundle.json")
    quality_path = stage / "quality.json"
    quality_path.write_text(json.dumps(bundle.quality or {}, indent=2,
                                       sort_keys=True) + "\n", encoding="utf-8")
    card_path = stage / "README.md"
    card_path.write_text(voice.render_model_card(bundle, target_repo, license_name),
                         encoding="utf-8")
    manifest = voice.load_manifest(manifest_path)
    remote = _remote_files(api, target_repo)
    operations = []
    for member in bundle.members:
        expected = member.manifest()
        actual = remote.get(member.remote_path)
        if (actual and actual.get("bytes") == expected["bytes"]
                and (not actual.get("sha256")
                     or actual["sha256"] == expected["sha256"])):
            continue
        operations.append(CommitOperationAdd(
            path_in_repo=member.remote_path, path_or_fileobj=str(member.path)))
    for local, remote_path in ((manifest_path, "bundle.json"),
                               (quality_path, "quality.json"),
                               (card_path, "README.md")):
        operations.append(CommitOperationAdd(
            path_in_repo=remote_path, path_or_fileobj=str(local)))
    if operations:
        api.create_commit(repo_id=target_repo, repo_type="model",
                          operations=operations,
                          commit_message=(f"Publish {bundle.backend.family} "
                                          f"{bundle.quant or 'base'} bundle"))
    remote = _remote_files(api, target_repo)
    if problems := voice.remote_bundle_problems(manifest, remote):
        raise voice.VoiceValidationError("published bundle failed verification: "
                                         + "; ".join(problems))
    return {"repo_id": target_repo, "uploaded": len(operations),
            "members": [member.remote_path for member in bundle.members],
            "verified": True}


def _review_for(path: Path, quant: str) -> dict | None:
    candidate = path / f"{quant}.json" if path.is_dir() else path
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    reviews = data.get("reviews") if isinstance(data, dict) else None
    entries = reviews if isinstance(reviews, list) else [data]
    return next((entry for entry in entries
                 if isinstance(entry, dict) and entry.get("quant") == quant), None)


def _review_skip_entry(review: dict | None) -> dict:
    return {
        "quality": None,
        "published": False,
        "reason": ("human listening review rejected this quant"
                   if review is not None else
                   "human listening review not recorded for this quant"),
    }


def _aggregate_bundles(bundles: list[voice.VoiceBundle]) -> voice.VoiceBundle:
    """One atomic publication containing every accepted quant variant."""
    if not bundles:
        raise voice.VoiceValidationError("cannot publish an empty voice bundle")
    first = bundles[0]
    primary_members = [bundle.primary for bundle in bundles]
    companion_by_path = {}
    for bundle in bundles:
        for member in bundle.companions:
            companion_by_path[member.remote_path] = member
    quality = {"quants": {bundle.quant: bundle.quality for bundle in bundles}}
    return voice.VoiceBundle(
        backend=first.backend,
        primary=primary_members[0],
        companions=tuple([*primary_members[1:], *companion_by_path.values()]),
        source_repo=first.source_repo,
        source_revision=first.source_revision,
        configuration=first.configuration,
        quality=quality,
    )


def run_audiocpp_tts_release(repo_id: str, target_repo: str,
                             options: VoiceReleaseOptions) -> dict:
    """Convert, validate, review, and publish an audio.cpp TTS source."""
    from ..runtime_control import prepare_voice_runtimes
    prepare_voice_runtimes(repo_id, "audio.cpp", voice.TTS, persist=True)
    backend = voice.backend_for(repo_id, track=voice.TTS, family=options.family)
    assert backend is not None
    package_mode = _audiocpp_package_reference(repo_id, backend)
    if package_mode:
        audio_dir, runtime, converter = ensure_audiocpp_tools()
        models = work_dir(repo_id) / "models"
        source = None
    else:
        backend, source, revision, runtime, converter, audio_dir, models = \
            prepare_audiocpp_source(repo_id, voice.TTS, options.family)
    audio_cpp_revision = _audiocpp_runtime_revision(audio_dir)
    if package_mode:
        revision = audio_cpp_revision
    if backend.speaker_reference == "required":
        if options.speaker is None:
            raise voice.VoiceValidationError(
                f"{backend.family} requires --speaker with a consented WAV reference")
        if not options.speaker.is_file():
            raise voice.VoiceValidationError(
                f"speaker reference does not exist: {options.speaker}")

    results = {}
    accepted = []
    for quant in options.quants or list(voice.available_quants(repo_id, backend)):
        review = (_review_for(Path(options.human_review), quant)
                  if options.human_review is not None else None)
        if options.human_review is not None and (
                review is None or not review.get("accepted")):
            results[quant] = _review_skip_entry(review)
            continue
        package_id = None
        runtime_model = None
        companions = ()
        if package_mode:
            runtime_model, paths, package_id = install_audiocpp_package(
                audio_dir, backend, repo_id, models, quant)
            bundle_source, bundle_revision = _audiocpp_package_provenance(
                backend, package_id)
            primary = next((path for path in paths
                            if path.suffix.casefold() == ".gguf"), paths[0])
            primary_member = voice.BundleMember(
                primary, "primary", hub_path=str(primary.relative_to(models)),
                quant=quant)
            companions = tuple(voice.BundleMember(
                path, "package-member", hub_path=str(path.relative_to(models)),
                quant=quant) for path in paths if path != primary)
        else:
            assert source is not None
            primary = convert_audiocpp_model(
                source, backend, converter, audio_dir, models, quant)
            primary_member = voice.BundleMember(primary, "primary", quant=quant)
            bundle_source, bundle_revision = repo_id, revision
        bundle_configuration = {
            "sample_rate": 16_000,
            "audio_cpp_revision": audio_cpp_revision,
            "package_format": "standalone GGUF with embedded model spec",
            "audio_cpp_package": package_id,
        }
        bundle = voice.VoiceBundle(
            backend=backend,
            primary=primary_member,
            companions=companions,
            source_repo=bundle_source,
            source_revision=bundle_revision,
            quant=quant,
            configuration={
                "language": options.language,
                "speaker_reference": backend.speaker_reference,
                "audio_cpp_revision": audio_cpp_revision,
                "package_format": "standalone GGUF with embedded model spec",
                "audio_cpp_package": package_id,
            },
            runtime_model=runtime_model,
        )
        quality = score_tts(
            bundle, runtime, options,
            existing_review=review)
        bundle = voice.VoiceBundle(**{**bundle.__dict__, "quality": quality})
        entry = {"quality": quality, "feasibility": feasibility(bundle, quality)}
        if not quality["passed"]:
            entry["published"] = False
            entry["reason"] = "automated quality gate failed"
        elif options.human_review is None:
            entry["published"] = False
            entry["reason"] = "human listening review required"
        else:
            accepted.append(bundle)
            entry["published"] = False
            entry["reason"] = "accepted and waiting for atomic bundle publication"
        results[quant] = entry
    if accepted and options.publish:
        publication = publish_bundle(_aggregate_bundles(accepted), target_repo)
        for bundle in accepted:
            results[bundle.quant]["publication"] = publication
            results[bundle.quant]["published"] = True
            results[bundle.quant].pop("reason", None)
    return {"backend": backend.id, "source": repo_id, "target": target_repo,
            "results": results}


def run_tts_release(repo_id: str, target_repo: str,
                    options: VoiceReleaseOptions | None = None) -> dict:
    options = _with_recorded_reviews(
        repo_id, options or VoiceReleaseOptions())
    selected = voice.backend_for(
        repo_id, track=voice.TTS, family=options.family)
    from ..runtime_control import prepare_voice_runtimes
    prepare_voice_runtimes(repo_id, selected.backend if selected else "llama.cpp", voice.TTS, persist=True)
    if selected is not None and selected.backend == "audio.cpp":
        return run_audiocpp_tts_release(repo_id, target_repo, options)
    backend, base, mmproj, revision, runtime, quantizer, llama_dir = \
        convert_tts_source(repo_id, options)
    quants = options.quants or list(
        voice.available_quants(repo_id, backend, llama_dir=llama_dir))
    results = {}
    accepted = []
    importance, gap_args, imatrix_error = None, [], None
    if any(quant in config.IQ_QUANTS or quant in config.IMATRIX_GUIDED
           for quant in quants):
        try:
            importance, gap_args = _tts_imatrix(base, llama_dir)
        except Exception as error:
            imatrix_error = str(error)
    for quant in quants:
        review = (_review_for(Path(options.human_review), quant)
                  if options.human_review is not None else None)
        if options.human_review is not None and (
                review is None or not review.get("accepted")):
            results[quant] = _review_skip_entry(review)
            continue
        if quant in config.IMATRIX_REQUIRED and importance is None:
            results[quant] = {
                "published": False,
                "reason": "skipped: this quant requires an importance matrix",
                "imatrix_error": imatrix_error,
            }
            continue
        primary = quantize_tts(
            base, quantizer, quant, imatrix=importance, gap_args=gap_args)
        bundle = voice.VoiceBundle(
            backend=backend,
            primary=voice.BundleMember(primary, "primary", quant=quant),
            companions=(voice.BundleMember(mmproj, "mmproj", quant="Q8_0"),),
            source_repo=repo_id, source_revision=revision, quant=quant,
            configuration={"language": options.language,
                           "speaker_reference": backend.speaker_reference,
                           "llama_cpp_revision": _git_revision(llama_dir)},
        )
        quality = score_tts(
            bundle, runtime, options,
            existing_review=review)
        bundle = voice.VoiceBundle(**{**bundle.__dict__, "quality": quality})
        entry = {"quality": quality, "feasibility": feasibility(bundle, quality)}
        if not quality["passed"]:
            entry["published"] = False
            entry["reason"] = "automated quality gate failed"
        elif options.human_review is None:
            entry["published"] = False
            entry["reason"] = "human listening review required"
        else:
            accepted.append(bundle)
            entry["published"] = False
            entry["reason"] = "accepted and waiting for atomic bundle publication"
        results[quant] = entry
    if accepted and options.publish:
        publication = publish_bundle(_aggregate_bundles(accepted), target_repo)
        for bundle in accepted:
            results[bundle.quant]["publication"] = publication
            results[bundle.quant]["published"] = True
            results[bundle.quant].pop("reason", None)
    return {"backend": backend.id, "source": repo_id, "target": target_repo,
            "results": results}


def run_asr_release(repo_id: str, target_repo: str,
                    options: VoiceReleaseOptions | None = None) -> dict:
    options = options or VoiceReleaseOptions()
    selected = voice.backend_for(
        repo_id, track=voice.ASR, family=options.family)
    from ..runtime_control import prepare_voice_runtimes
    prepare_voice_runtimes(repo_id, selected.backend if selected else "whisper.cpp", voice.ASR, persist=True)
    if selected is not None and selected.backend == "audio.cpp":
        return run_audiocpp_asr_release(repo_id, target_repo, options)
    backend, base, revision, runtime, quantizer, whisper_dir = \
        prepare_whisper_source(repo_id)
    baseline = score_asr(base, runtime, options)
    results = {}
    accepted = []
    for quant in options.quants or list(voice.available_quants(repo_id, backend)):
        model = quantize_whisper(base, quantizer, quant)
        quality = score_asr(model, runtime, options)
        quality["regression"] = voice.asr_regression(
            quality["wer"], baseline["wer"], options.max_wer_regression)
        bundle = voice.VoiceBundle(
            backend=backend,
            primary=voice.BundleMember(model, "asr-model", quant=quant),
            source_repo=repo_id, source_revision=revision, quant=quant,
            configuration={"sample_rate": 16_000,
                           "whisper_cpp_revision": _git_revision(whisper_dir)},
            quality=quality,
        )
        entry = {"quality": quality, "feasibility": feasibility(bundle, quality),
                 "published": False}
        if quality["regression"]["passed"]:
            accepted.append(bundle)
            entry["reason"] = "accepted and waiting for atomic bundle publication"
        elif not quality["regression"]["passed"]:
            entry["reason"] = "WER regression gate failed"
        results[quant] = entry
    if accepted and options.publish:
        publication = publish_bundle(_aggregate_bundles(accepted), target_repo)
        for bundle in accepted:
            results[bundle.quant]["publication"] = publication
            results[bundle.quant]["published"] = True
            results[bundle.quant].pop("reason", None)
    return {"backend": backend.id, "source": repo_id, "target": target_repo,
            "baseline": baseline, "results": results}


def _run_audiocpp_release(repo_id: str, target_repo: str,
                          options: VoiceReleaseOptions, track: str,
                          score, gate) -> dict:
    from ..runtime_control import prepare_voice_runtimes
    backend = voice.backend_for(repo_id, track=track, family=options.family)
    assert backend is not None
    source_info = (source_metadata(repo_id, family=options.family)
                   if track in (voice.SEP, voice.VC, voice.ALIGN) else {})
    license_name = source_info.get("license")
    if track in (voice.SEP, voice.VC, voice.ALIGN) and options.publish \
            and not license_name:
        raise voice.VoiceValidationError(
            "licence undetermined; cannot publish")
    prepare_voice_runtimes(repo_id, "audio.cpp", track, persist=True)
    package_mode = _audiocpp_package_reference(repo_id, backend)
    if package_mode:
        audio_dir, runtime, converter = ensure_audiocpp_tools()
        models = work_dir(repo_id) / "models"
        source = None
    else:
        backend, source, revision, runtime, converter, audio_dir, models = \
            prepare_audiocpp_source(repo_id, track, options.family)
    audio_cpp_revision = _audiocpp_runtime_revision(audio_dir)
    if package_mode:
        revision = audio_cpp_revision
    requested = options.quants or list(voice.available_quants(repo_id, backend))
    baseline_quant = next((quant for quant in ("BF16", "F16", "F32", "ORIG")
                           if quant in requested), requested[0])
    package_artifacts = {}

    def prepare(quant):
        started = time.perf_counter()
        if not package_mode:
            assert source is not None
            conversion_options = (
                {"match_gguf_namespaces": True}
                if track in (voice.SEP, voice.VC, voice.ALIGN) else {})
            model = convert_audiocpp_model(
                source, backend, converter, audio_dir, models, quant,
                **conversion_options)
            return model, (model,), None, time.perf_counter() - started
        runtime_model, paths, package_id = install_audiocpp_package(
            audio_dir, backend, repo_id, models, quant)
        elapsed = time.perf_counter() - started
        package_artifacts[quant] = (runtime_model, paths, package_id, elapsed)
        return runtime_model, paths, package_id, elapsed

    baseline_model, _, _, baseline_prepare_seconds = prepare(baseline_quant)
    baseline = score(backend, baseline_model, runtime, options, audio_dir)
    if track != voice.ASR:
        baseline["model_prepare_seconds"] = round(baseline_prepare_seconds, 6)
    results = {}
    accepted = []
    for quant in requested:
        if quant == baseline_quant:
            model, paths, package_id, prepare_seconds = (
                package_artifacts.get(quant)
                or (baseline_model, (baseline_model,), None,
                    baseline_prepare_seconds))
        else:
            model, paths, package_id, prepare_seconds = prepare(quant)
        quality = (baseline if quant == baseline_quant else
                   score(backend, model, runtime, options, audio_dir))
        quality = dict(quality)
        if track == voice.ASR:
            quality["regression"] = gate(
                quality, baseline, quant == baseline_quant)
        else:
            quality["task_gate"] = gate(
                quality, baseline, quant == baseline_quant)
            quality["model_prepare_seconds"] = round(prepare_seconds, 6)
        primary = next((path for path in paths
                        if path.suffix.casefold() == ".gguf"), paths[0])
        if package_id:
            bundle_source, bundle_revision = _audiocpp_package_provenance(
                backend, package_id)
        else:
            bundle_source, bundle_revision = repo_id, revision
        companions = tuple(voice.BundleMember(
            path, "package-member",
            hub_path=(str(path.relative_to(models)) if package_mode else None),
            quant=quant) for path in paths if path != primary)
        bundle_configuration = {
            "sample_rate": 16_000,
            "audio_cpp_revision": audio_cpp_revision,
            "package_format": "standalone GGUF with embedded model spec",
            "audio_cpp_package": package_id,
        }
        if track != voice.ASR:
            bundle_configuration["source_license"] = license_name
        role = {
            voice.ASR: "asr-model",
            voice.SEP: "separation-model",
            voice.VC: "vc-model",
            voice.ALIGN: "alignment-model",
        }[track]
        bundle = voice.VoiceBundle(
            backend=backend,
            primary=voice.BundleMember(
                primary, role,
                hub_path=(str(primary.relative_to(models))
                          if package_mode else None), quant=quant),
            companions=companions,
            source_repo=bundle_source, source_revision=bundle_revision,
            quant=quant,
            configuration=bundle_configuration,
            quality=quality,
            runtime_model=model if package_mode else None,
        )
        passed = (quality["regression"]["passed"] if track == voice.ASR
                  else quality["task_gate"]["passed"])
        entry = {"quality": quality, "feasibility": feasibility(bundle, quality),
                 "published": False,
                 "reason": ("accepted and waiting for atomic bundle publication"
                            if passed else (
                                "WER regression gate failed" if track == voice.ASR
                                else "native task quality gate failed"))}
        if passed:
            accepted.append(bundle)
        results[quant] = entry
    if accepted and options.publish:
        publication = publish_bundle(_aggregate_bundles(accepted), target_repo,
                                     license_name=license_name)
        for bundle in accepted:
            results[bundle.quant]["publication"] = publication
            results[bundle.quant]["published"] = True
            results[bundle.quant].pop("reason", None)
    return {"backend": backend.id, "source": repo_id, "target": target_repo,
            "baseline_quant": baseline_quant, "baseline": baseline,
            "results": results}


def _asr_quality_gate(candidate: dict, baseline: dict,
                      is_baseline: bool, options: VoiceReleaseOptions) -> dict:
    return voice.asr_regression(
        candidate["wer"], baseline["wer"], options.max_wer_regression)


def _separation_quality_gate(candidate: dict, baseline: dict,
                             is_baseline: bool) -> dict:
    if is_baseline:
        metrics = {
            "vocal_improvement_db": candidate["vocal_improvement_db"],
            "accompaniment_improvement_db":
                candidate["accompaniment_improvement_db"],
        }
        return {**metrics, "minimum_improvement_db": 6.0,
                "passed": all(value >= 6.0 for value in metrics.values())}
    deltas = {
        "vocal_delta_db": candidate["vocal_sisdr_db"]
                         - baseline["vocal_sisdr_db"],
        "accompaniment_delta_db": candidate["accompaniment_sisdr_db"]
                                  - baseline["accompaniment_sisdr_db"],
    }
    baseline_passed = (baseline["vocal_improvement_db"] >= 6.0
                       and baseline["accompaniment_improvement_db"] >= 6.0)
    return {**deltas, "minimum_delta_db": -1.0,
            "baseline_passed": baseline_passed,
            "passed": (baseline_passed
                       and all(value >= -1.0 for value in deltas.values()))}


def _vc_quality_gate(candidate: dict, baseline: dict,
                     is_baseline: bool) -> dict:
    waveform_passed = candidate.get("waveform_valid", True)
    if is_baseline:
        return {"roundtrip_wer": candidate["roundtrip_wer"],
                "maximum_wer": 0.35,
                "waveform_valid": waveform_passed,
                "passed": (waveform_passed
                           and candidate["roundtrip_wer"] is not None
                           and candidate["roundtrip_wer"] <= 0.35)}
    if baseline["roundtrip_wer"] is None:
        return {"roundtrip_wer": candidate["roundtrip_wer"],
                "baseline_wer": None, "maximum_wer": None,
                "waveform_valid": waveform_passed,
                "baseline_passed": False, "passed": False}
    limit = baseline["roundtrip_wer"] + 0.10
    baseline_passed = (baseline.get("waveform_valid", True)
                       and baseline["roundtrip_wer"] <= 0.35)
    return {"roundtrip_wer": candidate["roundtrip_wer"],
            "baseline_wer": baseline["roundtrip_wer"],
            "maximum_wer": round(limit, 6),
            "waveform_valid": waveform_passed,
            "baseline_passed": baseline_passed,
            "passed": (baseline_passed and waveform_passed
                       and candidate["roundtrip_wer"] is not None
                       and candidate["roundtrip_wer"] <= limit + 1e-12)}


def _alignment_quality_gate(candidate: dict, baseline: dict,
                            is_baseline: bool) -> dict:
    if is_baseline:
        return {"word_count": candidate["word_count"],
                "valid": candidate.get("valid", True),
                "passed": candidate.get("valid", True)}
    candidate_boundaries = [value for word in candidate["words"]
                            for value in (word["start"], word["end"])]
    baseline_boundaries = [value for word in baseline["words"]
                           for value in (word["start"], word["end"])]
    deltas = [abs(left - right) for left, right
              in zip(candidate_boundaries, baseline_boundaries)]
    mean_delta = sum(deltas) / len(deltas) if deltas else float("inf")
    max_delta = max(deltas, default=float("inf"))
    return {"mean_boundary_delta_seconds": round(mean_delta, 6),
            "max_boundary_delta_seconds": round(max_delta, 6),
            "maximum_mean_delta_seconds": 0.05,
            "maximum_boundary_delta_seconds": 0.20,
            "valid": candidate.get("valid", True),
            "baseline_valid": baseline.get("valid", True),
            "passed": (baseline.get("valid", True)
                       and candidate.get("valid", True)
                       and mean_delta <= 0.05 + 1e-12
                       and max_delta <= 0.20 + 1e-12)}


def run_audiocpp_asr_release(repo_id: str, target_repo: str,
                             options: VoiceReleaseOptions) -> dict:
    def score_asr(backend, model, runtime, score_options, _audio_dir):
        return score_audiocpp_asr(backend, model, runtime, score_options)

    def gate_asr(candidate, baseline, is_baseline):
        return _asr_quality_gate(candidate, baseline, is_baseline, options)

    return _run_audiocpp_release(
        repo_id, target_repo, options, voice.ASR, score_asr, gate_asr)


def run_audiocpp_sep_release(repo_id: str, target_repo: str,
                             options: VoiceReleaseOptions) -> dict:
    return _run_audiocpp_release(
        repo_id, target_repo, options, voice.SEP,
        score_audiocpp_separation, _separation_quality_gate)


def run_audiocpp_vc_release(repo_id: str, target_repo: str,
                            options: VoiceReleaseOptions) -> dict:
    return _run_audiocpp_release(
        repo_id, target_repo, options, voice.VC, score_audiocpp_vc,
        _vc_quality_gate)


def run_audiocpp_align_release(repo_id: str, target_repo: str,
                               options: VoiceReleaseOptions) -> dict:
    return _run_audiocpp_release(
        repo_id, target_repo, options, voice.ALIGN, score_audiocpp_align,
        _alignment_quality_gate)
