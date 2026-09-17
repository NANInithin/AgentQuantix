"""End-to-end TTS and ASR release pipelines.

The publication unit is a verified :class:`VoiceBundle`.  TTS uses the cached
llama.cpp or audio.cpp checkout; ASR has a separate whisper.cpp checkout,
build, converter, model format, quantizer, and acceptance corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .. import archsupport, config, voice
from . import build as build_mod, sanity


AUDIOCPP_REVISION = "v0.8.0"


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


def _git_revision(directory: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=20,
        ).stdout.strip()
    except Exception:
        return None


def ensure_llama_tts() -> tuple[Path, Path, Path]:
    """Return the llama.cpp checkout, llama-tts, and llama-quantize."""
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
                   for item in backend.package_ids)):
        return {
            "repo_id": repo_id,
            "revision": f"audio.cpp-{AUDIOCPP_REVISION}",
            "source_bytes": None,
            "gated": False,
            "private": False,
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
    return {
        "repo_id": repo_id,
        "revision": getattr(info, "sha", None),
        "source_bytes": total or None,
        "source_files": sorted(source_files) if source_files else None,
        "gated": bool(getattr(info, "gated", False)),
        "private": bool(getattr(info, "private", False)),
    }


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
    spec = voice.AUDIOCPP_SPEC_REGISTRY.get(backend.family, {})
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

    executable = build_mod.find_binary(llama_dir, "llama-imatrix")
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


def _audiocpp_tensor_inputs(source: Path, spec: dict) -> list[str]:
    """Resolve the generic safetensors input contract from a model spec."""
    source_contract = next((item for item in spec.get("sources", [])
                            if item.get("format") == "safetensors"), None)
    if source_contract is None:
        raise voice.VoiceValidationError(
            f"audio.cpp {spec['family']} has no safetensors conversion contract; "
            "use one of the GGUF packages declared by that backend catalog")
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
        inputs.append(f"{namespace}={path}")
    if missing:
        raise voice.VoiceValidationError(
            f"{spec['family']} source is missing audio.cpp-declared tensor "
            "inputs: " + ", ".join(missing))
    if not inputs:
        raise voice.VoiceValidationError(
            f"audio.cpp {spec['family']} declares no safetensors inputs")
    return inputs


def _prepare_audiocpp_tensor_sources(source: Path, spec: dict,
                                     audio_dir: Path) -> list[str]:
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
        return _audiocpp_tensor_inputs(source, spec)

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

    return _audiocpp_tensor_inputs(source, spec)


def convert_audiocpp_model(source: Path, backend: voice.VoiceBackend,
                           converter: Path, audio_dir: Path, output_dir: Path,
                           quant: str) -> Path:
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
                source, spec, audio_dir):
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
    spec = voice.AUDIOCPP_SPEC_REGISTRY.get(backend.family, {})
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
    spec = voice.AUDIOCPP_SPEC_REGISTRY.get(backend.family, {})
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


def score_tts(bundle: voice.VoiceBundle, runtime: Path, options: VoiceReleaseOptions,
              fixtures: list[dict] | None = None,
              asr_runtime: Path | None = None,
              asr_model: Path | None = None) -> dict:
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
    aggregate = {
        "fixtures": len(results),
        "roundtrip_wer": round(sum(r["roundtrip_wer"] for r in results)
                                / len(results), 6),
        "silence_ratio": round(max(r["silence_ratio"] for r in results), 6),
        "clipping_ratio": round(max(r["clipping_ratio"] for r in results), 6),
        "real_time_factor": round(sum(r["real_time_factor"] for r in results)
                                  / len(results), 4),
        "frames_per_second": round(sum(r["frames_per_second"] for r in results)
                                   / len(results), 2),
        "first_audio_seconds": round(sum(r["first_output_seconds"] for r in results)
                                     / len(results), 4),
        "minutes_per_audio_hour": round(
            sum(r["minutes_per_audio_hour"] for r in results) / len(results), 2),
    }
    aggregate["passed"] = voice.quality_passed(
        aggregate, options.max_roundtrip_wer)
    aggregate["results"] = results
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
    backend = voice.backend_for(repo_id, track=voice.TTS, family=options.family)
    assert backend is not None
    package_mode = _audiocpp_package_reference(repo_id, backend)
    if package_mode:
        audio_dir, runtime, converter = ensure_audiocpp_tools()
        models = work_dir(repo_id) / "models"
        source = None
        revision = f"audio.cpp-{AUDIOCPP_REVISION}"
    else:
        backend, source, revision, runtime, converter, audio_dir, models = \
            prepare_audiocpp_source(repo_id, voice.TTS, options.family)
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
                "audio_cpp_revision": _git_revision(audio_dir),
                "package_format": "standalone GGUF with embedded model spec",
                "audio_cpp_package": package_id,
            },
            runtime_model=runtime_model,
        )
        quality = score_tts(bundle, runtime, options)
        bundle = voice.VoiceBundle(**{**bundle.__dict__, "quality": quality})
        entry = {"quality": quality, "feasibility": feasibility(bundle, quality)}
        if not quality["passed"]:
            entry["published"] = False
            entry["reason"] = "automated quality gate failed"
        elif options.human_review is None:
            entry["published"] = False
            entry["reason"] = "human listening review required"
        else:
            review = _review_for(Path(options.human_review), quant)
            if not review or not review.get("accepted"):
                entry["published"] = False
                entry["reason"] = "human listening review did not accept this quant"
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
    options = options or VoiceReleaseOptions()
    selected = voice.backend_for(
        repo_id, track=voice.TTS, family=options.family)
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
        quality = score_tts(bundle, runtime, options)
        bundle = voice.VoiceBundle(**{**bundle.__dict__, "quality": quality})
        entry = {"quality": quality, "feasibility": feasibility(bundle, quality)}
        if not quality["passed"]:
            entry["published"] = False
            entry["reason"] = "automated quality gate failed"
        elif options.human_review is None:
            entry["published"] = False
            entry["reason"] = "human listening review required"
        else:
            review = _review_for(Path(options.human_review), quant)
            if not review or not review.get("accepted"):
                entry["published"] = False
                entry["reason"] = "human listening review did not accept this quant"
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


def run_audiocpp_asr_release(repo_id: str, target_repo: str,
                             options: VoiceReleaseOptions) -> dict:
    """Convert and quality-gate any ASR family in audio.cpp's catalog."""
    backend = voice.backend_for(repo_id, track=voice.ASR, family=options.family)
    assert backend is not None
    package_mode = _audiocpp_package_reference(repo_id, backend)
    if package_mode:
        audio_dir, runtime, converter = ensure_audiocpp_tools()
        models = work_dir(repo_id) / "models"
        source = None
        revision = f"audio.cpp-{AUDIOCPP_REVISION}"
    else:
        backend, source, revision, runtime, converter, audio_dir, models = \
            prepare_audiocpp_source(repo_id, voice.ASR, options.family)
    requested = options.quants or list(voice.available_quants(repo_id, backend))
    baseline_quant = next((quant for quant in ("BF16", "F16", "F32", "ORIG")
                           if quant in requested), requested[0])
    package_artifacts = {}

    def prepare(quant):
        if not package_mode:
            assert source is not None
            model = convert_audiocpp_model(
                source, backend, converter, audio_dir, models, quant)
            return model, (model,), None
        runtime_model, paths, package_id = install_audiocpp_package(
            audio_dir, backend, repo_id, models, quant)
        package_artifacts[quant] = (runtime_model, paths, package_id)
        return runtime_model, paths, package_id

    baseline_model, _, _ = prepare(baseline_quant)
    baseline = score_audiocpp_asr(backend, baseline_model, runtime, options)
    results = {}
    accepted = []
    for quant in requested:
        if quant == baseline_quant:
            model, paths, package_id = (package_artifacts.get(quant)
                                        or (baseline_model,
                                            (baseline_model,), None))
        else:
            model, paths, package_id = prepare(quant)
        quality = (baseline if quant == baseline_quant else
                   score_audiocpp_asr(backend, model, runtime, options))
        quality = dict(quality)
        quality["regression"] = voice.asr_regression(
            quality["wer"], baseline["wer"], options.max_wer_regression)
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
        bundle = voice.VoiceBundle(
            backend=backend,
            primary=voice.BundleMember(
                primary, "asr-model",
                hub_path=(str(primary.relative_to(models))
                          if package_mode else None), quant=quant),
            companions=companions,
            source_repo=bundle_source, source_revision=bundle_revision,
            quant=quant,
            configuration={
                "sample_rate": 16_000,
                "audio_cpp_revision": _git_revision(audio_dir),
                "package_format": "standalone GGUF with embedded model spec",
                "audio_cpp_package": package_id,
            },
            quality=quality,
            runtime_model=model if package_mode else None,
        )
        passed = quality["regression"]["passed"]
        entry = {"quality": quality, "feasibility": feasibility(bundle, quality),
                 "published": False,
                 "reason": ("accepted and waiting for atomic bundle publication"
                            if passed else "WER regression gate failed")}
        if passed:
            accepted.append(bundle)
        results[quant] = entry
    if accepted and options.publish:
        publication = publish_bundle(_aggregate_bundles(accepted), target_repo)
        for bundle in accepted:
            results[bundle.quant]["publication"] = publication
            results[bundle.quant]["published"] = True
            results[bundle.quant].pop("reason", None)
    return {"backend": backend.id, "source": repo_id, "target": target_repo,
            "baseline_quant": baseline_quant, "baseline": baseline,
            "results": results}
