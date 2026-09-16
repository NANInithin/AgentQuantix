"""Pinned adapter for the audited ``predict-woo/qwen3-tts.cpp`` backend.

This is not llama.cpp: it has a dedicated C++ runtime and converter pair for
Qwen3-TTS. Keeping it in a separate adapter prevents its two-file bundle and
family-specific quantization semantics from being mistaken for the normal text
GGUF sweep.

Audit boundary (pinned 2026-09-16): the converter supports only
Qwen3-TTS-12Hz-0.6B-Base. It writes a quantized talker plus an F16 tokenizer/
vocoder. A run is rejected if either source component is missing.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys

from .. import config, voice
from . import build as build_mod


BACKEND_REPO = "https://github.com/predict-woo/qwen3-tts.cpp.git"
BACKEND_REVISION = "b3ba14077cf1b3e11b86e5f84aa9184605c89b28"
BACKEND_LICENSE = "MIT"
BACKEND_NAME = "predict-woo/qwen3-tts.cpp"
SUPPORTED_SOURCE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TALKER_ROLE = "talker"
TOKENIZER_ROLE = "audio-tokenizer-vocoder"
_CONVERSION_SUMMARY = re.compile(r"Converted\s+(\d+)\s+tensors,\s+skipped\s+(\d+)",
                                 re.IGNORECASE)


def backend_dir() -> Path:
    """The agent-owned, revision-addressed checkout location."""
    return config.TEMP_DIR / "aqx-voice-backends" / f"qwen3-tts-{BACKEND_REVISION[:12]}"


def backend_audit() -> dict:
    """Provenance and limitations persisted with every produced bundle."""
    return {
        "name": BACKEND_NAME,
        "repo": BACKEND_REPO.removesuffix(".git"),
        "revision": BACKEND_REVISION,
        "license": BACKEND_LICENSE,
        "runtime": "qwen3-tts-cli",
        "source": SUPPORTED_SOURCE,
        "outputs": ["talker GGUF", "tokenizer/vocoder GGUF"],
        "limitations": [
            "Only Qwen3-TTS-12Hz-0.6B-Base is admitted.",
            "The tokenizer/vocoder remains F16; it is not part of the talker quant sweep.",
            "The external runtime is required; stock llama.cpp is not the runtime contract.",
        ],
    }


def source_problems(source: Path | str) -> list[str]:
    """Check the exact raw-checkpoint shape required by the audited scripts."""
    source = Path(source)
    problems = []
    for name in ("config.json", "vocab.json"):
        if not (source / name).is_file():
            problems.append(f"missing {name} in {source}")
    if not list(source.glob("*.safetensors")):
        problems.append(f"no talker safetensors in {source}")

    tokenizer = source / "speech_tokenizer"
    if not tokenizer.is_dir():
        problems.append(f"missing speech_tokenizer directory in {source}")
    else:
        if not (tokenizer / "config.json").is_file():
            problems.append(f"missing speech_tokenizer/config.json in {source}")
        if not list(tokenizer.glob("*.safetensors")):
            problems.append(f"no tokenizer safetensors in {tokenizer}")
    return problems


def converter_missing() -> list[str]:
    """External converter modules absent from the AgentQuantix environment."""
    requirements = {
        "torch": "torch", "numpy": "numpy", "safetensors": "safetensors",
        "tqdm": "tqdm", "gguf": "gguf",
    }
    missing = []
    for package, module in requirements.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return missing


def converter_hint(missing: list[str]) -> str:
    return ("The audited Qwen3-TTS converter needs " + ", ".join(missing)
            + ". Install the dedicated extra with:\n"
              "  uv tool install --force --reinstall "
              '"agentquantix[voice-qwen] @ git+https://github.com/NANInithin/AgentQuantix"')


def _revision(directory: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True,
                              timeout=20).stdout.strip()
    except Exception:
        return None


def _runtime_binary(directory: Path) -> Path | None:
    for candidate in (directory / "build" / "qwen3-tts-cli.exe",
                      directory / "build" / "Release" / "qwen3-tts-cli.exe",
                      directory / "build" / "qwen3-tts-cli",
                      directory / "build" / "Release" / "qwen3-tts-cli"):
        if candidate.is_file():
            return candidate
    return None


def _build_backend(directory: Path) -> None:
    """Build the pinned backend and its vendored GGML submodule.

    The backend chooses CUDA only when a usable toolkit exists, matching the
    project-wide policy that CPU is slow but valid. It remains independent from
    the user's llama.cpp checkout because this runtime needs its own GGML ABI.
    """
    cuda = "ON" if build_mod.has_cuda_toolkit() else "OFF"
    ggml = directory / "ggml"
    build_mod.run(["cmake", "-S", ggml, "-B", ggml / "build",
                   f"-DGGML_CUDA={cuda}"])
    build_mod.run(["cmake", "--build", ggml / "build", "--config", "Release",
                   "-j", build_mod.build_jobs()])
    build_mod.run(["cmake", "-S", directory, "-B", directory / "build"])
    build_mod.run(["cmake", "--build", directory / "build", "--config", "Release",
                   "-j", build_mod.build_jobs()])


def ensure_backend() -> tuple[Path, Path]:
    """Clone, pin, and build the audited backend; return (checkout, runtime)."""
    directory = backend_dir()
    if not directory.exists():
        if not shutil.which("git"):
            raise RuntimeError("git is required to install the Qwen3-TTS backend")
        directory.parent.mkdir(parents=True, exist_ok=True)
        build_mod.run(["git", "clone", "--recurse-submodules", BACKEND_REPO, directory])
        build_mod.run(["git", "-C", directory, "checkout", "--detach", BACKEND_REVISION])
        build_mod.run(["git", "-C", directory, "submodule", "update", "--init", "--recursive"])
    actual = _revision(directory)
    if actual != BACKEND_REVISION:
        raise RuntimeError(
            f"Qwen3-TTS backend at {directory} is {actual or 'unreadable'}, not "
            f"the audited revision {BACKEND_REVISION}. Refusing to run it.")
    if runtime := _runtime_binary(directory):
        return directory, runtime
    _build_backend(directory)
    if runtime := _runtime_binary(directory):
        return directory, runtime
    raise RuntimeError("Qwen3-TTS backend build finished without qwen3-tts-cli")


def conversion_plan(repo_id: str, source: Path | str, output_dir: Path | str,
                    quant: str = "f16") -> dict:
    """Commands and bundle shape for the only audited raw Qwen3-TTS source."""
    if repo_id != SUPPORTED_SOURCE:
        raise voice.VoiceValidationError(
            f"{repo_id} is not supported by the audited backend; only "
            f"{SUPPORTED_SOURCE} is admitted.")
    if quant not in {"f16", "q8_0", "q4_k"}:
        raise voice.VoiceValidationError(
            "Qwen3-TTS supports only f16, q8_0, or q4_k talker conversion")
    source, output_dir = Path(source), Path(output_dir)
    if problems := source_problems(source):
        raise voice.VoiceValidationError("invalid Qwen3-TTS source: "
                                         + "; ".join(problems))
    backend = backend_dir()
    talker = output_dir / f"qwen3-tts-0.6b-{quant}.gguf"
    tokenizer = output_dir / "qwen3-tts-tokenizer-f16.gguf"
    return {
        "talker": talker,
        "tokenizer": tokenizer,
        "commands": [
            [sys.executable, backend / "scripts" / "convert_tts_to_gguf.py",
             "--input", source, "--output", talker, "--type", quant],
            [sys.executable, backend / "scripts" / "convert_tokenizer_to_gguf.py",
             "--input", source, "--output", tokenizer, "--type", "f16"],
        ],
    }


def _converter_summary(output: str) -> tuple[int, int]:
    """Read the external converter's mapping summary, failing closed on drift."""
    match = _CONVERSION_SUMMARY.search(output or "")
    if not match:
        raise voice.VoiceValidationError(
            "Qwen3-TTS converter did not print its tensor mapping summary; "
            "the pinned adapter may have changed, so refusing to publish.")
    converted, skipped = (int(value) for value in match.groups())
    if converted < 1 or skipped:
        raise voice.VoiceValidationError(
            f"Qwen3-TTS converter mapped {converted} tensor(s) and skipped "
            f"{skipped}; every source tensor must be accounted for.")
    return converted, skipped


def _run_converter(command: list) -> tuple[int, int]:
    """Run one converter and reject a failed or incomplete tensor mapping."""
    printable = " ".join(map(str, command))
    print(f"\n{printable}\n", flush=True)
    completed = subprocess.run([str(part) for part in command], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=build_mod.build_env())
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.returncode:
        tail = (completed.stdout or "").strip().splitlines()[-1:]
        detail = tail[0] if tail else f"exit status {completed.returncode}"
        raise voice.VoiceValidationError(f"Qwen3-TTS conversion failed: {detail}")
    return _converter_summary(completed.stdout or "")


def convert(repo_id: str, source: Path | str, output_dir: Path | str,
            quant: str = "f16") -> voice.VoiceBundle:
    """Run both converters and return a complete staged bundle.

    Uploading is deliberately absent here. The caller must build the external
    runtime and pass its fresh smoke output through ``voice.run_qwen3_tts_smoke``
    before persisting or publishing the returned bundle.
    """
    plan = conversion_plan(repo_id, source, output_dir, quant)
    if missing := converter_missing():
        raise RuntimeError(converter_hint(missing))
    _, runtime = ensure_backend()
    for command in plan["commands"]:
        _run_converter(command)
    bundle = voice.VoiceBundle(
        family=voice.family_for(repo_id),
        primary=voice.BundleMember(plan["talker"], TALKER_ROLE),
        companions=(voice.BundleMember(plan["tokenizer"], TOKENIZER_ROLE),),
        source_repo=repo_id,
        adapter=backend_audit(),
    )
    if problems := bundle.problems():
        raise voice.VoiceValidationError("Qwen3-TTS conversion did not produce "
                                         "a complete bundle: " + "; ".join(problems))
    # Including the runtime path on the object would make manifests machine-
    # specific. Its pinned revision is already represented by source_revision.
    assert runtime.exists()
    return bundle
