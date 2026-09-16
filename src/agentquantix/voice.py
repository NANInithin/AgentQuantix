"""Voice backend registry, bundle contract, audio gates, and quality metrics.

TTS and ASR deliberately share data structures, not runtime semantics.  TTS
produces WAV audio through llama.cpp's ``llama-tts``.  ASR consumes 16-bit WAV
audio through the independently built ``whisper-cli`` and uses whisper.cpp's
GGML model format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time
import wave


TTS = "tts"
ASR = "asr"
TTS_QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
WHISPER_QUANTS = ("q8_0", "q5_0", "q4_0")


class VoiceValidationError(RuntimeError):
    """A voice artifact or runtime result is unsafe to publish."""


@dataclass(frozen=True)
class CompanionSpec:
    role: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class VoiceBackend:
    """One executable capability, including its artifact contract."""

    id: str
    family: str
    track: str
    backend: str
    converter: str
    runtime: str
    model_format: str
    repo_prefixes: tuple[str, ...]
    supported_quants: tuple[str, ...]
    companions: tuple[CompanionSpec, ...] = ()
    languages: tuple[str, ...] = ()
    sample_rate: int = 0
    speaker_reference: str = "none"  # none, optional, required
    milestone: str = ""
    enabled: bool = True

    def matches(self, repo_id: str) -> bool:
        value = (repo_id or "").casefold()
        return any(value.startswith(prefix.casefold())
                   for prefix in self.repo_prefixes)

    @property
    def required_companions(self) -> tuple[str, ...]:
        return tuple(spec.role for spec in self.companions if spec.required)


BACKENDS = (
    VoiceBackend(
        id="llama-qwen3-tts",
        family="qwen3-tts",
        track=TTS,
        backend="llama.cpp",
        converter="convert_hf_to_gguf.py",
        runtime="llama-tts",
        model_format="gguf",
        # Converter inputs only. Pre-converted ggml-org bundles have a
        # different import contract and are intentionally not matched here.
        repo_prefixes=("Qwen/Qwen3-TTS-",),
        supported_quants=TTS_QUANTS,
        companions=(CompanionSpec(
            "mmproj", True, "Audio tokenizer/projector GGUF used by libmtmd"),),
        languages=("zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"),
        sample_rate=24_000,
        speaker_reference="optional",
        milestone="v0.3.0",
    ),
    VoiceBackend(
        id="llama-pocket-tts",
        family="pocket-tts",
        track=TTS,
        backend="llama.cpp",
        converter="convert_hf_to_gguf.py",
        runtime="llama-tts",
        model_format="gguf",
        repo_prefixes=("kyutai/pocket-tts",),
        supported_quants=TTS_QUANTS,
        companions=(CompanionSpec(
            "mmproj", True, "Pocket TTS codec/projector GGUF"),),
        languages=("en",),
        sample_rate=24_000,
        speaker_reference="required",
        milestone="v0.3.1",
    ),
    VoiceBackend(
        id="whisper-cpp",
        family="whisper",
        track=ASR,
        backend="whisper.cpp",
        converter="models/convert-h5-to-ggml.py",
        runtime="whisper-cli",
        model_format="whisper-ggml-bin",
        repo_prefixes=("openai/whisper-",),
        supported_quants=WHISPER_QUANTS,
        languages=("multilingual",),
        sample_rate=16_000,
        milestone="v0.4.0",
    ),
)
BACKEND_REGISTRY = {backend.id: backend for backend in BACKENDS}


def backend_for(repo_id: str, track: str | None = None) -> VoiceBackend | None:
    return next((backend for backend in BACKENDS
                 if (track is None or backend.track == track)
                 and backend.matches(repo_id)), None)


def backend_named(name: str) -> VoiceBackend:
    try:
        return BACKEND_REGISTRY[name]
    except KeyError as error:
        raise VoiceValidationError(f"unknown voice backend: {name}") from error


def execution_gate(repo_id: str, track: str | None = None):
    backend = backend_for(repo_id, track=track)
    if backend is None:
        return False, (f"{repo_id} has no registered voice backend; add a "
                       "converter, bundle contract, runtime, and quality gate."), None
    if not backend.enabled:
        return False, f"{backend.family} is registered but disabled.", backend
    return True, (f"{backend.family} is supported through {backend.backend} "
                  f"for {backend.milestone}."), backend


def advisory_catalog() -> dict:
    return {
        "tracks": [TTS, ASR],
        "candidates": [{
            "backend": backend.id,
            "family": backend.family,
            "track": backend.track,
            "runtime": backend.runtime,
            "converter": backend.converter,
            "format": backend.model_format,
            "supported_quants": list(backend.supported_quants),
            "required_companions": list(backend.required_companions),
            "speaker_reference": backend.speaker_reference,
            "milestone": backend.milestone,
            "status": "supported" if backend.enabled else "disabled",
            "agent_run_available": backend.enabled,
            "repo_id": backend.repo_prefixes[0],
        } for backend in BACKENDS],
    }


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BundleMember:
    path: Path
    role: str
    hub_path: str | None = None
    required: bool = True
    quant: str | None = None

    @property
    def remote_path(self) -> str:
        return (self.hub_path or self.path.name).replace("\\", "/")

    def manifest(self) -> dict:
        return {
            "path": self.remote_path,
            "local_name": self.path.name,
            "role": self.role,
            "required": self.required,
            "quant": self.quant,
            "bytes": self.path.stat().st_size,
            "sha256": sha256(self.path),
        }


@dataclass(frozen=True)
class VoiceBundle:
    backend: VoiceBackend
    primary: BundleMember
    companions: tuple[BundleMember, ...] = ()
    source_repo: str | None = None
    source_revision: str | None = None
    quant: str | None = None
    configuration: dict = field(default_factory=dict)
    quality: dict | None = None
    created_at: str = field(default_factory=lambda:
                            datetime.now(timezone.utc).isoformat())

    @property
    def members(self) -> tuple[BundleMember, ...]:
        return (self.primary, *self.companions)

    def problems(self) -> list[str]:
        problems, local_seen, remote_seen = [], set(), set()
        roles = {member.role for member in self.members if member.required}
        for role in self.backend.required_companions:
            if role not in roles:
                problems.append(f"required companion role is absent: {role}")
        for member in self.members:
            local_key = str(member.path.resolve()).casefold()
            remote_key = member.remote_path.casefold()
            if local_key in local_seen:
                problems.append(f"local member appears twice: {member.path}")
            if remote_key in remote_seen:
                problems.append(f"Hub path appears twice: {member.remote_path}")
            local_seen.add(local_key)
            remote_seen.add(remote_key)
            parts = Path(member.remote_path).parts
            if (not member.remote_path or member.remote_path.startswith("/")
                    or ".." in parts):
                problems.append(f"unsafe Hub path: {member.remote_path!r}")
            if member.required and not member.path.is_file():
                problems.append(f"required {member.role} is missing: {member.path}")
            elif member.path.exists() and member.path.stat().st_size == 0:
                problems.append(f"{member.role} is empty: {member.path}")
        if self.quant and self.quant not in self.backend.supported_quants:
            problems.append(
                f"{self.quant} is not supported by {self.backend.id}; choose "
                + ", ".join(self.backend.supported_quants))
        return problems

    def manifest(self) -> dict:
        if problems := self.problems():
            raise VoiceValidationError("voice bundle is incomplete: "
                                       + "; ".join(problems))
        return {
            "format": 2,
            "backend": self.backend.id,
            "family": self.backend.family,
            "track": self.backend.track,
            "runtime": self.backend.runtime,
            "model_format": self.backend.model_format,
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "quant": self.quant,
            "configuration": self.configuration,
            "quality": self.quality,
            "created_at": self.created_at,
            "members": [member.manifest() for member in self.members],
        }


def write_manifest(bundle: VoiceBundle, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(bundle.manifest(), indent=2, sort_keys=True)
                         + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_manifest(path: Path | str) -> dict:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoiceValidationError(f"could not read voice manifest {path}: {error}") \
            from error
    if (not isinstance(data, dict) or data.get("format") != 2
            or data.get("backend") not in BACKEND_REGISTRY
            or not isinstance(data.get("members"), list) or not data["members"]):
        raise VoiceValidationError(f"invalid voice manifest: {path}")
    return data


def verify_manifest_files(manifest: dict, root: Path | str) -> list[str]:
    root = Path(root)
    problems = []
    for member in manifest.get("members", []):
        path = root / member["local_name"]
        if not path.is_file():
            problems.append(f"missing {member['role']}: {path}")
            continue
        if path.stat().st_size != member["bytes"]:
            problems.append(f"size mismatch for {path.name}")
        elif sha256(path) != member["sha256"]:
            problems.append(f"checksum mismatch for {path.name}")
    return problems


def remote_bundle_problems(manifest: dict, remote: dict[str, dict]) -> list[str]:
    """Compare a manifest with ``{hub_path: {bytes, sha256}}`` metadata."""
    problems = []
    for member in manifest["members"]:
        actual = remote.get(member["path"])
        if actual is None:
            problems.append(f"remote member missing: {member['path']}")
            continue
        if actual.get("bytes") not in (None, member["bytes"]):
            problems.append(f"remote size mismatch: {member['path']}")
        digest = actual.get("sha256")
        if digest and digest != member["sha256"]:
            problems.append(f"remote checksum mismatch: {member['path']}")
    return problems


def llama_tts_command(runtime: Path | str, bundle: VoiceBundle,
                      prompt: str, output: Path | str, language: str | None = None,
                      speaker: Path | str | None = None) -> list[str]:
    if bundle.backend.track != TTS:
        raise VoiceValidationError("llama-tts can only validate a TTS bundle")
    if bundle.backend.speaker_reference == "required" and not speaker:
        raise VoiceValidationError(
            f"{bundle.backend.family} requires a speaker reference")
    command = [str(runtime), "-m", str(bundle.primary.path)]
    if mmproj := next((m for m in bundle.companions if m.role == "mmproj"), None):
        command += ["-mm", str(mmproj.path)]
    command += ["-p", prompt, "--output", str(output)]
    if language and bundle.backend.family == "qwen3-tts":
        if language not in bundle.backend.languages:
            raise VoiceValidationError(f"unsupported Qwen3-TTS language: {language}")
        command += ["--tts-lang", language]
    if speaker:
        command += ["--tts-speaker-file", str(speaker)]
    return command


def whisper_command(runtime: Path | str, model: Path | str, audio: Path | str,
                    output_prefix: Path | str, language: str | None = None,
                    vad: bool = False) -> list[str]:
    command = [str(runtime), "-m", str(model), "-f", str(audio),
               "--output-txt", "--output-file", str(output_prefix),
               "--no-timestamps"]
    if language:
        command += ["--language", language]
    if vad:
        command.append("--vad")
    return command


def _decode_samples(raw: bytes, width: int):
    if width == 1:
        return ((byte - 128) / 128 for byte in raw)
    step = width
    maximum = float(1 << (8 * width - 1))

    def values():
        for offset in range(0, len(raw) - step + 1, step):
            chunk = raw[offset:offset + step]
            value = int.from_bytes(chunk, "little", signed=True)
            yield value / maximum
    return values()


def inspect_wav(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise VoiceValidationError(f"runtime did not create WAV output: {path}")
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getcomptype() != "NONE":
                raise VoiceValidationError("WAV output is not uncompressed PCM")
            channels, rate = audio.getnchannels(), audio.getframerate()
            width, frames = audio.getsampwidth(), audio.getnframes()
            raw = audio.readframes(frames)
    except (wave.Error, EOFError) as error:
        raise VoiceValidationError(f"invalid WAV output {path}: {error}") from error
    if channels < 1 or rate < 1 or width not in (1, 2, 3, 4) or frames < 1:
        raise VoiceValidationError(
            f"empty or unsupported WAV: {channels}ch {rate}Hz {width}B {frames} frames")
    samples = list(_decode_samples(raw, width))
    if not samples:
        raise VoiceValidationError("WAV contains no PCM samples")
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    silence = sum(abs(value) < 0.001 for value in samples) / len(samples)
    clipping = sum(abs(value) >= 0.999 for value in samples) / len(samples)
    return {
        "path": str(path), "channels": channels, "sample_rate": rate,
        "sample_width": width, "frames": frames,
        "seconds": round(frames / rate, 4), "rms": round(rms, 6),
        "silence_ratio": round(silence, 6),
        "clipping_ratio": round(clipping, 6),
    }


def validate_tts_audio(path: Path | str, *, min_seconds: float = 0.15,
                       max_seconds: float = 120.0,
                       max_silence: float = 0.98,
                       max_clipping: float = 0.02) -> dict:
    facts = inspect_wav(path)
    if not min_seconds <= facts["seconds"] <= max_seconds:
        raise VoiceValidationError(
            f"implausible TTS duration: {facts['seconds']} seconds")
    if facts["silence_ratio"] > max_silence or facts["rms"] < 0.0001:
        raise VoiceValidationError("TTS output is silent or almost entirely silent")
    if facts["clipping_ratio"] > max_clipping:
        raise VoiceValidationError("TTS output is clipped")
    return facts


def run_checked(command: list[str], *, output: Path | None = None,
                timeout: int = 600) -> dict:
    """Run a real inference binary and measure latency to its first output."""
    began = time.monotonic()
    first_output = None
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log:
        process = subprocess.Popen([str(part) for part in command], text=True,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if (first_output is None and output is not None and output.exists()
                        and output.stat().st_size > 44):
                    first_output = time.monotonic() - began
                if time.monotonic() - began > timeout:
                    process.kill()
                    process.wait(timeout=10)
                    raise VoiceValidationError(
                        f"runtime timed out after {timeout} seconds")
                time.sleep(0.02)
            log.seek(0)
            stdout = log.read()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    elapsed = time.monotonic() - began
    if process.returncode:
        tail = next((line for line in reversed(stdout.splitlines()) if line.strip()),
                    f"exit status {process.returncode}")
        raise VoiceValidationError(f"runtime inference failed: {tail}")
    return {"command": command, "elapsed_seconds": round(elapsed, 4),
            "first_output_seconds": round(first_output or elapsed, 4),
            "stdout": stdout}


def run_tts_smoke(runtime: Path | str, bundle: VoiceBundle, prompt: str,
                  output: Path | str, language: str | None = None,
                  speaker: Path | str | None = None, timeout: int = 600) -> dict:
    output = Path(output)
    if output.exists():
        raise VoiceValidationError(f"smoke output already exists: {output}")
    execution = run_checked(
        llama_tts_command(runtime, bundle, prompt, output, language, speaker),
        output=output, timeout=timeout)
    words = max(1, len(normalize_transcript(prompt)))
    audio = validate_tts_audio(
        output, min_seconds=max(0.15, words * 0.04),
        max_seconds=min(180.0, max(4.0, words * 1.5)))
    if (bundle.backend.sample_rate
            and audio["sample_rate"] != bundle.backend.sample_rate):
        raise VoiceValidationError(
            f"unexpected TTS sample rate: {audio['sample_rate']} Hz; "
            f"expected {bundle.backend.sample_rate} Hz")
    execution.update(audio)
    execution["real_time_factor"] = round(
        execution["elapsed_seconds"] / audio["seconds"], 4)
    execution["minutes_per_audio_hour"] = round(
        execution["real_time_factor"] * 60, 2)
    execution["frames_per_second"] = round(
        audio["frames"] / execution["elapsed_seconds"], 2)
    return execution


_WORD = re.compile(r"[^\w']+", re.UNICODE)


def normalize_transcript(text: str) -> list[str]:
    return [word for word in _WORD.sub(" ", text.casefold()).split() if word]


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected, actual = normalize_transcript(reference), normalize_transcript(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, word in enumerate(expected, 1):
        current = [row]
        for column, candidate in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (word != candidate)))
        previous = current
    return previous[-1] / len(expected)


def read_whisper_transcript(prefix: Path | str, stdout: str = "") -> str:
    text_path = Path(str(prefix) + ".txt")
    text = text_path.read_text(encoding="utf-8", errors="replace") \
        if text_path.is_file() else stdout
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith(("whisper_", "system_info:"))]
    return " ".join(lines)


def run_asr_smoke(runtime: Path | str, model: Path | str, audio: Path | str,
                  expected: str, output_prefix: Path | str,
                  language: str | None = None, vad: bool = False,
                  timeout: int = 600) -> dict:
    audio_facts = inspect_wav(audio)
    if audio_facts["sample_width"] != 2 or audio_facts["sample_rate"] != 16_000:
        raise VoiceValidationError("whisper-cli fixtures must be 16 kHz 16-bit PCM WAV")
    execution = run_checked(
        whisper_command(runtime, model, audio, output_prefix, language, vad),
        timeout=timeout)
    transcript = read_whisper_transcript(output_prefix, execution["stdout"])
    if not transcript:
        raise VoiceValidationError("whisper-cli produced no transcript")
    execution.update({
        "transcript": transcript,
        "wer": round(word_error_rate(expected, transcript), 6),
        "audio_seconds": audio_facts["seconds"],
        "real_time_factor": round(
            execution["elapsed_seconds"] / audio_facts["seconds"], 4),
        "audio_seconds_per_second": round(
            audio_facts["seconds"] / execution["elapsed_seconds"], 4),
    })
    return execution


def asr_regression(candidate_wer: float, baseline_wer: float,
                   max_regression: float = 0.05) -> dict:
    regression = candidate_wer - baseline_wer
    return {"candidate_wer": candidate_wer, "baseline_wer": baseline_wer,
            "regression": round(regression, 6),
            "max_regression": max_regression,
            "passed": regression <= max_regression}


def quality_passed(result: dict, max_roundtrip_wer: float = 0.35) -> bool:
    return (result.get("silence_ratio", 1) <= 0.98
            and result.get("clipping_ratio", 1) <= 0.02
            and result.get("roundtrip_wer", 1) <= max_roundtrip_wer)


def write_human_review(path: Path | str, *, quant: str, accepted: bool,
                       reviewer: str, notes: str = "") -> Path:
    if not reviewer.strip():
        raise VoiceValidationError("human listening review needs a reviewer")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "quant": quant, "accepted": bool(accepted), "reviewer": reviewer,
        "notes": notes, "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    return path


def render_model_card(bundle: VoiceBundle, target_repo: str,
                      license_name: str | None = None) -> str:
    manifest = bundle.manifest()
    quality = bundle.quality or {}
    command = (f"llama-tts -m {bundle.primary.remote_path}"
               + (f" -mm {next((m.remote_path for m in bundle.companions if m.role == 'mmproj'), '')}"
                  if any(m.role == "mmproj" for m in bundle.companions) else "")
               + " -p \"Hello world\" --output out.wav"
               if bundle.backend.track == TTS else
               f"whisper-cli -m {bundle.primary.remote_path} -f input.wav")
    rows = "\n".join(
        f"| `{item['path']}` | {item['role']} | {item['quant'] or '-'} | "
        f"{item['bytes'] / 1024 ** 2:.1f} MiB |"
        for item in manifest["members"])
    if isinstance(quality.get("quants"), dict):
        quality_rows = "\n".join(
            f"| {quant} round-trip WER | {result.get('roundtrip_wer', '-')} |\n"
            f"| {quant} real-time factor | {result.get('real_time_factor', '-')} |\n"
            f"| {quant} automated gate | {result.get('passed', '-')} |"
            for quant, result in quality["quants"].items())
    else:
        quality_rows = "\n".join(
            f"| {key.replace('_', ' ')} | {value} |"
            for key, value in quality.items()
            if isinstance(value, (str, int, float, bool)))
    quality_rows = quality_rows or "| status | Not measured |"
    language = ", ".join(bundle.backend.languages) or "See upstream model"
    return f"""---
base_model: {bundle.source_repo or ''}
license: {license_name or 'other'}
library_name: {bundle.backend.backend}
tags:
- {bundle.backend.track}
- {bundle.backend.family}
- quantized
---

# {target_repo.split('/')[-1]}

This is an AgentQuantix {bundle.backend.track.upper()} bundle for
`{bundle.source_repo}` using `{bundle.backend.runtime}`. The bundle is the unit
of publication: download every required file listed below.

## Runtime

```bash
{command}
```

Supported languages: {language}

Expected sample rate: {bundle.backend.sample_rate} Hz

Speaker reference: {bundle.backend.speaker_reference}

## Bundle files

| File | Role | Quant | Size |
| --- | --- | --- | ---: |
{rows}

## Quality results

| Check | Result |
| --- | --- |
{quality_rows}

## Voice cloning and consent

Do not clone or imitate a person's voice without explicit, lawful consent. Do
not use generated audio for impersonation, fraud, deception, harassment, or
privacy-invasive activity. Review the upstream model license and acceptable-use
terms before use.

## License

The upstream model license ({license_name or 'see upstream model'}) applies to
the model artifacts. llama.cpp and whisper.cpp retain their respective licenses.
"""
