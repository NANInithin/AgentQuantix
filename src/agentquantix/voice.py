"""Voice backend registry, bundle contract, audio gates, and quality metrics.

TTS and ASR deliberately share data structures, not runtime semantics. TTS
produces WAV audio through a family-specific native runtime: llama.cpp's
``llama-tts`` or audio.cpp's ``audiocpp_cli``. ASR consumes 16-bit WAV audio
through the independently built ``whisper-cli`` and uses whisper.cpp's GGML
model format.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import wave

from . import archsupport, config


TTS = "tts"
ASR = "asr"
# Fallback for planning before llama.cpp has been cloned.  When a checkout is
# present, available_quants() asks its quantize.cpp table instead so forks and
# newly-added upstream types are discovered rather than hard-coded here.
TTS_QUANTS = tuple(config.DEFAULT_QUANTS)
# Types accepted by audio.cpp's generic GGUF converter.  These are independent
# of the smaller set of prebuilt precisions listed in a family's package spec.
AUDIOCPP_QUANTS = (
    "ORIG", "F16", "BF16", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K",
    "Q6_K",
)
# The names printed and accepted by whisper.cpp/examples/common-ggml.cpp.
WHISPER_QUANTS = (
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k",
)
QWEN3_TTS_SOURCE_REPOS = ("Qwen/Qwen3-TTS-12Hz-1.7B-Base",)
AUDIOCPP_SPECS_DIR = (Path(__file__).parent / "fixtures"
                      / "audio_cpp_model_specs")


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
    repo_ids: tuple[str, ...] = ()
    example_repos: tuple[str, ...] = ()
    companions: tuple[CompanionSpec, ...] = ()
    languages: tuple[str, ...] = ()
    sample_rate: int = 0
    speaker_reference: str = "none"  # none, optional, required
    runtime_task: str = ""
    display_name: str = ""
    status: str = "supported"
    package_ids: tuple[str, ...] = ()
    match_aliases: tuple[str, ...] = ()
    milestone: str = ""
    enabled: bool = True

    def matches(self, repo_id: str) -> bool:
        value = (repo_id or "").casefold()
        return (any(value == candidate.casefold() for candidate in self.repo_ids)
                or any(value.startswith(prefix.casefold())
                       for prefix in self.repo_prefixes))

    def supports_language(self, language: str | None) -> bool:
        if not language or not self.languages:
            return True
        requested = language.casefold()
        aliases = {"en": "english", "fr": "french", "ja": "japanese",
                   "zh": "chinese", "es": "spanish"}
        for item in self.languages:
            value = item.casefold()
            if (value in ("auto", "multilingual") or "+ languages" in value
                    or value == requested or value.startswith(requested + "-")
                    or value == aliases.get(requested)):
                return True
        return False

    @property
    def catalog_repo(self) -> str:
        candidates = self.example_repos or self.repo_ids or self.repo_prefixes
        return candidates[0]

    @property
    def required_companions(self) -> tuple[str, ...]:
        return tuple(spec.role for spec in self.companions if spec.required)


CORE_BACKENDS = (
    VoiceBackend(
        id="llama-qwen3-tts",
        family="qwen3-tts",
        track=TTS,
        backend="llama.cpp",
        converter="convert_hf_to_gguf.py",
        runtime="llama-tts",
        model_format="gguf",
        # llama.cpp's converter and TTS documentation currently demonstrate
        # this exact Base checkpoint. Do not infer support from a family prefix.
        repo_prefixes=(),
        repo_ids=QWEN3_TTS_SOURCE_REPOS,
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
        repo_prefixes=(),
        repo_ids=("kyutai/pocket-tts",),
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
        example_repos=("openai/whisper-small",),
        supported_quants=WHISPER_QUANTS,
        languages=("multilingual",),
        sample_rate=16_000,
        milestone="v0.4.0",
    ),
)


def _audio_cpp_specs() -> tuple[dict, ...]:
    """Load the catalog shipped by the pinned audio.cpp revision.

    The files are copied unchanged from audio.cpp's ``model_specs`` directory.
    They are data, not an AgentQuantix allowlist: upgrading the pinned backend
    and syncing that directory is enough to expose newly declared families.
    """
    specs = []
    for path in sorted(AUDIOCPP_SPECS_DIR.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("family"):
            specs.append(value)
    return tuple(specs)


AUDIOCPP_SPECS = _audio_cpp_specs()
AUDIOCPP_SPEC_REGISTRY = {
    str(spec["family"]): spec for spec in AUDIOCPP_SPECS
}


def _audio_cpp_quant(value: str) -> str:
    return str(value).upper()


def _audio_cpp_source_repos(spec: dict) -> tuple[str, ...]:
    defaults = spec.get("package_defaults", {}).get("download", {})
    repos = []
    for package in spec.get("packages", []):
        if package.get("format") != "safetensors":
            continue
        download = {**defaults, **package.get("download", {})}
        if repo := download.get("repo"):
            repos.append(str(repo))
    return tuple(dict.fromkeys(repos))


def _audio_cpp_package_repos(spec: dict) -> tuple[str, ...]:
    defaults = spec.get("package_defaults", {}).get("download", {})
    repos = []
    for package in spec.get("packages", []):
        if package.get("format") != "gguf":
            continue
        download = {**defaults, **package.get("download", {})}
        if repo := download.get("repo"):
            repos.append(str(repo))
    return tuple(dict.fromkeys(repos))


def _audio_cpp_package_quants(spec: dict) -> tuple[str, ...]:
    values = [_audio_cpp_quant(package.get("precision", ""))
              for package in spec.get("packages", [])
              if package.get("format") == "gguf"
              and package.get("precision")]
    return tuple(dict.fromkeys(values))


def _audio_cpp_has_conversion(spec: dict) -> bool:
    return any(source.get("format") == "safetensors"
               for source in spec.get("sources", []))


def _audio_cpp_quants(spec: dict) -> tuple[str, ...]:
    """Every precision this family can consume or create.

    Package precision is not converter capability.  Some packages contain an
    F32 or Q4_0 artifact even though the generic converter does not emit that
    type; conversely a source-backed family can be converted to every type the
    converter exposes even when only Q8_0 has been uploaded by audio.cpp.
    """
    converter = AUDIOCPP_QUANTS if _audio_cpp_has_conversion(spec) else ()
    return tuple(dict.fromkeys((*converter, *_audio_cpp_package_quants(spec))))


def _audio_cpp_aliases(spec: dict) -> tuple[str, ...]:
    aliases = [str(spec.get("family", "")), str(spec.get("display_name", ""))]
    for package in spec.get("packages", []):
        aliases.extend((str(package.get("id", "")),
                        str(package.get("target_directory", ""))))
    return tuple(value for value in dict.fromkeys(aliases) if value)


def _audio_cpp_backends() -> tuple[VoiceBackend, ...]:
    backends = []
    for spec in AUDIOCPP_SPECS:
        tasks = tuple(str(task) for task in spec.get("tasks", []))
        tracks = []
        if "tts" in tasks and spec.get("category") == "tts":
            tracks.append((TTS, "tts"))
        elif "clone" in tasks and spec.get("category") == "tts":
            tracks.append((TTS, "clon"))
        if "asr" in tasks:
            tracks.append((ASR, "asr"))
        for track, runtime_task in tracks:
            clone = "speaker_reference" in spec.get(
                "capabilities", {}).get("clone", [])
            speaker = ("required" if runtime_task == "clon" else
                       "optional" if clone else "none")
            family = str(spec["family"])
            backends.append(VoiceBackend(
                id=f"audiocpp-{family}-{track}",
                family=family,
                track=track,
                backend="audio.cpp",
                converter="audiocpp_gguf",
                runtime="audiocpp_cli",
                model_format="audiocpp-gguf",
                repo_prefixes=(),
                repo_ids=_audio_cpp_source_repos(spec),
                example_repos=(f"audio.cpp:{family}",),
                supported_quants=_audio_cpp_quants(spec),
                languages=tuple(str(item) for item in spec.get("languages", [])),
                speaker_reference=speaker,
                runtime_task=runtime_task,
                display_name=str(spec.get("display_name", family)),
                status=str(spec.get("status", "unknown")),
                package_ids=tuple(str(item.get("id"))
                                  for item in spec.get("packages", [])
                                  if item.get("id")),
                match_aliases=_audio_cpp_aliases(spec),
                milestone="audio.cpp catalog",
                enabled=True,
            ))
    return tuple(backends)


AUDIOCPP_BACKENDS = _audio_cpp_backends()
BACKENDS = (*CORE_BACKENDS, *AUDIOCPP_BACKENDS)
BACKEND_REGISTRY = {backend.id: backend for backend in BACKENDS}
AUDIOCPP_PACKAGE_REPOS = {
    family: _audio_cpp_package_repos(spec)
    for family, spec in AUDIOCPP_SPEC_REGISTRY.items()
}
TTS_REVIEW_QUANTS = tuple(dict.fromkeys(
    (*TTS_QUANTS, *(quant for backend in AUDIOCPP_BACKENDS
                    if backend.track == TTS
                    for quant in backend.supported_quants))))


_GENERIC_MODEL_TOKENS = {
    "audio", "voice", "speech", "model", "base", "small", "large", "hf",
    "gguf", "tts", "asr", "stt", "v1", "v2", "v3",
}

_VOICE_NAME_HINTS = (
    "tts", "asr", "speech", "voice", "whisper", "audio", "parakeet",
    "voxcpm", "talker", "codec",
)


def looks_like_voice_model(repo_id: str) -> bool:
    """Conservative routing hint for models not yet in an installed catalog."""
    name = repo_id.split("/")[-1].casefold()
    tokens = set(re.findall(r"[a-z]+", name))
    return any(hint in tokens or hint in name for hint in _VOICE_NAME_HINTS)


def _name_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z]+|\d+", value.casefold())
            if token not in _GENERIC_MODEL_TOKENS and len(token) > 1}


def _compact_name(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _audio_cpp_backend_for(repo_id: str, track: str | None = None,
                           family: str | None = None) -> VoiceBackend | None:
    candidates = [backend for backend in AUDIOCPP_BACKENDS
                  if track is None or backend.track == track]
    if family:
        exact = [backend for backend in candidates
                 if backend.family.casefold() == family.casefold()]
        return exact[0] if len(exact) == 1 else None

    value = (repo_id or "").casefold()
    exact = [backend for backend in candidates
             if backend.matches(repo_id)
             or value == f"audio.cpp:{backend.family}".casefold()
             or any(value == package.casefold()
                    for package in backend.package_ids)]
    if len(exact) == 1:
        return exact[0]

    package_repo_matches = [
        backend for backend in candidates
        if any(value == repo.casefold()
               for repo in AUDIOCPP_PACKAGE_REPOS.get(backend.family, ()))
    ]
    if len(package_repo_matches) == 1:
        return package_repo_matches[0]

    # A third-party GGUF repository is not a convertible source checkpoint,
    # and GGUF schemas are runtime-specific. Only catalog package ids/repos
    # may select an already-converted audio.cpp artifact.
    if "gguf" in repo_id.split("/")[-1].casefold():
        return None

    compact = _compact_name(repo_id)
    tokens = _name_tokens(repo_id)
    scored = []
    for backend in candidates:
        score = 0
        family_name = _compact_name(backend.family)
        display_name = _compact_name(backend.display_name)
        if len(family_name) >= 5 and (family_name in compact or compact in family_name):
            score += 100
        if len(display_name) >= 5 and (display_name in compact or compact in display_name):
            score += 80
        overlap = tokens & _name_tokens(
            " ".join((backend.family, backend.display_name)))
        score += 12 * len(overlap)
        if backend.track == TTS and "tts" in value:
            score += 2
        if backend.track == ASR and any(word in value for word in ("asr", "stt")):
            score += 2
        if score:
            scored.append((score, backend))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] < 12:
        return None
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def backend_for(repo_id: str, track: str | None = None,
                family: str | None = None) -> VoiceBackend | None:
    if family:
        return _audio_cpp_backend_for(repo_id, track=track, family=family)
    core = next((backend for backend in CORE_BACKENDS
                 if (track is None or backend.track == track)
                 and backend.matches(repo_id)), None)
    return core or _audio_cpp_backend_for(repo_id, track=track)


def backend_named(name: str) -> VoiceBackend:
    try:
        return BACKEND_REGISTRY[name]
    except KeyError as error:
        raise VoiceValidationError(f"unknown voice backend: {name}") from error


def available_quants(repo_id: str, backend: VoiceBackend,
                     llama_dir: Path | None = None) -> tuple[str, ...]:
    """Quant types actually obtainable from this particular source.

    A normal HF safetensors source can use the converter's full type table. A
    virtual family, exact package id, or package repository can only install
    artifacts that audio.cpp has published. llama.cpp is queried from source
    so a fork's quant table is authoritative.
    """
    if backend.backend == "llama.cpp":
        discovered = archsupport.supported_quants(llama_dir)
        if discovered:
            preferred = [quant for quant in config.DEFAULT_QUANTS
                         if quant in discovered]
            preferred += sorted(discovered - set(preferred))
            return tuple(quant for quant in preferred
                         if quant not in {"COPY", "F16", "F32", "BF16"})
        return TTS_QUANTS
    if backend.backend == "whisper.cpp":
        return WHISPER_QUANTS
    if backend.backend == "audio.cpp":
        spec = AUDIOCPP_SPEC_REGISTRY.get(backend.family, {})
        package = next((item for item in spec.get("packages", [])
                        if str(item.get("id", "")).casefold() ==
                        repo_id.casefold()), None)
        if package is not None and package.get("precision"):
            return (_audio_cpp_quant(package["precision"]),)
        package_repos = AUDIOCPP_PACKAGE_REPOS.get(backend.family, ())
        is_package_reference = (
            repo_id.casefold() == f"audio.cpp:{backend.family}".casefold()
            or any(repo_id.casefold() == value.casefold()
                   for value in package_repos))
        if is_package_reference:
            return _audio_cpp_package_quants(spec)
        if _audio_cpp_has_conversion(spec):
            return AUDIOCPP_QUANTS
        return _audio_cpp_package_quants(spec)
    return backend.supported_quants


def default_quants(repo_id: str, backend: VoiceBackend) -> tuple[str, ...]:
    """Compatibility name for the complete source-specific quant sweep."""
    return available_quants(repo_id, backend)


def normalize_quant(backend: VoiceBackend, quant: str) -> str:
    """Use the spelling expected by the selected native converter."""
    return quant.casefold() if backend.backend == "whisper.cpp" else quant.upper()


def execution_gate(repo_id: str, track: str | None = None,
                   family: str | None = None):
    backend = backend_for(repo_id, track=track, family=family)
    if backend is None:
        if (repo_id or "").casefold().startswith("qwen/qwen3-tts"):
            supported = ", ".join(QWEN3_TTS_SOURCE_REPOS)
            return False, (f"{repo_id} is not a supported Qwen3-TTS source. "
                           f"Use the exact documented checkpoint: {supported}."), None
        return False, (f"{repo_id} does not resolve to a model family declared "
                       "by an installed voice backend. For an audio.cpp model "
                       "whose repository name is ambiguous, pass its catalog "
                       "family explicitly."), None
    if not backend.enabled:
        return False, f"{backend.family} is registered but disabled.", backend
    return True, (f"{backend.family} is supported through {backend.backend} "
                  f"for {backend.milestone}."), backend


def advisory_catalog() -> dict:
    return {
        "tracks": [TTS, ASR],
        "audio_cpp_catalog": {
            "spec_families": len(AUDIOCPP_SPECS),
            "release_routes": len(AUDIOCPP_BACKENDS),
            "revision_source": "bundled model_specs from pinned audio.cpp",
            "categories": sorted({str(spec.get("category", "unknown"))
                                  for spec in AUDIOCPP_SPECS}),
        },
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
            "status": backend.status if backend.enabled else "disabled",
            "agent_run_available": backend.enabled,
            "repo_id": backend.catalog_repo,
            "supported_repos": list(backend.repo_ids),
            "package_ids": list(backend.package_ids),
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
    runtime_model: Path | None = None
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


def audiocpp_tts_command(runtime: Path | str, bundle: VoiceBundle,
                         prompt: str, output: Path | str,
                         language: str | None = None,
                         speaker: Path | str | None = None) -> list[str]:
    """Build an audio.cpp CLI command for a standalone GGUF bundle."""
    if bundle.backend.track != TTS or bundle.backend.backend != "audio.cpp":
        raise VoiceValidationError(
            "audiocpp_cli can only validate an audio.cpp TTS bundle")
    if bundle.backend.speaker_reference == "required" and not speaker:
        raise VoiceValidationError(
            f"{bundle.backend.family} requires a speaker reference")
    command = [str(runtime), "--task", bundle.backend.runtime_task or "tts", "--family",
               bundle.backend.family, "--model",
               str(bundle.runtime_model or bundle.primary.path),
               "--backend", "best", "--text", prompt,
               "--out", str(output), "--metrics"]
    if language:
        command += ["--language", language]
    if speaker:
        command += ["--voice-ref", str(speaker)]
    return command


def audiocpp_asr_command(runtime: Path | str, backend: VoiceBackend,
                         model: Path | str, audio: Path | str,
                         output: Path | str,
                         language: str | None = None) -> list[str]:
    if backend.track != ASR or backend.backend != "audio.cpp":
        raise VoiceValidationError(
            "audiocpp_cli can only validate an audio.cpp ASR bundle")
    command = [str(runtime), "--task", backend.runtime_task or "asr",
               "--family", backend.family, "--model", str(model),
               "--backend", "best", "--audio", str(audio),
               "--text-out", str(output), "--metrics"]
    if language:
        command += ["--language", language]
    return command


def tts_command(runtime: Path | str, bundle: VoiceBundle, prompt: str,
                output: Path | str, language: str | None = None,
                speaker: Path | str | None = None) -> list[str]:
    """Dispatch TTS validation to the bundle's declared native runtime."""
    if bundle.backend.backend == "llama.cpp":
        return llama_tts_command(runtime, bundle, prompt, output, language, speaker)
    if bundle.backend.backend == "audio.cpp":
        return audiocpp_tts_command(
            runtime, bundle, prompt, output, language, speaker)
    raise VoiceValidationError(
        f"no TTS command builder for backend {bundle.backend.backend}")


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
    if facts["seconds"] < min_seconds:
        raise VoiceValidationError(
            f"TTS output is too short: {facts['seconds']} seconds; "
            f"minimum {round(min_seconds, 4)}")
    if facts["seconds"] > max_seconds:
        raise VoiceValidationError(
            f"TTS output is too long: {facts['seconds']} seconds; "
            f"maximum {round(max_seconds, 4)}")
    if facts["silence_ratio"] > max_silence or facts["rms"] < 0.0001:
        raise VoiceValidationError("TTS output is silent or almost entirely silent")
    if facts["clipping_ratio"] > max_clipping:
        raise VoiceValidationError("TTS output is clipped")
    return facts


def resample_pcm16_wav(source: Path | str, destination: Path | str,
                       target_rate: int = 16_000) -> Path:
    """Write a mono 16-bit PCM WAV suitable for whisper.cpp.

    TTS backends emit 24 kHz audio, while whisper.cpp's command-line examples
    and our ASR contract use 16 kHz PCM. A small linear resampler keeps this
    mandatory quality path dependency-free and deterministic in CI.
    """
    source, destination = Path(source), Path(destination)
    try:
        with wave.open(str(source), "rb") as input_wav:
            if input_wav.getcomptype() != "NONE" or input_wav.getsampwidth() != 2:
                raise VoiceValidationError(
                    "ASR round-trip resampling requires 16-bit PCM WAV input")
            channels = input_wav.getnchannels()
            source_rate = input_wav.getframerate()
            frames = input_wav.getnframes()
            samples = array("h")
            samples.frombytes(input_wav.readframes(frames))
    except (wave.Error, EOFError) as error:
        raise VoiceValidationError(f"cannot resample invalid WAV {source}: {error}") \
            from error
    if sys.byteorder != "little":
        samples.byteswap()
    if channels < 1 or source_rate < 1 or not samples:
        raise VoiceValidationError(f"cannot resample empty WAV: {source}")
    mono = ([int(sum(samples[offset:offset + channels]) / channels)
             for offset in range(0, len(samples), channels)]
            if channels > 1 else list(samples))
    output_count = max(1, round(len(mono) * target_rate / source_rate))
    if output_count == 1 or len(mono) == 1:
        converted = array("h", [mono[0]])
    else:
        scale = (len(mono) - 1) / (output_count - 1)
        converted = array("h")
        for index in range(output_count):
            position = index * scale
            left = int(position)
            right = min(left + 1, len(mono) - 1)
            fraction = position - left
            converted.append(round(mono[left] * (1 - fraction)
                                   + mono[right] * fraction))
    if sys.byteorder != "little":
        converted.byteswap()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as output_wav:
        output_wav.setnchannels(1)
        output_wav.setsampwidth(2)
        output_wav.setframerate(target_rate)
        output_wav.writeframes(converted.tobytes())
    return destination


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
        tts_command(runtime, bundle, prompt, output, language, speaker),
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


_CJK_RANGES = (
    (0x3040, 0x30FF),   # Hiragana and Katakana
    (0x31F0, 0x31FF),   # Katakana phonetic extensions
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),   # Half-width Katakana
    (0x20000, 0x2FA1F), # CJK extensions and compatibility supplement
)


def _is_cjk_character(character: str) -> bool:
    value = ord(character)
    return any(start <= value <= end for start, end in _CJK_RANGES)


def normalize_transcript(text: str) -> list[str]:
    """Tokenize for multilingual ASR comparison.

    Whitespace languages use normal word tokens. Chinese and Japanese do not
    reliably place spaces between words, so their Han/kana characters are
    individual tokens. This makes the existing edit-distance gate behave as
    WER for whitespace languages and CER for CJK instead of scoring one
    character substitution as a 50-100% sentence error.
    """
    normalized = unicodedata.normalize("NFKC", text.casefold())
    tokens, word = [], []

    def flush_word():
        if word:
            tokens.append("".join(word))
            word.clear()

    for character in normalized:
        if _is_cjk_character(character):
            flush_word()
            tokens.append(character)
        elif character.isalnum() or character in ("'", "_"):
            word.append(character)
        else:
            flush_word()
    flush_word()
    return tokens


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


def run_audiocpp_asr_smoke(runtime: Path | str, backend: VoiceBackend,
                           model: Path | str, audio: Path | str,
                           expected: str, output: Path | str,
                           language: str | None = None,
                           timeout: int = 600) -> dict:
    audio_facts = inspect_wav(audio)
    if audio_facts["sample_width"] != 2 or audio_facts["sample_rate"] != 16_000:
        raise VoiceValidationError(
            "audio.cpp ASR fixtures must be 16 kHz 16-bit PCM WAV")
    output = Path(output)
    execution = run_checked(
        audiocpp_asr_command(runtime, backend, model, audio, output, language),
        timeout=timeout)
    transcript = (output.read_text(encoding="utf-8", errors="replace").strip()
                  if output.is_file() else "")
    if not transcript:
        match = re.search(r"(?m)^text_output=(.*)$", execution["stdout"])
        transcript = match.group(1).strip() if match else ""
    if not transcript:
        raise VoiceValidationError("audiocpp_cli produced no transcript")
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
    package_members = any(member.role == "package-member"
                          for member in bundle.companions)
    runtime_artifact = bundle.primary.remote_path
    if package_members:
        parent = Path(runtime_artifact).parent.as_posix()
        runtime_artifact = parent if parent != "." else runtime_artifact
    if bundle.backend.track == TTS and bundle.backend.backend == "audio.cpp":
        command = (f"audiocpp_cli --task {bundle.backend.runtime_task or 'tts'} "
                   f"--family {bundle.backend.family} "
                   f"--model {runtime_artifact} --backend best "
                   "--text \"Hello world\""
                   + (" --voice-ref reference.wav"
                      if bundle.backend.speaker_reference == "required" else "")
                   + " --out out.wav")
    elif bundle.backend.track == TTS:
        command = (f"llama-tts -m {bundle.primary.remote_path}"
                   + (f" -mm {next((m.remote_path for m in bundle.companions if m.role == 'mmproj'), '')}"
                      if any(m.role == "mmproj" for m in bundle.companions) else "")
                   + " -p \"Hello world\" --output out.wav")
    elif bundle.backend.backend == "audio.cpp":
        command = (f"audiocpp_cli --task asr --family {bundle.backend.family} "
                   f"--model {runtime_artifact} --backend best "
                   "--audio input.wav --text-out transcript.txt")
    else:
        command = f"whisper-cli -m {bundle.primary.remote_path} -f input.wav"
    rows = "\n".join(
        f"| `{item['path']}` | {item['role']} | {item['quant'] or '-'} | "
        f"{item['bytes'] / 1024 ** 2:.1f} MiB |"
        for item in manifest["members"])
    if isinstance(quality.get("quants"), dict):
        quality_rows = "\n".join(
            f"| {quant} round-trip WER/CER | {result.get('roundtrip_wer', '-')} |\n"
            f"| {quant} real-time factor | {result.get('real_time_factor', '-')} |\n"
            f"| {quant} automated gate | {result.get('passed', '-')} |"
            for quant, result in quality["quants"].items())
    else:
        quality_rows = "\n".join(
            f"| {'roundtrip WER/CER' if key == 'roundtrip_wer' else key.replace('_', ' ')} | {value} |"
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

Expected sample rate: {f'{bundle.backend.sample_rate} Hz' if bundle.backend.sample_rate else 'runtime-reported; see quality.json'}

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
the model artifacts. {bundle.backend.backend} retains its own license.
"""
