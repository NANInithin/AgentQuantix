"""Voice-model capabilities, bundle contracts, and runtime smoke validation.

This is deliberately separate from the text quantization pipeline. A voice
model can require audio codecs/projectors and a dedicated runtime; an unproven
family must not enter an unattended text sweep merely because it has a familiar
Hub tag. Qwen3-TTS is the only enabled family in the v0.3.0 foundation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import wave


TIER_1 = "tier-1"
TIER_2 = "tier-2"
TIER_3 = "tier-3"


@dataclass(frozen=True)
class VoiceFamily:
    """One family and the evidence needed to place it on the roadmap."""

    name: str
    tier: str
    release: str | None
    modality: str
    runtime: str | None
    repo_prefixes: tuple[str, ...]
    enabled: bool = False

    def matches(self, repo_id: str) -> bool:
        normalized = (repo_id or "").casefold()
        return any(normalized.startswith(prefix.casefold())
                   for prefix in self.repo_prefixes)


# A prefix is deliberately narrower than a Hub pipeline tag. Tags are useful
# discovery hints but are not a converter/runtime contract.
VOICE_FAMILIES = (
    VoiceFamily(
        name="qwen3-tts", tier=TIER_1, release="v0.3.0", modality="tts",
        runtime="qwen3-tts-cli",
        repo_prefixes=("Qwen/Qwen3-TTS-12Hz-0.6B-Base",), enabled=True,
    ),
    VoiceFamily(
        name="pocket-tts", tier=TIER_1, release="v0.3.1", modality="tts",
        runtime="llama-tts", repo_prefixes=("kyutai/Pocket-TTS",),
    ),
    VoiceFamily(
        name="voxtral", tier=TIER_2, release="v0.4.0", modality="audio-input",
        runtime="llama-mtmd-cli", repo_prefixes=("mistralai/Voxtral-",),
    ),
    VoiceFamily(
        name="qwen3-asr", tier=TIER_2, release="v0.4.0", modality="audio-input",
        runtime="llama-mtmd-cli", repo_prefixes=("Qwen/Qwen3-ASR-",),
    ),
)


def family_for(repo_id: str) -> VoiceFamily | None:
    """The known voice family for a Hub repo, or ``None`` when unknown."""
    return next((family for family in VOICE_FAMILIES if family.matches(repo_id)), None)


def execution_gate(repo_id: str) -> tuple[bool, str, VoiceFamily | None]:
    """Whether this model may enter a voice execution path today.

    Unknown models are deliberately denied: Tier 3 is a research state, not a
    fallback execution mode. The returned reason is safe to display in the CLI
    or agent surface.
    """
    family = family_for(repo_id)
    if family is None:
        return False, (f"{repo_id} is not an approved voice family. Add a "
                       "converter, bundle contract and runtime-quality gate "
                       "before enabling it."), None
    if family.enabled:
        return True, f"{family.name} is enabled for {family.release}.", family
    return False, (f"{family.name} is planned for {family.release} "
                   f"({family.tier}, {family.modality}); it is not runnable "
                   "in this release."), family


@dataclass(frozen=True)
class BundleMember:
    """A local artifact needed to run a voice model bundle."""

    path: Path
    role: str
    required: bool = True
    hub_path: str | None = None

    def manifest(self) -> dict:
        return {
            "path": self.path.name,
            "hub_path": self.hub_path or self.path.name,
            "role": self.role,
            "required": self.required,
            "bytes": self.path.stat().st_size,
            "sha256": sha256(self.path),
        }


@dataclass(frozen=True)
class VoiceBundle:
    """The model artifacts that must travel together for a runnable release."""

    family: VoiceFamily
    primary: BundleMember
    companions: tuple[BundleMember, ...] = ()
    source_repo: str | None = None
    source_revision: str | None = None
    adapter: dict | None = None

    @property
    def members(self) -> tuple[BundleMember, ...]:
        return (self.primary, *self.companions)

    def problems(self) -> list[str]:
        """Local bundle defects, before any remote state can be changed."""
        problems = []
        seen, remote_seen = set(), set()
        for member in self.members:
            key = member.path.name.casefold()
            if key in seen:
                problems.append(f"bundle names {member.path.name} more than once")
            seen.add(key)
            hub_path = (member.hub_path or member.path.name).replace("\\", "/")
            if not hub_path or hub_path.startswith("/") or ".." in hub_path.split("/"):
                problems.append(f"unsafe Hub path for {member.role}: {hub_path!r}")
            if hub_path.casefold() in remote_seen:
                problems.append(f"bundle maps more than one member to {hub_path}")
            remote_seen.add(hub_path.casefold())
            if member.required and not member.path.is_file():
                problems.append(f"required {member.role} is missing: {member.path}")
            elif member.path.exists() and member.path.stat().st_size == 0:
                problems.append(f"{member.role} is empty: {member.path}")
        return problems

    def manifest(self) -> dict:
        """Stable metadata for persistence/upload after :meth:`problems` is empty."""
        if not isinstance(self.family, VoiceFamily):
            raise VoiceValidationError("voice bundle has no approved voice family")
        problems = self.problems()
        if problems:
            raise VoiceValidationError("voice bundle is incomplete: "
                                       + "; ".join(problems))
        return {
            "format": 1,
            "family": self.family.name,
            "tier": self.family.tier,
            "modality": self.family.modality,
            "runtime": self.family.runtime,
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "adapter": self.adapter,
            "members": [member.manifest() for member in self.members],
        }


class VoiceValidationError(RuntimeError):
    """A voice bundle or runtime result is not safe to publish."""


def sha256(path: Path | str) -> str:
    """The exact bytes identity used by resumable voice-bundle publication."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(bundle: VoiceBundle, path: Path | str) -> Path:
    """Persist a complete bundle manifest for later resumable publication."""
    path = Path(path)
    data = bundle.manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return path


def load_manifest(path: Path | str) -> dict:
    """Load a persisted bundle manifest, rejecting corrupt or wrong-shape data."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoiceValidationError(f"could not read voice bundle manifest {path}: {error}") \
            from error
    if (not isinstance(data, dict) or data.get("format") != 1
            or not isinstance(data.get("members"), list) or not data["members"]):
        raise VoiceValidationError(f"invalid voice bundle manifest: {path}")
    return data


def qwen3_tts_smoke_command(qwen_tts: Path | str, model_dir: Path | str,
                            output: Path | str,
                            prompt: str = "Hello from AgentQuantix.") -> list[str]:
    """The audited external runtime invocation for the local two-file bundle."""
    return [str(qwen_tts), "-m", str(model_dir), "-t", prompt,
            "-o", str(output)]


def qwen3_tts_runtime_environment(qwen_tts: Path | str) -> dict[str, str]:
    """Return an environment that can load an AgentQuantix-built Qwen runtime.

    The audited Windows build keeps ``qwen3tts.dll`` beside the executable but
    places GGML's DLLs under ``<checkout>/ggml/build/bin/Release``.  Supplying
    that directory here makes a smoke result depend on the actual pinned
    runtime, rather than an incidental developer ``PATH``.  Other layouts are
    harmless: nonexistent candidate directories are simply omitted.
    """
    executable = Path(qwen_tts)
    checkout = executable.parent.parent.parent
    candidates = [executable.parent]
    if os.name == "nt":
        candidates.append(checkout / "ggml" / "build" / "bin" / "Release")
    else:
        candidates.append(checkout / "ggml" / "build" / "bin")
    existing = [str(candidate) for candidate in candidates if candidate.is_dir()]
    environment = os.environ.copy()
    if existing:
        environment["PATH"] = os.pathsep.join([*existing, environment.get("PATH", "")])
    return environment


def validate_wav(path: Path | str) -> dict:
    """Verify that a runtime actually produced non-empty PCM WAV audio.

    This is a v0.3.0 liveness/integrity gate, not a subjective speech-quality
    score. The latter belongs to v0.3.1's benchmarked quality stage.
    """
    path = Path(path)
    if not path.is_file():
        raise VoiceValidationError(f"runtime did not create WAV output: {path}")
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            sample_width = audio.getsampwidth()
            frames = audio.getnframes()
    except (wave.Error, EOFError) as error:
        raise VoiceValidationError(f"runtime produced an invalid WAV: {path}: {error}") \
            from error
    if channels < 1 or sample_rate < 1 or sample_width < 1 or frames < 1:
        raise VoiceValidationError(
            f"runtime produced empty/invalid audio: {path} "
            f"({channels} channel(s), {sample_rate} Hz, {frames} frames)")
    return {
        "path": str(path), "channels": channels, "sample_rate": sample_rate,
        "sample_width": sample_width, "frames": frames,
        "seconds": round(frames / sample_rate, 3),
    }


def run_qwen3_tts_smoke(qwen_tts: Path | str, model_dir: Path | str,
                         output: Path | str,
                         prompt: str = "Hello from AgentQuantix.",
                         timeout: int = 180) -> dict:
    """Run the Tier-1 smoke test and return only verified WAV facts."""
    output = Path(output)
    # A stale successful WAV must never make a failed run look healthy. The
    # caller supplies a fresh per-run staging path; refusing to overwrite also
    # protects a user-provided recording from a bad invocation.
    if output.exists():
        raise VoiceValidationError(
            f"smoke output already exists; choose a fresh staging path: {output}")
    command = qwen3_tts_smoke_command(qwen_tts, model_dir, output, prompt)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=timeout,
                               env=qwen3_tts_runtime_environment(qwen_tts))
    if completed.returncode:
        tail = (completed.stdout or "").strip().splitlines()[-1:]
        detail = tail[0] if tail else f"exit status {completed.returncode}"
        raise VoiceValidationError(f"qwen3-tts-cli smoke test failed: {detail}")
    return validate_wav(output)
