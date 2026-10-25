"""Runtime identities shared by CLI, MCP and native pipeline adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path


class RuntimeErrorBase(RuntimeError):
    """An actionable runtime lifecycle failure."""


class RuntimeCancelled(RuntimeErrorBase):
    """An explicit user cancellation, rather than a failed native build."""


class RuntimeDecisionRequired(RuntimeErrorBase):
    def __init__(self, details):
        self.details = details
        super().__init__(details.get("message", "runtime decision required"))


@dataclass(frozen=True)
class RuntimeSpec:
    id: str
    repository: str
    directory: str
    adapter_version: int = 1
    aliases: tuple[str, ...] = ()

    @property
    def url(self):
        return f"https://github.com/{self.repository}.git"


RUNTIMES = {
    spec.id: spec for spec in (
        RuntimeSpec("llama.cpp", "ggml-org/llama.cpp", "llama.cpp"),
        RuntimeSpec("audio.cpp", "0xShug0/audio.cpp", "audio.cpp"),
        RuntimeSpec("whisper.cpp", "ggml-org/whisper.cpp", "whisper.cpp"),
        RuntimeSpec("stable-diffusion.cpp", "leejet/stable-diffusion.cpp", "stable-diffusion.cpp"),
        RuntimeSpec("vision.cpp", "Acly/vision.cpp", "vision.cpp"),
        RuntimeSpec("ultralytics-ggml", "Asher-1/ultralytics-ggml", "ultralytics-ggml"),
        RuntimeSpec("rf-detr.cpp", "localai-org/rf-detr.cpp", "rf-detr.cpp", aliases=("mudler/rf-detr.cpp",)),
        RuntimeSpec("depth-anything.cpp", "localai-org/depth-anything.cpp", "depth-anything.cpp", aliases=("mudler/depth-anything.cpp",)),
        RuntimeSpec("clip.cpp", "monatis/clip.cpp", "clip.cpp"),
        RuntimeSpec("sam3cpp", "ropoctl/sam3cpp", "sam3cpp"),
        RuntimeSpec("timeseries.cpp", "liodon-ai/timeseries.cpp", "timeseries.cpp"),
    )
}


def runtime_spec(engine):
    try:
        return RUNTIMES[engine]
    except KeyError as error:
        raise RuntimeErrorBase(f"unknown runtime: {engine}") from error


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RuntimeHandle:
    engine: str
    source: Path
    build_dir: Path
    binaries: dict[str, Path]
    manifest: dict = field(repr=False)

    @property
    def revision(self):
        return self.manifest["revision"]

    @property
    def identity(self):
        return self.manifest["identity"]

    def snapshot(self):
        # Deep copy prevents a caller changing the manager's in-memory manifest.
        return json.loads(json.dumps(self.manifest))

    @classmethod
    def from_manifest(cls, manifest):
        return cls(manifest["engine"], Path(manifest["source"]),
                   Path(manifest["build_dir"]),
                   {name: Path(value["path"]) for name, value in manifest["binaries"].items()},
                   json.loads(json.dumps(manifest)))
