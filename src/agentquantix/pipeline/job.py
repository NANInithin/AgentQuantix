"""One approved model, frozen at the moment it was approved.

A Job is built from a research assessment and then does not change. That
matters more than it looks: the approval gate shows the user a quant list, an
imatrix strategy and a peak-disk figure, and a run started an hour later must
use those, not freshly computed ones that may have drifted because the machine
got busier or the Hub listing moved.

The Job is also the resume record. It is written to state/runs/<name>.json
before any work begins, so an interrupted run — a reboot, a power cut, a
Ctrl-C six hours in — can be resumed from the file with no re-research.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import time

from .. import config


@dataclass
class Job:
    repo_id: str                     # the source model on the Hub
    base_name: str                   # our GGUF naming prefix
    target_repo: str                 # where the quants get published
    quants: list                     # what this run will BUILD (remaining work)
    # The full intended sweep, including anything already published. Kept
    # separately because verification must check the repo against the whole
    # sweep — a resume run that builds 4 quants has not made the repo
    # incomplete, and reporting "26 missing" would be nonsense.
    all_quants: list = field(default_factory=list)

    # How the BF16 is obtained: "official-gguf" downloads the publisher's own
    # file, "convert" downloads safetensors and runs convert_hf_to_gguf.py.
    source_kind: str = "convert"
    official_gguf_repo: str | None = None
    official_gguf_file: str | None = None

    imatrix_source: str = "BF16"     # "BF16" or a quant name to compute it on
    imatrix_ngl: int = 0
    imatrix_chunks: int | None = None

    is_multimodal: bool = False
    fork: dict | None = None         # {"repo": ..., "ref": ...} or None

    assessment: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    # ---- derived paths -------------------------------------------------
    # Deliberately laid out exactly like the existing hand-written scripts, so
    # a job the agent starts and a script the user runs share scratch space
    # and can resume each other's work.
    @property
    def work_dir(self) -> Path:
        return config.TEMP_DIR / self.base_name

    @property
    def source_dir(self) -> Path:
        return self.work_dir / "source"

    @property
    def models_dir(self) -> Path:
        return self.work_dir / "models"

    @property
    def bf16_path(self) -> Path:
        return self.models_dir / f"{self.base_name}-BF16.gguf"

    @property
    def mmproj_path(self) -> Path:
        return self.models_dir / f"mmproj-{self.base_name}-F16.gguf"

    @property
    def imatrix_path(self) -> Path:
        return self.models_dir / f"{self.base_name}-imatrix.dat"

    @property
    def status_file(self) -> Path:
        return self.work_dir / "status" / "status.json"

    @property
    def record_file(self) -> Path:
        return config.RUNS_DIR / f"{self.base_name}.json"

    def quant_path(self, quant) -> Path:
        return self.models_dir / f"{self.base_name}-{quant}.gguf"

    # ---- persistence ---------------------------------------------------
    def save(self):
        self.record_file.parent.mkdir(parents=True, exist_ok=True)
        self.record_file.write_text(
            json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return self.record_file

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_assessment(cls, assessment):
        """Build a job from what the approval gate actually showed the user."""
        imatrix = assessment.get("imatrix") or {}
        fork = None
        if not assessment.get("arch_ok") and assessment.get("fork_leads"):
            best = assessment["fork_leads"][0]
            fork = {"repo": best["repo"], "ref": best["ref"],
                    "why": best.get("why"), "confidence": best.get("confidence")}
        return cls(
            repo_id=assessment["repo_id"],
            base_name=assessment["repo_id"].split("/")[-1],
            target_repo=assessment["target_repo"],
            quants=list(assessment["quants"]),
            all_quants=list(assessment.get("all_quants")
                            or assessment["quants"]),
            source_kind=assessment.get("source_kind", "convert"),
            official_gguf_repo=assessment.get("official_gguf"),
            official_gguf_file=assessment.get("official_bf16_file"),
            imatrix_source=imatrix.get("source", "BF16"),
            imatrix_ngl=int(imatrix.get("ngl", 0)),
            is_multimodal=bool(assessment.get("is_multimodal")),
            fork=fork,
            assessment=assessment,
        )


# =====================================================
# STATUS  (resume across runs)
# =====================================================
class Status:
    """What has already been generated and uploaded for one job.

    Two independent sources of truth are consulted at run time: this file, and
    the Hub file listing. The Hub is authoritative — it is what people can
    actually download — while the file makes the common case fast and survives
    a Hub outage. Anything in either one is not redone.
    """

    def __init__(self, path: Path, lock=None):
        self.path = path
        self.lock = lock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"generated": [], "uploaded": []},
                                            indent=2))

    def _read(self):
        # Both keys are always present in the returned dict, whatever the file
        # holds. The quant loop indexes ["uploaded"] on every iteration, and a
        # status.json written by an older script — or half-written by a run
        # that was killed mid-flush — would otherwise raise a KeyError deep
        # inside an otherwise recoverable batch.
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            data = {}
        return {"generated": list(data.get("generated") or []),
                "uploaded": list(data.get("uploaded") or [])}

    def load(self):
        if self.lock:
            with self.lock:
                return self._read()
        return self._read()

    def mark(self, kind, name):
        def write():
            status = self._read()
            status.setdefault(kind, [])
            if name not in status[kind]:
                status[kind].append(name)
            self.path.write_text(json.dumps(status, indent=2))
        if self.lock:
            with self.lock:
                write()
        else:
            write()
