"""Run identifiers shared by resume, review, verify and publish actions.

Each pipeline keeps its existing on-disk record, so work started before the
shared CLI existed stays reachable without being moved:

  text:<base_name>   state/runs/<base_name>.json (the frozen Job)
  voice:<slug>       TEMP_DIR/voice/<slug>/ (run.json added on first use)
  media:<plan_id>    TEMP_DIR/aqx-media-runs/<plan_id>/plan.json

Resolution is deterministic. An identifier that matches several runs returns
those choices instead of picking one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import time

from . import config
from .tasks import MEDIA, PIPELINES, RETRIEVAL, TEXT, VOICE


class AmbiguousRun(ValueError):
    def __init__(self, identifier, matches):
        self.matches = matches
        super().__init__(f"{identifier!r} matches several runs: "
                         + ", ".join(item.id for item in matches)
                         + "; pass the full run id")


@dataclass
class RunRecord:
    id: str
    pipeline: str
    repo_id: str
    location: str
    target: str | None = None
    task: str | None = None
    engine: str | None = None
    created: str | None = None
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def voice_root():
    return config.TEMP_DIR / "voice"


def media_root():
    return config.TEMP_DIR / "aqx-media-runs"


def retrieval_root():
    return config.TEMP_DIR / "aqx-retrieval-runs"


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _text_runs():
    for path in sorted(config.RUNS_DIR.glob("*.json")) if config.RUNS_DIR.is_dir() else ():
        data = _read(path)
        if not isinstance(data, dict) or "repo_id" not in data or "base_name" not in data:
            continue
        yield RunRecord(f"text:{data['base_name']}", TEXT, data["repo_id"], str(path),
                        target=data.get("target_repo"), task="text-generation", engine="llama.cpp",
                        created=data.get("created_at"),
                        details={"quants": data.get("quants", []), "fork": data.get("fork")})


def _voice_runs():
    root = voice_root()
    for directory in sorted(root.iterdir()) if root.is_dir() else ():
        if not directory.is_dir():
            continue
        record = _read(directory / "run.json") or {}
        repo = record.get("repo_id") or (
            directory.name.replace("--", ":", 1) if directory.name.startswith("audio.cpp--")
            else directory.name.replace("--", "/", 1))
        reviews = sorted(p.stem for p in (directory / "reviews").glob("*.json")) \
            if (directory / "reviews").is_dir() else []
        yield RunRecord(f"voice:{directory.name}", VOICE, repo, str(directory),
                        target=record.get("target_repo"), task=record.get("task"),
                        engine=record.get("engine"), created=record.get("created"),
                        details={"reviews": reviews, "inferred": not record,
                                 "plan": record.get("plan")})


def _media_runs():
    root = media_root()
    for directory in sorted(root.iterdir()) if root.is_dir() else ():
        plan = _read(directory / "plan.json")
        if not isinstance(plan, dict) or not re.fullmatch(r"[0-9a-f]{64}", directory.name):
            continue
        reviews = sorted(p.stem for p in (directory / "reviews").glob("*.json")) \
            if (directory / "reviews").is_dir() else []
        yield RunRecord(f"media:{directory.name}", MEDIA, plan.get("repo_id", ""), str(directory),
                        target=", ".join(sorted(set((plan.get("target_repositories") or {}).values()))) or None,
                        task=plan.get("task"), engine="stable-diffusion.cpp",
                        details={"quants": plan.get("quants", []), "reviews": reviews})


def _retrieval_runs():
    root = retrieval_root()
    for directory in sorted(root.iterdir()) if root.is_dir() else ():
        data = _read(directory / "run.json")
        journal = _read(directory / "journal.json") or {}
        plan = data.get("plan") if isinstance(data, dict) else None
        if not isinstance(plan, dict) or not plan.get("model"):
            continue
        yield RunRecord(f"retrieval:{directory.name}", RETRIEVAL, plan["model"],
                        str(directory), target=plan.get("target_repo"), task=plan.get("task"),
                        engine=plan.get("engine"), created=data.get("created"),
                        details={"plan": plan, "journal": journal})


def discover(pipeline=None):
    found = [*_text_runs(), *_voice_runs(), *_media_runs(), *_retrieval_runs()]
    return [item for item in found if pipeline in (None, item.pipeline)]


def resolve(identifier, pipeline=None):
    value = identifier.strip()
    runs = discover(pipeline)
    prefix, _, key = value.partition(":")
    if prefix in PIPELINES and key:
        exact = [item for item in runs if item.id == value]
        if not exact and prefix == MEDIA and re.fullmatch(r"[0-9a-f]{8,63}", key):
            exact = [item for item in runs if item.pipeline == MEDIA and item.id.startswith(value)]
    else:
        folded = value.casefold()
        exact = [item for item in runs if item.id.partition(":")[2].casefold() == folded
                 or item.repo_id.casefold() == folded
                 or item.repo_id.split("/")[-1].casefold() == folded]
        if not exact and re.fullmatch(r"[0-9a-f]{8,64}", value):
            exact = [item for item in runs if item.pipeline == MEDIA
                     and item.id.partition(":")[2].startswith(value)]
    if not exact:
        raise ValueError(f"no recorded run matches {identifier!r}; `aqx runs` lists them")
    if len(exact) > 1:
        raise AmbiguousRun(identifier, exact)
    record = exact[0]
    if record.pipeline == VOICE and record.details.get("inferred"):
        # Older voice workspaces had no record. Adding one is the migration;
        # existing artifacts and reviews are left exactly where they are.
        write_voice_record(record.repo_id, {}, directory=Path(record.location))
        record.details["inferred"] = False
    return record


def write_voice_record(repo_id, plan, *, directory=None):
    from .pipeline import voice_release
    directory = Path(directory or voice_release.work_dir(repo_id))
    existing = _read(directory / "run.json") or {}
    record = {**existing, "schema": 1, "repo_id": repo_id,
              "created": existing.get("created") or time.strftime("%Y-%m-%d %H:%M:%S")}
    if plan:
        record.update({"target_repo": plan.get("target_repo"), "task": plan.get("task"),
                       "engine": plan.get("engine"), "plan": plan})
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "run.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")
    temporary.replace(directory / "run.json")
    return f"voice:{directory.name}"
