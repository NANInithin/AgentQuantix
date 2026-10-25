"""Shared runtime maintenance actions; MCP never reads terminal input."""
from __future__ import annotations

import re
from pathlib import Path

from ..runtimes import RUNTIMES, RuntimeDecisionRequired, RuntimeManager


def _tool(name, description, properties, required=()):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties,
                             "required": list(required)}}


ENGINE = {"type": "string", "enum": list(RUNTIMES)}
TOOLS = [
    _tool("get_runtime_status", "Inspect installed native runtime revisions, storage and provenance. Does not build or contact upstream.",
          {"engine": ENGINE}),
    _tool("check_runtime_updates", "Check one runtime's upstream stable release, or default branch when no stable release exists. Returns an immutable target for review; never installs or updates.",
          {"engine": ENGINE}, ["engine"]),
    _tool("prepare_runtime", "Install an original runtime only after runtime-specific user approval of the exact target_revision returned by check_runtime_updates. persistent=false selects a temporary original build. Updating an installed runtime additionally requires update_approved=true. Model approval is not runtime-update approval.",
          {"engine": ENGINE, "persistent": {"type": "boolean"},
           "user_approved": {"type": "boolean"}, "update_approved": {"type": "boolean"},
           "target_revision": {"type": "object"}, "compute_backend": {"type": "string"},
           "targets": {"type": "array", "items": {"type": "string"}}},
          ["engine", "persistent", "target_revision", "user_approved"]),
    _tool("adopt_runtime", "Register and verify a user-owned existing original runtime without modifying its checkout or compiling. Requires explicit approval of the source path.",
          {"engine": ENGINE, "source": {"type": "string"},
           "user_approved": {"type": "boolean"}}, ["engine", "source", "user_approved"]),
    _tool("rollback_runtime", "Restore the previous verified installation. Running jobs keep their recorded runtime; requires explicit rollback approval.",
          {"engine": ENGINE, "user_approved": {"type": "boolean"}}, ["engine", "user_approved"]),
    _tool("prune_runtime", "Inventory obsolete managed runtime builds. Defaults to dry_run=true; removal requires explicit user approval. Active, previous, external and leased installations are protected.",
          {"engine": ENGINE, "dry_run": {"type": "boolean", "default": True},
           "user_approved": {"type": "boolean"}}, ["engine"]),
]
NAMES = {tool["name"] for tool in TOOLS}


def _approved_target(target):
    if not isinstance(target, dict):
        raise ValueError("target_revision must be the immutable target returned by check_runtime_updates")
    revision = target.get("sha") or target.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        raise ValueError("target_revision requires an immutable commit SHA")
    return {**target, "revision": revision}


def _status(manager, engine):
    handle = manager.inspect(engine)
    return {"engine": engine, "repository": RUNTIMES[engine].repository,
            "installed": handle is not None,
            "runtime": handle.snapshot() if handle else None}


def call(name, arguments=None):
    arguments = arguments or {}
    if name not in NAMES:
        raise ValueError(f"unknown runtime tool: {name}")
    engine = arguments.get("engine")
    if engine is not None and engine not in RUNTIMES:
        raise ValueError(f"unknown runtime: {engine}")
    if name == "prune_runtime" and type(arguments.get("dry_run", True)) is not bool:
        raise ValueError("dry_run must be a boolean")
    # Refuse before creating a manager, checking the network or resolving a
    # builder. A model's user_approved field cannot authorize this operation.
    if name == "prepare_runtime":
        if arguments.get("user_approved") is not True:
            raise ValueError("prepare_runtime requires user_approved=true for runtime setup")
        if type(arguments.get("persistent")) is not bool:
            raise ValueError("prepare_runtime requires an explicit persistent boolean")
        target = _approved_target(arguments.get("target_revision"))
    elif name in ("adopt_runtime", "rollback_runtime") or (
            name == "prune_runtime" and arguments.get("dry_run", True) is False):
        if arguments.get("user_approved") is not True:
            raise ValueError(f"{name} requires user_approved=true for this runtime action")
        if name == "adopt_runtime" and not arguments.get("source"):
            raise ValueError("source path is required for runtime adoption")
        target = None
    else:
        target = None
    if name != "get_runtime_status" and engine is None:
        raise ValueError("engine is required")

    manager = RuntimeManager()
    if name == "get_runtime_status":
        if engine:
            return _status(manager, engine)
        return {"runtimes": [_status(manager, item) for item in RUNTIMES]}
    if name == "check_runtime_updates":
        return manager.check_update(engine)
    if name == "rollback_runtime":
        return {"status": "ready", "runtime": manager.rollback(engine).snapshot()}
    if name == "prune_runtime":
        return manager.prune(engine, dry_run=arguments.get("dry_run", True))

    from .. import runtime_builders
    probe = lambda source, build_dir: runtime_builders.probe_existing(engine, source)
    if name == "adopt_runtime":
        handle = manager.adopt(engine, Path(arguments["source"]), probe)
        return {"status": "ready", "runtime": handle.snapshot()}
    compute_backend = arguments.get("compute_backend", "auto")
    targets = tuple(arguments.get("targets") or ())
    fingerprint = runtime_builders.fingerprint(engine, compute_backend=compute_backend, required_targets=targets)
    options = {"compute_backend": compute_backend, "required_targets": list(targets)}
    try:
        handle = manager.prepare(
            engine,
            builder=lambda source, build_dir: runtime_builders.build_runtime(engine, source, build_dir, options),
            build_fingerprint=fingerprint,
            probe=probe,
            required_binaries=tuple(fingerprint["targets"]),
            persistent=arguments["persistent"],
            update=arguments.get("update_approved") is True,
            revision=target,
        )
    except RuntimeDecisionRequired as error:
        return {"status": "decision-required", "requirements": error.details}
    return {"status": "ready", "runtime": handle.snapshot()}
