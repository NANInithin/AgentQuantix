"""Execution-scoped native runtime selection shared by CLI and tool harnesses.

The lifecycle core never prompts. This module supplies the interactive policy
and binds its immutable handles to existing pipeline entry points.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import sys

from . import config
from .runtimes import (
    RuntimeCancelled, RuntimeDecisionRequired, RuntimeErrorBase, RuntimeManager,
    RuntimeSession, current_session,
)


def _interactive_decision(details):
    engine = details["engine"]
    target = details.get("target") or {}
    print(f"Runtime {engine}: {target.get('ref', '')} {target.get('revision', '')}")
    if details["kind"] == "storage":
        print(f"Regular use: {details['persistent_root']}")
        print(f"Temporary use: {details['temporary_root']}")
        answer = input(f"Keep {engine} installed for regular use? [Y/n] ").strip().casefold()
        return {"persistent": answer not in ("n", "no")}
    if details["kind"] == "update":
        print(f"Installed: {details.get('installed_revision')} ({details.get('comparison', 'different')})")
        if details.get("future_only"):
            print("This resumed job will keep its frozen runtime; the update is for future runs.")
        answer = input("Build and validate this runtime update? [y/N/cancel] ").strip().casefold()
        if answer in ("cancel", "c"):
            return {"cancel": True}
        return {"update": answer in ("y", "yes")}
    if details["kind"] == "fallback":
        print(f"Update failed: {details['error']}")
        answer = input("Continue with the previous installed runtime? [Y/n] ").strip().casefold()
        return {"use_installed": answer not in ("n", "no")}
    raise RuntimeDecisionRequired(details)


@contextmanager
def execution_session(*, interactive=False, manager=None, decide=None):
    """Deduplicate update checks and hold runtime leases for one invocation."""
    existing = current_session()
    if existing is not None:
        yield existing
        return
    callback = decide or (_interactive_decision if interactive else None)
    with RuntimeSession(manager=manager, decide=callback) as session:
        yield session


def handle_for(engine):
    session = current_session()
    return session.get(engine) if session else None


def selected_runtime(engine):
    session = current_session()
    handle = session.get(engine) if session else None
    if handle is not None:
        return handle
    manager = session.manager if session else RuntimeManager()
    return manager.inspect(engine, verify=False)


def ensure_runtime(engine, *, compute_backend="auto", frozen=None, revision=None, required_targets=None):
    """Select an approved installation inside an execution session.

    Legacy low-level helpers remain usable by callers without a session; the
    public CLI and tool execution paths always establish one.
    """
    session = current_session()
    if session is None:
        return None
    from . import runtime_builders
    options = (frozen.get("build_fingerprint") if frozen else None) or runtime_builders.fingerprint(
        engine, compute_backend, required_targets=required_targets)
    if engine == "stable-diffusion.cpp" and not frozen and not session.get(engine) and not session.manager.inspect(engine):
        legacy_root = config.TEMP_DIR / "aqx-media-backends"
        candidates = sorted(legacy_root.glob("*/source"), key=lambda p: p.parent.stat().st_mtime, reverse=True)
        for source in candidates:
            try:
                available = runtime_builders.probe_existing(engine, source)
            except runtime_builders.RuntimeBuildError as error:
                print(f"Runtime {engine}: legacy cache could not be adopted: {error}", file=sys.stderr)
                continue
            if not available:
                continue
            if not session.decide:
                raise RuntimeDecisionRequired({"kind": "storage", "engine": engine,
                    "message": "Choose persistent or temporary use for the existing media runtime",
                    "persistent_root": str(config.WORK_ROOT / ".aqx-runtime-builds" / engine),
                    "temporary_root": str(source.parent), "legacy_source": str(source)})
            answer = session.decide({"kind": "storage", "engine": engine,
                "message": "Keep the existing media runtime for regular use?",
                "persistent_root": str(config.WORK_ROOT / ".aqx-runtime-builds" / engine),
                "temporary_root": str(source.parent), "legacy_source": str(source)})
            if not answer or answer.get("cancel") or "persistent" not in answer:
                raise RuntimeErrorBase("runtime migration cancelled")
            session.manager.migrate(engine, source,
                builder=lambda s, b: runtime_builders.build_runtime(engine, s, b, options),
                probe=lambda s, b: runtime_builders.probe_existing(engine, s),
                persistent=answer["persistent"], build_fingerprint=options,
                legacy_build_dir=source.parent / "build")
            break
    try:
        handle = session.ensure(
            engine,
            builder=lambda source, build_dir: runtime_builders.build_runtime(engine, source, build_dir, options),
            build_fingerprint=options,
            probe=lambda source, build_dir: runtime_builders.probe_existing(engine, source),
            required_binaries=tuple(options.get("targets", ())),
            frozen=frozen, revision=revision)
    except (RuntimeDecisionRequired, RuntimeCancelled):
        raise
    except RuntimeErrorBase as error:
        active = session.manager.inspect(engine) if frozen is None and session.decide else None
        if active is None:
            raise
        answer = session.decide({"kind": "fallback", "engine": engine, "error": str(error),
                                 "installed_revision": active.revision})
        if not answer or not answer.get("use_installed"):
            raise
        handle = session.put(engine, active)
    check = session.checks.get(engine, {})
    if check.get("status") == "unavailable":
        print(f"Runtime {engine}: update check unavailable; using {handle.revision}. "
              f"{check.get('error', '')}", file=sys.stderr)
    offered = getattr(session, "future_updates_offered", set())
    if (frozen and engine not in offered and check.get("target", {}).get("revision") != handle.revision
            and check.get("status") == "available"):
        offered.add(engine)
        session.future_updates_offered = offered
        if session.decide:
            answer = session.decide({"kind": "update", "engine": engine, "future_only": True,
                "installed_revision": handle.revision, "target": check["target"],
                "comparison": check.get("comparison", "different")})
            if answer and answer.get("cancel"):
                raise RuntimeCancelled(f"runtime update cancelled: {engine}")
            if answer and answer.get("update"):
                future_options = runtime_builders.fingerprint(engine, compute_backend, required_targets=required_targets)
                try:
                    session.manager.prepare(engine,
                        builder=lambda source, build_dir: runtime_builders.build_runtime(engine, source, build_dir, future_options),
                        build_fingerprint=future_options, persistent=handle.manifest["storage_policy"] == "persistent",
                        update=True, revision=check["target"], check=check)
                except RuntimeErrorBase as error:
                    print(f"Runtime {engine}: future update failed; resumed job keeps {handle.revision}. {error}", file=sys.stderr)
        else:
            print(f"Runtime {engine}: an update is available; this resumed job keeps {handle.revision}.", file=sys.stderr)
    return handle


def prepare_voice_runtimes(repo_id, engine, track, *, persist=False):
    """Bind conversion and scoring engines to an interrupted voice release."""
    if current_session() is None:
        return {}
    from .pipeline import voice_release
    from .media import atomic_json
    record = voice_release.work_dir(repo_id) / "runtime-snapshots.json"
    snapshots = json.loads(record.read_text(encoding="utf-8")) if record.is_file() else {}
    if not isinstance(snapshots, dict):
        raise RuntimeErrorBase("invalid voice runtime snapshot journal")
    required = list(dict.fromkeys(
        [engine, *(["whisper.cpp"] if track in ("tts", "vc") else [])]))
    selected = {}
    for item in required:
        handle = ensure_runtime(item, frozen=snapshots.get(item),
                                required_targets=("llama-tts",) if item == "llama.cpp" and track == "tts" else None)
        selected[item] = handle.snapshot()
    if persist:
        atomic_json(record, selected)
    return selected


def mark_failed():
    session = current_session()
    if session:
        session.mark_failed()


def frozen_runtime(engine):
    handle = handle_for(engine)
    return handle.snapshot() if handle else None


def require_binaries(handle, *names):
    missing = [name for name in names if name not in handle.binaries]
    if missing:
        raise RuntimeErrorBase(
            f"{handle.engine}: installed runtime lacks {', '.join(missing)}; "
            "use aqx runtime update to build the required targets in a separate installation")
    return tuple(handle.binaries[name] for name in names)
