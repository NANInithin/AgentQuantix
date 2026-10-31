"""Revision-bound native builds with explicit decisions and atomic activation.

This module never prompts. CLI and MCP callers receive the same structured
decision, while engine adapters own their native build/probe commands.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from .. import config
from .spec import (RuntimeCancelled, RuntimeDecisionRequired, RuntimeErrorBase, RuntimeHandle,
                   canonical_hash, file_hash, runtime_spec)


_SESSION = ContextVar("agentquantix_runtime_session", default=None)
_SHA = re.compile(r"[0-9a-f]{40,64}\Z")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        pending.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path != root and root in path.parents


def _pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        process = kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return False
        try:
            exit_code = wintypes.DWORD()
            return (bool(kernel32.GetExitCodeProcess(
                process, ctypes.byref(exit_code))) and exit_code.value == 259)
        finally:
            kernel32.CloseHandle(process)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as error:
        return error.errno == 1
    return True


def _request_json(url):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "AgentQuantix/runtime-manager"}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + token
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as response:
        return json.load(response)


def _run(command, **kwargs):
    command = [str(value) for value in command]
    try:
        return subprocess.run(command, capture_output=True,
                              text=True, errors="replace", timeout=300, **kwargs)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeErrorBase(f"runtime command could not execute {command[0]}: {error}") from error


@contextmanager
def _lock(path, timeout=30):
    """OS locks are released by process exit, including crashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        started = time.monotonic()
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if time.monotonic() - started >= timeout:
                    raise RuntimeErrorBase(f"runtime is busy; lock: {path}") from error
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class RuntimeManager:
    def __init__(self, work_root=None, state_root=None, temp_root=None, *,
                 request_json=None, run_command=None):
        self.work_root = Path(work_root or config.WORK_ROOT).resolve()
        self.state_root = Path(state_root or config.STATE_DIR / "runtimes").resolve()
        self.temp_root = Path(temp_root or config.TEMP_DIR).resolve()
        self.request_json = request_json or _request_json
        self.run_command = run_command or _run

    def _state(self, engine):
        runtime_spec(engine)
        return self.state_root / engine / "installation.json"

    def _lock(self, engine):
        runtime_spec(engine)
        return _lock(self.state_root / "locks" / (engine + ".lock"))

    def _live_leases(self, identity):
        directory = self.state_root / "leases" / identity
        alive = []
        if not directory.is_dir():
            return alive
        for path in directory.glob("*.json"):
            try:
                lease = _read(path, {})
            except (OSError, ValueError):
                lease = {}
            pid = lease.get("pid") if isinstance(lease, dict) else None
            if pid == os.getpid():
                continue
            if _pid_alive(pid):
                alive.append(pid)
            else:
                path.unlink(missing_ok=True)
        return sorted(set(alive))

    def _legacy_runtime_root(self, root):
        return _inside(root, self.work_root / ".aqx-runtime-builds")

    def _remove_legacy_root(self, root):
        root = Path(root).resolve()
        base = self.work_root / ".aqx-runtime-builds"
        if not self._legacy_runtime_root(root):
            return
        if root.is_dir():
            shutil.rmtree(root)
        parent = root.parent
        while parent != base and base in parent.parents:
            if not parent.is_dir() or any(parent.iterdir()):
                break
            parent.rmdir()
            parent = parent.parent
        if base.is_dir() and not any(base.iterdir()):
            base.rmdir()

    def _git(self, source, *arguments, optional=False):
        result = self.run_command(["git", "-C", str(source), *arguments])
        if result.returncode and not optional:
            raise RuntimeErrorBase(f"runtime git command failed: {result.stderr.strip()}")
        return result.stdout.strip() if not result.returncode else ""

    def original_path(self, engine):
        defaults = {"llama.cpp": config.UPSTREAM_LLAMA,
                    "audio.cpp": config.UPSTREAM_AUDIOCPP,
                    "whisper.cpp": config.UPSTREAM_WHISPER}
        # Explicit roots in tests/maintenance take priority over global config.
        if self.work_root == Path(config.WORK_ROOT).resolve() and engine in defaults:
            return Path(defaults[engine]).resolve()
        return self.work_root / runtime_spec(engine).directory

    def inspect(self, engine, *, verify=True):
        state = _read(self._state(engine), {})
        manifest = state.get("active")
        if not manifest:
            return None
        handle = RuntimeHandle.from_manifest(manifest)
        if verify:
            self.verify(handle)
        return handle

    def verify(self, handle):
        data = handle.manifest
        if data.get("engine") != handle.engine or not _SHA.fullmatch(data.get("revision", "")):
            raise RuntimeErrorBase("invalid frozen runtime identity")
        runtime_spec(handle.engine)
        if data.get("origin_kind") != "upstream" or data.get("repository") != runtime_spec(handle.engine).repository:
            raise RuntimeErrorBase("frozen runtime does not identify its designated original upstream")
        recorded = dict(data)
        identity = recorded.pop("identity", None)
        if identity != canonical_hash(recorded):
            raise RuntimeErrorBase("runtime manifest was modified")
        for kind in ("binaries", "resources"):
            for name, artifact in data.get(kind, {}).items():
                path = Path(artifact["path"])
                if not path.is_file() or file_hash(path) != artifact["sha256"]:
                    raise RuntimeErrorBase(f"runtime {kind} changed or missing: {name} ({path})")
        if not handle.source.is_dir():
            raise RuntimeErrorBase(f"runtime source is missing: {handle.source}")
        actual = self._git(handle.source, "rev-parse", "HEAD")
        if actual != data["revision"]:
            raise RuntimeErrorBase("runtime source revision changed; restore its recorded version")
        status = self._git(handle.source, "status", "--porcelain", "--untracked-files=no")
        diff = self._git(handle.source, "diff", "--submodule=diff", "--binary", "HEAD")
        if canonical_hash({"status": status, "diff": diff}) != data["source_changes"]:
            raise RuntimeErrorBase("runtime source changes differ from the recorded build")
        if self._git(handle.source, "submodule", "status", "--recursive") != data["submodules"]:
            raise RuntimeErrorBase("runtime submodule revisions changed")
        return handle

    def _manifest(self, engine, source, output, *, revision, build_fingerprint,
                  storage_policy, ownership_kind, root):
        native_build_dir = Path(output.get("build_dir", source / "build")).resolve()

        def artifacts(values, *, retain_external=False):
            result = {}
            for name, value in values.items():
                path = Path(value["path"] if isinstance(value, dict) else value).resolve()
                if not path.is_file():
                    raise RuntimeErrorBase(f"runtime probe did not produce {name}: {path}")
                digest = file_hash(path)
                if isinstance(value, dict) and value.get("sha256") and value["sha256"] != digest:
                    raise RuntimeErrorBase(f"runtime resource hash mismatch: {name}")
                if retain_external and not (_inside(path, source) or _inside(path, native_build_dir)):
                    # Shipped catalogs/conversion metadata must outlive a tool
                    # virtualenv reinstall. Store immutable content in user state.
                    retained = self.state_root / "resources" / digest / path.name
                    retained.parent.mkdir(parents=True, exist_ok=True)
                    if retained.exists() and file_hash(retained) != digest:
                        raise RuntimeErrorBase(f"retained runtime resource changed: {name}")
                    if not retained.exists():
                        temporary = retained.with_name(retained.name + "." + uuid.uuid4().hex + ".tmp")
                        try:
                            shutil.copyfile(path, temporary)
                            if file_hash(temporary) != digest:
                                raise RuntimeErrorBase(f"runtime resource changed while retaining it: {name}")
                            os.replace(temporary, retained)
                        finally:
                            temporary.unlink(missing_ok=True)
                    path = retained
                result[name] = {"path": str(path), "sha256": digest}
            return result
        binaries = artifacts(output.get("binaries", {}))
        if not binaries:
            raise RuntimeErrorBase("runtime build/probe returned no verified binaries")
        actual = self._git(source, "rev-parse", "HEAD")
        if actual != revision:
            raise RuntimeErrorBase("runtime build changed the approved source revision")
        status = self._git(source, "status", "--porcelain", "--untracked-files=no")
        diff = self._git(source, "diff", "--submodule=diff", "--binary", "HEAD")
        manifest = {"schema_version": 1, "engine": engine,
                    "repository": runtime_spec(engine).repository,
                    "revision": revision, "origin_kind": "upstream",
                    "storage_policy": storage_policy, "ownership_kind": ownership_kind,
                    "source": str(source.resolve()),
                    "build_dir": str(native_build_dir),
                    "root": str(Path(root).resolve()), "binaries": binaries,
                    "resources": artifacts(output.get("resources", {}), retain_external=True),
                    "build_fingerprint": build_fingerprint,
                    "source_changes": canonical_hash({"status": status, "diff": diff}),
                    "dirty": bool(status),
                    "submodules": self._git(source, "submodule", "status", "--recursive"),
                    "provenance": output.get("provenance", {}), "created_at": _now()}
        manifest["identity"] = canonical_hash(manifest)
        return manifest

    def _activate(self, engine, manifest, *, record_previous=True):
        path = self._state(engine)
        state = _read(path, {})
        previous = state.get("active")
        same_persistent_root = (previous
            and previous["storage_policy"] == "persistent"
            and manifest["storage_policy"] == "persistent"
            and Path(previous["root"]).resolve() == Path(manifest["root"]).resolve())
        if previous and previous["identity"] != manifest["identity"] and record_previous and not same_persistent_root:
            state["previous"] = previous
        elif manifest["storage_policy"] == "persistent" and manifest["ownership_kind"] == "managed":
            state.pop("previous", None)
        state["active"] = manifest
        if manifest["storage_policy"] == "persistent":
            state["preferred_storage"] = "persistent"
        _write(self.state_root / engine / (manifest["identity"] + ".json"), manifest)
        _write(path, state)
        return RuntimeHandle.from_manifest(manifest)

    def _check_original(self, engine, source):
        spec = runtime_spec(engine)
        origin = self._git(source, "config", "--local", "--get", "remote.origin.url")
        origin = origin.rstrip("/").removesuffix(".git")
        origins = {f"https://github.com/{name}".casefold() for name in (spec.repository, *spec.aliases)}
        normalized = origin.replace("git@github.com:", "https://github.com/").replace(
            "ssh://git@github.com/", "https://github.com/").casefold()
        if normalized not in origins:
            raise RuntimeErrorBase(f"checkout origin does not match original {engine}: {origin}")
        revision = self._git(source, "rev-parse", "HEAD")
        if not _SHA.fullmatch(revision):
            raise RuntimeErrorBase("existing runtime did not expose an immutable source revision")
        return revision

    def _expected_ggml_patches(self, engine, source, status):
        layouts = {
            "ultralytics-ggml": (
                "M cpp_ggml/third_party/ggml",
                "cpp_ggml/third_party/ggml",
                "cpp_ggml/third_party/ggml-patches",
            ),
            "rf-detr.cpp": (
                "M third_party/ggml",
                "third_party/ggml",
                "third_party/ggml-patches",
            ),
        }
        if engine not in layouts:
            return False
        expected_status, submodule_path, patch_path = layouts[engine]
        if status != expected_status:
            return False
        source = Path(source)
        submodule = source / submodule_path
        patches = sorted((source / patch_path).glob("*.patch"))
        submodule_status = self._git(
            submodule, "status", "--porcelain", "--untracked-files=all")
        patch_paths = set()
        try:
            for patch in reversed(patches):
                for line in patch.read_text(encoding="utf-8").splitlines():
                    if line.startswith("diff --git "):
                        names = line[len("diff --git "):].split()
                        if len(names) != 2 or not names[0].startswith("a/") or not names[1].startswith("b/"):
                            return False
                        patch_paths.update((names[0][2:], names[1][2:]))
                self._git(submodule, "apply", "--check", "--reverse", str(patch))
        except RuntimeErrorBase:
            return False
        changed_paths = set(self._git(submodule, "diff", "--name-only").splitlines())
        staged_paths = self._git(submodule, "diff", "--cached", "--name-only")
        return (bool(patches) and bool(patch_paths) and patch_paths == changed_paths and
                not any(line.startswith("??") for line in submodule_status.splitlines()) and
                not staged_paths)

    def _complete_cached_source(self, engine, source):
        """Require a clean, fully materialized native checkout before offline use."""
        source = Path(source).resolve()
        cmake_path = ("cpp_ggml/CMakeLists.txt" if engine == "ultralytics-ggml"
                      else "CMakeLists.txt")
        if not (source / cmake_path).is_file():
            raise RuntimeErrorBase(f"{engine}: cached source is incomplete (missing CMakeLists.txt)")
        revision = self._check_original(engine, source)
        self._git(source, "ls-files", "--error-unmatch", cmake_path)
        if self._git(source, "ls-files", "--deleted"):
            raise RuntimeErrorBase(f"{engine}: cached source has missing tracked files")
        status = self._git(source, "status", "--porcelain", "--untracked-files=no")
        if status and not self._expected_ggml_patches(engine, source, status):
            raise RuntimeErrorBase(f"{engine}: cached source has tracked changes; leaving it untouched")
        submodules = self._git(source, "submodule", "status", "--recursive")
        if any(line.startswith(("-", "+", "U")) for line in submodules.splitlines()):
            raise RuntimeErrorBase(f"{engine}: cached source has uninitialized or divergent submodules")
        return revision

    def adopt(self, engine, source, probe, *, build_fingerprint=None, storage_policy="persistent"):
        """Register an existing original without building or editing its checkout."""
        source = Path(source).resolve()
        if storage_policy not in ("persistent", "temporary"):
            raise RuntimeErrorBase("invalid runtime storage policy")
        with self._lock(engine):
            revision = self._check_original(engine, source)
            output = probe(source, source / "build")
            if not output:
                return None
            manifest = self._manifest(engine, source, output, revision=revision,
                build_fingerprint=build_fingerprint or {}, storage_policy=storage_policy,
                ownership_kind="external-adopted", root=source)
            return self._activate(engine, manifest)

    def migrate(self, engine, source, *, builder, probe, persistent,
                build_fingerprint=None, legacy_build_dir=None):
        """Adopt a legacy temporary original or copy it into regular-use storage."""
        source = Path(source).resolve()
        legacy_build_dir = Path(legacy_build_dir or source / "build").resolve()
        if not persistent:
            return self.adopt(engine, source,
                lambda path, unused: probe(path, legacy_build_dir),
                build_fingerprint=build_fingerprint, storage_policy="temporary")
        spec = runtime_spec(engine)
        revision = self._check_original(engine, source)
        fingerprint = {"adapter_version": spec.adapter_version, "platform": platform.platform(),
                       "machine": platform.machine(), **(build_fingerprint or {}),
                       "migrated_from": str(source)}
        root = self.original_path(engine).resolve()
        candidate_source, candidate_build = root, root / "build"
        with self._lock(engine):
            if root.exists():
                raise RuntimeErrorBase(
                    f"{root} already exists; adopt or update the persistent {engine} checkout instead")
            setup_path = self.state_root / engine / "setup.json"
            failure_path = self.state_root / engine / "failure.json"
            commands = []
            try:
                _write(setup_path, {"engine": engine, "revision": revision,
                                    "source_cache": str(source), "commands": commands})
                shutil.copytree(source, candidate_source,
                    ignore=lambda directory, names: ["build"]
                    if Path(directory).resolve() == source and "build" in names else [])
                output = None
                cache = legacy_build_dir / "CMakeCache.txt"
                cache_text = cache.read_text(encoding="utf-8", errors="replace") if cache.is_file() else ""
                bound_paths = (str(source), source.as_posix(), str(legacy_build_dir), legacy_build_dir.as_posix())
                if legacy_build_dir.is_dir() and not any(value in cache_text for value in bound_paths):
                    shutil.copytree(legacy_build_dir, candidate_build)
                    try:
                        output = probe(candidate_source, candidate_build)
                    except Exception:
                        output = None
                if output is None:
                    if candidate_build.exists():
                        shutil.rmtree(candidate_build)
                    output = builder(candidate_source, candidate_build)
                    probed = probe(candidate_source, candidate_build)
                    if probed:
                        output = {**output, **probed,
                            "resources": {**output.get("resources", {}), **probed.get("resources", {})},
                            "provenance": {**output.get("provenance", {}), **probed.get("provenance", {})}}
                manifest = self._manifest(engine, candidate_source, output, revision=revision,
                    build_fingerprint=fingerprint, storage_policy="persistent",
                    ownership_kind="managed", root=root)
                return self._activate(engine, manifest, record_previous=False)
            except BaseException as error:
                _write(failure_path, {"engine": engine, "revision": revision,
                                      "error": str(error), "failed_at": _now()})
                if not isinstance(error, Exception):
                    raise
                raise RuntimeErrorBase(
                    f"runtime migration failed; legacy installation retained; log: {failure_path}: {error}") from error

    def check_update(self, engine):
        """Check metadata only; failed attempts never replace a resolved target."""
        spec = runtime_spec(engine)
        active = self.inspect(engine, verify=False)
        result = {"engine": engine, "checked_at": _now(),
                  "installed_revision": active.revision if active else None}
        base = f"https://api.github.com/repos/{spec.repository}"
        try:
            try:
                release = self.request_json(base + "/releases/latest")
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                release = None
            if release and not isinstance(release, dict):
                raise ValueError("invalid release metadata")
            if release and not release.get("draft") and not release.get("prerelease"):
                ref, channel = release["tag_name"], "stable"
            else:
                ref, channel = self.request_json(base)["default_branch"], "default-branch"
            revision = self.request_json(base + "/commits/" + urllib.parse.quote(ref, safe=""))["sha"]
            if not _SHA.fullmatch(revision):
                raise ValueError("upstream did not return an immutable commit SHA")
            target = {"revision": revision, "ref": ref, "channel": channel,
                      "repository": spec.repository}
            result.update(status="current" if active and active.revision == revision else "available",
                          target=target, comparison="identical" if active and active.revision == revision else "different")
            if active and active.revision != revision:
                comparison = self.request_json(base + "/compare/" + active.revision + "..." + revision)
                result["comparison"] = comparison.get("status", "different")
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update(status="unavailable", error=f"{type(error).__name__}: {error}")
            result.pop("target", None)
        with self._lock(engine):
            path = self.state_root / engine / "update-check.json"
            old = _read(path, {})
            state = {"last_attempt": result, "last_success": old.get("last_success")}
            if result["status"] != "unavailable":
                state["last_success"] = result
            _write(path, state)
        return result

    def prepare(self, engine, *, builder, build_fingerprint=None, probe=None,
                persistent=None, update=None, revision=None, frozen=None, check=None,
                required_binaries=()):
        """Return a verified handle, or a structured install/update decision.

        ``persistent=False`` chooses a temporary original, not cancellation.
        ``builder``/``probe`` return binaries/resources/provenance dictionaries.
        Frozen resumes never change revision merely because an update exists.
        """
        spec = runtime_spec(engine)
        if persistent is not None and not isinstance(persistent, bool):
            raise RuntimeErrorBase("runtime storage decision must be a boolean")
        if update is not None and not isinstance(update, bool):
            raise RuntimeErrorBase("runtime update decision must be a boolean")
        if frozen:
            handle = RuntimeHandle.from_manifest(frozen)
            if handle.engine != engine:
                raise RuntimeErrorBase("frozen runtime belongs to another engine")
            if not handle.source.exists() and handle.manifest.get("storage_policy") == "temporary":
                handle = self.restore_frozen(handle, builder=builder, probe=probe)
            else:
                try:
                    self.verify(handle)
                except RuntimeErrorBase as error:
                    persistent_root = self.original_path(engine).resolve()
                    if (handle.manifest.get("storage_policy") == "persistent"
                            and handle.source.resolve() == persistent_root
                            and handle.source.is_dir()):
                        current = self._git(handle.source, "rev-parse", "HEAD")
                        if current != handle.revision:
                            raise RuntimeErrorBase(
                                f"this job was started with {engine} {handle.revision[:12]}, but "
                                f"{persistent_root} is now {current[:12]}; re-run without --resume, "
                                f"or reinstall {handle.revision[:12]} with `aqx runtime install {engine} "
                                f"--persistent --revision {handle.revision}`") from error
                    raise
            if set(required_binaries) - set(handle.binaries):
                raise RuntimeErrorBase("frozen runtime lacks required binaries; approve a new runtime plan")
            if check is None:
                self.check_update(engine)
            return handle

        original = self.original_path(engine).resolve()
        try:
            active = self.inspect(engine)
        except RuntimeErrorBase as error:
            manifest = _read(self._state(engine), {}).get("active")
            if (manifest and manifest.get("storage_policy") == "persistent"
                    and Path(manifest["source"]).resolve() == original and original.is_dir()):
                status = self._git(original, "status", "--porcelain", "--untracked-files=no")
                if status and not self._expected_ggml_patches(engine, original, status):
                    raise RuntimeErrorBase(
                        f"{original} has local changes; commit/stash them first") from error
            raise
        if active is None and probe and original.is_dir():
            try:
                self._check_original(engine, original)
            except RuntimeErrorBase as error:
                raise RuntimeErrorBase(
                    f"{original} exists but is not a checkout of {spec.repository}; "
                    "move it away or set AQX_WORK_ROOT") from error
            status = self._git(original, "status", "--porcelain", "--untracked-files=no")
            if status and not self._expected_ggml_patches(engine, original, status):
                raise RuntimeErrorBase(f"{original} has local changes; commit/stash them first")
            try:
                fingerprint = {"adapter_version": spec.adapter_version,
                               "platform": platform.platform(), "machine": platform.machine(),
                               **(build_fingerprint or {}),
                               "required_binaries": sorted(required_binaries)}
                active = self.adopt(engine, original, probe, build_fingerprint=fingerprint)
            except Exception:
                active = None

        check = check or self.check_update(engine)
        target = revision or check.get("target")
        if target and (not isinstance(target, dict) or not _SHA.fullmatch(target.get("revision", ""))):
            raise RuntimeErrorBase("runtime target must contain an exact immutable revision")

        local_source = None
        if check.get("status") == "unavailable" and active is None:
            cached = original
            if cached.is_dir():
                cached_revision = self._complete_cached_source(engine, cached)
                local_source = cached
                target = {"revision": cached_revision, "repository": spec.repository,
                          "ref": cached_revision, "channel": "local-cache"}

        state = _read(self._state(engine), {})
        if persistent is None:
            preference = state.get("preferred_storage")
            if active:
                persistent = active.manifest["storage_policy"] == "persistent"
            elif preference:
                persistent = preference == "persistent"
            else:
                raise RuntimeDecisionRequired({"kind": "storage", "engine": engine,
                    "message": f"Keep {engine} installed for regular use? Declining builds it in temp.",
                    "persistent_root": str(original),
                    "temporary_root": str(self.temp_root),
                    "target": target, "choices": ["persistent", "temporary", "cancel"]})

        legacy_to_remove = None
        force_rebuild = False
        if (persistent and active and active.manifest.get("storage_policy") == "persistent"
                and self._legacy_runtime_root(active.manifest["root"])):
            legacy_root = Path(active.manifest["root"]).resolve()
            old_revision = active.revision
            with self._lock(engine):
                live = self._live_leases(active.identity)
                if live:
                    raise RuntimeErrorBase(
                        f"{engine} is in use by a running aqx job (pid {live[0]}); update after it finishes")
                if original.exists():
                    if update is None:
                        relocation_target = {"revision": old_revision, "repository": spec.repository,
                                             "ref": old_revision, "channel": "installed"}
                        raise RuntimeDecisionRequired({"kind": "update", "engine": engine,
                            "message": f"Move {engine} to {original}: rebuild {old_revision[:12]} there (in place)",
                            "installed_revision": old_revision, "target": relocation_target,
                            "comparison": "relocate", "legacy_root": str(legacy_root),
                            "persistent_root": str(original),
                            "choices": ["update", "installed", "cancel"]})
                    if update is False:
                        return active
                    target = {"revision": old_revision, "repository": spec.repository,
                              "ref": old_revision, "channel": "installed"}
                    legacy_to_remove = legacy_root
                    force_rebuild = True
                else:
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(active.source), str(original))
                    state = _read(self._state(engine), {})
                    state.pop("active", None)
                    state.pop("previous", None)
                    _write(self._state(engine), state)
                    old_fingerprint = active.manifest.get("build_fingerprint", {})
                    try:
                        relocated_output = probe(original, original / "build") if probe else None
                    except Exception:
                        relocated_output = None
                    required_missing = set(required_binaries) - set(
                        (relocated_output or {}).get("binaries", {}))
                    requested = {"adapter_version": spec.adapter_version, "platform": platform.platform(),
                                 "machine": platform.machine(), **(build_fingerprint or {}),
                                 "required_binaries": sorted(required_binaries)}
                    config_changed = any(old_fingerprint.get(key) != value for key, value in requested.items()
                                         if key != "required_binaries")
                    if relocated_output and not required_missing and not config_changed:
                        manifest = self._manifest(engine, original, relocated_output,
                            revision=old_revision, build_fingerprint=old_fingerprint,
                            storage_policy="persistent", ownership_kind="managed", root=original)
                        active = self._activate(engine, manifest, record_previous=False)
                        self._remove_legacy_root(legacy_root)
                        if not target or target["revision"] == old_revision or update is False:
                            return active
                    else:
                        active = None
                        force_rebuild = True
                        legacy_to_remove = legacy_root
                        target = {"revision": old_revision, "repository": spec.repository,
                                  "ref": old_revision, "channel": "installed"}

        missing = set(required_binaries) - set(active.binaries) if active else set()
        storage_change = bool(active and
            (active.manifest["storage_policy"] == "persistent") != persistent)
        requested_configuration = {"adapter_version": spec.adapter_version,
                                   "platform": platform.platform(),
                                   "machine": platform.machine(),
                                   **(build_fingerprint or {}),
                                   "required_binaries": sorted(required_binaries)}
        configuration_change = bool(active and any(
            active.manifest.get("build_fingerprint", {}).get(key) != value
            for key, value in requested_configuration.items() if key != "required_binaries"))
        rebuild = bool(missing or storage_change or configuration_change or force_rebuild)
        if active and not rebuild and (not target or target["revision"] == active.revision or update is False):
            return active
        if active and rebuild and (not target or update is False):
            target = {"revision": active.revision, "repository": spec.repository,
                      "ref": active.revision, "channel": "installed"}
        if active and update is None and target and target["revision"] != active.revision:
            raise RuntimeDecisionRequired({"kind": "update", "engine": engine,
                "message": f"A different {engine} revision is available; build and validate it?",
                "installed_revision": active.revision, "target": target,
                "comparison": check.get("comparison", "different"),
                "choices": ["update", "installed", "cancel"]})

        if check.get("status") == "unavailable" and active and target and target["revision"] == active.revision:
            cached = active.source
            if cached.is_dir():
                cached_revision = self._complete_cached_source(engine, cached)
                local_source = cached
                target = {"revision": cached_revision, "repository": spec.repository,
                          "ref": cached_revision, "channel": "local-cache"}

        if not target:
            raise RuntimeErrorBase(f"{engine}: update check unavailable and no installed runtime is usable")
        sha = target.get("revision", "")
        if not _SHA.fullmatch(sha):
            raise RuntimeErrorBase("runtime installation requires an exact commit SHA")
        if target.get("repository", spec.repository) != spec.repository:
            raise RuntimeErrorBase("runtime target repository differs from the approved original")

        fingerprint = {"adapter_version": spec.adapter_version, "platform": platform.platform(),
                       "machine": platform.machine(), **(build_fingerprint or {}),
                       "required_binaries": sorted(required_binaries)}
        key = canonical_hash(fingerprint)
        root = original if persistent else self.temp_root / f"{engine}-{sha[:12]}-{key[:8]}"
        source = root
        build_dir = root / "build"
        manifest_path = (self.state_root / engine / (canonical_hash({
            "engine": engine, "revision": sha, "root": str(root.resolve()),
            "build_fingerprint": fingerprint}) + ".json") if persistent else
            root / "aqx-manifest.json")
        setup_path = self.state_root / engine / "setup.json" if persistent else root / "aqx-setup.json"
        failure_path = self.state_root / engine / "failure.json" if persistent else root / "aqx-failure.json"

        with self._lock(engine):
            previous = self.inspect(engine)
            if (persistent and previous
                    and previous.manifest.get("storage_policy") == "persistent"
                    and previous.source.resolve() == root.resolve()):
                live = self._live_leases(previous.identity)
                if live:
                    raise RuntimeErrorBase(
                        f"{engine} is in use by a running aqx job (pid {live[0]}); update after it finishes")
            if not persistent and (root / "aqx-manifest.json").is_file():
                manifest = _read(root / "aqx-manifest.json", {})
                if (manifest.get("revision") != sha or
                        canonical_hash(manifest.get("build_fingerprint", {})) != key):
                    raise RuntimeErrorBase(f"runtime directory collision at {root}")
                handle = RuntimeHandle.from_manifest(manifest)
                self.verify(handle)
                if set(required_binaries) - set(handle.binaries):
                    raise RuntimeErrorBase("cached runtime lacks required binary targets")
                return self._activate(engine, handle.manifest)

            if persistent and original.exists():
                try:
                    self._check_original(engine, original)
                except RuntimeErrorBase as error:
                    raise RuntimeErrorBase(
                        f"{original} exists but is not a checkout of {spec.repository}; "
                        "move it away or set AQX_WORK_ROOT") from error
                status = self._git(original, "status", "--porcelain", "--untracked-files=no")
                expected_patches = bool(status and self._expected_ggml_patches(engine, original, status))
                if status and not expected_patches:
                    raise RuntimeErrorBase(f"{original} has local changes; commit/stash them first")
            else:
                expected_patches = False

            if persistent:
                state = _read(self._state(engine), {})
                state.pop("active", None)
                state.pop("previous", None)
                _write(self._state(engine), state)

            commands = []

            def record_setup(**extra):
                _write(setup_path, {"engine": engine, "revision": sha,
                                    "commands": commands, **extra})

            def run(args):
                print(f"  .. {engine}: {' '.join(map(str, args[:3]))}...",
                      file=sys.stderr, flush=True)
                result = self.run_command(args)
                commands.append({"argv": list(map(str, args)), "returncode": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr})
                record_setup()
                if result.returncode:
                    raise RuntimeErrorBase(f"{engine}: native source setup failed; log: {setup_path}")

            try:
                if local_source is not None:
                    if self._complete_cached_source(engine, local_source) != sha:
                        raise RuntimeErrorBase("cached runtime revision changed after selection")
                    if local_source.resolve() != source.resolve():
                        if not source.exists():
                            shutil.copytree(local_source, source,
                                ignore=lambda directory, names: ["build"]
                                if Path(directory).resolve() == local_source and "build" in names else [])
                        if self._complete_cached_source(engine, source) != sha:
                            raise RuntimeErrorBase("copied offline runtime source differs from the selected revision")
                    record_setup(source_cache=str(local_source), network_setup=False)
                elif not source.exists():
                    run(["git", "clone", "--no-checkout", "--filter=blob:none", spec.url, source])
                else:
                    self._check_original(engine, source)

                if local_source is None:
                    run(["git", "-C", source, "fetch", "--depth", "1", "origin", sha])
                    run(["git", "-C", source, "checkout", "--detach", sha])
                    submodule_command = ["git", "-C", source, "submodule", "update",
                                         "--init", "--recursive"]
                    if expected_patches:
                        submodule_command.append("--force")
                    run(submodule_command)
                if self._git(source, "rev-parse", "HEAD") != sha:
                    raise RuntimeErrorBase("fetched source differs from the approved runtime revision")

                old_fingerprint = previous.manifest.get("build_fingerprint", {}) if previous else {}
                configuration_changed = bool(previous and any(
                    old_fingerprint.get(name) != value for name, value in fingerprint.items()
                    if name != "required_binaries"))
                if (persistent and (configuration_changed or force_rebuild) and build_dir.exists()):
                    if build_dir.is_dir():
                        shutil.rmtree(build_dir)
                    else:
                        build_dir.unlink()

                output = builder(source, build_dir)
                if probe:
                    probed = probe(source, build_dir)
                    if probed:
                        output = {**output, **probed,
                            "binaries": {**output.get("binaries", {}), **probed.get("binaries", {})},
                            "resources": {**output.get("resources", {}), **probed.get("resources", {})},
                            "provenance": {**output.get("provenance", {}), **probed.get("provenance", {})}}
                missing = set(required_binaries) - set(output.get("binaries", {}))
                if missing:
                    raise RuntimeErrorBase("runtime build lacks required targets: " +
                        ", ".join(sorted(missing)))
                manifest = self._manifest(engine, source, output, revision=sha,
                    build_fingerprint=fingerprint,
                    storage_policy="persistent" if persistent else "temporary",
                    ownership_kind="managed", root=root)
                if not persistent:
                    _write(manifest_path, manifest)
                handle = self._activate(engine, manifest, record_previous=not persistent)
                if legacy_to_remove is not None:
                    if not self._live_leases(previous.identity if previous else ""):
                        self._remove_legacy_root(legacy_to_remove)
                return handle
            except BaseException as error:
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                if not persistent:
                    root.mkdir(parents=True, exist_ok=True)
                _write(failure_path, {"engine": engine, "revision": sha,
                    "failed_at": _now(), "error": str(error),
                    "previous_identity": previous.identity if previous else None})
                if not isinstance(error, Exception):
                    raise
                raise RuntimeErrorBase(
                    f"{engine}: runtime setup/build failed; log: {failure_path} ({error})") from error

    def rollback(self, engine):
        with self._lock(engine):
            state = _read(self._state(engine), {})
            active = state.get("active")
            if (active and active.get("storage_policy") == "persistent"
                    and Path(active["root"]).resolve() == self.original_path(engine).resolve()):
                root = self.original_path(engine)
                raise RuntimeErrorBase(
                    f"{engine} is installed in place at {root}; reinstall an older version with "
                    f"`aqx runtime install {engine} --persistent --revision <sha>`")
            if not state.get("previous"):
                raise RuntimeErrorBase(f"{engine}: no previous installation to restore")
            handle = RuntimeHandle.from_manifest(state["previous"])
            self.verify(handle)
            return self._activate(engine, handle.manifest)

    def restore_frozen(self, handle, *, builder, probe=None):
        """Reconstruct a cleaned managed temporary original at its recorded SHA.

        Only bit-identical executable/resource outputs may reuse the old approval.
        A nondeterministic rebuild requires a new plan rather than weakening the
        original job's binary binding. The globally active version is untouched.
        """
        data = handle.snapshot()
        expected = dict(data)
        expected.pop("identity", None)
        if canonical_hash(expected) != handle.identity:
            raise RuntimeErrorBase("frozen runtime manifest was modified")
        if (data.get("ownership_kind") != "managed" or data.get("storage_policy") != "temporary"
                or not _SHA.fullmatch(data.get("revision", ""))):
            raise RuntimeErrorBase("only cleaned managed temporary runtimes can be reconstructed")
        if data.get("origin_kind") != "upstream" or data.get("repository") != runtime_spec(handle.engine).repository:
            raise RuntimeErrorBase("frozen runtime does not identify its designated original upstream")
        root = Path(data["root"]).resolve()
        if (root.parent != self.temp_root or not root.name.startswith(handle.engine + "-")
                or handle.source.resolve() != root):
            raise RuntimeErrorBase("frozen runtime reconstruction path is outside managed temporary storage")
        spec = runtime_spec(handle.engine)
        with self._lock(handle.engine):
            if handle.source.exists():
                return self.verify(handle)
            commands = []
            try:
                for command in (
                    ["git", "clone", "--no-checkout", "--filter=blob:none", spec.url, handle.source],
                    ["git", "-C", handle.source, "fetch", "--depth", "1", "origin", handle.revision],
                    ["git", "-C", handle.source, "checkout", "--detach", handle.revision],
                    ["git", "-C", handle.source, "submodule", "update", "--init", "--recursive"],
                ):
                    result = self.run_command(command)
                    commands.append({"argv": list(map(str, command)), "returncode": result.returncode,
                                     "stdout": result.stdout, "stderr": result.stderr})
                    _write(root / "aqx-setup.json", {"identity": handle.identity, "commands": commands})
                    if result.returncode:
                        raise RuntimeErrorBase(f"runtime reconstruction source setup failed: {handle.engine}")
                builder(handle.source, handle.build_dir)
                if probe:
                    probe(handle.source, handle.build_dir)
                self.verify(handle)
                _write(root / "aqx-manifest.json", data)
                return handle
            except BaseException as error:
                root.mkdir(parents=True, exist_ok=True)
                _write(root / "aqx-failure.json", {"identity": handle.identity,
                    "error": str(error), "failed_at": _now()})
                if not isinstance(error, Exception):
                    raise
                raise RuntimeErrorBase("runtime reconstruction differs from the approved build or failed; "
                    "approve a new runtime plan before reusing affected artifacts: " + str(error)) from error

    def lease(self, handle):
        path = self.state_root / "leases" / handle.identity / f"{os.getpid()}-{uuid.uuid4().hex}.json"
        with self._lock(handle.engine):
            self.verify(handle)
            _write(path, {"pid": os.getpid(), "identity": handle.identity, "started_at": _now()})
        return path

    def release(self, path):
        path = Path(path).resolve()
        if not _inside(path, self.state_root / "leases"):
            raise RuntimeErrorBase("refusing to release a lease outside runtime state")
        path.unlink(missing_ok=True)

    def cleanup_temporary(self, handle):
        data = handle.manifest
        if data["storage_policy"] != "temporary" or data["ownership_kind"] != "managed":
            return False
        root = Path(data["root"]).resolve()
        if root.parent != self.temp_root or not root.name.startswith(handle.engine + "-"):
            raise RuntimeErrorBase("refusing runtime cleanup outside its temporary managed root")
        with self._lock(handle.engine):
            leases = self.state_root / "leases" / handle.identity
            if leases.is_dir() and any(leases.glob("*.json")):
                return False
            state = _read(self._state(handle.engine), {})
            if state.get("active", {}).get("identity") == handle.identity:
                state.pop("active")
                _write(self._state(handle.engine), state)
            if root.is_dir():
                shutil.rmtree(root)
            return True

    def prune(self, engine, *, dry_run=True):
        """Inventory/remove obsolete managed runtimes and failed legacy candidates."""
        spec = runtime_spec(engine)
        result = {"engine": engine, "dry_run": dry_run, "candidates": [], "protected": []}
        with self._lock(engine):
            state = _read(self._state(engine), {})
            protected = {item["identity"] for item in (state.get("active"), state.get("previous")) if item}
            protected_roots = {str(Path(item["root"]).resolve())
                               for item in (state.get("active"), state.get("previous")) if item}
            recorded_roots = set()
            records = self.state_root / spec.id
            for path in sorted(records.glob("*.json")):
                if path.name in ("installation.json", "update-check.json"):
                    continue
                manifest = _read(path)
                if not isinstance(manifest, dict) or manifest.get("engine") != engine or "identity" not in manifest:
                    continue
                identity = manifest["identity"]
                recorded = dict(manifest)
                recorded.pop("identity", None)
                if identity != canonical_hash(recorded):
                    raise RuntimeErrorBase(f"refusing prune of modified runtime manifest: {path}")
                root = Path(manifest["root"]).resolve()
                recorded_roots.add(str(root))
                if manifest.get("ownership_kind") != "managed" or not root.exists():
                    continue
                if identity in protected or self._live_leases(identity):
                    result["protected"].append(str(root))
                    continue
                if root == self.original_path(engine).resolve():
                    continue
                legacy_base = self.work_root / ".aqx-runtime-builds" / engine
                temporary_root = root.parent == self.temp_root and root.name.startswith(engine + "-")
                if not temporary_root and not _inside(root, legacy_base):
                    raise RuntimeErrorBase("refusing prune outside managed runtime roots")
                result["candidates"].append(str(root))
                if not dry_run:
                    if temporary_root:
                        shutil.rmtree(root)
                    else:
                        self._remove_legacy_root(root)

            legacy_base = self.work_root / ".aqx-runtime-builds" / engine
            if legacy_base.is_dir():
                lease_root = self.state_root / "leases"
                live_lease_ids = []
                if lease_root.is_dir():
                    for identity_dir in lease_root.iterdir():
                        if self._live_leases(identity_dir.name):
                            live_lease_ids.append(identity_dir.name)
                leased_roots = set()
                for identity in live_lease_ids:
                    record = _read(records / (identity + ".json"), {})
                    if isinstance(record, dict) and record.get("root"):
                        leased_roots.add(str(Path(record["root"]).resolve()))
                for revision_dir in sorted(path for path in legacy_base.iterdir() if path.is_dir()):
                    for candidate in sorted(path for path in revision_dir.iterdir() if path.is_dir()):
                        candidate_key = str(candidate.resolve())
                        if candidate_key in protected_roots or candidate_key in leased_roots:
                            if candidate_key not in result["protected"]:
                                result["protected"].append(candidate_key)
                            continue
                        if (candidate_key not in recorded_roots
                                and not (candidate / "manifest.json").is_file()):
                            result["candidates"].append(candidate_key)
                            if not dry_run:
                                self._remove_legacy_root(candidate)
        return result


class RuntimeSession:
    """One invocation's deduplicated checks, decisions, handles and leases."""
    def __init__(self, manager=None, decide=None):
        self.manager = manager or RuntimeManager()
        self.decide = decide
        self.handles = {}
        self._retained = []
        self.checks = {}
        self.leases = []
        self.failed = False
        self._token = None

    def __enter__(self):
        self._token = _SESSION.set(self)
        return self

    def get(self, engine):
        return self.handles.get(engine)

    def put(self, engine, handle, *, replace=False):
        if handle.engine != engine:
            raise RuntimeErrorBase("runtime handle engine mismatch")
        if engine in self.handles and self.handles[engine].identity != handle.identity and not replace:
            raise RuntimeErrorBase("cannot change a runtime during an active session")
        if engine not in self.handles or self.handles[engine].identity != handle.identity:
            self.leases.append(self.manager.lease(handle))
            self._retained.append(handle)
        self.handles[engine] = handle
        return handle

    def ensure(self, engine, **kwargs):
        if engine in self.handles:
            frozen = kwargs.get("frozen")
            if frozen and frozen.get("identity") != self.handles[engine].identity:
                raise RuntimeErrorBase("runtime differs from frozen job in this session")
            handle = self.handles[engine]
            missing = set(kwargs.get("required_binaries", ())) - set(handle.binaries)
            storage_change = kwargs.get("persistent") is not None and (
                handle.manifest["storage_policy"] == "persistent") != kwargs["persistent"]
            configuration_change = any(handle.manifest.get("build_fingerprint", {}).get(key) != value
                for key, value in (kwargs.get("build_fingerprint") or {}).items())
            if not (missing or storage_change or configuration_change):
                return self.handles[engine]
            # A later task in the same batch may need another binary at the
            # same frozen revision. Keep the old lease while extending targets.
            kwargs["revision"] = {"revision": self.handles[engine].revision,
                                  "repository": runtime_spec(engine).repository}
            kwargs["update"] = False
        if engine not in self.checks:
            # Adoption/probe in prepare can precede update discovery; check once
            # here and keep its outcome across consent retries.
            self.checks[engine] = self.manager.check_update(engine)
        kwargs["check"] = self.checks[engine]
        for _ in range(3):
            try:
                return self.put(engine, self.manager.prepare(engine, **kwargs), replace=True)
            except RuntimeDecisionRequired as error:
                if not self.decide:
                    raise
                answer = self.decide(error.details)
                if not answer or answer.get("cancel"):
                    raise RuntimeCancelled(f"runtime setup cancelled: {engine}") from error
                kwargs.update({name: answer[name] for name in ("persistent", "update") if name in answer})
        raise RuntimeErrorBase(f"runtime decisions did not resolve setup for {engine}")

    def mark_failed(self):
        self.failed = True

    def __exit__(self, kind, value, traceback):
        try:
            for lease in self.leases:
                self.manager.release(lease)
            if kind is None and not self.failed:
                for handle in self._retained:
                    self.manager.cleanup_temporary(handle)
        finally:
            if self._token is not None:
                _SESSION.reset(self._token)


def current_session():
    return _SESSION.get()


def selected_source(engine, fallback):
    session = current_session()
    handle = session.get(engine) if session else None
    if handle is None:
        manager = session.manager if session else RuntimeManager()
        handle = manager.inspect(engine, verify=False)
    return handle.source if handle else Path(fallback)
