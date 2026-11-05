"""Offline lifecycle tests: user decisions, revision pins and failure isolation."""
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import urllib.error

import pytest

from agentquantix.runtimes import (RUNTIMES, RuntimeDecisionRequired,
                                   RuntimeErrorBase, RuntimeManager, RuntimeSession,
                                   current_session, selected_source)
from agentquantix.runtimes.spec import canonical_hash


OLD = "a" * 40
NEW = "b" * 40


class FakeGit:
    def __init__(self):
        self.revisions = {}
        self.calls = []
        self.origins = {}
        self.effective_origins = {}
        self.dirty = {}
        self.submodules = {}
        self.diffs = {}
        self.diff_names = {}

    def __call__(self, command):
        command = list(map(str, command))
        self.calls.append(command)
        if command[1] == "clone":
            path = Path(command[-1])
            path.mkdir(parents=True)
            self.revisions[str(path)] = OLD
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        source, args = command[2], command[3:]
        output = ""
        if args[:2] == ["rev-parse", "HEAD"]:
            output = self.revisions.get(source, OLD)
        elif args[:2] == ["checkout", "--detach"]:
            self.revisions[source] = args[-1]
        elif args[:4] == ["config", "--local", "--get", "remote.origin.url"]:
            output = self.origins.get(source, "https://github.com/ggml-org/llama.cpp.git")
        elif args[:3] == ["remote", "get-url", "origin"]:
            output = self.effective_origins.get(
                source, self.origins.get(source, "https://github.com/ggml-org/llama.cpp.git"))
        elif args[:1] == ["status"]:
            output = self.dirty.get(source, "")
        elif args[:2] == ["submodule", "status"]:
            output = self.submodules.get(source, "")
        elif args[:2] == ["diff", "--name-only"]:
            output = self.diff_names.get(source, "")
        elif args[:1] == ["diff"]:
            output = self.diffs.get(source, "")
        return SimpleNamespace(returncode=0, stdout=output, stderr="")


class Upstream:
    def __init__(self, revision=NEW):
        self.revision = revision
        self.calls = []
        self.offline = False
        self.no_release = False

    def __call__(self, url):
        self.calls.append(url)
        if self.offline:
            raise urllib.error.URLError("offline")
        if url.endswith("/releases/latest"):
            if self.no_release:
                raise urllib.error.HTTPError(url, 404, "no release", None, None)
            return {"tag_name": "v2", "draft": False, "prerelease": False}
        if "/commits/" in url:
            return {"sha": self.revision}
        if "/compare/" in url:
            return {"status": "ahead"}
        return {"default_branch": "trunk"}


@pytest.fixture
def lifecycle(tmp_path):
    git, upstream = FakeGit(), Upstream()
    manager = RuntimeManager(tmp_path / "work", tmp_path / "state", tmp_path / "scratch",
                             run_command=git, request_json=upstream)
    return SimpleNamespace(manager=manager, git=git, upstream=upstream)


def build(source, build_dir):
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "native.exe"
    binary.write_bytes(b"verified native executable")
    resource = source / "converter.py"
    resource.write_text("# conversion contract", encoding="utf-8")
    return {"binaries": {"native": binary}, "resources": {"converter": {"path": resource}},
            "provenance": {"probe": "passed"}}


def installed(lifecycle, **kwargs):
    return lifecycle.manager.prepare("llama.cpp", builder=build, persistent=True, **kwargs)


def _default_candidate_fingerprint():
    import platform
    from agentquantix.runtimes.spec import runtime_spec
    spec = runtime_spec("llama.cpp")
    return {"adapter_version": spec.adapter_version, "platform": platform.platform(),
            "machine": platform.machine(), "required_binaries": []}


def test_registry_has_all_designated_originals():
    assert len(RUNTIMES) == 11
    assert RUNTIMES["ultralytics-ggml"].repository == "Asher-1/ultralytics-ggml"
    assert RUNTIMES["ultralytics-ggml"].adapter_version == 2


def test_cached_ultralytics_source_checks_cpp_ggml_cmake(lifecycle, tmp_path):
    source = tmp_path / "ultralytics-ggml"
    cmake_file = source / "cpp_ggml/CMakeLists.txt"
    cmake_file.parent.mkdir(parents=True)
    cmake_file.write_text("project(yolo)\n", encoding="utf-8")
    lifecycle.git.origins[str(source)] = "https://github.com/Asher-1/ultralytics-ggml.git"

    assert lifecycle.manager._complete_cached_source("ultralytics-ggml", source) == OLD
    assert any(command[-2:] == ["--error-unmatch", "cpp_ggml/CMakeLists.txt"]
               for command in lifecycle.git.calls)


def test_ultralytics_source_validator_accepts_its_exact_ggml_patch(lifecycle, tmp_path):
    source = tmp_path / "ultralytics-ggml"
    patch_dir = source / "cpp_ggml/third_party/ggml-patches"
    patch_dir.mkdir(parents=True)
    (source / "cpp_ggml/CMakeLists.txt").write_text("project(yolo)\n", encoding="utf-8")
    (patch_dir / "0001-yolo.patch").write_text(
        "diff --git a/src/ggml.c b/src/ggml.c\n", encoding="utf-8")
    submodule = str(source / "cpp_ggml/third_party/ggml")
    lifecycle.git.dirty[submodule] = " M src/ggml.c"
    lifecycle.git.diff_names[submodule] = "src/ggml.c"

    assert lifecycle.manager._expected_ggml_patches(
        "ultralytics-ggml", source, "M cpp_ggml/third_party/ggml")


def test_rfdetr_source_validator_accepts_its_exact_patch_series(lifecycle, tmp_path):
    source = tmp_path / "rf-detr.cpp"
    patch_dir = source / "third_party/ggml-patches"
    patch_dir.mkdir(parents=True)
    for name, path in (
            ("0001-first.patch", "src/ggml.c"),
            ("0002-second.patch", "src/ggml-cpu/ggml-cpu.c")):
        (patch_dir / name).write_text(f"diff --git a/{path} b/{path}\n", encoding="utf-8")
    submodule = str(source / "third_party/ggml")
    lifecycle.git.dirty[submodule] = " M src/ggml.c\n M src/ggml-cpu/ggml-cpu.c"
    lifecycle.git.diff_names[submodule] = "src/ggml.c\nsrc/ggml-cpu/ggml-cpu.c"

    assert lifecycle.manager._expected_ggml_patches(
        "rf-detr.cpp", source, "M third_party/ggml")


def test_latest_stable_release_resolves_exact_sha(lifecycle):
    check = lifecycle.manager.check_update("llama.cpp")
    assert check["target"] == {"revision": NEW, "ref": "v2", "channel": "stable",
                               "repository": "ggml-org/llama.cpp"}
    assert lifecycle.upstream.calls[-1].endswith("/commits/v2")
    assert not lifecycle.git.calls


def test_no_stable_release_falls_back_to_actual_default_branch(lifecycle):
    lifecycle.upstream.no_release = True
    check = lifecycle.manager.check_update("llama.cpp")
    assert check["target"]["channel"] == "default-branch"
    assert check["target"]["ref"] == "trunk"
    assert lifecycle.upstream.calls[-1].endswith("/commits/trunk")


def test_clip_runtime_installs_from_default_branch_when_no_release_exists(lifecycle):
    lifecycle.upstream.no_release = True
    handle = lifecycle.manager.prepare("clip.cpp", builder=build, persistent=True)
    assert handle.manifest["repository"] == "monatis/clip.cpp"
    assert handle.revision == NEW
    assert lifecycle.upstream.calls[-1].endswith("/commits/trunk")
    assert any(command[1] == "clone" and
               any("monatis/clip.cpp.git" in arg for arg in command)
               for command in lifecycle.git.calls)


def test_failed_update_check_keeps_last_success_separately(lifecycle):
    lifecycle.manager.check_update("llama.cpp")
    lifecycle.upstream.offline = True
    check = lifecycle.manager.check_update("llama.cpp")
    assert check["status"] == "unavailable" and "target" not in check
    record = json.loads((lifecycle.manager.state_root / "llama.cpp/update-check.json").read_text())
    assert record["last_attempt"]["status"] == "unavailable"
    assert record["last_success"]["target"]["revision"] == NEW


def test_missing_runtime_requests_storage_before_cloning(lifecycle):
    with pytest.raises(RuntimeDecisionRequired) as raised:
        lifecycle.manager.prepare("llama.cpp", builder=build)
    assert raised.value.details["kind"] == "storage"
    assert raised.value.details["persistent_root"] == str(
        lifecycle.manager.original_path("llama.cpp"))
    assert raised.value.details["temporary_root"] == str(lifecycle.manager.temp_root)
    assert not lifecycle.git.calls


@pytest.mark.parametrize("persistent", [True, False])
def test_original_build_storage_policy_and_exact_checkout(lifecycle, persistent):
    handle = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=persistent)
    if persistent:
        assert handle.source == lifecycle.manager.original_path("llama.cpp")
        assert handle.build_dir == handle.source / "build"
        assert not (lifecycle.manager.work_root / ".aqx-runtime-builds").exists()
    else:
        key = canonical_hash(handle.manifest["build_fingerprint"])[:8]
        assert handle.source == lifecycle.manager.temp_root / f"llama.cpp-{NEW[:12]}-{key}"
        assert handle.build_dir == handle.source / "build"
    assert handle.manifest["origin_kind"] == "upstream"
    assert handle.manifest["storage_policy"] == ("persistent" if persistent else "temporary")
    assert handle.manifest["resources"]["converter"]["sha256"]
    checkout = [args for args in lifecycle.git.calls if "checkout" in args]
    assert checkout[0][-2:] == ["--detach", NEW]
    assert lifecycle.manager.verify(handle) is handle


def test_offline_installed_revision_continues_without_build(lifecycle):
    old = installed(lifecycle)
    lifecycle.upstream.offline = True
    handle = lifecycle.manager.prepare("llama.cpp", builder=lambda *args: pytest.fail("offline rebuild"))
    assert handle.identity == old.identity


def test_new_revision_requires_update_approval_and_decline_reuses_current(lifecycle):
    old = installed(lifecycle)
    lifecycle.upstream.revision = OLD
    with pytest.raises(RuntimeDecisionRequired) as raised:
        lifecycle.manager.prepare("llama.cpp", builder=build)
    assert raised.value.details["kind"] == "update"
    assert raised.value.details["installed_revision"] == NEW
    handle = lifecycle.manager.prepare("llama.cpp", builder=build, update=False)
    assert handle.identity == old.identity


def test_failed_in_place_update_clears_active_and_logs_failure(lifecycle):
    old = installed(lifecycle)
    lifecycle.upstream.revision = OLD

    def fail(*args):
        raise RuntimeError("compiler failed")

    with pytest.raises(RuntimeErrorBase, match="failure.json"):
        lifecycle.manager.prepare("llama.cpp", builder=fail, update=True)
    assert lifecycle.manager.inspect("llama.cpp") is None
    failure = lifecycle.manager.state_root / "llama.cpp" / "failure.json"
    assert json.loads(failure.read_text())["previous_identity"] == old.identity
    rebuilt = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=True)
    assert rebuilt.revision == OLD and rebuilt.source == old.source


def test_successful_persistent_update_is_in_place_without_rollback(lifecycle):
    first = installed(lifecycle)
    lifecycle.upstream.revision = OLD
    second = lifecycle.manager.prepare("llama.cpp", builder=build, update=True)
    assert first.source == second.source == lifecycle.manager.original_path("llama.cpp")
    assert second.revision == OLD
    state = json.loads((lifecycle.manager.state_root / "llama.cpp" / "installation.json").read_text())
    assert "previous" not in state
    with pytest.raises(RuntimeErrorBase, match="installed in place"):
        lifecycle.manager.rollback("llama.cpp")


def test_keyboard_interrupt_during_persistent_build_is_recorded(lifecycle):
    installed(lifecycle)
    lifecycle.upstream.revision = OLD

    def interrupt(*_args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        lifecycle.manager.prepare("llama.cpp", builder=interrupt, update=True)
    failure = lifecycle.manager.state_root / "llama.cpp" / "failure.json"
    assert json.loads(failure.read_text())["error"] == ""
    assert lifecycle.manager.inspect("llama.cpp") is None


def test_persistent_update_refuses_tracked_changes(lifecycle):
    handle = installed(lifecycle)
    lifecycle.upstream.revision = OLD
    lifecycle.git.dirty[str(handle.source)] = " M tracked.cpp"
    with pytest.raises(RuntimeErrorBase, match="has local changes; commit/stash them first"):
        lifecycle.manager.prepare("llama.cpp", builder=build, update=True)


def test_persistent_install_refuses_non_checkout_at_original_path(lifecycle, monkeypatch):
    source = lifecycle.manager.original_path("llama.cpp")
    source.mkdir(parents=True)
    original_git = lifecycle.manager._git

    def no_checkout(path, *args, **kwargs):
        if Path(path) == source and args[:3] == ("config", "--local", "--get"):
            raise RuntimeErrorBase("not a git checkout")
        return original_git(path, *args, **kwargs)

    monkeypatch.setattr(lifecycle.manager, "_git", no_checkout)
    with pytest.raises(RuntimeErrorBase, match="exists but is not a checkout of ggml-org/llama.cpp"):
        lifecycle.manager.prepare("llama.cpp", builder=build, persistent=True)


def test_live_lease_blocks_in_place_update_and_stale_lease_is_removed(lifecycle, monkeypatch):
    import agentquantix.runtimes.manager as manager_module

    handle = installed(lifecycle)
    lifecycle.upstream.revision = OLD
    leases = lifecycle.manager.state_root / "leases" / handle.identity
    leases.mkdir(parents=True)
    stale = leases / "dead.json"
    stale.write_text('{"pid": 87654321}', encoding="utf-8")
    active_lease = leases / "live.json"
    active_lease.write_text('{"pid": 12345}', encoding="utf-8")
    monkeypatch.setattr(manager_module, "_pid_alive", lambda pid: pid == 12345)
    with pytest.raises(RuntimeErrorBase, match="in use by a running aqx job \\(pid 12345\\)"):
        lifecycle.manager.prepare("llama.cpp", builder=build, update=True)
    assert not stale.exists()
    active_lease.unlink()
    monkeypatch.setattr(manager_module, "_pid_alive", lambda _pid: False)
    updated = lifecycle.manager.prepare("llama.cpp", builder=build, update=True)
    assert updated.revision == OLD
    assert not active_lease.exists()


def test_revision_only_update_keeps_persistent_build_directory(lifecycle):
    first = installed(lifecycle)
    marker = first.build_dir / "incremental-marker"
    marker.write_text("retain", encoding="utf-8")
    lifecycle.upstream.revision = OLD
    second = lifecycle.manager.prepare("llama.cpp", builder=build, update=True)
    assert second.source == first.source
    assert marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="exercises the POSIX signal-zero path")
def test_pid_alive_checks_current_and_missing_posix_pid():
    from agentquantix.runtimes.manager import _pid_alive

    assert _pid_alive(os.getpid())
    assert not _pid_alive(987654321)


def test_pid_alive_uses_windows_process_api(monkeypatch):
    import ctypes
    from ctypes import wintypes
    import agentquantix.runtimes.manager as manager_module

    class Function:
        restype = None

        def __init__(self, callback):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    class Kernel:
        OpenProcess = Function(lambda *_args: 123)
        GetExitCodeProcess = Function(
            lambda _handle, output: (ctypes.cast(
                output, ctypes.POINTER(wintypes.DWORD)).contents.__setattr__("value", 259) or 1))
        CloseHandle = Function(lambda *_args: 1)

    monkeypatch.setattr(manager_module.os, "name", "nt")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: Kernel, raising=False)
    assert manager_module._pid_alive(1234)


def test_frozen_persistent_resume_reports_in_place_revision_change(lifecycle):
    first = installed(lifecycle)
    lifecycle.upstream.revision = OLD
    installed(lifecycle, update=True)
    persistent_root = lifecycle.manager.original_path("llama.cpp")
    with pytest.raises(RuntimeErrorBase) as error:
        lifecycle.manager.prepare("llama.cpp", builder=build, frozen=first.snapshot())
    assert str(error.value) == (
        f"this job was started with llama.cpp {first.revision[:12]}, but "
        f"{persistent_root} is now {OLD[:12]}; re-run without --resume, or reinstall "
        f"{first.revision[:12]} with `aqx runtime install llama.cpp --persistent --revision "
        f"{first.revision}`")
    assert lifecycle.manager.inspect("llama.cpp").revision == OLD


def _legacy_active(lifecycle):
    root = lifecycle.manager.work_root / ".aqx-runtime-builds" / "llama.cpp" / OLD / "legacy-key"
    source = root / "source"
    source.mkdir(parents=True)
    lifecycle.git.revisions[str(source)] = OLD
    output = build(source, source / "build")
    manifest = lifecycle.manager._manifest("llama.cpp", source, output, revision=OLD,
        build_fingerprint=_default_candidate_fingerprint(), storage_policy="persistent",
        ownership_kind="managed", root=root)
    return lifecycle.manager._activate("llama.cpp", manifest, record_previous=False), root


def test_legacy_persistent_root_moves_to_flat_path_when_destination_absent(lifecycle):
    legacy, root = _legacy_active(lifecycle)
    lifecycle.upstream.revision = OLD

    def probe(source, build_dir):
        return build(source, build_dir)

    handle = lifecycle.manager.prepare("llama.cpp",
        builder=lambda *_args: pytest.fail("successful probe should avoid rebuilding"),
        probe=probe, persistent=True)
    assert handle.revision == legacy.revision
    assert handle.source == lifecycle.manager.original_path("llama.cpp")
    assert handle.manifest["root"] == str(handle.source)
    assert not root.exists()
    assert not (lifecycle.manager.work_root / ".aqx-runtime-builds").exists()


def test_legacy_move_rebuilds_in_place_when_probe_fails(lifecycle):
    _, root = _legacy_active(lifecycle)
    lifecycle.upstream.revision = OLD
    built = []

    def rebuild(source, build_dir):
        built.append(source)
        return build(source, build_dir)

    handle = lifecycle.manager.prepare("llama.cpp", builder=rebuild,
        probe=lambda *_args: None, persistent=True)
    assert built == [lifecycle.manager.original_path("llama.cpp")]
    assert handle.revision == OLD
    assert handle.source == lifecycle.manager.original_path("llama.cpp")
    assert not root.exists()


def test_legacy_relocation_prompts_and_acceptance_updates_flat_path(lifecycle):
    legacy, root = _legacy_active(lifecycle)
    destination = lifecycle.manager.original_path("llama.cpp")
    destination.mkdir(parents=True)
    lifecycle.git.revisions[str(destination)] = OLD
    with pytest.raises(RuntimeDecisionRequired) as raised:
        lifecycle.manager.prepare("llama.cpp", builder=build, persistent=True)
    details = raised.value.details
    assert details["comparison"] == "relocate"
    assert details["target"]["revision"] == OLD
    assert details["persistent_root"] == str(destination)
    assert details["legacy_root"] == str(root)
    declined = lifecycle.manager.prepare("llama.cpp", builder=build,
        persistent=True, update=False)
    assert declined.source == legacy.source
    assert root.exists()
    handle = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=True, update=True)
    assert handle.source == destination
    assert handle.revision == OLD
    assert not root.exists()


def test_adopted_original_is_never_cloned_or_modified(lifecycle):
    source = lifecycle.manager.original_path("llama.cpp")
    source.mkdir(parents=True)
    lifecycle.git.revisions[str(source)] = OLD
    output = build(source, source / "build")
    handle = lifecycle.manager.prepare("llama.cpp", builder=lambda *args: pytest.fail("adopted rebuild"),
                                      probe=lambda *args: output, update=False)
    assert handle.source == source
    assert handle.manifest["ownership_kind"] == "external-adopted"
    assert not any(args[1] == "clone" or "checkout" in args for args in lifecycle.git.calls)


def test_checkout_with_wrong_origin_is_not_adopted_as_original(lifecycle):
    source = lifecycle.manager.original_path("llama.cpp")
    source.mkdir(parents=True)
    lifecycle.git.revisions[str(source)] = OLD
    lifecycle.git.origins[str(source)] = "https://github.com/other/llama.cpp.git"
    with pytest.raises(RuntimeErrorBase, match="does not match original"):
        lifecycle.manager.adopt("llama.cpp", source, build)


def test_proxy_rewrite_does_not_change_original_repository_check(lifecycle):
    source = lifecycle.manager.original_path("vision.cpp")
    source.mkdir(parents=True)
    lifecycle.git.revisions[str(source)] = OLD
    lifecycle.git.origins[str(source)] = "https://github.com/Acly/vision.cpp.git"
    lifecycle.git.effective_origins[str(source)] = (
        "https://git-manager.devin.ai/proxy/github.com/Acly/vision.cpp.git")

    assert lifecycle.manager._check_original("vision.cpp", source) == OLD


def test_binary_resource_and_source_drift_are_detected(lifecycle):
    handle = installed(lifecycle)
    resource = Path(handle.manifest["resources"]["converter"]["path"])
    resource.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeErrorBase, match="changed or missing"):
        lifecycle.manager.verify(handle)


def test_temporary_cleanup_respects_leases_and_preserves_manifest(lifecycle):
    handle = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=False)
    assert handle.source.parent == lifecycle.manager.temp_root
    assert handle.source.name.startswith(f"llama.cpp-{handle.revision[:12]}-")
    assert handle.build_dir == handle.source / "build"
    assert (handle.source / "aqx-manifest.json").is_file()
    assert (handle.source / "aqx-setup.json").is_file()
    lease = lifecycle.manager.lease(handle)
    assert not lifecycle.manager.cleanup_temporary(handle)
    lifecycle.manager.release(lease)
    assert lifecycle.manager.cleanup_temporary(handle)
    assert not handle.source.exists()
    assert (lifecycle.manager.state_root / "llama.cpp" / (handle.identity + ".json")).is_file()


def test_session_checks_once_uses_context_and_cleans_successful_temp(lifecycle):
    with RuntimeSession(lifecycle.manager, decide=lambda details: {"persistent": False}) as session:
        first = session.ensure("llama.cpp", builder=build)
        count = len(lifecycle.upstream.calls)
        assert session.ensure("llama.cpp", builder=build).identity == first.identity
        assert len(lifecycle.upstream.calls) == count
        assert current_session() is session
        assert selected_source("llama.cpp", Path("fallback")) == first.source
    assert current_session() is None
    assert not first.source.exists()


def test_failed_session_retains_temporary_runtime_for_resume(lifecycle):
    with RuntimeSession(lifecycle.manager, decide=lambda details: {"persistent": False}) as session:
        handle = session.ensure("llama.cpp", builder=build)
        session.mark_failed()
    assert handle.source.is_dir()


def test_failed_temporary_build_writes_failure_and_keeps_flat_root(lifecycle):
    def fail(*_args):
        raise RuntimeError("compiler failed")

    with pytest.raises(RuntimeErrorBase, match="aqx-failure.json"):
        lifecycle.manager.prepare("llama.cpp", builder=fail, persistent=False)
    roots = list(lifecycle.manager.temp_root.glob("llama.cpp-*"))
    assert len(roots) == 1
    assert roots[0].is_dir()
    assert (roots[0] / "aqx-failure.json").is_file()


def test_additional_required_target_builds_same_sha_without_updating(lifecycle):
    first = installed(lifecycle)

    def extended(source, build_dir):
        output = build(source, build_dir)
        extra = build_dir / "tts.exe"
        extra.write_bytes(b"tts runtime")
        output["binaries"]["tts"] = extra
        return output

    handle = lifecycle.manager.prepare("llama.cpp", builder=extended, update=False,
                                      required_binaries=["tts"])
    assert handle.revision == first.revision and "tts" in handle.binaries
    assert handle.source == first.source


def test_prune_never_removes_flat_persistent_install(lifecycle):
    handle = installed(lifecycle)
    result = lifecycle.manager.prune("llama.cpp")
    assert str(handle.source) not in result["candidates"]
    lifecycle.manager.prune("llama.cpp", dry_run=False)
    assert handle.source.is_dir()


def test_migration_temporary_adopts_legacy_build_without_deleting_it(lifecycle):
    source = lifecycle.manager.temp_root / "aqx-media-backends/old/source"
    source.mkdir(parents=True)
    lifecycle.git.revisions[str(source)] = OLD
    output = build(source, source.parent / "build")
    output["build_dir"] = str(source.parent / "build")
    handle = lifecycle.manager.migrate("llama.cpp", source, builder=build,
        probe=lambda *args: output, persistent=False, legacy_build_dir=source.parent / "build")
    assert handle.manifest["storage_policy"] == "temporary"
    assert handle.manifest["ownership_kind"] == "external-adopted"
    assert handle.build_dir == source.parent / "build"
    assert not lifecycle.manager.cleanup_temporary(handle)
    assert source.is_dir()


def test_migration_copy_preserves_legacy_source_and_rebuilds_bound_cache(lifecycle):
    source = lifecycle.manager.temp_root / "aqx-media-backends/old/source"
    source.mkdir(parents=True)
    lifecycle.git.revisions[str(source)] = OLD
    legacy_build = source.parent / "build"
    build(source, legacy_build)
    (legacy_build / "CMakeCache.txt").write_text("SOURCE=" + str(source), encoding="utf-8")
    calls = []

    def rebuilt(candidate, build_dir):
        calls.append(candidate)
        # FakeGit uses an in-memory map rather than a real copied .git dir.
        lifecycle.git.revisions[str(candidate)] = OLD
        return build(candidate, build_dir)

    handle = lifecycle.manager.migrate("llama.cpp", source, builder=rebuilt,
        probe=lambda *args: None, persistent=True, legacy_build_dir=legacy_build)
    assert len(calls) == 1
    assert handle.source == lifecycle.manager.original_path("llama.cpp")
    assert source.is_dir() and legacy_build.is_dir()
    assert handle.manifest["ownership_kind"] == "managed"
    assert Path(handle.manifest["root"]) == lifecycle.manager.original_path("llama.cpp")


def test_persistent_migration_refuses_existing_original_path(lifecycle):
    source = lifecycle.manager.temp_root / "aqx-media-backends/old/source"
    source.mkdir(parents=True)
    destination = lifecycle.manager.original_path("llama.cpp")
    destination.mkdir(parents=True)
    with pytest.raises(RuntimeErrorBase, match="adopt or update the persistent llama.cpp checkout"):
        lifecycle.manager.migrate("llama.cpp", source, builder=build,
            probe=lambda *_args: None, persistent=True)


def test_new_install_clones_into_flat_persistent_root(lifecycle):
    handle = installed(lifecycle)
    assert handle.source == lifecycle.manager.original_path("llama.cpp")
    assert handle.build_dir == handle.source / "build"
    assert (lifecycle.manager.state_root / "llama.cpp" / "setup.json").is_file()
    assert not (handle.source / "setup.json").exists()
    assert not (lifecycle.manager.work_root / ".aqx-runtime-builds").exists()


def test_prune_removes_manifestless_legacy_failure_candidates(lifecycle):
    root = lifecycle.manager.work_root / ".aqx-runtime-builds" / "llama.cpp" / NEW / "failed-key"
    root.mkdir(parents=True)
    (root / "failure.json").write_text("failed", encoding="utf-8")
    result = lifecycle.manager.prune("llama.cpp", dry_run=False)
    assert str(root) in result["candidates"]
    assert not root.exists()
    assert not (lifecycle.manager.work_root / ".aqx-runtime-builds").exists()


def test_modified_manifest_and_changed_source_are_rejected(lifecycle):
    handle = installed(lifecycle)
    snapshot = handle.snapshot()
    snapshot["revision"] = OLD
    from agentquantix.runtimes import RuntimeHandle
    with pytest.raises(RuntimeErrorBase, match="manifest was modified"):
        lifecycle.manager.verify(RuntimeHandle.from_manifest(snapshot))
    lifecycle.git.dirty[str(handle.source)] = " M model.cpp"
    with pytest.raises(RuntimeErrorBase, match="source changes"):
        lifecycle.manager.verify(handle)


def test_cleanup_cannot_escape_the_managed_temporary_root(lifecycle):
    from agentquantix.runtimes import RuntimeHandle
    from agentquantix.runtimes.spec import canonical_hash
    handle = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=False)
    snapshot = handle.snapshot()
    snapshot["root"] = str(lifecycle.manager.work_root)
    snapshot.pop("identity")
    snapshot["identity"] = canonical_hash(snapshot)
    with pytest.raises(RuntimeErrorBase, match="outside"):
        lifecycle.manager.cleanup_temporary(RuntimeHandle.from_manifest(snapshot))


def test_no_local_runtime_and_offline_check_does_not_build_moving_ref(lifecycle):
    lifecycle.upstream.offline = True
    with pytest.raises(RuntimeErrorBase, match="no installed runtime"):
        lifecycle.manager.prepare("llama.cpp", builder=build, persistent=True)
    assert not lifecycle.git.calls


def test_frozen_temporary_runtime_reconstructs_exact_sha_without_activation(lifecycle):
    frozen = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=False)
    assert frozen.source.parent == lifecycle.manager.temp_root
    assert frozen.source.name.startswith(f"llama.cpp-{NEW[:12]}-")
    lifecycle.manager.cleanup_temporary(frozen)
    lifecycle.upstream.revision = OLD
    active = installed(lifecycle)
    resumed = lifecycle.manager.prepare("llama.cpp", builder=build, frozen=frozen.snapshot())
    assert resumed.identity == frozen.identity and resumed.revision == NEW
    assert resumed.source == frozen.source
    assert resumed.source.parent == lifecycle.manager.temp_root
    assert lifecycle.manager.inspect("llama.cpp").identity == active.identity


def test_changed_reconstructed_binary_cannot_reuse_frozen_approval(lifecycle):
    frozen = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=False)
    lifecycle.manager.cleanup_temporary(frozen)

    def different(source, build_dir):
        result = build(source, build_dir)
        result["binaries"]["native"].write_bytes(b"different compiler output")
        return result

    with pytest.raises(RuntimeErrorBase, match="approve a new runtime plan"):
        lifecycle.manager.prepare("llama.cpp", builder=different, frozen=frozen.snapshot())


def test_external_catalog_resource_survives_package_reinstall(lifecycle, tmp_path):
    catalog = tmp_path / "tool-virtualenv/catalog.json"
    catalog.parent.mkdir()
    catalog.write_text('{"families": ["sdxl"]}', encoding="utf-8")

    def with_catalog(source, build_dir):
        result = build(source, build_dir)
        result["resources"]["capability_catalog"] = {"path": catalog}
        return result

    handle = lifecycle.manager.prepare("llama.cpp", builder=with_catalog, persistent=True)
    retained = Path(handle.manifest["resources"]["capability_catalog"]["path"])
    assert lifecycle.manager.state_root in retained.parents
    catalog.unlink()
    assert lifecycle.manager.verify(handle) is handle


def test_explicit_persistent_install_promotes_temporary_original(lifecycle):
    temporary = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=False)
    regular = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=True, update=False)
    assert regular.manifest["storage_policy"] == "persistent"
    assert regular.revision == temporary.revision
    assert regular.source == lifecycle.manager.original_path("llama.cpp")
    assert temporary.source.is_dir()


def test_changed_build_configuration_is_not_silently_reused(lifecycle):
    first = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=True,
                                      build_fingerprint={"compute_backend": "cpu"})
    (first.build_dir / "old-config.txt").write_text("old", encoding="utf-8")
    second = lifecycle.manager.prepare("llama.cpp", builder=build, update=False,
                                       build_fingerprint={"compute_backend": "cuda"})
    assert second.source == first.source == lifecycle.manager.original_path("llama.cpp")
    assert second.revision == first.revision
    assert second.manifest["build_fingerprint"]["compute_backend"] == "cuda"
    assert not (second.build_dir / "old-config.txt").exists()


@pytest.mark.parametrize("persistent", [True, False])
def test_offline_complete_source_can_build_without_network_setup(lifecycle, persistent):
    source = lifecycle.manager.original_path("llama.cpp")
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("# tracked native project", encoding="utf-8")
    lifecycle.git.revisions[str(source)] = OLD
    lifecycle.upstream.offline = True
    with pytest.raises(RuntimeDecisionRequired) as raised:
        lifecycle.manager.prepare("llama.cpp", builder=build)
    assert raised.value.details["target"]["revision"] == OLD
    assert raised.value.details["target"]["channel"] == "local-cache"
    handle = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=persistent)
    assert handle.revision == OLD
    assert source.is_dir()
    if persistent:
        assert handle.source == source
    else:
        assert handle.source != source
    assert not any(args[1] == "clone" or "fetch" in args or "checkout" in args or "update" in args
                   for args in lifecycle.git.calls)
    assert handle.manifest["storage_policy"] == ("persistent" if persistent else "temporary")


def test_offline_source_with_uninitialized_submodules_cannot_build(lifecycle):
    source = lifecycle.manager.original_path("llama.cpp")
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("# native project", encoding="utf-8")
    lifecycle.git.revisions[str(source)] = OLD
    lifecycle.git.submodules[str(source)] = "-" + OLD + " thirdparty/ggml"
    lifecycle.upstream.offline = True
    with pytest.raises(RuntimeErrorBase, match="uninitialized or divergent submodules"):
        lifecycle.manager.prepare("llama.cpp", builder=lambda *args: pytest.fail("incomplete build"),
                                  persistent=True)


@pytest.mark.parametrize("error", [FileNotFoundError("git missing"),
                                  subprocess.TimeoutExpired(["git"], 300)])
def test_native_execution_setup_failures_are_actionable(monkeypatch, error):
    from agentquantix.runtimes.manager import _run

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeErrorBase, match="runtime command could not execute git"):
        _run(["git", "--version"])


def test_frozen_fork_cannot_be_treated_as_an_original(lifecycle):
    from agentquantix.runtimes import RuntimeHandle
    from agentquantix.runtimes.spec import canonical_hash
    handle = installed(lifecycle)
    snapshot = handle.snapshot()
    snapshot["origin_kind"] = "fork"
    snapshot.pop("identity")
    snapshot["identity"] = canonical_hash(snapshot)
    with pytest.raises(RuntimeErrorBase, match="designated original upstream"):
        lifecycle.manager.verify(RuntimeHandle.from_manifest(snapshot))


def test_source_setup_commands_are_reported_before_execution(lifecycle, capsys):
    handle = installed(lifecycle)

    progress = [line for line in capsys.readouterr().err.splitlines()
                if line.startswith("  .. llama.cpp: ")]
    setup = json.loads((lifecycle.manager.state_root / "llama.cpp" / "setup.json")
                       .read_text(encoding="utf-8"))

    assert len(progress) == 4
    assert progress[0] == "  .. llama.cpp: git clone --no-checkout..."
    assert len(setup["commands"]) == 4
    assert all(set(command) == {"argv", "returncode", "stdout", "stderr"}
               for command in setup["commands"])


def test_deleted_persistent_root_drops_record_instead_of_failing(lifecycle, tmp_path):
    handle = installed(lifecycle)
    state_path = lifecycle.manager._state("llama.cpp")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    gone = tmp_path / "work" / ".aqx-runtime-builds" / "llama.cpp" / "deleted"
    state["active"] = {**handle.manifest, "root": str(gone), "source": str(gone / "source")}
    state_path.write_text(json.dumps(state), encoding="utf-8")

    assert lifecycle.manager.inspect("llama.cpp") is None
    assert "active" not in json.loads(state_path.read_text(encoding="utf-8"))


def test_changed_files_in_existing_root_still_fail_verification(lifecycle):
    handle = installed(lifecycle)
    Path(handle.binaries["native"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeErrorBase, match="changed or missing"):
        lifecycle.manager.inspect("llama.cpp")
