"""Offline lifecycle tests: user decisions, revision pins and failure isolation."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import urllib.error

import pytest

from agentquantix.runtimes import (RUNTIMES, RuntimeDecisionRequired,
                                   RuntimeErrorBase, RuntimeManager, RuntimeSession,
                                   current_session, selected_source)


OLD = "a" * 40
NEW = "b" * 40


class FakeGit:
    def __init__(self):
        self.revisions = {}
        self.calls = []
        self.origins = {}
        self.dirty = {}
        self.submodules = {}

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
        elif args[:3] == ["remote", "get-url", "origin"]:
            output = self.origins.get(source, "https://github.com/ggml-org/llama.cpp.git")
        elif args[:1] == ["status"]:
            output = self.dirty.get(source, "")
        elif args[:2] == ["submodule", "status"]:
            output = self.submodules.get(source, "")
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


def test_registry_has_all_designated_originals():
    assert len(RUNTIMES) == 11
    assert RUNTIMES["ultralytics-ggml"].repository == "Asher-1/ultralytics-ggml"


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
    assert not lifecycle.git.calls


@pytest.mark.parametrize("persistent", [True, False])
def test_original_build_storage_policy_and_exact_checkout(lifecycle, persistent):
    handle = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=persistent)
    expected = lifecycle.manager.work_root if persistent else lifecycle.manager.temp_root
    assert expected in handle.source.parents
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


def test_failed_candidate_preserves_active_version_and_logs_failure(lifecycle):
    old = installed(lifecycle)
    lifecycle.upstream.revision = OLD

    def fail(*args):
        raise RuntimeError("compiler failed")

    with pytest.raises(RuntimeErrorBase, match="active installation preserved"):
        lifecycle.manager.prepare("llama.cpp", builder=fail, update=True)
    assert lifecycle.manager.inspect("llama.cpp").identity == old.identity
    logs = list(lifecycle.manager.work_root.rglob("failure.json"))
    assert len(logs) == 1
    assert json.loads(logs[0].read_text())["previous_identity"] == old.identity


def test_successful_update_and_rollback_are_side_by_side(lifecycle):
    first = installed(lifecycle)
    lifecycle.upstream.revision = OLD
    second = lifecycle.manager.prepare("llama.cpp", builder=build, update=True)
    assert first.source != second.source and first.source.is_dir()
    assert lifecycle.manager.rollback("llama.cpp").identity == first.identity


def test_frozen_resume_does_not_follow_active_runtime(lifecycle):
    first = installed(lifecycle)
    lifecycle.upstream.revision = OLD
    installed(lifecycle, update=True)
    handle = lifecycle.manager.prepare("llama.cpp", builder=build, frozen=first.snapshot())
    assert handle.identity == first.identity
    assert lifecycle.manager.inspect("llama.cpp").revision == OLD


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


def test_binary_resource_and_source_drift_are_detected(lifecycle):
    handle = installed(lifecycle)
    resource = Path(handle.manifest["resources"]["converter"]["path"])
    resource.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeErrorBase, match="changed or missing"):
        lifecycle.manager.verify(handle)


def test_temporary_cleanup_respects_leases_and_preserves_manifest(lifecycle):
    handle = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=False)
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
    assert handle.source != first.source


def test_prune_preserves_active_previous_and_external_roots(lifecycle):
    first = installed(lifecycle)
    lifecycle.upstream.revision = OLD
    second = installed(lifecycle, update=True)
    lifecycle.upstream.revision = "c" * 40
    third = installed(lifecycle, update=True)
    result = lifecycle.manager.prune("llama.cpp")
    assert str(first.manifest["root"]) in result["candidates"]
    assert first.source.is_dir()
    lifecycle.manager.prune("llama.cpp", dry_run=False)
    assert not first.source.exists()
    assert second.source.is_dir() and third.source.is_dir()


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
    assert lifecycle.manager.work_root in handle.source.parents
    assert source.is_dir() and legacy_build.is_dir()
    assert handle.manifest["ownership_kind"] == "managed"


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
    lifecycle.manager.cleanup_temporary(frozen)
    lifecycle.upstream.revision = OLD
    active = installed(lifecycle)
    resumed = lifecycle.manager.prepare("llama.cpp", builder=build, frozen=frozen.snapshot())
    assert resumed.identity == frozen.identity and resumed.revision == NEW
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
    assert lifecycle.manager.work_root in regular.source.parents
    assert temporary.source.is_dir()


def test_changed_build_configuration_is_not_silently_reused(lifecycle):
    first = lifecycle.manager.prepare("llama.cpp", builder=build, persistent=True,
                                      build_fingerprint={"compute_backend": "cpu"})
    second = lifecycle.manager.prepare("llama.cpp", builder=build, update=False,
                                       build_fingerprint={"compute_backend": "cuda"})
    assert second.source != first.source
    assert second.revision == first.revision
    assert second.manifest["build_fingerprint"]["compute_backend"] == "cuda"


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
    assert handle.source != source and source.is_dir()
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
