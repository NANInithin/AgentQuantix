"""Execution session and existing pipeline integration; no native/network work."""
import json

import pytest

from agentquantix import config, media, media_catalog, runtime_builders, runtime_control
from agentquantix.pipeline import build, media_release, run, voice_release
from agentquantix.pipeline.job import Job
from agentquantix.runtimes import RuntimeCancelled, RuntimeDecisionRequired, RuntimeErrorBase, RuntimeHandle, current_session


SHA = "a" * 40


def runtime_handle(tmp_path, engine="llama.cpp", *, revision=SHA, identity=None, resources=None):
    names = {
        "llama.cpp": ("llama-quantize", "llama-imatrix", "llama-tts"),
        "audio.cpp": ("audiocpp_cli", "audiocpp_gguf"),
        "whisper.cpp": ("whisper-cli", "whisper-quantize"),
        "stable-diffusion.cpp": ("sd-cli",),
    }[engine]
    source, build_dir = tmp_path / engine / "source", tmp_path / engine / "build"
    manifest = {"engine": engine, "revision": revision, "identity": identity or engine + "-" + revision,
                "source": str(source), "build_dir": str(build_dir),
                "binaries": {name: {"path": str(build_dir / name), "sha256": "fake"} for name in names},
                "resources": resources or {}, "storage_policy": "temporary", "provenance": {"test": True}}
    return RuntimeHandle.from_manifest(manifest)


@pytest.fixture
def integration(tmp_path, monkeypatch):
    class Manager:
        def __init__(self):
            self.calls = []
            self.handles = {}
            self.offline = False

        def check_update(self, engine):
            self.calls.append(("check", engine))
            if self.offline:
                return {"status": "unavailable", "error": "offline"}
            return {"status": "current", "target": {"revision": SHA}}

        def prepare(self, engine, **kwargs):
            self.calls.append(("prepare", engine, kwargs))
            if kwargs.get("frozen"):
                return RuntimeHandle.from_manifest(kwargs["frozen"])
            handle = self.handles.setdefault(engine, runtime_handle(tmp_path, engine))
            handle.manifest.setdefault("build_fingerprint", dict(kwargs.get("build_fingerprint") or {}))
            return handle

        def inspect(self, engine, **kwargs):
            return self.handles.get(engine)

        def lease(self, handle):
            self.calls.append(("lease", handle.identity))
            return "lease-" + handle.identity

        def release(self, path):
            self.calls.append(("release", path))

        def cleanup_temporary(self, handle):
            self.calls.append(("cleanup", handle.identity))

    monkeypatch.setattr(runtime_builders, "fingerprint", lambda engine, *args, **kwargs:
                        {"compute_backend": "cpu", "required_targets": kwargs.get("required_targets"),
                         "targets": list(runtime_builders._TARGETS[engine]) + list(kwargs.get("required_targets") or [])})
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "WORK_ROOT", tmp_path / "runtime-root")
    return Manager()


def test_execution_session_deduplicates_runtime_checks_and_holds_lease(integration):
    with runtime_control.execution_session(manager=integration) as session:
        first = runtime_control.ensure_runtime("llama.cpp")
        assert runtime_control.ensure_runtime("llama.cpp") is first
        assert runtime_control.handle_for("llama.cpp") is first
        assert runtime_control.frozen_runtime("llama.cpp") == first.snapshot()
        with runtime_control.execution_session(manager=integration) as nested:
            assert nested is session
    assert current_session() is None
    assert [call[0] for call in integration.calls] == ["check", "prepare", "lease", "release", "cleanup"]


def test_failed_result_preserves_temporary_runtime_for_resume(integration):
    with runtime_control.execution_session(manager=integration):
        runtime_control.ensure_runtime("llama.cpp")
        runtime_control.mark_failed()
    assert any(call[0] == "release" for call in integration.calls)
    assert not any(call[0] == "cleanup" for call in integration.calls)


def test_exception_preserves_temporary_runtime_and_resets_context(integration):
    with pytest.raises(ValueError):
        with runtime_control.execution_session(manager=integration):
            runtime_control.ensure_runtime("llama.cpp")
            raise ValueError("conversion failed")
    assert current_session() is None
    assert not any(call[0] == "cleanup" for call in integration.calls)


def test_frozen_resume_keeps_old_revision_and_still_checks_update(integration, tmp_path):
    frozen = runtime_handle(tmp_path, revision="b" * 40).snapshot()
    with runtime_control.execution_session(manager=integration):
        handle = runtime_control.ensure_runtime("llama.cpp", frozen=frozen)
        assert handle.revision == "b" * 40
        assert integration.calls[0] == ("check", "llama.cpp")
        assert integration.calls[1][2]["frozen"] == frozen


def test_conflicting_frozen_job_cannot_replace_session_runtime(integration, tmp_path):
    with runtime_control.execution_session(manager=integration):
        runtime_control.ensure_runtime("llama.cpp")
        with pytest.raises(RuntimeErrorBase, match="frozen"):
            runtime_control.ensure_runtime("llama.cpp", frozen=runtime_handle(tmp_path, revision="b" * 40).snapshot())


def test_offline_check_reports_installed_revision(integration, capsys):
    integration.offline = True
    with runtime_control.execution_session(manager=integration):
        handle = runtime_control.ensure_runtime("llama.cpp")
    message = capsys.readouterr().err
    assert "update check unavailable" in message and handle.revision in message


def test_noninteractive_session_returns_runtime_decision_without_prompt(integration, monkeypatch):
    def prepare(*args, **kwargs):
        raise RuntimeDecisionRequired({"kind": "storage", "engine": "llama.cpp"})
    monkeypatch.setattr(integration, "prepare", prepare)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("noninteractive session prompted"))
    with runtime_control.execution_session(manager=integration):
        with pytest.raises(RuntimeDecisionRequired) as error:
            runtime_control.ensure_runtime("llama.cpp")
        assert error.value.details["kind"] == "storage"


def test_cli_readonly_runtime_list_does_not_prepare_runtime(monkeypatch):
    from agentquantix import cli
    from agentquantix.agent import runtime_tools
    calls = []
    monkeypatch.setattr(config, "migrate_legacy_state", lambda: [])
    monkeypatch.setattr(runtime_builders, "fingerprint", lambda *args, **kwargs: pytest.fail("read-only command prepared runtime"))
    monkeypatch.setattr(runtime_tools, "call", lambda name, arguments: calls.append((name, arguments)) or {"runtimes": []})
    assert cli.main(["runtime", "list"]) == 0
    assert calls == [("get_runtime_status", {"engine": None})]


def test_low_level_legacy_call_has_no_lifecycle_side_effects(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_builders, "fingerprint", lambda *args, **kwargs: pytest.fail("legacy call prepared runtime"))
    assert runtime_control.ensure_runtime("llama.cpp") is None
    source = tmp_path / "llama.cpp"
    source.mkdir()
    monkeypatch.setattr(config, "UPSTREAM_LLAMA", source)
    monkeypatch.setattr(build, "binaries", lambda path: (path / "quantize", path / "imatrix"))
    assert build.ensure_upstream() == (source, source / "quantize", source / "imatrix")


@pytest.mark.parametrize("engine,call,names", [
    ("llama.cpp", build.ensure_upstream, ("llama-quantize", "llama-imatrix")),
    ("audio.cpp", voice_release.ensure_audiocpp_tools, ("audiocpp_cli", "audiocpp_gguf")),
    ("whisper.cpp", voice_release.ensure_whisper_tools, ("whisper-cli", "whisper-quantize")),
])
def test_native_helpers_use_selected_immutable_handle(integration, engine, call, names):
    with runtime_control.execution_session(manager=integration):
        source, *binaries = call()
        handle = runtime_control.handle_for(engine)
        assert source == handle.source
        assert binaries == [handle.binaries[name] for name in names]


def test_media_helper_returns_selected_binary_and_provenance(integration):
    with runtime_control.execution_session(manager=integration):
        binary, provenance = media_release.ensure_backend("cpu")
        handle = runtime_control.handle_for("stable-diffusion.cpp")
        assert binary == handle.binaries["sd-cli"]
        assert provenance["engine_revision"] == handle.revision
        assert provenance["runtime_snapshot"] == handle.snapshot()


def test_media_catalog_follows_selected_runtime_and_rejects_wrong_revision(integration, tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "revision": SHA}), encoding="utf-8")
    handle = runtime_handle(tmp_path, "stable-diffusion.cpp", resources={"capability_catalog": {
        "path": str(catalog_path), "sha256": media.file_hash(catalog_path)}})
    integration.handles["stable-diffusion.cpp"] = handle
    with runtime_control.execution_session(manager=integration):
        runtime_control.ensure_runtime("stable-diffusion.cpp")
        assert media_catalog.selected_catalog_path() == catalog_path
        assert media_catalog.load_catalog()["revision"] == SHA
        catalog_path.write_text(json.dumps({"schema_version": 1, "revision": "b" * 40}), encoding="utf-8")
        with pytest.raises((ValueError, RuntimeErrorBase), match="pinned engine|changed|hash|integrity"):
            media_catalog.load_catalog()
    assert media_catalog.selected_catalog_path() == media_catalog.CATALOG_PATH


@pytest.mark.parametrize("with_status,target,expected", [
    (True, "owner/Demo-GGUF", True),
    (False, "owner/Demo-GGUF", False),
    (True, "other/Demo-GGUF", False),
])
def test_job_reuses_frozen_runtime_only_for_matching_resumable_work(tmp_path, monkeypatch, with_status, target, expected):
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    frozen = runtime_handle(tmp_path).snapshot()
    previous = Job("org/Demo", "Demo", "owner/Demo-GGUF", ["Q4_0"], runtime_snapshot=frozen)
    previous.save()
    if with_status:
        previous.status_file.parent.mkdir(parents=True)
        previous.status_file.write_text("{}", encoding="utf-8")
    assessment = {"repo_id": "org/Demo", "target_repo": target, "quants": ["Q4_0"], "arch_ok": True}
    resumed = Job.from_assessment(assessment)
    assert bool(resumed.runtime_snapshot) is expected
    if expected:
        assert resumed.runtime_snapshot == frozen


def test_failed_text_orchestrator_marks_session_failed_and_records_runtime(integration, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "TOKEN", "offline-test")
    from agentquantix import archsupport
    monkeypatch.setattr(archsupport, "supported_quants", lambda source: ["Q4_0"])
    monkeypatch.setattr(run, "process", lambda *args, **kwargs: {"failures": [("Q4_0", "offline failure")]})
    job = Job("org/Demo", "Demo", "owner/Demo-GGUF", ["Q4_0"])
    with runtime_control.execution_session(manager=integration):
        result = run.run_jobs([job])
        assert current_session().failed
        assert result["incomplete"] == ["Demo"]
    assert not any(call[0] == "cleanup" for call in integration.calls)
    assert Job.load(job.record_file).runtime_snapshot["revision"] == SHA


def test_plain_text_runtime_does_not_require_tts_target(integration):
    with runtime_control.execution_session(manager=integration):
        build.ensure_upstream()
    options = next(call[2] for call in integration.calls if call[0] == "prepare")
    assert set(options["required_binaries"]) == {"llama-quantize", "llama-imatrix"}
    assert "llama-tts" not in options["build_fingerprint"]["targets"]


def test_llama_voice_helper_explicitly_requires_tts_target(integration):
    with runtime_control.execution_session(manager=integration):
        _, runtime, quantizer = voice_release.ensure_llama_tts()
    options = next(call[2] for call in integration.calls if call[0] == "prepare")
    assert "llama-tts" in options["required_binaries"]
    assert runtime.name == "llama-tts" and quantizer.name == "llama-quantize"


def test_voice_preview_selects_conversion_and_scoring_runtimes_without_persisting(integration):
    repo = "org/Voice"
    record = voice_release.work_dir(repo) / "runtime-snapshots.json"
    with runtime_control.execution_session(manager=integration):
        snapshots = runtime_control.prepare_voice_runtimes(repo, "audio.cpp", "tts", persist=False)
        assert set(snapshots) == {"audio.cpp", "whisper.cpp"}
        assert not record.exists()
    assert [call[1] for call in integration.calls if call[0] == "check"] == ["audio.cpp", "whisper.cpp"]


def test_voice_execution_records_both_runtime_snapshots_and_resume_keeps_them(integration, tmp_path):
    repo = "org/Voice"
    with runtime_control.execution_session(manager=integration):
        original = runtime_control.prepare_voice_runtimes(repo, "audio.cpp", "tts", persist=True)
    record = voice_release.work_dir(repo) / "runtime-snapshots.json"
    assert json.loads(record.read_text(encoding="utf-8")) == original
    integration.calls.clear()
    integration.handles = {engine: runtime_handle(tmp_path, engine, revision="b" * 40)
                           for engine in ("audio.cpp", "whisper.cpp")}
    with runtime_control.execution_session(manager=integration):
        resumed = runtime_control.prepare_voice_runtimes(repo, "audio.cpp", "tts", persist=True)
        assert resumed == original
        assert runtime_control.handle_for("audio.cpp").revision == SHA
        assert runtime_control.handle_for("whisper.cpp").revision == SHA
    for call in integration.calls:
        if call[0] == "prepare":
            assert call[2]["frozen"] == original[call[1]]


def test_asr_runtime_selection_does_not_add_unrelated_engine(integration):
    with runtime_control.execution_session(manager=integration):
        snapshots = runtime_control.prepare_voice_runtimes("org/Whisper", "whisper.cpp", "asr")
    assert set(snapshots) == {"whisper.cpp"}
    assert [call[1] for call in integration.calls if call[0] == "check"] == ["whisper.cpp"]


@pytest.mark.parametrize("approve", [False, True])
def test_frozen_resume_update_is_future_only_and_does_not_replace_handle(integration, tmp_path, monkeypatch, approve):
    frozen = runtime_handle(tmp_path, revision=SHA).snapshot()
    target = {"revision": "b" * 40, "ref": "v2", "channel": "stable"}
    future = runtime_handle(tmp_path, revision=target["revision"])
    decisions = []

    def check_update(engine):
        integration.calls.append(("check", engine))
        return {"status": "available", "target": target, "comparison": "ahead"}

    def prepare(engine, **kwargs):
        integration.calls.append(("prepare", engine, kwargs))
        if kwargs.get("frozen"):
            return RuntimeHandle.from_manifest(kwargs["frozen"])
        assert kwargs["update"] is True and kwargs["revision"] == target
        integration.handles[engine] = future
        return future

    def decide(details):
        decisions.append(details)
        assert details["future_only"] is True
        return {"update": approve}

    monkeypatch.setattr(integration, "check_update", check_update)
    monkeypatch.setattr(integration, "prepare", prepare)
    with runtime_control.execution_session(manager=integration, decide=decide):
        selected = runtime_control.ensure_runtime("llama.cpp", frozen=frozen)
        assert selected.revision == SHA
        assert runtime_control.handle_for("llama.cpp").identity == frozen["identity"]
        runtime_control.ensure_runtime("llama.cpp", frozen=frozen)
        assert runtime_control.handle_for("llama.cpp").revision == SHA
    assert len(decisions) == 1
    assert decisions[0]["target"] == target
    future_builds = [call for call in integration.calls if call[0] == "prepare" and not call[2].get("frozen")]
    assert len(future_builds) == int(approve)
    assert [call[1] for call in integration.calls if call[0] == "lease"] == [frozen["identity"]]


@pytest.mark.parametrize("accept_fallback", [False, True])
def test_failed_update_uses_previous_installation_only_after_fallback_decision(integration, tmp_path, monkeypatch, accept_fallback):
    installed = runtime_handle(tmp_path)
    integration.handles["llama.cpp"] = installed
    target = {"revision": "b" * 40}
    decisions = []

    def check_update(engine):
        integration.calls.append(("check", engine))
        return {"status": "available", "target": target}

    def prepare(engine, **kwargs):
        integration.calls.append(("prepare", engine, kwargs))
        if kwargs.get("update") is None:
            raise RuntimeDecisionRequired({"kind": "update", "engine": engine, "target": target})
        assert kwargs["update"] is True
        raise RuntimeErrorBase("candidate compiler failed; installed runtime preserved")

    def decide(details):
        decisions.append(details["kind"])
        if details["kind"] == "update":
            return {"update": True}
        assert details["kind"] == "fallback"
        assert details["installed_revision"] == installed.revision
        return {"use_installed": accept_fallback}

    monkeypatch.setattr(integration, "check_update", check_update)
    monkeypatch.setattr(integration, "prepare", prepare)
    with runtime_control.execution_session(manager=integration, decide=decide):
        if accept_fallback:
            handle = runtime_control.ensure_runtime("llama.cpp")
            assert handle is installed
        else:
            with pytest.raises(RuntimeErrorBase, match="compiler failed"):
                runtime_control.ensure_runtime("llama.cpp")
            assert runtime_control.handle_for("llama.cpp") is None
    assert decisions == ["update", "fallback"]
    assert any(call[0] == "lease" for call in integration.calls) is accept_fallback


def test_failed_update_without_decision_policy_does_not_silently_fallback(integration, tmp_path, monkeypatch):
    integration.handles["llama.cpp"] = runtime_handle(tmp_path)
    monkeypatch.setattr(integration, "prepare", lambda *args, **kwargs:
                        (_ for _ in ()).throw(RuntimeErrorBase("candidate failed")))
    with runtime_control.execution_session(manager=integration):
        with pytest.raises(RuntimeErrorBase, match="candidate failed"):
            runtime_control.ensure_runtime("llama.cpp")
        assert runtime_control.handle_for("llama.cpp") is None


def test_failed_future_update_keeps_resumed_job_on_frozen_runtime(integration, tmp_path, monkeypatch, capsys):
    frozen = runtime_handle(tmp_path).snapshot()
    target = {"revision": "b" * 40}
    decisions = []

    def prepare(engine, **kwargs):
        integration.calls.append(("prepare", engine, kwargs))
        if kwargs.get("frozen"):
            return RuntimeHandle.from_manifest(kwargs["frozen"])
        raise RuntimeErrorBase("future candidate compiler failed")

    def decide(details):
        decisions.append(details)
        return {"update": True}

    monkeypatch.setattr(integration, "check_update", lambda engine: {"status": "available", "target": target})
    monkeypatch.setattr(integration, "prepare", prepare)
    with runtime_control.execution_session(manager=integration, decide=decide):
        handle = runtime_control.ensure_runtime("llama.cpp", frozen=frozen)
        assert handle.identity == frozen["identity"]
        assert runtime_control.handle_for("llama.cpp").revision == SHA
    assert decisions[0]["future_only"] is True
    error = capsys.readouterr().err
    assert "future candidate compiler failed" in error


def test_future_update_without_decision_policy_only_reports_available_update(integration, tmp_path, monkeypatch, capsys):
    frozen = runtime_handle(tmp_path).snapshot()
    monkeypatch.setattr(integration, "check_update", lambda engine:
                        {"status": "available", "target": {"revision": "b" * 40}})
    with runtime_control.execution_session(manager=integration):
        handle = runtime_control.ensure_runtime("llama.cpp", frozen=frozen)
        assert handle.identity == frozen["identity"]
    prepared = [call for call in integration.calls if call[0] == "prepare"]
    assert len(prepared) == 1 and prepared[0][2]["frozen"] == frozen
    assert "resumed job keeps" in capsys.readouterr().err


def test_mcp_text_resume_resolves_saved_runtime_before_selecting_installed_newer_runtime(integration, tmp_path, monkeypatch):
    from agentquantix.agent import tools
    frozen = runtime_handle(tmp_path).snapshot()
    installed = runtime_handle(tmp_path, revision="b" * 40)
    integration.handles["llama.cpp"] = installed
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    previous = Job("org/Demo", "Demo", "owner/Demo-GGUF", ["Q4_0"], runtime_snapshot=frozen)
    previous.save()
    previous.status_file.parent.mkdir(parents=True)
    previous.status_file.write_text("{}", encoding="utf-8")
    assessment = {"repo_id": "org/Demo", "target_repo": "owner/Demo-GGUF", "quants": ["Q4_0"],
                  "arch_ok": True, "verdict": "ok", "hours": {"total": 1}}
    monkeypatch.setattr(tools, "_assessments_for", lambda models: [assessment])
    monkeypatch.setattr(tools.hub, "forget_our_repos", lambda: None)
    captured = []

    def run_jobs(jobs, **kwargs):
        captured.extend(jobs)
        assert jobs[0].runtime_snapshot == frozen
        assert runtime_control.handle_for("llama.cpp").identity == frozen["identity"]
        return {"hours": 0, "incomplete": ["Demo"],
                "jobs": {"Demo": {"failures": [("<model>", "test stops before publication")]}}}

    monkeypatch.setattr(tools.run_mod, "run_jobs", run_jobs)
    with runtime_control.execution_session(manager=integration):
        tools.call("start_quantization", {"models": ["org/Demo"], "user_approved": True})
    assert len(captured) == 1
    prepared = [call for call in integration.calls if call[0] == "prepare"]
    assert prepared[0][2]["frozen"] == frozen
    assert not any(call[0] == "lease" and call[1] == installed.identity for call in integration.calls)


def test_user_runtime_cancel_propagates_without_installed_fallback(integration, tmp_path, monkeypatch):
    integration.handles["llama.cpp"] = runtime_handle(tmp_path)
    decisions = []

    def prepare(*args, **kwargs):
        raise RuntimeDecisionRequired({"kind": "update", "engine": "llama.cpp"})

    def decide(details):
        decisions.append(details["kind"])
        return {"cancel": True}

    monkeypatch.setattr(integration, "prepare", prepare)
    with runtime_control.execution_session(manager=integration, decide=decide):
        with pytest.raises(RuntimeCancelled, match="cancel"):
            runtime_control.ensure_runtime("llama.cpp")
        assert runtime_control.handle_for("llama.cpp") is None
    assert decisions == ["update"]
    assert not any(call[0] == "lease" for call in integration.calls)


def test_readonly_media_plan_uses_installed_catalog_without_runtime_preparation(tmp_path, monkeypatch):
    from agentquantix.runtimes import RuntimeManager
    catalog = json.loads(media_catalog.CATALOG_PATH.read_text(encoding="utf-8"))
    catalog["revision"] = "b" * 40
    installed_catalog = tmp_path / "installed-catalog.json"
    installed_catalog.write_text(json.dumps(catalog), encoding="utf-8")
    handle = runtime_handle(tmp_path, "stable-diffusion.cpp", revision="b" * 40,
        resources={"capability_catalog": {"path": str(installed_catalog), "sha256": media.file_hash(installed_catalog)}})
    inspections = []

    def inspect(self, engine, **kwargs):
        inspections.append((engine, kwargs))
        return handle if engine == "stable-diffusion.cpp" else None

    monkeypatch.setattr(RuntimeManager, "inspect", inspect)
    monkeypatch.setattr(runtime_control, "ensure_runtime", lambda *args, **kwargs:
                        pytest.fail("read-only plan prepared runtime"))
    monkeypatch.setattr(runtime_builders, "build_runtime", lambda *args, **kwargs:
                        pytest.fail("read-only plan built runtime"))
    monkeypatch.setattr(media_release, "converter_types", lambda **kwargs: {"types": ["q4_0"], "probed": False})
    source = {"repo_id": "org/Synthetic", "revision": "c" * 40, "pipeline_tag": "text-to-image",
              "library_name": "diffusers", "tags": [], "license": "apache-2.0",
              "configs": {"model_index.json": {"_class_name": "FluxPipeline"}},
              "files": {"model.safetensors": {"bytes": 8192, "sha256": "d" * 64}}}
    contract = {"task": "text-to-image", "required_roles": ["diffusion_model"],
                "evidence": {"native_recipe": "docs/flux.md"},
                "components": [{"role": "diffusion_model", "source_path": "model.safetensors", "quantizable": True}]}
    plan = media.plan_release(source["repo_id"], metadata=source, contract=contract, family="flux",
                              target_repo="user/Synthetic-GGUF", quants=["q4_0"], hunt_forks=False)
    assert plan["engine_revision"] == handle.revision
    assert plan["capability_manifest_sha256"] == media.file_hash(installed_catalog)
    assert plan["runtime_snapshot"] == handle.snapshot()
    assert inspections and all(kwargs.get("verify") is False for _, kwargs in inspections)
    assert not plan["blockers"]


@pytest.mark.parametrize("tampered", [False, True])
def test_media_run_validates_blocked_or_modified_plan_before_runtime_selection(tmp_path, monkeypatch, tampered):
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "work")
    monkeypatch.setattr(runtime_control, "ensure_runtime", lambda *args, **kwargs:
                        pytest.fail("invalid media plan selected/built runtime"))
    plan = {"engine_revision": media_catalog.ENGINE_REVISION,
            "capability_manifest_sha256": media.file_hash(media_catalog.CATALOG_PATH),
            "fixture_sha256": media.fixture_hashes(), "compute_backend": "cpu",
            "state": "blocked", "blockers": ["family unresolved"]}
    plan["plan_id"] = media.canonical_hash(plan)
    if tampered:
        plan["compute_backend"] = "cuda"
    with pytest.raises(media.MediaValidationError, match="modified|blocked"):
        media_release.run_release(plan, user_approved=True)
    assert not (config.TEMP_DIR / "aqx-media-runs").exists()
