"""Runtime-specific approval and CLI/MCP parity without network/native builds."""
import argparse
import json
from types import SimpleNamespace

import pytest

from agentquantix import runtime_builders, runtime_cli
from agentquantix.agent import runtime_tools
from agentquantix.runtimes import RuntimeDecisionRequired


SHA = "a" * 40
TARGET = {"revision": SHA, "ref": "v1.0", "channel": "stable", "repository": "ggml-org/llama.cpp"}


@pytest.fixture
def surface(monkeypatch):
    calls = []
    handle = SimpleNamespace(snapshot=lambda: {"engine": "llama.cpp", "revision": SHA,
                                               "storage_policy": "persistent"})

    class Manager:
        installed = False

        def inspect(self, engine):
            calls.append(("inspect", engine))
            return handle if self.installed else None

        def check_update(self, engine):
            calls.append(("check", engine))
            return {"engine": engine, "status": "available", "target": dict(TARGET)}

        def prepare(self, engine, **kwargs):
            calls.append(("prepare", engine, kwargs))
            return handle

        def rollback(self, engine):
            calls.append(("rollback", engine))
            return handle

        def adopt(self, engine, source, probe):
            calls.append(("adopt", engine, str(source)))
            return handle

        def prune(self, engine, *, dry_run=True):
            calls.append(("prune", engine, dry_run))
            return {"engine": engine, "dry_run": dry_run, "candidates": [], "protected": []}

    manager = Manager()
    monkeypatch.setattr(runtime_tools, "RuntimeManager", lambda: manager)
    monkeypatch.setattr(runtime_builders, "fingerprint", lambda *args, **kwargs:
                        {"compute_backend": "cpu", "targets": ["llama-quantize", "llama-imatrix"]})
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("MCP action read terminal input"))
    return SimpleNamespace(calls=calls, manager=manager, handle=handle)


@pytest.mark.parametrize("name,args", [
    ("prepare_runtime", {"persistent": False}),
    ("prepare_runtime", {"persistent": False, "user_approved": False}),
    ("adopt_runtime", {"source": "somewhere"}),
    ("rollback_runtime", {}),
    ("prune_runtime", {"dry_run": False}),
])
def test_runtime_mutation_requires_its_own_approval(surface, name, args):
    with pytest.raises(ValueError, match="user_approved"):
        runtime_tools.call(name, {"engine": "llama.cpp", **args})
    assert surface.calls == []


@pytest.mark.parametrize("dry_run", [None, "false", 0])
def test_prune_rejects_nonboolean_dryrun_before_manager(surface, dry_run):
    with pytest.raises(ValueError, match="boolean"):
        runtime_tools.call("prune_runtime", {"engine": "llama.cpp", "dry_run": dry_run})
    assert surface.calls == []


def test_runtime_inventory_is_json_and_does_not_check_upstream(surface):
    result = runtime_tools.call("get_runtime_status", {})
    assert len(result["runtimes"]) == 11
    assert not any(item["installed"] for item in result["runtimes"])
    assert all(call[0] == "inspect" for call in surface.calls)
    json.dumps(result)


def test_runtime_update_check_only_resolves_metadata(surface):
    assert runtime_tools.call("check_runtime_updates", {"engine": "llama.cpp"})["target"] == TARGET
    assert surface.calls == [("check", "llama.cpp")]


@pytest.mark.parametrize("target", [None, {"revision": "main"}, {"revision": "v1.0"}])
def test_update_requires_exact_approved_target(surface, target):
    with pytest.raises(ValueError, match="target_revision|immutable"):
        runtime_tools.call("prepare_runtime", {"engine": "llama.cpp", "persistent": True,
            "user_approved": True, "update_approved": True, "target_revision": target})
    assert surface.calls == []


def test_setup_refusal_selects_temporary_without_approving_update(surface):
    result = runtime_tools.call("prepare_runtime", {"engine": "llama.cpp", "persistent": False,
                                                     "target_revision": TARGET, "user_approved": True})
    assert result["status"] == "ready"
    kwargs = surface.calls[-1][2]
    assert kwargs["persistent"] is False
    assert kwargs["update"] is False


def test_initial_setup_also_requires_displayed_immutable_target(surface):
    with pytest.raises(ValueError, match="target_revision"):
        runtime_tools.call("prepare_runtime", {"engine": "llama.cpp", "persistent": False,
                                                 "user_approved": True})
    assert surface.calls == []


def test_runtime_update_passes_exact_target_without_prompting(surface):
    result = runtime_tools.call("prepare_runtime", {"engine": "llama.cpp", "persistent": True,
        "user_approved": True, "update_approved": True, "target_revision": dict(TARGET)})
    assert result["runtime"]["revision"] == SHA
    assert surface.calls[-1][2]["revision"] == TARGET
    assert surface.calls[-1][2]["update"] is True


def test_missing_runtime_decisions_are_structured_for_mcp(surface, monkeypatch):
    def prepare(*args, **kwargs):
        raise RuntimeDecisionRequired({"kind": "storage", "engine": "llama.cpp"})
    monkeypatch.setattr(surface.manager, "prepare", prepare)
    result = runtime_tools.call("prepare_runtime", {"engine": "llama.cpp", "persistent": False,
                                                     "target_revision": TARGET, "user_approved": True})
    assert result == {"status": "decision-required", "requirements": {"kind": "storage", "engine": "llama.cpp"}}


def test_prune_defaults_to_inventory_and_needs_explicit_removal_approval(surface):
    assert runtime_tools.call("prune_runtime", {"engine": "llama.cpp"})["dry_run"] is True
    assert surface.calls == [("prune", "llama.cpp", True)]
    runtime_tools.call("prune_runtime", {"engine": "llama.cpp", "dry_run": False, "user_approved": True})
    assert surface.calls[-1] == ("prune", "llama.cpp", False)


def parser():
    result = argparse.ArgumentParser()
    runtime_cli.add_parser(result.add_subparsers(required=True))
    return result


def test_cli_exposes_only_implemented_runtime_actions():
    for command in ("list", "show llama.cpp", "check llama.cpp", "install llama.cpp --temporary --yes",
                    f"install audio.cpp --persistent --yes --revision {SHA}",
                    "update llama.cpp --persistent --yes", "rollback llama.cpp --yes",
                    "adopt llama.cpp existing --yes", "prune llama.cpp"):
        args = parser().parse_args(["runtime", *command.split()])
        assert args.func is runtime_cli.command


def test_noninteractive_install_needs_explicit_storage(surface, monkeypatch):
    monkeypatch.setattr(runtime_cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    args = parser().parse_args(["runtime", "install", "llama.cpp", "--yes"])
    assert args.func(args) == 2
    assert not any(call[0] == "prepare" for call in surface.calls)


def test_cli_runtime_update_reuses_storage_and_approved_exact_target(surface, monkeypatch):
    surface.manager.installed = True
    monkeypatch.setattr(runtime_cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    args = parser().parse_args(["runtime", "update", "llama.cpp", "--yes"])
    assert args.func(args) == 0
    kwargs = surface.calls[-1][2]
    assert kwargs["revision"] == TARGET
    assert kwargs["persistent"] is True
    assert kwargs["update"] is True


def test_cli_install_revision_passes_exact_sha_to_prepare(surface, monkeypatch):
    surface.manager.installed = True
    monkeypatch.setattr(runtime_cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    revision = "4af143229384fb6da3f373dc87de145ae954609b"
    args = parser().parse_args([
        "runtime", "install", "audio.cpp", "--persistent", "--yes",
        "--revision", revision,
    ])
    assert args.func(args) == 0
    kwargs = surface.calls[-1][2]
    assert kwargs["revision"] == {
        "revision": revision, "ref": revision, "channel": "pinned",
        "repository": "0xShug0/audio.cpp",
    }
    assert kwargs["persistent"] is True
    assert kwargs["update"] is True


def test_cli_install_revision_rejects_tags_before_upstream_check(surface, capsys):
    args = parser().parse_args([
        "runtime", "install", "audio.cpp", "--persistent", "--yes",
        "--revision", "v0.8.0",
    ])
    assert args.func(args) == 1
    assert "40-hex commit SHA" in capsys.readouterr().out
    assert surface.calls == []


def test_cli_install_revision_help_requires_exact_sha(capsys):
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["runtime", "install", "audio.cpp", "--help"])
    assert error.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--revision REVISION" in help_text
    assert "40-hex commit SHA" in help_text
    assert "tags and branches are not accepted" in help_text


def test_cli_failed_update_check_keeps_runtime_without_prepare(surface, monkeypatch):
    monkeypatch.setattr(surface.manager, "check_update", lambda engine: {"engine": engine, "status": "unavailable"})
    args = parser().parse_args(["runtime", "update", "llama.cpp", "--persistent", "--yes"])
    assert args.func(args) == 1
    assert not any(call[0] == "prepare" for call in surface.calls)
