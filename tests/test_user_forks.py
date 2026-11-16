from types import SimpleNamespace
from urllib.parse import unquote

import pytest

from agentquantix import actions, archsupport, cli, config, research
from agentquantix.agent import runtime_tools, tools
from agentquantix.pipeline.job import Job
from agentquantix.pipeline import build
import subprocess


@pytest.fixture
def candidate():
    return SimpleNamespace(repo_id="zenteiq-ai/Dhir-2B", org="zenteiq-ai", name="Dhir-2B",
                           model_type="dhir", architectures=["DhirForCausalLM"],
                           pipeline_tag="text-generation", tags=[], error=None)


@pytest.mark.parametrize("value,ref", [
    ("https://github.com/zenteiq-research/llama.cpp", None),
    ("https://github.com/zenteiq-research/llama.cpp.git", None),
    ("https://github.com/zenteiq-research/llama.cpp/tree/dhir", "dhir"),
    ("https://github.com/zenteiq-research/llama.cpp.git#dhir", "dhir"),
    ("zenteiq-research/llama.cpp@dhir-support", "dhir-support"),
    ("zenteiq-research/llama.cpp#feature/dhir", "feature/dhir"),
    ("zenteiq-research/llama.cpp", None),
])
def test_parse(value, ref):
    assert archsupport.parse_fork(value) == {"repo": "zenteiq-research/llama.cpp", "ref": ref}


@pytest.mark.parametrize("value", ["", "https://evil.test/a/b", "a/b#", "a/b@-bad",
                                         "a/b/tree/dhir#other", "a/b/c", "a/b#has space"])
def test_parse_invalid(value):
    with pytest.raises(ValueError):
        archsupport.parse_fork(value)


def test_auto_ref(monkeypatch, candidate):
    monkeypatch.setattr(archsupport, "_github_json", lambda url: [
        {"name": "master"}, {"name": "dhir"}, {"name": "dhir-support"}])
    assert archsupport.user_fork("a/b", candidate)["ref"] == "dhir-support"
    monkeypatch.setattr(archsupport, "_github_json", lambda url:
                        [] if "branches?" in url else {"default_branch": "dhir"})
    assert archsupport.user_fork("a/b", candidate)["ref"] == "dhir"
    monkeypatch.setattr(archsupport, "_github_json", lambda url: None)
    with pytest.raises(ValueError, match="explicit ref"):
        archsupport.user_fork("a/b", candidate)
    assert archsupport.user_fork("a/b#dhir", candidate)["ref"] == "dhir"


def test_org_stem_and_stale_miss(monkeypatch, candidate, tmp_path):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    archsupport._fork_cache_put("DhirForCausalLM", [])
    seen = []

    def github(url):
        seen.append(unquote(url))
        if "branches?" in url:
            return [{"name": "dhir"}]
        if "zenteiq llama.cpp" in unquote(url):
            return {"items": [{"full_name": "zenteiq-research/llama.cpp"}]}
        return {"items": []}

    monkeypatch.setattr(archsupport, "_github_json", github)
    leads = archsupport.find_fork(candidate)
    assert leads[0]["repo"] == "zenteiq-research/llama.cpp"
    assert leads[0]["ref"] == "dhir"
    assert leads[0]["kind"] == "publisher-fork"
    assert any("zenteiq-ai llama.cpp" in url for url in seen)


def test_assessment_job_and_resume(monkeypatch, candidate, tmp_path):
    monkeypatch.setattr(research.hub, "one", lambda repo: candidate)
    monkeypatch.setattr(research.hub, "enrich", lambda *a, **k: None)
    monkeypatch.setattr(archsupport, "check", lambda c: (False, "unsupported"))
    monkeypatch.setattr(archsupport, "supported_quants", lambda: set())
    monkeypatch.setattr(archsupport, "find_fork", lambda c: [{"repo": "other/fork", "ref": "main"}])
    monkeypatch.setattr(research.feasibility, "assess", lambda c, s, ok, detail, fork_leads:
                        {"repo_id": c.repo_id, "target_repo": "me/Dhir-GGUF", "quants": ["Q4_K_M"],
                         "arch_ok": ok, "fork_leads": fork_leads, "warnings": []})
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    assessment = research.assess_one(candidate.repo_id, sysinfo={}, fork="a/b#dhir")
    assert assessment["fork_leads"][0]["kind"] == "user-fork"
    job = Job.from_assessment(assessment)
    assert job.fork["repo"] == "a/b"
    assert Job.load(job.save()).fork == job.fork
    monkeypatch.setattr(archsupport, "check", lambda c: (True, "supported"))
    assessment = research.assess_one(candidate.repo_id, sysinfo={}, fork="a/b")
    assert not assessment["fork_leads"]
    assert "ignored" in assessment["warnings"][0]


def test_batch_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        actions.assessments_for(["a/b", "a/c"], fork="a/fork#dhir")


def test_fork_reassesses_researched_shorthand(monkeypatch):
    saved = {"repo_id": "zenteiq-ai/Dhir-2B", "fork_leads": []}
    monkeypatch.setattr(research, "load_latest", lambda: {"assessments": [saved]})
    monkeypatch.setattr(actions.sysprobe, "probe", lambda **kwargs: {})
    calls = []
    fresh = {"repo_id": saved["repo_id"], "fork_leads": [{"kind": "user-fork"}]}

    def assess(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return fresh

    monkeypatch.setattr(research, "assess_one", assess)
    assert actions.assessments_for(["Dhir-2B"], fork="a/b#dhir") == [fresh]
    assert calls == [("zenteiq-ai/Dhir-2B", {"sysinfo": {}, "fork": "a/b#dhir"})]
    assert saved["fork_leads"] == []


@pytest.mark.parametrize("source", ["https://github.com/a/b", "http://github.com/a/b",
                                    "git@github.com:a/b", "github.com/a/b"])
def test_adopt_rejects_url(source, monkeypatch):
    monkeypatch.setattr(runtime_tools, "RuntimeManager", lambda: pytest.fail("must reject before manager"))
    with pytest.raises(ValueError, match="local built checkout.*--fork"):
        runtime_tools.call("adopt_runtime", {"engine": "llama.cpp", "source": source,
                                             "user_approved": True})


def test_tool_schemas():
    for name in ("inspect_model", "plan_release", "start_release"):
        schema = next(t for t in tools.TOOLS if t["name"] == name)
        assert "fork" in schema["input_schema"]["properties"]


@pytest.mark.parametrize("existing", [False, True])
def test_commit_fork_fetch(monkeypatch, tmp_path, existing):
    monkeypatch.setattr(config, "FORKS_DIR", tmp_path)
    ref = "ab12" * 10
    root = tmp_path / build._slug("a/b", ref)
    if existing:
        root.mkdir()
    commands = []
    monkeypatch.setattr(build, "run", lambda cmd: commands.append([str(c) for c in cmd]))
    monkeypatch.setattr(build, "binaries", lambda path: None)
    monkeypatch.setattr(build, "compile_tools", lambda *args: ("quantize", "imatrix"))
    assert build.ensure_fork("a/b", ref) == (root, "quantize", "imatrix")
    prefix = ["git", "-C", str(root)]
    if existing:
        assert commands == [prefix + ["fetch", "--force", "--depth", "1", "origin", ref],
                            prefix + ["reset", "--hard", "FETCH_HEAD"]]
    else:
        assert commands == [["git", "init", str(root)],
                            prefix + ["remote", "add", "origin", "https://github.com/a/b.git"],
                            prefix + ["fetch", "--depth", "1", "origin", ref],
                            prefix + ["checkout", "--detach", "FETCH_HEAD"]]


def test_commit_fork_refresh_failure_does_not_build(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "FORKS_DIR", tmp_path)
    ref = "a" * 40
    (tmp_path / build._slug("a/b", ref)).mkdir()
    monkeypatch.setattr(build, "binaries", lambda path: None)
    monkeypatch.setattr(build, "compile_tools", lambda *args: pytest.fail("must not build stale checkout"))

    def fail(cmd):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(build, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        build.ensure_fork("a/b", ref)


@pytest.mark.parametrize("command", ["inspect", "plan", "run"])
def test_cli_fork(command, monkeypatch):
    monkeypatch.setattr(actions, "inspect", lambda *a, **k: {"route": {"ready": True}, "fork": k["fork"]})
    parser = cli.build_parser()
    args = parser.parse_args([command, "zenteiq-ai/Dhir-2B", "--fork", "a/b#dhir"])
    assert args.fork == "a/b#dhir"


def test_fork_keeps_full_repo_id_over_saved_variant(monkeypatch):
    saved = {"repo_id": "org/Model-Instruct", "fork_leads": []}
    monkeypatch.setattr(research, "load_latest", lambda: {"assessments": [saved]})
    monkeypatch.setattr(actions.sysprobe, "probe", lambda **kwargs: {})
    calls = []
    monkeypatch.setattr(research, "assess_one",
                        lambda repo_id, **kwargs: calls.append(repo_id) or {"repo_id": repo_id})
    actions.assessments_for(["org/Model"], fork="a/b#dhir")
    assert calls == ["org/Model"]
