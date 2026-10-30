from types import SimpleNamespace

import pytest

from agentquantix import cli, hub, report, research
from agentquantix.agent import tools
from agentquantix.tasks import router


def _candidates():
    return [
        hub.Candidate("acme/text-model", 1, pipeline_tag="text-generation",
                      tags=["transformers"], library_name="transformers",
                      params=7_000_000_000),
        hub.Candidate("BAAI/bge-small-en-v1.5", 2,
                      pipeline_tag="feature-extraction",
                      library_name="sentence-transformers"),
        hub.Candidate("org/flux-text-to-image", 3,
                      pipeline_tag="text-to-image", library_name="diffusers"),
        hub.Candidate("org/depth-model", 4, pipeline_tag="depth-estimation",
                      library_name="transformers"),
        hub.Candidate("org/tokenizer", 5, pipeline_tag="token-classification",
                      library_name="transformers"),
        hub.Candidate("org/foo-awq", 6,
                      pipeline_tag="automatic-speech-recognition",
                      library_name="transformers"),
        hub.Candidate("org/audio-model", 7, pipeline_tag="audio-to-audio",
                      library_name="transformers"),
        hub.Candidate("org/asr-diffusion-model", 8,
                      pipeline_tag="automatic-speech-recognition",
                      library_name="transformers"),
        hub.Candidate("org/enrichment-error", 9,
                      pipeline_tag="feature-extraction",
                      library_name="sentence-transformers"),
    ]


def _assessment(candidate, sysinfo, _arch_ok, _arch_detail, fork_leads=None):
    return {
        "repo_id": candidate.repo_id,
        "rank": candidate.rank,
        "url": candidate.url,
        "target_repo": candidate.target_repo,
        "verdict": "ok",
        "blockers": [],
        "warnings": [],
        "params_b": 7.0,
        "n_experts": 0,
        "bf16_gb": 14.0,
        "peak_disk_gb": 20.0,
        "upload_gb": 10.0,
        "hours": {
            "total": 2.0,
            "download": 0.1,
            "convert": 0.2,
            "imatrix": 0.3,
            "sweep_overlapped": 1.0,
            "quantize": 0.8,
            "upload": 0.9,
            "fork_build": 0,
        },
        "quants": ["Q4_K_M"],
        "all_quants": ["Q4_K_M"],
        "published_count": 0,
        "published_quants": [],
        "architecture": "LlamaForCausalLM",
        "model_type": candidate.model_type,
        "n_layers": 32,
        "arch_detail": "supported",
        "source_kind": "convert",
        "official_gguf": None,
        "official_bf16_file": None,
        "imatrix": {
            "source": "BF16",
            "source_gb": 14.0,
            "ngl": 0,
            "fits_fast_memory": True,
        },
        "free_disk_gb": sysinfo["disk_free_gb"],
        "download_gb": 14.0,
        "rates": {"upload_mbps": 100, "upload_learned": True},
        "fork_leads": [],
    }


@pytest.fixture
def offline_research(monkeypatch):
    candidates = _candidates()
    enrich_calls = []
    monkeypatch.setattr(research.hub, "trending", lambda _limit: candidates)
    monkeypatch.setattr(
        research.sysprobe, "probe",
        lambda **_kwargs: {"disk_free_gb": 2.5, "cpu": "test"})

    def enrich(candidate, check_ggufs=True, want_readme=True):
        enrich_calls.append((candidate.repo_id, check_ggufs, want_readme))
        if candidate.repo_id == "org/enrichment-error":
            candidate.error = "ConnectionError: offline"
            return candidate
        candidate.architectures = (
            ["LlamaForCausalLM"]
            if candidate.pipeline_tag == "text-generation" else [])
        candidate.model_type = (
            "llama" if candidate.pipeline_tag == "text-generation" else None)
        candidate.source_bytes = {
            "acme/text-model": 14_000_000_000,
            "BAAI/bge-small-en-v1.5": 1_000_000_000,
            "org/flux-text-to-image": 500_000_000,
        }.get(candidate.repo_id, 0)
        candidate.license = "apache-2.0"
        return candidate

    monkeypatch.setattr(research.hub, "enrich", enrich)
    monkeypatch.setattr(
        research.archsupport, "supported_architectures",
        lambda: ["LlamaForCausalLM"])
    monkeypatch.setattr(research.archsupport, "supported_quants",
                        lambda: ["Q4_K_M"])
    monkeypatch.setattr(
        research.archsupport, "check",
        lambda _candidate, supported=None: (True, "supported"))
    monkeypatch.setattr(
        research.archsupport, "find_fork",
        lambda _candidate: pytest.fail("fork hunt is not needed"))
    monkeypatch.setattr(research.feasibility, "assess", _assessment)
    monkeypatch.setattr(
        research.feasibility, "rank_key", lambda assessment: assessment["repo_id"])
    monkeypatch.setattr(
        router, "hub_metadata",
        lambda _repo_id: pytest.fail("metadata routing must not fetch"))
    monkeypatch.setattr(report.sysprobe, "summary", lambda _system: "test system")
    return candidates, enrich_calls


def test_research_routes_non_text_tasks_without_fetching(offline_research):
    candidates, enrich_calls = offline_research
    result = research.research(probe_disk=False, hunt_forks=False)

    assert [item["repo_id"] for item in result["assessments"]] == [
        "acme/text-model"]
    assert result["task_candidate_count"] == 7
    routes = result["task_routes"]
    assert [item["status"] for item in routes] == [
        "ready", "experimental", "choose", "blocked", "blocked"]
    by_repo = {item["repo_id"]: item for item in routes}

    embedding = by_repo["BAAI/bge-small-en-v1.5"]
    assert embedding["task"] == "embeddings"
    assert embedding["status"] == "ready"
    assert embedding["source_gb"] == 1.0
    assert embedding["warnings"] == [
        "source is 1 GB; conversion may need ~3x free disk"]
    assert embedding["next"] == (
        "aqx run BAAI/bge-small-en-v1.5 --task embeddings --no-publish")

    diffusion = by_repo["org/flux-text-to-image"]
    assert diffusion["status"] == "experimental"
    assert diffusion["engine"] == "stable-diffusion.cpp"
    assert diffusion["task"] == "text-to-image"

    depth = by_repo["org/depth-model"]
    assert depth["status"] == "blocked"
    assert "no vision.cpp family matches depth-estimation" in depth["detail"]
    assert depth["next"] is None

    expected_audio = router.route(
        "org/audio-model",
        metadata={
            "pipeline_tag": "audio-to-audio",
            "library_name": "transformers",
            "tags": [],
            "model_type": None,
            "architectures": [],
        })
    audio = by_repo["org/audio-model"]
    expected_status = (
        "choose" if expected_audio.choices
        else "blocked" if expected_audio.blocker
        else "experimental" if expected_audio.pipeline == "media"
        else "ready")
    assert audio["task"] == expected_audio.task
    assert audio["detail"] == (
        expected_audio.blocker or "; ".join(expected_audio.reasons))
    assert audio["status"] == expected_status
    assert audio["status"] in ("choose", "blocked")

    choice = by_repo["org/asr-diffusion-model"]
    assert choice["status"] == "choose"
    assert choice["next"] == (
        "aqx run org/asr-diffusion-model --task "
        "<automatic-speech-recognition> --no-publish")

    dropped = {item["repo_id"]: item["reason"] for item in result["dropped"]}
    assert dropped["org/tokenizer"].startswith("no AgentQuantix task (")
    assert dropped["org/foo-awq"] == "derivative (name: awq)"
    assert "org/enrichment-error" not in dropped
    assert result["errors"] == [{
        "repo_id": "org/enrichment-error",
        "error": "ConnectionError: offline",
    }]

    assert ("acme/text-model", True, False) in enrich_calls
    assert all(
        check_ggufs is False and want_readme is False
        for repo_id, check_ggufs, want_readme in enrich_calls
        if repo_id != "acme/text-model")
    assert candidates[1].dropped is None


def test_text_only_mode_keeps_the_previous_drop_reasons(offline_research):
    candidates, enrich_calls = offline_research
    result = research.research(
        probe_disk=False, hunt_forks=False, include_tasks=False)

    assert "task_routes" not in result
    assert "task_candidate_count" not in result
    assert [item["repo_id"] for item in result["assessments"]] == [
        "acme/text-model"]
    assert result["dropped"] == [
        {
            "repo_id": candidate.repo_id,
            "rank": candidate.rank,
            "reason": f"not text-capable ({candidate.pipeline_tag})",
        }
        for candidate in candidates[1:]
    ]
    assert enrich_calls == [("acme/text-model", True, False)]


def test_hub_listing_candidates_keep_library_name(monkeypatch):
    model = SimpleNamespace(
        id="org/model", pipeline_tag="feature-extraction",
        library_name="sentence-transformers", tags=[])
    fake_api = SimpleNamespace(
        list_models=lambda **_kwargs: [model],
        model_info=lambda *_args, **_kwargs: model)
    monkeypatch.setattr(hub, "api", lambda: fake_api)

    assert hub.trending(limit=1)[0].library_name == "sentence-transformers"
    assert hub.one("org/model").library_name == "sentence-transformers"


def test_reports_render_and_filter_other_tasks(offline_research):
    _candidates, _enrich_calls = offline_research
    result = research.research(probe_disk=False, hunt_forks=False)

    terminal = report.table(result)
    assert "OTHER TASKS" in terminal
    assert "BAAI/bge-small-en-v1.5" in terminal
    assert ("7 other-task candidates -> 1 ready, 1 experimental, "
            "1 need --task, 2 blocked.") in terminal
    assert ("ready = a native adapter exists for this task; `aqx run` still "
            "runs its native quality gates before anything is published. "
            "Accepted models: docs/SUPPORT_MATRIX.md.") in terminal

    markdown = report.markdown(result)
    assert "## Other tasks" in markdown
    assert "[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)" in markdown
    assert "`aqx run BAAI/bge-small-en-v1.5 --task embeddings --no-publish`" in markdown

    hidden = report.table(result, include_blocked=False)
    assert "org/depth-model" not in hidden
    assert "org/audio-model" not in hidden
    assert "org/asr-diffusion-model" in hidden

    top = report.table(result, limit=1)
    assert "BAAI/bge-small-en-v1.5" in top
    assert "org/flux-text-to-image" not in top
    markdown_top = report.markdown(result, limit=1, include_blocked=False)
    assert "BAAI/bge-small-en-v1.5" in markdown_top
    assert "org/depth-model" not in markdown_top

    old_result = {
        key: value for key, value in result.items()
        if key not in ("task_routes", "task_candidate_count")
    }
    assert "OTHER TASKS" not in report.table(old_result)
    assert "## Other tasks" not in report.markdown(old_result)


def test_research_cli_text_only_option_forwards_and_is_documented(
        monkeypatch, tmp_path, capsys):
    calls = {}
    result = {
        "generated_at": "now",
        "elapsed_s": 0,
        "system": {},
        "trending_count": 0,
        "kept_count": 0,
        "dropped": [],
        "errors": [],
        "assessments": [],
    }
    monkeypatch.setattr(
        cli.research, "research",
        lambda **kwargs: calls.update(kwargs) or result)
    monkeypatch.setattr(cli.research, "save", lambda _result: tmp_path / "report.json")

    args = cli.build_parser().parse_args([
        "research", "--text-only", "--json", "--no-disk", "--no-fork-hunt"])
    assert cli.cmd_research(args) == 0
    assert calls["include_tasks"] is False

    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(["research", "--help"])
    assert raised.value.code == 0
    assert "only rank text-generation candidates (previous behaviour)" in (
        capsys.readouterr().out)


def test_agent_research_counts_task_route_statuses(monkeypatch, tmp_path):
    result = {
        "system": {},
        "trending_count": 4,
        "kept_count": 1,
        "assessments": [],
        "task_routes": [
            {"status": "ready"},
            {"status": "experimental"},
            {"status": "choose"},
            {"status": "blocked"},
            {"status": "blocked"},
        ],
    }
    monkeypatch.setattr(tools.research, "research", lambda **_kwargs: result)
    monkeypatch.setattr(tools.research, "save", lambda _result: tmp_path / "saved.json")
    monkeypatch.setattr(tools.sysprobe, "summary", lambda _system: "test")
    monkeypatch.setattr(tools.report, "table", lambda _result: "table")
    monkeypatch.setattr(tools.voice, "advisory_catalog", lambda: [])

    output = tools._call("research_trending", {})
    assert output["counts"]["task_routes"] == {
        "ready": 1,
        "experimental": 1,
        "choose": 1,
        "blocked": 2,
    }
