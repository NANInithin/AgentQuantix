"""Running a model that was never in a research report.

The sweep is built around the trending list, but "it has to have been trending
when you last ran research" is an arbitrary wall the moment you want to re-cut
an older release, retry a fix, or take a request. Naming a full repo id is
itself the judgement the base-model filter was standing in for, so a named repo
is assessed on demand and skips the filter entirely.
"""

import pytest

from agentquantix import cli, research


@pytest.fixture
def report():
    """A research result holding exactly one model."""
    return {"assessments": [{"repo_id": "org/trending",
                             "verdict": "ok",
                             "warnings": [],
                             "quants": ["Q4_K_M"],
                             "hours": {"total": 1.0},
                             "peak_disk_gb": 10.0}]}


def test_a_named_repo_is_assessed_on_demand(report, monkeypatch):
    asked = []

    def fake_assess_one(repo_id, **kwargs):
        asked.append(repo_id)
        return {"repo_id": repo_id, "verdict": "ok", "warnings": [],
                "quants": ["Q4_K_M"], "hours": {"total": 1.0},
                "peak_disk_gb": 10.0}

    monkeypatch.setattr(research, "assess_one", fake_assess_one)

    chosen = cli._select_interactively(report, ["Qwen/Qwen3-0.6B"])

    assert asked == ["Qwen/Qwen3-0.6B"]
    assert [c["repo_id"] for c in chosen] == ["Qwen/Qwen3-0.6B"]


def test_the_report_still_wins_when_it_has_the_model(report, monkeypatch):
    """A model already assessed must not be re-priced. The run has to use the
    same quant list and imatrix plan that were shown at the approval gate."""
    def fail(*a, **k):
        raise AssertionError("should not re-assess a model already in the report")

    monkeypatch.setattr(research, "assess_one", fail)
    chosen = cli._select_interactively(report, ["org/trending"])
    assert [c["repo_id"] for c in chosen] == ["org/trending"]


def test_a_bare_name_not_in_the_report_is_not_guessed_at(report, monkeypatch):
    """Without an org there is nothing to look up, and guessing which
    `Qwen3-0.6B` was meant would be worse than saying so."""
    monkeypatch.setattr(research, "assess_one",
                        lambda *a, **k: pytest.fail("should not be called"))
    assert cli._select_interactively(report, ["mystery-model"]) == []


def test_a_bad_repo_id_is_reported_not_crashed(report, monkeypatch):
    def fake_assess_one(repo_id, **kwargs):
        raise ValueError(f"{repo_id}: cannot read this repo from the Hub")

    monkeypatch.setattr(research, "assess_one", fake_assess_one)
    assert cli._select_interactively(report, ["org/typo"]) == []


def test_a_named_repo_works_with_no_report_at_all(monkeypatch):
    """REGRESSION. `aqx run org/model` on a fresh machine used to refuse with
    "No research on file" — but a named repo needs no report."""
    monkeypatch.setattr(
        research, "assess_one",
        lambda repo_id, **k: {"repo_id": repo_id, "verdict": "ok",
                              "warnings": [], "quants": ["Q4_K_M"],
                              "hours": {"total": 1.0}, "peak_disk_gb": 10.0})

    chosen = cli._select_interactively(None, ["Qwen/Qwen3-0.6B"])
    assert [c["repo_id"] for c in chosen] == ["Qwen/Qwen3-0.6B"]


def test_a_named_repo_that_is_blocked_still_says_why(report, monkeypatch):
    monkeypatch.setattr(
        research, "assess_one",
        lambda repo_id, **k: {"repo_id": repo_id, "verdict": "blocked",
                              "blockers": ["peak disk 900G exceeds 400G free"],
                              "warnings": [], "quants": [],
                              "hours": {"total": 0.0}, "peak_disk_gb": 900.0})

    assert cli._select_interactively(report, ["org/enormous"]) == []


def test_hub_one_marks_it_as_not_from_the_trending_list():
    """rank 0 keeps an on-demand model from being compared against a list it
    was never part of."""
    import inspect

    from agentquantix import hub

    source = inspect.getsource(hub.one)
    assert "rank=0" in source
