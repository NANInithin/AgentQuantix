"""Which models the agent may be handed, and what happens to an unknown one.

Separate from test_tools_cards.py because this is about REACHING a model at
all, rather than about what gets written once its quants exist.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentquantix.agent import tools                             # noqa: E402


def _fake_assessment(repo_id):
    return {"repo_id": repo_id, "verdict": "runnable"}


def test_a_model_not_in_the_report_is_assessed_on_demand(monkeypatch):
    """REGRESSION. This raised "not in the last research run: X", and the only
    remedy that error suggests is to research again — which cannot help, since
    the model was never going to be in a trending sweep.

    Observed: a user asked for ibm-granite/granite-4.2-3b and the agent ran
    research_trending at limit=1, then 100, then 300, and never answered.
    """
    monkeypatch.setattr(tools.research, "load_latest", lambda: None)
    monkeypatch.setattr(tools.sysprobe, "probe", lambda **k: {"cpu": "fake"})
    seen = []

    def fake_assess_one(repo_id, sysinfo=None, **kwargs):
        seen.append(repo_id)
        return _fake_assessment(repo_id)

    monkeypatch.setattr(tools.research, "assess_one", fake_assess_one)

    out = tools._assessments_for(["ibm-granite/granite-4.2-3b"])
    assert [a["repo_id"] for a in out] == ["ibm-granite/granite-4.2-3b"]
    assert seen == ["ibm-granite/granite-4.2-3b"]


def test_the_machine_is_probed_once_for_a_batch(monkeypatch):
    """assess_one() probes when given no sysinfo. Left to itself that is the
    same measurement repeated per model."""
    monkeypatch.setattr(tools.research, "load_latest", lambda: None)
    probes = []
    monkeypatch.setattr(tools.sysprobe, "probe",
                        lambda **k: probes.append(1) or {"cpu": "fake"})
    monkeypatch.setattr(tools.research, "assess_one",
                        lambda r, sysinfo=None, **k: _fake_assessment(r))

    tools._assessments_for(["a/one", "b/two", "c/three"])
    assert len(probes) == 1


def test_the_report_is_preferred_over_a_fresh_assessment(monkeypatch):
    """Researched models must not be re-assessed - that is a Hub round trip
    for an answer already on disk."""
    report = {"assessments": [_fake_assessment("org/known")]}
    monkeypatch.setattr(tools.research, "load_latest", lambda: report)
    monkeypatch.setattr(tools.sysprobe, "probe", lambda **k: {"cpu": "fake"})
    monkeypatch.setattr(
        tools.research, "assess_one",
        lambda *a, **k: pytest.fail("should not assess a researched model"))

    assert tools._assessments_for(["org/known"])[0]["repo_id"] == "org/known"


def test_a_mixed_batch_keeps_the_order_it_was_asked_in(monkeypatch):
    report = {"assessments": [_fake_assessment("org/known")]}
    monkeypatch.setattr(tools.research, "load_latest", lambda: report)
    monkeypatch.setattr(tools.sysprobe, "probe", lambda **k: {"cpu": "fake"})
    monkeypatch.setattr(tools.research, "assess_one",
                        lambda r, sysinfo=None, **k: _fake_assessment(r))

    out = tools._assessments_for(["new/first", "org/known", "new/last"])
    assert [a["repo_id"] for a in out] == ["new/first", "org/known", "new/last"]


def test_an_ambiguous_short_name_still_asks_rather_than_guessing(monkeypatch):
    """find() raises on an ambiguous match. That is a real question for the
    user and must not be swallowed into a Hub lookup for a repo named 'b'."""
    report = {"assessments": [_fake_assessment("org/b-one"),
                              _fake_assessment("org/b-two")]}
    monkeypatch.setattr(tools.research, "load_latest", lambda: report)
    with pytest.raises(ValueError, match="matches 2"):
        tools._assessments_for(["b-"])
