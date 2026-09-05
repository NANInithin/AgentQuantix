"""Finding quants we have already published, wherever we put them.

The point of this check is that work already on the Hub is not work. Getting
it wrong is expensive in both directions: miss a repo and the agent proposes a
sweep that is mostly finished, at full price; match too loosely and it skips
work that was never done.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentquantix import config, hub                             # noqa: E402


class _FakeApi:
    """Answers list_repo_files for the repos it was given, 404s for the rest."""

    def __init__(self, repos):
        self.repos = repos
        self.asked = []

    def list_repo_files(self, repo_id, repo_type="model"):
        self.asked.append(repo_id)
        if repo_id not in self.repos:
            raise FileNotFoundError(repo_id)
        return self.repos[repo_id]


def _candidate(repo_id="ibm-granite/granite-4.2-3b"):
    return hub.Candidate(repo_id=repo_id, rank=0)


def _setup(monkeypatch, namespace="us", ours=()):
    monkeypatch.setattr(config, "namespace", lambda: namespace)
    monkeypatch.setattr(hub, "our_repos", lambda: tuple(ours))


def test_the_predicted_repo_is_found(monkeypatch):
    _setup(monkeypatch)
    api = _FakeApi({"us/granite-4.2-3b-GGUF": [
        "granite-4.2-3b-Q4_K_M.gguf", "granite-4.2-3b-BF16.gguf"]})
    candidate = _candidate()
    files, quants = hub._already_ours(candidate, api)
    assert quants == {"Q4_K_M", "BF16"}
    assert candidate.published_repos == ["us/granite-4.2-3b-GGUF"]


def test_a_differently_spelled_repo_of_ours_is_found(monkeypatch):
    """REGRESSION. Only `{namespace}/{name}-GGUF` was consulted, so a repo
    published by an older script, under a lowercase suffix, or renamed at any
    point read as "nothing published" and bought the whole sweep again."""
    _setup(monkeypatch, ours=["us/granite-4.2-3b-gguf"])
    api = _FakeApi({"us/granite-4.2-3b-gguf": ["granite-4.2-3b-Q4_K_M.gguf"]})
    candidate = _candidate()
    files, quants = hub._already_ours(candidate, api)
    assert quants == {"Q4_K_M"}
    assert candidate.published_repos == ["us/granite-4.2-3b-gguf"]


def test_quants_are_pooled_across_our_repos(monkeypatch):
    _setup(monkeypatch, ours=["us/granite-4.2-3b-gguf"])
    api = _FakeApi({
        "us/granite-4.2-3b-GGUF": ["granite-4.2-3b-Q4_K_M.gguf"],
        "us/granite-4.2-3b-gguf": ["granite-4.2-3b-Q8_0.gguf"],
    })
    _, quants = hub._already_ours(_candidate(), api)
    assert quants == {"Q4_K_M", "Q8_0"}


def test_a_different_model_is_never_matched(monkeypatch):
    """Substring matching would read granite-4.2-3b-instruct as granite-4.2-3b
    and skip a sweep that was never run. Missing a repo costs a redo; this
    costs work that silently never happens."""
    _setup(monkeypatch, ours=["us/granite-4.2-3b-instruct-GGUF",
                              "us/granite-4.2-3b-base-GGUF"])
    api = _FakeApi({"us/granite-4.2-3b-instruct-GGUF": ["x-Q4_K_M.gguf"],
                    "us/granite-4.2-3b-base-GGUF": ["y-Q8_0.gguf"]})
    candidate = _candidate()
    _, quants = hub._already_ours(candidate, api)
    assert quants == set()
    assert candidate.published_repos == []


def test_someone_elses_repo_is_never_ours(monkeypatch):
    """our_repos() is scoped to our namespace, so a community quant of the
    same model cannot be mistaken for work we did."""
    _setup(monkeypatch, ours=["us/other-model-GGUF"])
    api = _FakeApi({"them/granite-4.2-3b-GGUF": ["granite-4.2-3b-Q4_K_M.gguf"]})
    _, quants = hub._already_ours(_candidate(), api)
    assert quants == set()


def test_nothing_published_is_not_an_error(monkeypatch):
    _setup(monkeypatch)
    files, quants = hub._already_ours(_candidate(), _FakeApi({}))
    assert (files, quants) == (set(), set())


def test_a_hub_failure_degrades_to_proposing_the_full_sweep(monkeypatch):
    """This feeds an optimisation. Unable to answer must mean "assume nothing
    is published", never an exception that fails the assessment."""
    monkeypatch.setattr(config, "namespace", lambda: "us")

    class _Boom:
        def list_models(self, **kwargs):
            raise ConnectionError("hub down")

    monkeypatch.setattr(hub, "api", lambda: _Boom())
    hub.forget_our_repos()
    try:
        assert hub.our_repos() == ()
    finally:
        hub.forget_our_repos()


def test_normalisation_folds_case_and_suffix():
    for spelling in ("us/Granite-4.2-3b-GGUF", "us/granite-4.2-3b-gguf",
                     "us/granite-4.2-3b.gguf", "us/granite-4.2-3b_GGUF",
                     "granite-4.2-3b"):
        assert hub._normalised(spelling) == "granite-4.2-3b"
