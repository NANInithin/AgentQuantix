"""The card tools' repo handling.

Split from test_card.py because these are about the AGENT boundary — what the
model is allowed to name — rather than about card content.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentquantix import config                                  # noqa: E402
from agentquantix.agent import tools                             # noqa: E402


def test_a_source_repo_id_is_mapped_to_our_gguf_repo(monkeypatch):
    """REGRESSION, and the dangerous one.

    The old check was `if "/" not in repo`, so a bare model name was rewritten
    but a SOURCE repo id was not — the slash made it look already-qualified.
    Passing `ibm-granite/granite-4.2-3b`, which is the obvious thing to pass
    when that is the model being worked on, sent the card step at IBM's own
    repo. It would have tried to upload a README.md over the publisher's model
    card; only write access stopped it.
    """
    monkeypatch.setattr(config, "namespace", lambda: "us")
    assert tools._our_gguf_repo("ibm-granite/granite-4.2-3b") == \
        "us/granite-4.2-3b-GGUF"


def test_every_spelling_lands_on_the_same_repo(monkeypatch):
    monkeypatch.setattr(config, "namespace", lambda: "us")
    for given in ("granite-4.2-3b",
                  "granite-4.2-3b-GGUF",
                  "ibm-granite/granite-4.2-3b",
                  "us/granite-4.2-3b-GGUF",
                  "someone-else/granite-4.2-3b-GGUF",
                  "  ibm-granite/granite-4.2-3b/  "):
        assert tools._our_gguf_repo(given) == "us/granite-4.2-3b-GGUF"


def test_an_empty_repo_id_is_refused(monkeypatch):
    monkeypatch.setattr(config, "namespace", lambda: "us")
    with pytest.raises(ValueError):
        tools._our_gguf_repo("/")
