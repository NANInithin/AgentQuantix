"""Regression tests for the model card renderer.

These exist because of a specific, expensive failure: `fork_leads` is an empty
list for every architecture llama.cpp already supports, and

    assessment.get("fork_leads", [None])[0]

raised IndexError on it — the .get() default only applies when the key is
ABSENT, and it never is. The card step runs at the very END of a run, so the
exception surfaced as "the job failed" hours after the quants had actually been
published, and the natural response was to start the whole thing again.

Run with:  python -m pytest tests -q     (from the AgentQuantix directory)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentquantix import card                                    # noqa: E402


def _verification(**overrides):
    base = {
        "repo_id": "someone/Model-GGUF",
        "base_name": "Model",
        "url": "https://huggingface.co/someone/Model-GGUF",
        "count": 2,
        "total_gb": 1.5,
        "files": [
            {"name": "Model-BF16.gguf", "quant": "BF16", "bytes": 10 ** 9,
             "gb": 1.0, "suspect": False},
            {"name": "Model-Q4_K_M.gguf", "quant": "Q4_K_M", "bytes": 5 * 10 ** 8,
             "gb": 0.5, "suspect": False},
        ],
        "missing": [],
        "suspect": [],
        "last_modified": "2026-09-04 12:00:00",
        "error": None,
    }
    base.update(overrides)
    return base


def test_renders_with_empty_fork_leads():
    """The regression. Every supported architecture has fork_leads == []."""
    text = card.render(_verification(),
                       assessment={"repo_id": "someone/Model",
                                   "fork_leads": [],
                                   "imatrix": {"source": "BF16"}})
    assert "base_model: someone/Model" in text
    assert "Runtime requirement" not in text      # no fork, so no fork section


def test_renders_with_fork_leads_absent():
    """A job record written by an older version has no fork_leads key at all."""
    text = card.render(_verification(),
                       assessment={"repo_id": "someone/Model",
                                   "imatrix": {"source": "BF16"}})
    assert "Runtime requirement" not in text


def test_renders_with_a_fork_lead():
    """When a fork IS needed the card has to say so — people cannot run these
    files on stock llama.cpp, and finding that out by trying is miserable."""
    text = card.render(
        _verification(),
        assessment={"repo_id": "someone/Model",
                    "fork_leads": [{"repo": "org/llama.cpp", "ref": "model/X"}],
                    "imatrix": {"source": "BF16"}})
    assert "Runtime requirement" in text
    assert "org/llama.cpp" in text
    assert "model/X" in text


def test_omits_base_model_when_source_is_unknown():
    """A wrong base_model points the Hub's model tree at a repo that does not
    exist, which is worse than omitting the field."""
    text = card.render(_verification(), assessment={"fork_leads": []})
    assert "base_model:" not in text


def test_imatrix_source_note_appears_only_when_degraded():
    guided = _verification(files=[
        {"name": "Model-IQ3_M.gguf", "quant": "IQ3_M", "bytes": 5 * 10 ** 8,
         "gb": 0.5, "suspect": False}])

    on_bf16 = card.render(guided, assessment={"repo_id": "a/b",
                                              "fork_leads": [],
                                              "imatrix": {"source": "BF16"}})
    assert "approximation of the weights" not in on_bf16

    on_quant = card.render(guided, assessment={"repo_id": "a/b",
                                               "fork_leads": [],
                                               "imatrix": {"source": "Q2_K"}})
    assert "approximation of the weights" in on_quant


if __name__ == "__main__":
    # Runnable without pytest, so a quick check never needs a dependency.
    failures = 0
    for name, function in sorted(globals().items()):
        if not name.startswith("test_") or not callable(function):
            continue
        try:
            function()
            print(f"  PASS  {name}")
        except Exception as e:
            failures += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{'all passed' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
