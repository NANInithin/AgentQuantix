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


def test_the_card_says_nothing_about_the_imatrix():
    """Dropped at the user's request. The card used to carry a
    "Quantization details" block, including a disclosure when the matrix was
    computed on a smaller quant instead of the BF16.

    Asserted rather than merely deleted so a later change does not quietly
    reintroduce it.
    """
    guided = _verification(files=[
        {"name": "Model-IQ3_M.gguf", "quant": "IQ3_M", "bytes": 5 * 10 ** 8,
         "gb": 0.5, "suspect": False}])
    text = card.render(guided, assessment={"repo_id": "a/b", "fork_leads": [],
                                           "imatrix": {"source": "Q2_K"}})
    assert "imatrix" not in text.lower()
    assert "approximation of the weights" not in text


# =====================================================
# VALIDATION OF A COMPOSED CARD
# =====================================================
# The agent owns the document; these check only the CLAIMS in it. Sizes here
# are the rounded GiB figures of _verification()'s byte counts: 10**9 bytes is
# 0.93 GiB and 5*10**8 is 0.47 GiB — deliberately NOT the decimal 1.0/0.5, so
# a card that echoed the wrong unit would be caught.
_GOOD = """---
base_model: someone/Model
library_name: gguf
---

# Model GGUF

| File | Size |
|---|---|
| [Model-BF16.gguf](x) | 0.93 GB |
| [Model-Q4_K_M.gguf](x) | 0.47 GB |
"""


def _facts(**overrides):
    return card.facts(_verification(**overrides),
                      assessment={"repo_id": "someone/Model", "fork_leads": []})


def test_a_correct_card_passes():
    assert card.validate(_GOOD, _facts()) == []


def test_an_invented_filename_is_caught():
    text = _GOOD.replace("| [Model-Q4_K_M.gguf](x) | 0.47 GB |",
                         "| [Model-Q4_K_M.gguf](x) | 0.47 GB |\n"
                         "| [Model-Q2_K.gguf](x) | 0.2 GB |")
    problems = card.validate(text, _facts())
    assert any("not in the repo" in p and "Model-Q2_K.gguf" in p
               for p in problems)


def test_an_omitted_file_is_caught():
    """Silently dropping a published file is as wrong as inventing one - it
    is the file nobody downloads because the card never mentioned it."""
    text = _GOOD.replace("| [Model-Q4_K_M.gguf](x) | 0.47 GB |\n", "")
    problems = card.validate(text, _facts())
    assert any("omits" in p and "Model-Q4_K_M.gguf" in p for p in problems)


def test_a_wrong_size_is_caught():
    text = _GOOD.replace("0.47 GB", "3.10 GB")
    problems = card.validate(text, _facts())
    assert any("Model-Q4_K_M.gguf" in p and "3.1" in p for p in problems)


def test_rounding_is_allowed():
    """0.93 and 0.9 describe the same file. The check is for claims that are
    WRONG, not for a formatting preference."""
    assert card.validate(_GOOD.replace("0.93 GB", "0.9 GB"), _facts()) == []


def test_a_size_on_a_line_with_no_file_is_ignored():
    """Prose says things like "20 GB total". Only a figure sharing a line with
    exactly one filename is a claim about that file."""
    text = _GOOD + "\nThe whole repo is about 1.4 GB.\n"
    assert card.validate(text, _facts()) == []


def test_a_wrong_base_model_is_caught():
    problems = card.validate(_GOOD.replace("someone/Model", "someone/Other", 1),
                             _facts())
    assert any("base_model" in p for p in problems)


def test_base_model_must_be_omitted_when_the_source_is_unknown():
    """The Hub builds its model tree from this field. Unresolved means say
    nothing - a guess points the tree at a repo that does not exist."""
    facts = card.facts(_verification(), assessment={"fork_leads": []})
    facts["source_repo"] = None
    problems = card.validate(_GOOD, facts)
    assert any("never established" in p for p in problems)


def test_base_model_in_prose_is_not_mistaken_for_front_matter():
    """Only the front matter declares the relation; the same characters in a
    fenced example or a sentence are just text."""
    text = _GOOD + "\n```yaml\nbase_model: someone/Something-Else\n```\n"
    assert card.validate(text, _facts()) == []


def test_a_missing_fork_requirement_is_caught():
    """Without this note the files silently do not run."""
    facts = _facts()
    facts["fork"] = {"repo": "someone/llama.cpp-fork", "ref": "abc123"}
    problems = card.validate(_GOOD, facts)
    assert any("llama.cpp-fork" in p for p in problems)


def test_an_unsupported_arxiv_citation_is_caught():
    """The failure mode worth catching: a plausible id recalled rather than
    copied. A fabricated citation on a public repo is worse than none."""
    facts = _facts()
    facts["source"] = {"arxiv_ids": ["2502.16161"], "readme": "# Model"}
    problems = card.validate(_GOOD + "\nSee arXiv:2401.99999.\n", facts)
    assert any("2401.99999" in p for p in problems)


def test_an_arxiv_id_the_source_provides_is_allowed():
    facts = _facts()
    facts["source"] = {"arxiv_ids": ["2502.16161"], "readme": "# Model"}
    assert card.validate(_GOOD + "\nSee arXiv:2502.16161.\n", facts) == []


def test_an_arxiv_id_only_in_the_source_readme_is_allowed():
    """Publishers cite themselves in prose without the Hub emitting a tag."""
    facts = _facts()
    facts["source"] = {"arxiv_ids": [], "readme": "Paper: arXiv:2502.16161"}
    assert card.validate(_GOOD + "\nSee arXiv:2502.16161.\n", facts) == []


def test_layout_is_not_validated():
    """The whole point of the split: the agent may restructure freely as long
    as the facts hold."""
    text = """---
base_model: someone/Model
---

# Something completely different

## The big one
`Model-BF16.gguf` — 0.93 GB. Start here.

## The practical one
`Model-Q4_K_M.gguf` — 0.47 GB. Most people want this.

Extra sections, different tone, no table at all.
"""
    assert card.validate(text, _facts()) == []


def test_an_invented_quant_name_in_prose_is_caught():
    """REGRESSION, from a real card. It recommended "Q8_0_00", "Q6_K_S",
    "Q4_0_1" and "Q2_K_M" — none of which exist in llama.cpp at all — in a
    bulleted list rather than as filenames, so the filename rule never saw
    them."""
    text = _GOOD + """
- **High Performance:** Q8_0_00, Q6_K_S
- **Memory Efficient:** Q4_0_1, Q2_K_M
"""
    problems = card.validate(text, _facts())
    joined = " ".join(problems)
    assert "does not contain" in joined
    for invented in ("Q8_0_00", "Q6_K_S", "Q4_0_1", "Q2_K_M"):
        assert invented in joined


def test_recommending_a_quant_the_repo_has_is_fine():
    text = _GOOD + "\nMost people should take Q4_K_M. BF16 is the source.\n"
    assert card.validate(text, _facts()) == []


def test_a_filename_is_not_also_reported_as_a_bare_quant():
    """Model-Q4_K_M.gguf must be judged once, by the filename rule."""
    problems = card.validate(_GOOD, _facts())
    assert not any("does not contain" in p for p in problems)


def test_publish_refuses_a_card_that_fails_validation():
    """It must raise rather than upload - the check is worthless if the bad
    card goes up anyway. dry_run does not exempt it."""
    import pytest
    bad = _GOOD.replace("0.47 GB", "9.90 GB")
    with pytest.raises(card.CardRejected) as caught:
        card.publish(_verification(), assessment={"repo_id": "someone/Model",
                                                  "fork_leads": []},
                     content=bad, dry_run=True)
    assert caught.value.problems


def test_publish_without_content_still_renders_the_template():
    text = card.publish(_verification(),
                        assessment={"repo_id": "someone/Model",
                                    "fork_leads": []},
                        dry_run=True)
    assert "# Model GGUF" in text


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
