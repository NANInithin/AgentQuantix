"""The harness making sure a written card is actually published.

Observed twice on the same repo, through two prompt revisions: the model
composes the card, replies with the markdown, and stops. loop.py hands the
turn back to the human, nothing is published, and the repo keeps the generic
placeholder that start_quantization wrote. The run looks finished.

Instruction was not enough, so the loop detects it and asks for the tool call.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentquantix.agent import loop                              # noqa: E402

REPO = "NANI-Nithin/granite-4.2-3b-GGUF"

# Shortened from the card the model actually replied with.
REAL_CARD = """```yaml
---
base_model: ibm-granite/granite-4.2-3b
license: apache-2.0
---

# Granite-4.2-3B GGUF Quantization

This repository contains GGUF quantized versions of IBM Granite-4.2-3B.

| **Quant** | **Size** |
|-----------|----------|
| **BF16** | 6.82 GB |
| **Q4_K_M** | 2.09 GB |

granite-4.2-3b-Q4_K_M.gguf is the usual default for most people, and the
rest of the sweep is published alongside it in the same repository.
```
"""


MUSE = "NANI-Nithin/Muse-Glimmer-30B-GGUF"

# The second observed shape, shortened: a sentence of preamble, then the
# document, and no front matter at all. Anchoring on the reply's first
# character misses this one.
PREAMBLE_THEN_CARD = """I'll rewrite the model card for NANI-Nithin/Muse-Glimmer-30B-GGUF based on the source model facts I
retrieved. Here's my rewrite:

---

# Muse Glimmer-30B GGUF

**Authors:** Meta Superintelligence Lab

## Model Overview

Muse Glimmer-30B is a 30-billion-parameter causal language model built for
autonomous agentic tasks on consumer hardware. This GGUF repository provides
quantized versions for local deployment on device without cloud access.

## Quantization Options

| Quantization | Size (GB) | Best Use Case |
|--------------|-----------|---------------|
| **Q4_K_M** | 15.03 | Most users, standard deployment |
| **Q6_K** | 21.3 | When RAM isn't a constraint |

## Usage

huggingface-cli download NANI-Nithin/Muse-Glimmer-30B-GGUF Muse-Glimmer-30B-Q4_K_M.gguf --local-dir .
"""


def test_a_card_written_into_the_chat_is_detected():
    assert loop._unpublished_card({REPO}, REAL_CARD) == REPO


def test_a_card_behind_preamble_prose_is_detected():
    """REGRESSION. The reply opened with "I'll rewrite the model card for..."
    and carried no front matter, so both the first-character test and the
    base_model test missed it."""
    assert loop._unpublished_card({MUSE}, PREAMBLE_THEN_CARD) == MUSE


def test_an_ordinary_answer_is_left_alone():
    """Nudging when the model merely discussed a card would talk past the
    user, which is worse than the bug being fixed."""
    for reply in ("Done - the card is published.",
                  "Which quant would you like me to recommend first?",
                  "I cannot find a model called granite-4.2-3b on the Hub.",
                  ""):
        assert loop._unpublished_card({REPO}, reply) is None


def test_nothing_pending_means_no_nudge():
    assert loop._unpublished_card(set(), REAL_CARD) is None


def test_the_named_repo_wins_when_several_are_pending():
    other = "NANI-Nithin/Qwen3-0.6B-GGUF"
    assert loop._unpublished_card({REPO, other}, REAL_CARD) == REPO


def test_ambiguity_is_left_alone():
    """Two pending repos and a card naming neither: guessing would publish a
    card to the wrong repo, so say nothing."""
    card = "# Some GGUF\n\n" + "x" * 700 + "\nfile.gguf\n"
    assert loop._unpublished_card({"a/one-GGUF", "b/two-GGUF"}, card) is None


# =====================================================
# TRACKING
# =====================================================
def test_fetching_facts_opens_the_debt():
    pending = set()
    loop._track_card(pending, "get_card_facts", json.dumps({"repo_id": REPO}))
    assert pending == {REPO}


def test_publishing_settles_it():
    pending = {REPO}
    loop._track_card(pending, "write_model_card",
                     json.dumps({"repo": REPO, "published": True}))
    assert pending == set()


def test_a_rejected_card_stays_outstanding():
    """write_model_card returns published: false with the problems when
    validation fails. The card is still owed."""
    pending = {REPO}
    loop._track_card(pending, "write_model_card", json.dumps(
        {"repo": REPO, "published": False, "problems": ["bad size"]}))
    assert pending == {REPO}


def test_a_dry_run_does_not_settle_it():
    pending = {REPO}
    loop._track_card(pending, "write_model_card",
                     json.dumps({"repo": REPO, "published": False}))
    assert pending == {REPO}


def test_a_tool_error_changes_nothing():
    pending = set()
    loop._track_card(pending, "get_card_facts",
                     json.dumps({"error": "no GGUF files"}))
    assert pending == set()


def test_unparseable_output_changes_nothing():
    pending = {REPO}
    for junk in ("not json", "", "[1,2,3]", None):
        loop._track_card(pending, "write_model_card", junk)
    assert pending == {REPO}


def test_other_tools_are_ignored():
    pending = set()
    loop._track_card(pending, "start_quantization",
                     json.dumps({"repo": REPO, "published": True}))
    assert pending == set()
