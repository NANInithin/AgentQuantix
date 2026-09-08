"""The imatrix: will the file load, what runs the pass, and over how much text.

All three exist because of one run. `Nex-N2.5-mini` (qwen35moe, block_count 41,
nextn_predict_layers 1) converted to a BF16 whose tensors stop at block 39.
llama-quantize did not care — it streams tensors and never builds a graph — so
the sweep cut every quant and uploaded them. llama-imatrix was the first step
to actually load the model:

    check_tensor_dims: tensor 'blk.40.attn_norm.weight' not found

and the pipeline logged "continuing with the quants that do not need one".
The repo ended up full of files that build could not open.

The first attempt at a fix read the header and refused any model with a block
missing. That was the wrong question, and it blocked this model outright —
block 40 is a declared NextN head, and another publisher quantized the same
checkpoint the same day with IQ types, so a build that loads it exists.

Whether a model loads is answered by loading it. The header check only reports
a block missing for no declared reason, and the imatrix now runs before
anything is published so a load failure costs nothing.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentquantix import feasibility, hub                        # noqa: E402
from agentquantix.pipeline import sanity                         # noqa: E402


# =====================================================
# WILL IT LOAD
# =====================================================
def _fake_gguf(monkeypatch, architecture, block_count, blocks_present,
               nextn=None):
    """A gguf module whose reader describes one synthetic header."""

    class Field:
        def __init__(self, value):
            self._value = value

        def contents(self):
            return self._value

    class Tensor:
        def __init__(self, name):
            self.name = name

    class Reader:
        def __init__(self, _path):
            self.fields = {"general.architecture": Field(architecture)}
            if block_count is not None:
                self.fields[f"{architecture}.block_count"] = Field(block_count)
            if nextn is not None:
                self.fields[f"{architecture}.nextn_predict_layers"] = \
                    Field(nextn)
            self.tensors = [Tensor(f"blk.{i}.attn_norm.weight")
                            for i in blocks_present]
            self.tensors.append(Tensor("token_embd.weight"))

    module = types.ModuleType("gguf")
    module.GGUFReader = Reader
    monkeypatch.setitem(sys.modules, "gguf", module)


def test_a_declared_nextn_head_is_not_a_defect(monkeypatch):
    """REGRESSION, and an expensive false positive.

    Nex-N2.5-mini declares 41 blocks, exports 40, and sets
    nextn_predict_layers = 1: block 40 is a multi-token-prediction head the
    converter is right not to emit. The first version of this check reported
    it as fatal and blocked the model outright — while another publisher was
    quantizing the same checkpoint the same day, imatrix and IQ types
    included, so a build that loads it plainly exists.

    A missing block is not evidence that a model will not load. Only loading
    it is, which is why the pipeline runs the imatrix before it publishes.
    """
    _fake_gguf(monkeypatch, "qwen35moe", 41, range(40), nextn=1)
    assert sanity.missing_blocks("x.gguf") == []
    assert sanity.unloadable_reason("x.gguf") is None


def test_more_missing_than_nextn_explains_is_still_reported(monkeypatch):
    _fake_gguf(monkeypatch, "qwen35moe", 41, range(38), nextn=1)
    # 38 and 39 are unexplained; only 40 is covered by the NextN count.
    assert sanity.missing_blocks("x.gguf") == [38, 39]


def test_a_missing_block_with_no_nextn_declared_is_reported(monkeypatch):
    _fake_gguf(monkeypatch, "qwen35moe", 41, range(40))
    assert sanity.missing_blocks("x.gguf") == [40]
    assert "nextn_predict_layers" in sanity.unloadable_reason("x.gguf")


def test_a_complete_model_is_not_flagged(monkeypatch):
    _fake_gguf(monkeypatch, "llama", 32, range(32))
    assert sanity.missing_blocks("x.gguf") == []
    assert sanity.unloadable_reason("x.gguf") is None


def test_a_hole_in_the_middle_is_caught(monkeypatch):
    _fake_gguf(monkeypatch, "llama", 8, [0, 1, 2, 4, 5, 6, 7])
    assert sanity.missing_blocks("x.gguf") == [3]


def test_an_unreadable_header_blocks_nothing(monkeypatch):
    """Refusing to publish on the strength of a header we failed to parse
    would be a worse failure than the one being prevented."""
    _fake_gguf(monkeypatch, "llama", None, range(32))     # no block_count
    assert sanity.missing_blocks("x.gguf") == []
    assert sanity.unloadable_reason("x.gguf") is None


def test_a_missing_gguf_package_blocks_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "gguf", None)
    assert sanity.unloadable_reason("x.gguf") is None


def test_a_reader_that_raises_blocks_nothing(monkeypatch):
    module = types.ModuleType("gguf")

    def boom(_path):
        raise OSError("truncated file")

    module.GGUFReader = boom
    monkeypatch.setitem(sys.modules, "gguf", module)
    assert sanity.unloadable_reason("x.gguf") is None


# =====================================================
# LOAD FAILURE vs IMATRIX FAILURE
# =====================================================
# The distinction the original bug turned on. llama-imatrix exiting 1 because
# the model will not load and llama-imatrix exiting 1 because it ran out of
# memory need opposite responses: abandon the model, or carry on without the
# IQ types.
def test_a_load_failure_is_recognised():
    from agentquantix.pipeline import imatrix

    real = ("0.00.417.566 E llama_model_load: error loading model: "
            "check_tensor_dims: tensor 'blk.40.attn_norm.weight' not found\n"
            "0.00.812.127 E cmn  common_init_: failed to load model")
    assert imatrix._is_load_failure(real)


def test_an_ordinary_imatrix_failure_is_not_a_load_failure():
    """These must NOT abandon the model. The sweep loses the IQ types and
    keeps everything else, which is the behaviour that was always right for
    this class of failure."""
    from agentquantix.pipeline import imatrix

    for output in ("ggml_backend_cpu_buffer_type_alloc_buffer: failed to "
                   "allocate buffer of size 12884901888",
                   "std::bad_alloc",
                   "compute_imatrix: failed to eval",
                   "terminate called after throwing an instance of",
                   ""):
        assert not imatrix._is_load_failure(output)


def test_the_command_failure_carries_its_output():
    """run_verbose raises with one summary line. Classifying needs the lines
    around it, so the exception carries the tail."""
    from agentquantix.pipeline.build import CommandFailed

    error = CommandFailed("llama-imatrix failed: boom", output="line\nboom")
    assert error.output == "line\nboom"
    assert "boom" in str(error)


# =====================================================
# WHAT RUNS THE PASS
# =====================================================
def _candidate(params=30_000_000_000, layers=40):
    candidate = hub.Candidate(repo_id="x/M", rank=0)
    candidate.params = params
    candidate.n_layers = layers
    return candidate


def _plan(fast_gb, candidate=None):
    candidate = candidate or _candidate()
    return feasibility.imatrix_plan(
        candidate, {"fast_memory_gb": fast_gb, "vram_free_gb": 0},
        feasibility.bf16_size_gb(candidate), history={})


def test_the_bf16_is_used_when_it_fits():
    assert _plan(128)["source"] == "BF16"


def test_the_ladder_takes_the_largest_quant_that_fits():
    """REGRESSION. The ladder was Q8_0 / Q4_K_M / Q2_K, so a box with room for
    six bits was given four and a box with room for three was given two. The
    imatrix is only as good as the weights it is measured on, and the cost of
    a rung is entirely paid in quality."""
    assert _plan(32)["source"] == "Q6_K"       # was Q4_K_M
    assert _plan(24)["source"] == "Q5_K_M"     # was Q4_K_M
    assert _plan(16)["source"] == "Q3_K_M"     # was Q2_K


def test_the_ladder_shrinks_monotonically():
    sizes = [_plan(fast)["source_gb"] for fast in (128, 64, 48, 32, 24, 16)]
    assert sizes == sorted(sizes, reverse=True)


def test_nothing_on_the_ladder_needs_an_imatrix_itself():
    """Computing the matrix on a file that could not exist without one is
    circular. The IQ set and Q2_K_S are excluded by construction."""
    from agentquantix import config
    for quant in feasibility.IMATRIX_SOURCE_LADDER:
        assert quant not in config.IMATRIX_REQUIRED


def test_an_impossible_box_still_gets_a_source():
    """An imatrix from a thrashing Q2_K still beats no IQ quants at all."""
    plan = _plan(4)
    assert plan["source"] == feasibility.IMATRIX_SOURCE_LADDER[-1]
    assert plan["fits_fast_memory"] is False


# =====================================================
# HOW MUCH TEXT
# =====================================================
def test_a_small_model_gets_the_ceiling():
    """It costs seconds a chunk, so there is nothing to save by reading less.
    Every model used to get the same calibration regardless of price."""
    plan = feasibility.calibration_plan(0.6, history={})
    assert plan["chunks"] == feasibility.IMATRIX_MAX_CHUNKS
    assert plan["at_ceiling"] is True


def test_a_huge_model_is_floored_not_starved():
    """Below the floor the matrix is estimated from too little text. A model
    that cannot afford the floor overruns the budget and says so, rather than
    being given calibration that does not work."""
    plan = feasibility.calibration_plan(66.0, thrash=4.0, history={})
    assert plan["chunks"] == feasibility.IMATRIX_MIN_CHUNKS
    assert plan["at_floor"] is True
    assert plan["minutes"] > plan["target_minutes"]


def test_chunks_fall_as_the_model_grows():
    counts = [feasibility.calibration_plan(gb, history={})["chunks"]
              for gb in (3.2, 13.6, 20.2, 40.0)]
    assert counts == sorted(counts, reverse=True)


def test_thrashing_buys_fewer_chunks():
    """A source that pages off disk costs several times more per chunk, so it
    should be given fewer of them, not the same number at four times the
    price."""
    fast = feasibility.calibration_plan(20.0, thrash=1.0, history={})
    slow = feasibility.calibration_plan(20.0, thrash=4.0, history={})
    assert slow["chunks"] < fast["chunks"]


def test_the_rate_is_learned_once_a_run_records_chunks():
    history = {"imatrix": [{"chunks": 60, "gb": 13.6, "minutes": 15.0}]}
    plan = feasibility.calibration_plan(13.6, history=history)
    assert plan["rate_learned"] is True
    # 15 min for 60 chunks over 13.6 GB, so 25 min buys 100.
    assert plan["chunks"] == 100


def test_samples_without_a_chunk_count_are_ignored():
    """Every run before the plan chose a count read the whole pool, and how
    many chunks that came to is not recoverable. Inferring it would put a
    guessed denominator under every future estimate."""
    history = {"imatrix": [{"gb": 13.6, "minutes": 15.0}]}
    assert feasibility.calibration_plan(13.6, history=history)["rate_learned"] \
        is False


def test_the_estimate_and_the_plan_agree():
    """The time estimate reads the plan's own figure. Deriving it twice from
    the same inputs invites the two to drift."""
    candidate = _candidate()
    assessment = feasibility.assess(
        candidate, {"fast_memory_gb": 16, "vram_free_gb": 0, "disk_free_gb": 900,
                    "disk_gbs": 1.0},
        arch_ok=True, arch_detail="ok")
    planned = assessment["imatrix"]["calibration"]["minutes"] / 60
    assert assessment["hours"]["imatrix"] >= round(planned, 2) - 0.01
