"""The estimator: disk arithmetic, verdicts, and the already-published logic.

Several of these are regressions for bugs that shipped. They are marked as
such, because a test whose reason is understood survives a refactor and one
that reads as an arbitrary assertion gets deleted.
"""

from agentquantix import feasibility


# =====================================================
# CONCURRENCY -> DISK
# =====================================================
def test_default_concurrency_is_three_not_two():
    """REGRESSION. Peak disk was documented and computed as BF16 + 2 quants.

    With a one-slot queue there are three: one being written, one waiting, one
    uploading. Undercounting by a whole quant matters because peak disk is a
    BLOCKER criterion — the estimate said a model fit when it did not.
    """
    assert feasibility.concurrent_quants() == 3


def test_sequential_holds_exactly_one():
    assert feasibility.concurrent_quants(sequential=True) == 1


def test_concurrency_is_the_sum_of_its_parts():
    assert feasibility.concurrent_quants(quantize_workers=2, upload_workers=3,
                                         queue_depth=4) == 9


def test_sequential_overrides_the_worker_counts():
    """--sequential means no overlap at all, whatever else was asked for."""
    assert feasibility.concurrent_quants(quantize_workers=4, upload_workers=4,
                                         queue_depth=4, sequential=True) == 1


def test_peak_disk_shrinks_under_sequential():
    assessment = {
        "bf16_gb": 50.0,
        "peak_convert_gb": 50.0,
        "quant_sizes_gb": {"Q8_0": 30.0, "Q6_K": 23.0, "Q5_K_M": 20.0,
                           "Q4_K_M": 17.0},
        "imatrix": {"source": "BF16", "source_gb": 50.0},
    }
    default = feasibility.peak_disk_for(assessment)
    sequential = feasibility.peak_disk_for(assessment, sequential=True)

    assert default == 50.0 + 30.0 + 23.0 + 20.0     # BF16 + three largest
    assert sequential == 50.0 + 30.0                # BF16 + one
    assert sequential < default


def test_peak_disk_counts_a_quant_imatrix_source():
    """An imatrix cut from a quant is an extra file on disk for the whole run."""
    base = {"bf16_gb": 50.0, "peak_convert_gb": 0.0,
            "quant_sizes_gb": {"Q8_0": 30.0},
            "imatrix": {"source": "BF16", "source_gb": 50.0}}
    on_quant = dict(base, imatrix={"source": "Q2_K", "source_gb": 9.0})

    assert (feasibility.peak_disk_for(on_quant, sequential=True)
            - feasibility.peak_disk_for(base, sequential=True)) == 9.0


def test_peak_disk_never_below_the_conversion_moment():
    """Making the sweep sequential does not shrink safetensors + BF16."""
    assessment = {"bf16_gb": 10.0, "peak_convert_gb": 24.0,
                  "quant_sizes_gb": {"Q4_K_M": 4.0},
                  "imatrix": {"source": "BF16", "source_gb": 10.0}}
    assert feasibility.peak_disk_for(assessment, sequential=True) == 24.0


# =====================================================
# SIZES
# =====================================================
def test_quant_size_scales_with_bits_per_weight():
    small = feasibility.quant_size_gb(7e9, "IQ1_S")
    large = feasibility.quant_size_gb(7e9, "Q8_0")
    assert small < large
    # Q8_0 is ~8.5 bpw: 7e9 * 8.5 / 8 bytes is a shade over 7 GiB.
    assert 6.5 < large < 8.0


def test_moe_models_get_the_moe_only_quant(candidate):
    dense = feasibility.quant_list(candidate)
    assert "MXFP4_MOE" not in dense

    candidate.n_experts = 128
    assert "MXFP4_MOE" in feasibility.quant_list(candidate)


# =====================================================
# IMATRIX STRATEGY
# =====================================================
def test_imatrix_uses_bf16_when_it_fits(candidate, sysinfo):
    plan = feasibility.imatrix_plan(candidate, sysinfo, bf16_gb=13.0)
    assert plan["source"] == "BF16"
    assert plan["fits_fast_memory"]


def test_imatrix_falls_back_to_the_largest_quant_that_fits(candidate, sysinfo):
    """A BF16 far larger than memory would page for every chunk, turning a
    fifteen-minute job into a three-hour one.

    `fits_fast_memory` describes the SOURCE that was chosen, not the BF16 —
    the whole point of falling back is to end up with one that fits, so it
    reads True here.
    """
    plan = feasibility.imatrix_plan(candidate, sysinfo, bf16_gb=140.0)
    assert plan["source"] != "BF16"
    assert plan["fits_fast_memory"]
    # Q8_0 is near-lossless, so the ladder must not skip past it to something
    # rougher while it still fits.
    assert plan["source"] == "Q8_0"


def test_imatrix_admits_when_even_the_smallest_will_thrash(candidate, sysinfo):
    """On a machine too small for any source, say so rather than pretending."""
    tiny = dict(sysinfo, fast_memory_gb=0.5, vram_free_gb=0.0)
    plan = feasibility.imatrix_plan(candidate, tiny, bf16_gb=140.0)
    assert plan["source"] == "Q2_K"
    assert not plan["fits_fast_memory"]


def test_imatrix_offload_is_bounded_by_free_vram(candidate, sysinfo):
    plan = feasibility.imatrix_plan(candidate, dict(sysinfo, vram_free_gb=0.0),
                                    bf16_gb=13.0)
    assert plan["ngl"] == 0


# =====================================================
# VERDICTS
# =====================================================
def _assess(candidate, sysinfo, **overrides):
    for key, value in overrides.items():
        setattr(candidate, key, value)
    return feasibility.assess(candidate, sysinfo, True, "supported: Llama")


def test_a_plain_supported_model_is_runnable(candidate, sysinfo):
    result = _assess(candidate, sysinfo)
    assert result["verdict"] in ("ok", "warn")
    assert not result["blockers"]
    assert result["quants"]


def test_blocked_when_peak_disk_exceeds_free_space(candidate, sysinfo):
    result = _assess(candidate, dict(sysinfo, disk_free_gb=5.0))
    assert result["verdict"] == "blocked"
    assert any("peak disk" in b for b in result["blockers"])


def test_blocked_without_a_token(candidate, sysinfo):
    result = _assess(candidate, dict(sysinfo, hf_token=False))
    assert result["verdict"] == "blocked"


def test_unsupported_with_no_fork_is_blocked(candidate, sysinfo):
    result = feasibility.assess(candidate, sysinfo, False, "unsupported: Xyz")
    assert result["verdict"] == "blocked"


def test_unsupported_with_a_fork_is_only_a_warning(candidate, sysinfo):
    result = feasibility.assess(
        candidate, sysinfo, False, "unsupported: Xyz",
        fork_leads=[{"repo": "org/llama.cpp", "ref": "model/x",
                     "confidence": "high", "why": "branch"}])
    assert result["verdict"] == "warn"
    assert result["hours"]["fork_build"] > 0


# =====================================================
# ALREADY PUBLISHED
# =====================================================
def test_fully_published_is_done_and_costs_nothing(candidate, sysinfo):
    """REGRESSION. A finished repo was reported as hours of work, because
    nothing checked the user's own namespace."""
    candidate.published_quants = set(feasibility.quant_list(candidate)) | {"BF16"}
    result = _assess(candidate, sysinfo)

    assert result["verdict"] == "done"
    assert result["quants"] == []
    assert result["hours"]["total"] == 0
    assert result["peak_disk_gb"] == 0
    assert not result["blockers"]


def test_partly_published_is_priced_on_the_remainder(candidate, sysinfo):
    everything = feasibility.quant_list(candidate)
    full = _assess(candidate, sysinfo)

    candidate.published_quants = set(everything[:-3]) | {"BF16"}
    partial = _assess(candidate, sysinfo)

    assert len(partial["quants"]) == 3
    assert partial["all_quants"] == everything
    assert partial["hours"]["total"] < full["hours"]["total"]
    # The BF16 is already up there, so it is not an upload cost again.
    assert partial["upload_gb"] < full["upload_gb"]
    assert partial["source_kind"] == "refetch-ours"


def test_published_count_ignores_non_sweep_files(candidate, sysinfo):
    """REGRESSION. BF16 and mmproj are published but are not sweep members;
    counting them produced "30 of 29 quants already published"."""
    candidate.published_quants = {"BF16", "Q4_K_M"}
    result = _assess(candidate, sysinfo)
    assert result["published_count"] == 1
    assert result["published_count"] <= len(result["all_quants"])


# =====================================================
# RANKING
# =====================================================
def test_ranking_puts_runnable_first_and_blocked_last():
    def row(verdict, hours, rank=1):
        return {"verdict": verdict, "hours": {"total": hours}, "rank": rank}

    rows = [row("blocked", 1), row("done", 1), row("warn", 9), row("ok", 5)]
    order = [r["verdict"] for r in sorted(rows, key=feasibility.rank_key)]
    assert order == ["ok", "warn", "done", "blocked"]


def test_cheaper_runnable_models_sort_higher():
    def row(hours):
        return {"verdict": "ok", "hours": {"total": hours}, "rank": 1}

    rows = [row(40), row(2), row(9)]
    assert [r["hours"]["total"] for r in sorted(rows, key=feasibility.rank_key)] \
        == [2, 9, 40]


# =====================================================
# TIMING HISTORY
# =====================================================
def test_record_accepts_a_field_called_kind(tmp_path, monkeypatch):
    """REGRESSION. record(kind, **fields) collided with a caller recording its
    own "kind" field:

        TypeError: record() got multiple values for argument 'kind'

    Every download the pipeline timed hit this, aborting the run right after
    the weights finished downloading. Windows never saw it because the BF16
    was already on disk from an earlier hand-run, so the download path -- the
    only caller that passes kind= -- never executed.
    """
    from agentquantix import config

    monkeypatch.setattr(config, "TIMING_HISTORY", tmp_path / "timings.json")
    feasibility.record("downloads", kind="safetensors", mbps=12.5, gb=2.0)

    samples = feasibility.load_history()["downloads"]
    assert samples[-1]["kind"] == "safetensors"
    assert samples[-1]["mbps"] == 12.5


def test_record_never_raises_on_a_bad_history_file(tmp_path, monkeypatch):
    """Timing data is a nicety. It must never be able to fail a run."""
    from agentquantix import config

    broken = tmp_path / "timings.json"
    broken.write_text("{ not json")
    monkeypatch.setattr(config, "TIMING_HISTORY", broken)

    feasibility.record("uploads", mbps=1.0)          # must not raise
    assert feasibility.load_history()["uploads"][-1]["mbps"] == 1.0
