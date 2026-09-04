"""The xet policy: the size boundary, the per-phase decision, and toggling.

The boundary test is the important one. huggingface_hub's cap is 50 DECIMAL GB
(46.57 GiB), and this codebase measures files in GiB — so a bare `50` in the
comparison quietly admitted files between 46.6 and 50 GiB that then failed to
download partway into a run.
"""

import pytest

from agentquantix import transfer


# =====================================================
# THE BOUNDARY
# =====================================================
def test_the_cap_is_decimal_gb_expressed_in_gib():
    assert transfer.HTTP_DOWNLOAD_LIMIT_BYTES == 50 * 1000 * 1000 * 1000
    assert 46.5 < transfer.HTTP_DOWNLOAD_LIMIT_GIB < 46.6


def test_files_between_46_and_50_gib_are_over_the_cap():
    """REGRESSION. These are the ones a bare `> 50` comparison let through."""
    assert not transfer.exceeds_http_limit(size_gib=46.0)
    assert transfer.exceeds_http_limit(size_gib=48.0)
    assert transfer.exceeds_http_limit(size_gib=53.0)


def test_the_byte_form_agrees_with_the_gib_form():
    just_over = transfer.HTTP_DOWNLOAD_LIMIT_BYTES + 1
    assert transfer.exceeds_http_limit(size_bytes=just_over)
    assert not transfer.exceeds_http_limit(
        size_bytes=transfer.HTTP_DOWNLOAD_LIMIT_BYTES)


# =====================================================
# POLICY
# =====================================================
def test_uploads_never_use_xet_by_default():
    """It is ~10x slower on the step that dominates almost every run."""
    enabled, _ = transfer.for_upload()
    assert enabled is False


def test_small_downloads_leave_the_setting_alone():
    """None means 'do not touch it' - there is no evidence xet is slower on
    the download side, so churning the global for no reason is wrong."""
    enabled, _ = transfer.for_download(size_gib=2.0)
    assert enabled is None


def test_oversized_downloads_request_xet(monkeypatch):
    monkeypatch.setattr(transfer, "installed", lambda: True)
    enabled, why = transfer.for_download(size_gib=75.0)
    assert enabled is True
    assert "cap" in why


def test_oversized_download_without_the_backend_is_refused(monkeypatch):
    monkeypatch.setattr(transfer, "installed", lambda: False)
    enabled, why = transfer.for_download(size_gib=75.0)
    assert enabled is False
    assert "not installed" in why


@pytest.mark.parametrize("mode,expected", [("on", True), ("off", False)])
def test_an_explicit_mode_pins_both_phases(monkeypatch, mode, expected):
    monkeypatch.setenv("AQX_XET", mode)
    monkeypatch.setattr(transfer, "installed", lambda: True)
    assert transfer.for_upload()[0] is expected
    assert transfer.for_download(size_gib=75.0)[0] is expected
    assert transfer.for_download(size_gib=1.0)[0] is expected


def test_an_unrecognised_mode_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("AQX_XET", "nonsense")
    assert transfer.mode() == "auto"


# =====================================================
# APPLYING IT
# =====================================================
def test_pin_changes_the_constant_and_the_environment(monkeypatch):
    from huggingface_hub import constants
    monkeypatch.setattr(transfer, "installed", lambda: True)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)

    assert transfer.pin(False, "test") is True
    assert constants.HF_HUB_DISABLE_XET is True
    # Mirrored so subprocesses inherit the decision.
    import os
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_pin_is_a_no_op_when_already_correct(monkeypatch):
    """Uploads pin the policy once; repeat calls must not churn global state
    while other threads are mid-transfer."""
    from huggingface_hub import constants
    monkeypatch.setattr(transfer, "installed", lambda: True)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    assert transfer.pin(False, "again") is False


def test_applied_restores_the_previous_setting(monkeypatch):
    from huggingface_hub import constants
    monkeypatch.setattr(transfer, "installed", lambda: True)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)

    with transfer.applied(True, "big download"):
        assert constants.HF_HUB_DISABLE_XET is False
    assert constants.HF_HUB_DISABLE_XET is True


def test_applied_with_none_touches_nothing(monkeypatch):
    from huggingface_hub import constants
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    with transfer.applied(None):
        assert constants.HF_HUB_DISABLE_XET is True


def test_summary_mentions_the_limit():
    assert "47 GiB" in transfer.summary() or "46" in transfer.summary()
