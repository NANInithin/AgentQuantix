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
def test_uploads_use_xet_by_default(monkeypatch):
    """REGRESSION. This asserted the exact opposite, on a "~10x slower"
    figure measured once on a home connection and never reproduced.

    It was wrong. huggingface_hub 1.x removed hf_transfer, and plain LFS
    uploads one file's parts sequentially over a single connection, so "xet
    off" meant a single TCP stream with no way to widen it. Measured on a
    1.12 GB freshly-cut quant into a fresh repo: 11.65 MB/s plain against
    33.6 MB/s on the wire and 64.4 MB/s effective with xet.
    """
    monkeypatch.setattr(transfer, "installed", lambda: True)
    enabled, why = transfer.for_upload()
    assert enabled is True
    assert "single connection" in why


def test_uploads_do_not_claim_xet_when_it_is_absent(monkeypatch):
    """pin() cannot enable a backend that is not installed, so the policy must
    not ask for one -- it would print a misleading "enabled" line."""
    monkeypatch.setattr(transfer, "installed", lambda: False)
    assert transfer.for_upload()[0] is False


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


def test_pin_on_requests_xets_high_performance_mode(monkeypatch):
    """HF_XET_HIGH_PERFORMANCE is the documented successor to
    HF_HUB_ENABLE_HF_TRANSFER, which huggingface_hub 1.x removed."""
    import os

    from huggingface_hub import constants
    monkeypatch.setattr(transfer, "installed", lambda: True)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)

    assert transfer.pin(True, "test") is True
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"


def test_pin_leaves_an_explicit_high_performance_choice_alone(monkeypatch):
    """It is a performance hint, not correctness. A user who turned it off
    meant it, and pin() must not quietly turn it back on."""
    import os

    from huggingface_hub import constants
    monkeypatch.setattr(transfer, "installed", lambda: True)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    monkeypatch.setenv("HF_XET_HIGH_PERFORMANCE", "0")

    transfer.pin(True, "test")
    assert os.environ["HF_XET_HIGH_PERFORMANCE"] == "0"


def test_pin_off_does_not_set_high_performance(monkeypatch):
    import os

    from huggingface_hub import constants
    monkeypatch.setattr(transfer, "installed", lambda: True)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", False)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)

    transfer.pin(False, "test")
    assert "HF_XET_HIGH_PERFORMANCE" not in os.environ


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


# =====================================================
# THE CAP IS NOT DOWNLOAD-ONLY
# =====================================================
def test_oversized_uploads_require_xet(monkeypatch):
    """REGRESSION. for_upload() took no size and always answered False in
    auto mode, on the stated belief that "there is no size at which xet
    becomes necessary to upload". That is wrong: huggingface_hub's own
    upload_file documents "up to 50 GB" and the large-folder path calls 50GB
    a hard limit.

    A 51.90 GiB BF16 was therefore pushed over plain LFS, failed, and retried
    on a 60s backoff into a limit that does not move.
    """
    monkeypatch.setattr(transfer, "installed", lambda: True)
    enabled, why = transfer.for_upload(size_gib=51.90)
    assert enabled is True
    assert "limit" in why


def test_the_cap_is_what_overrides_an_explicit_off(monkeypatch):
    """AQX_XET=off is a speed preference, and below the cap it is honoured.

    Above the cap it stops being a preference: plain LFS cannot send the file
    at all, so obeying it would turn a slow upload into a guaranteed failure.
    Same shape as for_download().
    """
    monkeypatch.setenv("AQX_XET", "off")
    monkeypatch.setattr(transfer, "installed", lambda: True)
    assert transfer.for_upload(size_gib=45.0)[0] is False
    assert transfer.for_upload(size_gib=75.0)[0] is True


def test_the_upload_cap_matches_the_download_cap(monkeypatch):
    """One limit, one threshold. A quant that can be fetched can be sent."""
    monkeypatch.setenv("AQX_XET", "off")
    monkeypatch.setattr(transfer, "installed", lambda: True)
    just_under = transfer.HTTP_DOWNLOAD_LIMIT_GIB - 0.01
    just_over = transfer.HTTP_DOWNLOAD_LIMIT_GIB + 0.01
    assert transfer.for_upload(size_gib=just_under)[0] is False
    assert transfer.for_upload(size_gib=just_over)[0] is True


def test_an_oversized_upload_without_the_backend_is_refused(monkeypatch):
    """Saying False here is what lets the caller drop the model at plan time
    rather than after the download and the conversion."""
    monkeypatch.setattr(transfer, "installed", lambda: False)
    enabled, why = transfer.for_upload(size_gib=75.0)
    assert enabled is False
    assert "cannot succeed" in why
