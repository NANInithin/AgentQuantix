"""Calls into huggingface_hub, checked against whatever version is installed.

These are offline: they never reach the network. What they check is that the
arguments we intend to pass are ones the installed huggingface_hub will
actually accept — which is the failure this file exists for.

`list_models(direction=-1)` was valid for the whole of 0.x and is a TypeError
on 1.x, so a machine that resolved a newer hub crashed on the very first
command while the machine it was developed on kept working. Nothing caught it
because the only test that would have needed the network.
"""

import inspect

from huggingface_hub import HfApi

from agentquantix import hub


def test_trending_kwargs_are_accepted_by_the_installed_hub():
    """REGRESSION. The 1.x signature dropped `direction`:

        TypeError: HfApi.list_models() got an unexpected keyword argument
        'direction'
    """
    parameters = inspect.signature(HfApi.list_models).parameters
    for name in hub.trending_kwargs(10):
        assert name in parameters, (
            f"list_models has no {name!r} in this huggingface_hub; "
            f"accepted: {sorted(parameters)}")


def test_trending_kwargs_ask_for_the_trending_sort():
    """Whatever the spelling, it must be the trending key -- sorting by
    downloads would silently return a completely different list."""
    sort = hub.trending_kwargs(10)["sort"]
    assert sort in ("trendingScore", "trending_score")


def test_trending_kwargs_carry_the_limit_and_full_flag():
    kwargs = hub.trending_kwargs(42)
    assert kwargs["limit"] == 42
    # full=True is what makes the one-request filter possible; without it the
    # results carry no tags and every candidate needs its own round trip.
    assert kwargs["full"] is True


def test_direction_is_only_sent_to_versions_that_take_it():
    parameters = inspect.signature(HfApi.list_models).parameters
    kwargs = hub.trending_kwargs(10)
    if "direction" in parameters:
        # 0.x sorts ascending without it, which would return the LEAST
        # trending models -- the exact opposite of the intent.
        assert kwargs["direction"] == -1
    else:
        assert "direction" not in kwargs


def test_every_other_hub_call_we_make_still_exists():
    """A cheap guard over the rest of the surface. These were all checked
    against 1.30 by hand; this keeps them checked."""
    import huggingface_hub as hub_pkg

    expected = {
        HfApi.model_info: ("repo_id", "files_metadata"),
        HfApi.list_repo_files: ("repo_id", "repo_type"),
        HfApi.create_repo: ("repo_id", "repo_type", "exist_ok"),
        HfApi.upload_file: ("path_or_fileobj", "path_in_repo", "repo_id"),
        hub_pkg.hf_hub_download: ("repo_id", "filename", "local_dir", "token"),
        hub_pkg.snapshot_download: ("repo_id", "local_dir", "token"),
    }
    for function, names in expected.items():
        parameters = inspect.signature(function).parameters
        missing = [n for n in names if n not in parameters]
        assert not missing, f"{function.__qualname__} lost {missing}"


def test_the_transfer_policy_constants_still_exist():
    """transfer.py flips HF_HUB_DISABLE_XET at runtime and compares against
    MAX_HTTP_DOWNLOAD_SIZE; both are internal-ish and worth pinning down."""
    from huggingface_hub import constants

    assert hasattr(constants, "HF_HUB_DISABLE_XET"), (
        "the xet policy toggles this at runtime; without it the whole "
        "per-phase transfer decision is inert")
    # We hardcode the cap rather than importing it, because it is absent
    # in older versions -- but when it IS present it must agree with us,
    # or our 46.57 GiB boundary is wrong.
    from agentquantix import transfer
    declared = getattr(constants, "MAX_HTTP_DOWNLOAD_SIZE", None)
    if declared is not None:
        assert declared == transfer.HTTP_DOWNLOAD_LIMIT_BYTES
