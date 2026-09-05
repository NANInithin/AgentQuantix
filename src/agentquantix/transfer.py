"""When to use xet, decided per phase rather than per run.

The naive framing is "big model, turn xet on". The real rule is finer, and the
K2-Horizon script already found it by hand:

  * hf_xet is REQUIRED to DOWNLOAD any file over huggingface_hub's plain-HTTP
    cap. Without it `hf_hub_download` raises outright — a hard failure, not a
    slowdown.

  * hf_xet is the ONLY multi-connection upload path left. This module used to
    assert the opposite -- "roughly 10x SLOWER to UPLOAD, ~2 MB/s against ~20
    MB/s plain HTTP" -- from one measurement on a home connection, and forced
    xet off for every upload under the cap on that basis. It was wrong, and
    expensively so. Two things it missed:

      - huggingface_hub 1.x REMOVED hf_transfer ("hf_transfer is not used
        anymore", constants.py). Xet is the only accelerated backend that
        still exists.

      - The plain-LFS path uploads a single file's parts SEQUENTIALLY, one
        connection at a time -- `_upload_parts_iteratively` in lfs.py is a
        plain for-loop over `http_backoff("PUT", ...)`. And `upload_file`
        commits one file at a time, so the `num_threads=5` pool in
        `_upload_lfs_files` (which parallelises across files in a commit,
        not within one) always gets a work queue of exactly 1.

    So "xet off" meant one TCP stream for the whole run, with no way to use
    more. Measured on a 1.12 GB freshly-cut quant, fresh repo, nothing
    available to deduplicate against:

        plain HTTP   11.65 MB/s   (median of 11 recorded samples)
        xet          33.60 MB/s   on the wire, new bytes only
                     64.40 MB/s   effective, after chunk dedup/compression

    2.9x on the wire and 5.5x end to end, at one upload worker.

  * The 50 GB cap applies to UPLOADS too, which this module originally got
    wrong. `upload_file` documents "up to 50 GB" and the large-folder path
    calls it a hard limit. So a BF16 over the cap needs xet in BOTH
    directions, and every quant cut from it -- all smaller by construction --
    still wants it off.

What remains genuinely per-phase is the CAP: xet is REQUIRED above it and
merely preferable below, and only the required case may override an explicit
user preference. A single environment variable set once cannot express that,
which is why this still lives here rather than in the environment.

Two facts from huggingface_hub 0.36 make this implementable, both verified
against the installed package rather than assumed:

  * `MAX_HTTP_DOWNLOAD_SIZE` is 50 * 1000^3 bytes — 50 DECIMAL GB, which is
    46.57 GiB. Comparing a GiB figure against a bare 50 silently allows files
    that will then fail to download.

  * `constants.HF_HUB_DISABLE_XET` is read at CALL time by both consumers
    (`file_download` and `is_xet_available`, which the commit path uses), not
    captured at import. So it can be flipped around a phase and put back.
"""

from __future__ import annotations

import contextlib
import os

# huggingface_hub's own cap, in bytes, and the same figure in the GiB units the
# rest of the codebase measures files in. Kept as both so no call site has to
# remember which one it is holding.
HTTP_DOWNLOAD_LIMIT_BYTES = 50 * 1000 * 1000 * 1000
HTTP_DOWNLOAD_LIMIT_GIB = HTTP_DOWNLOAD_LIMIT_BYTES / 1024 ** 3   # 46.57

MODES = ("auto", "on", "off")


def mode():
    """The configured policy: "auto" (default), "on" or "off"."""
    value = (os.getenv("AQX_XET") or "auto").strip().lower()
    return value if value in MODES else "auto"


def installed():
    """Is hf_xet importable at all. Nothing here can turn on what is absent."""
    try:
        __import__("hf_xet")
        return True
    except ImportError:
        return False


def env_disabled():
    """Did the user explicitly disable xet in the environment.

    Respected as an intent signal, but not as a veto: in auto mode a download
    that cannot happen without xet still gets it, because failing the run to
    honour a speed preference would be obtuse.
    """
    return os.environ.get("HF_HUB_DISABLE_XET", "").lower() in ("1", "true", "yes")


def exceeds_http_limit(size_bytes=None, size_gib=None):
    """Is this file too large for the plain-HTTP download path."""
    if size_bytes is not None:
        return size_bytes > HTTP_DOWNLOAD_LIMIT_BYTES
    return (size_gib or 0) > HTTP_DOWNLOAD_LIMIT_GIB


# =====================================================
# POLICY
# =====================================================
def for_download(size_bytes=None, size_gib=None):
    """Should xet be used to fetch this file. Returns (enabled, reason).

    Only the size cap forces the decision. Below it, the setting is left
    exactly as the environment has it — the 10x penalty that justifies turning
    xet off was measured on the UPLOAD side, and there is no evidence it
    applies to downloads, so churning the setting on that basis would be
    inventing a reason.
    """
    configured = mode()
    if configured == "on":
        return True, "AQX_XET=on"
    if configured == "off":
        return False, "AQX_XET=off"

    if exceeds_http_limit(size_bytes, size_gib):
        if not installed():
            return False, "over the plain-HTTP cap but hf_xet is not installed"
        return True, (f"file is over huggingface_hub's "
                      f"{HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB plain-HTTP cap, "
                      "so xet is required")
    return None, "under the cap; leaving the setting alone"


def for_upload(size_bytes=None, size_gib=None):
    """Should xet be used to upload this file. Returns (enabled, reason).

    ON in auto mode whenever the backend is installed. Above the cap it is
    required — huggingface_hub's own upload_file documents "up to 50 GB", and
    a BF16 over that fails on the LFS path however many times it is retried.
    Below the cap it is merely much faster, for the reasons in the module
    docstring: plain LFS gives one sequential connection per file and nothing
    can widen it.

    Call it with no size for the ordinary case (every quant is smaller than
    the BF16 it was cut from, so only the BF16 can ever exceed this).
    """
    configured = mode()
    if configured == "on":
        return True, "AQX_XET=on"

    over_cap = exceeds_http_limit(size_bytes, size_gib)

    # AQX_XET=off is honoured right up to the point where it would guarantee a
    # failed upload rather than a slow one, which mirrors for_download().
    if configured == "off" and not over_cap:
        return False, "AQX_XET=off"

    if not installed():
        if over_cap:
            return False, ("over the plain-HTTP cap and hf_xet is not "
                           "installed - this upload cannot succeed")
        return False, "hf_xet is not installed"

    if over_cap:
        return True, (f"file is over the Hub's {HTTP_DOWNLOAD_LIMIT_GIB:.0f} "
                      "GiB per-file limit, so xet is required")
    return True, ("plain LFS uploads one file's parts sequentially over a "
                  "single connection; xet is the only parallel path")


# =====================================================
# APPLYING IT
# =====================================================
def enabled():
    """Whether xet is currently enabled in this process.

    getattr rather than a plain attribute read: the constant arrived in a
    particular huggingface_hub release, and a version without it should
    degrade to "xet is on" rather than crashing the whole command on an
    AttributeError from a policy helper.
    """
    from huggingface_hub import constants
    return not getattr(constants, "HF_HUB_DISABLE_XET", False)


def pin(want, reason=""):
    """Set the xet policy for everything that follows, and leave it set.

    This is the thread-safe half of the pair. The setting is process-global —
    a module constant plus an environment variable — so a context manager that
    restored it on exit would fight itself the moment two upload workers
    overlapped: one thread's restore would land in the middle of another
    thread's transfer.

    So phases that run concurrently (the uploads) pin the policy ONCE before
    any thread starts, and only genuinely single-threaded moments (fetching the
    BF16, before the uploaders exist) use the scoped form below.

    `want=None` means "leave it alone". Returns True if anything changed.
    """
    if want is None or want == enabled():
        return False

    from huggingface_hub import constants

    if want and not installed():
        # Nothing to enable. Say so rather than pretending it worked — the
        # caller is about to attempt something that may now fail.
        print("  [xet] requested but hf_xet is not installed - "
              "continuing without it.")
        return False

    constants.HF_HUB_DISABLE_XET = not want
    # Mirrored into the environment so any subprocess inherits the decision.
    os.environ["HF_HUB_DISABLE_XET"] = "0" if want else "1"
    if want:
        # Xet's high-throughput mode. Unset by default, and it is the
        # documented successor to HF_HUB_ENABLE_HF_TRANSFER, which 1.x
        # removed. Only ever turned ON here - an explicit 0 from the user is
        # left standing, since this is a performance hint and not correctness.
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if reason:
        print(f"  [xet] {'enabled' if want else 'disabled'}: {reason}")
    return True


@contextlib.contextmanager
def applied(want, reason=""):
    """Force xet on or off for one block, then restore. Single-threaded only.

    `want=None` means "leave it alone", which keeps call sites free of
    conditionals. A no-op when the setting already matches, which is the common
    case and avoids touching global state for nothing.
    """
    from huggingface_hub import constants

    previous_constant = constants.HF_HUB_DISABLE_XET
    previous_env = os.environ.get("HF_HUB_DISABLE_XET")

    if not pin(want, reason):
        yield
        return
    try:
        yield
    finally:
        constants.HF_HUB_DISABLE_XET = previous_constant
        if previous_env is None:
            os.environ.pop("HF_HUB_DISABLE_XET", None)
        else:
            os.environ["HF_HUB_DISABLE_XET"] = previous_env


def summary():
    """One line describing the effective policy, for the probe and reports."""
    configured = mode()
    if not installed():
        return ("hf_xet not installed - files over "
                f"{HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB cannot be downloaded")
    if configured == "on":
        return "xet forced ON for every transfer (AQX_XET=on)"
    if configured == "off":
        return ("xet forced OFF for every transfer (AQX_XET=off) - files over "
                f"{HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB cannot be downloaded, and "
                "uploads fall back to a single sequential connection")
    return ("auto: xet on for uploads (plain LFS is single-connection) and "
            f"for any transfer over {HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB")
