"""When to use xet, decided per phase rather than per run.

The naive framing is "big model, turn xet on". The real rule is finer, and the
K2-Horizon script already found it by hand:

  * hf_xet is REQUIRED to DOWNLOAD any file over huggingface_hub's plain-HTTP
    cap. Without it `hf_hub_download` raises outright — a hard failure, not a
    slowdown.

  * hf_xet is roughly 10x SLOWER to UPLOAD on this connection. Measured ~2 MB/s
    against ~20 MB/s plain HTTP, with the CPU at 0.1 cores and the disk 67%
    idle, so it is the backend and not the link.

A run of a 75 GB model therefore wants xet ON for one download and OFF for the
thirty uploads that follow. A single environment variable set once cannot
express that, which is why it lives here instead.

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


def for_upload():
    """Should xet be used to upload. Returns (enabled, reason).

    Always off in auto mode. There is no size at which xet becomes necessary
    to upload — the commit path merely OFFERS it as a transfer and falls back
    to LFS when it is unavailable — and it is an order of magnitude slower
    here, on a step that dominates almost every run.
    """
    configured = mode()
    if configured == "on":
        return True, "AQX_XET=on"
    if configured == "off":
        return False, "AQX_XET=off"
    return False, "xet uploads ~10x slower than plain HTTP; not needed for uploads"


# =====================================================
# APPLYING IT
# =====================================================
def enabled():
    """Whether xet is currently enabled in this process."""
    from huggingface_hub import constants
    return not constants.HF_HUB_DISABLE_XET


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
                f"{HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB cannot be downloaded")
    return (f"auto: xet on for downloads over {HTTP_DOWNLOAD_LIMIT_GIB:.0f} GiB, "
            "off for all uploads")
