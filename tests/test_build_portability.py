"""Things that differ between platforms and Python versions.

CI runs the test suite on 3.10 and 3.13, Windows and Linux — but it never
builds llama.cpp, so the build module's platform handling is only covered by
what is asserted here. That is exactly where a version-specific API slips
through: it does not fail at import, only when a fork is finally cleaned up,
hours into a run on someone else's machine.
"""

import shutil
import sys
from pathlib import Path

from agentquantix.pipeline import build as build_mod


# =====================================================
# PYTHON VERSION
# =====================================================
def test_rmtree_hook_matches_this_interpreter():
    """REGRESSION. `onexc` is 3.12+; on 3.10/3.11 it is `onerror`, and passing
    the wrong one is a TypeError. The package supports 3.10, which is what
    Ubuntu 22.04 ships."""
    import inspect

    accepted = inspect.signature(shutil.rmtree).parameters
    expected = "onexc" if sys.version_info >= (3, 12) else "onerror"
    assert expected in accepted


def test_fork_cleanup_removes_a_read_only_tree(tmp_path, monkeypatch):
    """The functional half: git marks pack files read-only, so a plain rmtree
    dies partway through on Windows. This exercises the real call, including
    whichever error hook this interpreter takes."""
    from agentquantix import config

    forks = tmp_path / "aqx-forks"
    fork = forks / "some-fork"
    (fork / ".git" / "objects").mkdir(parents=True)
    stubborn = fork / ".git" / "objects" / "pack.idx"
    stubborn.write_text("x")
    stubborn.chmod(0o444)

    monkeypatch.setattr(config, "FORKS_DIR", forks)
    monkeypatch.delenv("KEEP_FORK", raising=False)

    build_mod.cleanup_fork(fork)
    assert not fork.exists()


def test_cleanup_refuses_to_touch_the_users_own_checkout(tmp_path, monkeypatch):
    """Fork cleanup deletes a whole tree. It must only ever be able to delete
    one it created, never the long-lived llama.cpp the user built themselves."""
    from agentquantix import config

    theirs = tmp_path / "llama.cpp"
    (theirs / "build").mkdir(parents=True)
    monkeypatch.setattr(config, "FORKS_DIR", tmp_path / "aqx-forks")

    build_mod.cleanup_fork(theirs)
    assert theirs.exists(), "cleanup must not delete a checkout it did not make"


# =====================================================
# TOOL DISCOVERY
# =====================================================
def test_find_tool_locates_something_on_path():
    assert build_mod.find_tool("python") or build_mod.find_tool("python3")


def test_find_tool_looks_beside_our_interpreter(tmp_path, monkeypatch):
    """REGRESSION. pip-installed ninja lands in the environment's Scripts dir,
    which uv and pipx deliberately keep off PATH — so the [build] extra
    installed a tool nothing could find and every build used the slow
    generator."""
    suffix = ".exe" if sys.platform == "win32" else ""
    fake = tmp_path / f"pretend-tool{suffix}"
    fake.write_text("")
    fake.chmod(0o755)

    monkeypatch.setattr(build_mod, "_interpreter_scripts", lambda: tmp_path)
    assert build_mod.find_tool("pretend-tool")


def test_build_env_puts_that_directory_on_path(tmp_path, monkeypatch):
    """Finding the tool is not enough: cmake resolves ninja from PATH itself."""
    monkeypatch.setattr(build_mod, "_interpreter_scripts", lambda: tmp_path)
    import os
    assert str(tmp_path) in build_mod.build_env()["PATH"].split(os.pathsep)


def test_build_env_preserves_the_rest_of_the_environment(monkeypatch):
    monkeypatch.setenv("AQX_CANARY", "present")
    assert build_mod.build_env().get("AQX_CANARY") == "present"


# =====================================================
# PLATFORM
# =====================================================
def test_binary_lookup_covers_both_layouts(tmp_path):
    """Windows multi-config generators nest binaries under Release/; Ninja and
    every Unix generator do not."""
    for relative in ("build/bin", "build/bin/Release"):
        root = tmp_path / relative.replace("/", "_")
        target = root / relative
        target.mkdir(parents=True)
        suffix = ".exe" if sys.platform == "win32" else ""
        (target / f"llama-quantize{suffix}").write_text("")
        assert build_mod.find_binary(root, "llama-quantize"), relative


def test_msvc_lookup_is_a_no_op_off_windows():
    if sys.platform != "win32":
        assert build_mod.msvc_env_bat() is None


def test_cuda_detection_does_not_require_a_gpu():
    """A CPU-only VM must get a clean False, not an exception - that decision
    is what keeps GGML_CUDA from being forced ON and failing the configure."""
    assert isinstance(build_mod.has_cuda_toolkit(), bool)


def test_build_jobs_is_a_sane_positive_number():
    jobs = int(build_mod.build_jobs())
    assert 1 <= jobs <= 256


def test_upstream_repo_is_the_official_one():
    """A bare machine clones this unattended; it must not drift to a fork."""
    assert build_mod.UPSTREAM_LLAMA_REPO.startswith("https://github.com/ggml-org/")


def test_work_root_falls_back_to_a_home_directory(monkeypatch, tmp_path):
    """On a machine with no ~/Documents/INF the work root must land somewhere
    writable, not on a Windows path that does not exist."""
    monkeypatch.delenv("AQX_WORK_ROOT", raising=False)
    monkeypatch.delenv("INF_ROOT", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    import importlib
    from agentquantix import config
    reloaded = importlib.reload(config)
    try:
        assert tmp_path in reloaded.WORK_ROOT.parents or \
            reloaded.WORK_ROOT == tmp_path / ".agentquantix"
    finally:
        importlib.reload(config)
