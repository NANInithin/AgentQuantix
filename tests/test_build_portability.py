"""Things that differ between platforms and Python versions.

CI runs the test suite on 3.10 and 3.13, Windows and Linux — but it never
builds llama.cpp, so the build module's platform handling is only covered by
what is asserted here. That is exactly where a version-specific API slips
through: it does not fail at import, only when a fork is finally cleaned up,
hours into a run on someone else's machine.
"""

import shutil
import sys

import pytest
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
# THE BUILD INVOCATION
# =====================================================
def _capture_build(monkeypatch, tmp_path, *, vcvars, ninja=True,
                   platform="linux"):
    """Run compile_tools far enough to see what it would execute.

    The platform is simulated so both branches are covered from either host —
    the Linux path is the one that broke, and it cannot be reached from a
    Windows test run otherwise.
    """
    commands = []

    monkeypatch.setattr(build_mod.sys, "platform", platform)
    monkeypatch.setattr(build_mod, "msvc_env_bat", lambda: vcvars)
    monkeypatch.setattr(build_mod, "find_tool",
                        lambda name: "/usr/bin/ninja" if ninja and name == "ninja"
                        else None)
    monkeypatch.setattr(build_mod, "build_settings", lambda: ["-DGGML_CUDA=OFF"])
    monkeypatch.setattr(build_mod, "build_jobs", lambda: "4")
    monkeypatch.setattr(build_mod, "build_is_stale", lambda *a: False)
    monkeypatch.setattr(build_mod, "run",
                        lambda cmd, **kw: commands.append([str(c) for c in cmd]))
    # Pretend the compile produced its binaries.
    monkeypatch.setattr(build_mod, "binaries", lambda d: ("quantize", "imatrix"))
    monkeypatch.setattr(build_mod.feasibility, "record", lambda *a, **k: None)

    build_mod.compile_tools(tmp_path, tmp_path)
    return commands


def test_no_vcvars_means_cmake_is_invoked_directly(monkeypatch, tmp_path):
    """REGRESSION. The batch-file detour was gated on the generator alone, so
    Linux picked Ninja and then tried to run it through cmd.exe:

        cmd /c /home/.../llama.cpp/_build.bat
        Bootstrap failed: FileNotFoundError: No such file or directory: 'cmd'
    """
    commands = _capture_build(monkeypatch, tmp_path, vcvars=None)

    assert all(cmd[0] != "cmd" for cmd in commands), commands
    assert commands[0][0] == "cmake" and "-G" in commands[0]
    assert "Ninja" in commands[0]
    assert commands[1][:2] == ["cmake", "--build"]
    assert not (tmp_path / "_build.bat").exists()


def test_windows_with_vcvars_still_uses_the_batch_wrapper(monkeypatch, tmp_path):
    """The detour is required there: Ninja cannot find cl.exe without it."""
    commands = _capture_build(monkeypatch, tmp_path,
                              vcvars=tmp_path / "vcvars64.bat",
                              platform="win32")
    assert commands[0][0] == "cmd"


def test_cuda_arch_is_not_probed_for_a_cpu_build(monkeypatch, capsys):
    """REGRESSION. A CPU-only machine got 'could not determine CUDA arch - the
    build will target every supported architecture and take far longer', which
    is alarming and untrue: that build compiles no CUDA at all."""
    monkeypatch.setattr(build_mod, "_cmake_cache_text", lambda: "")
    monkeypatch.setattr(build_mod, "has_cuda_toolkit", lambda: False)
    monkeypatch.setattr(build_mod, "cuda_arch_from_gpu",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("must not probe the GPU")))

    defines = build_mod.build_settings()
    assert "-DGGML_CUDA=OFF" in defines
    assert not any("CUDA_ARCHITECTURES" in d for d in defines)
    assert "far longer" not in capsys.readouterr().out


def test_cuda_arch_is_pinned_when_cuda_is_on(monkeypatch):
    """Pinning to this GPU alone is worth ~8x on the ggml-cuda compile."""
    monkeypatch.setattr(build_mod, "_cmake_cache_text", lambda: "")
    monkeypatch.setattr(build_mod, "has_cuda_toolkit", lambda: True)
    monkeypatch.setattr(build_mod, "cuda_arch_from_gpu", lambda: "89")

    assert "-DCMAKE_CUDA_ARCHITECTURES=89" in build_mod.build_settings()


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


# =====================================================
# THE ARCHITECTURE GATE
# =====================================================
def test_registrations_parse_out_of_source(tmp_path):
    """REGRESSION. The gate ran convert_hf_to_gguf.py, which imports torch --
    not one of our dependencies. On a machine without it every candidate came
    back "could not read the supported-architecture list", so the whole
    trending report showed as unsupported."""
    from agentquantix import archsupport

    conversion = tmp_path / "conversion"
    conversion.mkdir()
    (conversion / "llama.py").write_text(
        '@ModelBase.register("LlamaForCausalLM")\nclass X: pass\n')
    (conversion / "multi.py").write_text(
        '@ModelBase.register(\n    "AForCausalLM",\n    "BForCausalLM",\n)\n'
        'class Y: pass\n')

    names = archsupport._parse_registrations(tmp_path)
    assert {"LlamaForCausalLM", "AForCausalLM", "BForCausalLM"} <= names


def test_the_gate_falls_back_when_the_converter_cannot_run(tmp_path, monkeypatch):
    from agentquantix import archsupport

    conversion = tmp_path / "conversion"
    conversion.mkdir()
    (conversion / "m.py").write_text('@ModelBase.register("OnlyOne")\n')
    monkeypatch.setattr(archsupport, "_ask_converter",
                        lambda d: (set(), "ModuleNotFoundError: torch"))

    assert archsupport.supported_architectures(tmp_path) == {"OnlyOne"}


def test_converter_status_reports_why_it_failed(tmp_path, monkeypatch):
    """The old message named no cause. 'No module named torch' is fixable;
    'could not read the supported-architecture list' is a mystery."""
    from agentquantix import archsupport

    conversion = tmp_path / "conversion"
    conversion.mkdir()
    (conversion / "m.py").write_text('@ModelBase.register("OnlyOne")\n')
    monkeypatch.setattr(archsupport, "_ask_converter",
                        lambda d: (set(), "ModuleNotFoundError: torch"))

    status = archsupport.converter_status(tmp_path)
    assert status["ok"] and status["source"] == "source-parse"
    assert "torch" in status["error"]


# =====================================================
# CONVERTER DEPENDENCIES
# =====================================================
def test_converter_requirements_are_detected(monkeypatch):
    """REGRESSION. Missing torch surfaced as a subprocess exit code AFTER the
    weights downloaded -- 5.6 GB for gpt2 -- and the message named neither the
    package nor the fix."""
    from agentquantix.pipeline import source

    real_import = __import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_torch)
    assert "torch" in source.converter_missing()


def test_the_hint_names_the_packages_and_the_fix():
    from agentquantix.pipeline import source

    hint = source.converter_hint(["torch", "transformers"])
    assert "torch" in hint and "transformers" in hint
    assert "uv tool install" in hint
    # And that models with a publisher GGUF do not need any of it, so nobody
    # installs a gigabyte they did not have to.
    assert "already ships a BF16 GGUF" in hint


def test_conversion_refuses_before_downloading(tmp_path, monkeypatch):
    """The check must happen before snapshot_download, not after."""
    from agentquantix.pipeline import source
    from agentquantix.pipeline.job import Job

    monkeypatch.setattr(source, "converter_missing", lambda: ["torch"])
    monkeypatch.setattr(source, "snapshot_download",
                        lambda **kw: pytest.fail("downloaded despite no torch"))

    job = Job(repo_id="org/M", base_name="M", target_repo="me/M-GGUF",
              quants=["Q4_K_M"], source_kind="convert")
    monkeypatch.setattr(type(job), "work_dir",
                        property(lambda self: tmp_path / "M"))

    with pytest.raises(RuntimeError, match="torch"):
        source.ensure_bf16(job, tmp_path, set())
