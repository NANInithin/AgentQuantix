"""Getting a llama-quantize and a llama-imatrix that know this architecture.

Two paths:

  * The local checkout at INF, already CUDA-built. Used whenever upstream
    supports the architecture, which is the common case, and costs nothing.

  * A fork. When upstream cannot convert the model, the agent clones the
    branch the research step found and builds it in temp/aqx-forks/<slug>.
    Forks are keyed by repo+ref and REUSED, because the CUDA compile is by far
    the most expensive step here and two models needing the same branch should
    pay for it once.

The build configuration is copied from the user's existing upstream build
rather than invented, so a fork build matches how llama.cpp is already built on
this machine — same generator, same CUDA arch pin, same native flags.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import stat
import subprocess
import sys
import time

from .. import config, feasibility


def run(cmd, **kwargs):
    cmd = [str(c) for c in cmd]
    if cmd[0].lower() in {"python", "python.exe"}:
        cmd[0] = sys.executable
    print("\n" + " ".join(cmd) + "\n", flush=True)
    # Our own Scripts directory goes on PATH for every build subprocess — see
    # build_env(). cmake looks up ninja itself, so finding it is not enough.
    kwargs.setdefault("env", build_env())
    subprocess.run(cmd, check=True, **kwargs)


ERROR_MARKERS = ("Error:", "error:", "ValueError", "TypeError", "KeyError",
                 "RuntimeError", "NotImplementedError", "AssertionError",
                 "ModuleNotFoundError", "ImportError", "OSError", "raise ")


def run_verbose(cmd, label="command"):
    """Run a command, stream its output, and raise with the REAL error.

    subprocess.run(check=True) raises CalledProcessError, whose message is the
    argv and an exit code. For a converter that failed with

        ValueError: Can not map tensor 'h.0.attn.bias'

    that reduces the one useful line to "returned non-zero exit status 1",
    and the summary at the end of a multi-model run then repeats the argv
    instead of the reason. The output is still streamed live, because a long
    conversion with no visible progress looks hung.
    """
    import collections

    cmd = [str(c) for c in cmd]
    if cmd[0].lower() in {"python", "python.exe"}:
        cmd[0] = sys.executable
    print("\n" + " ".join(cmd) + "\n", flush=True)

    tail = collections.deque(maxlen=60)
    # stdin is closed, not inherited. A model with custom code makes
    # transformers ask "Do you wish to run the custom code? [y/N]" on stdin,
    # and a converter running in a quantize worker thread would sit on that
    # prompt forever with its question buried in interleaved output. Closed
    # stdin turns the hang into an immediate, reportable failure.
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, text=True,
                               bufsize=1, env=build_env())
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    code = process.wait()
    if code == 0:
        return

    # The last line that looks like an exception is almost always the point.
    # Falling back to the last non-empty line beats an exit code either way.
    reason = next((l.strip() for l in reversed(tail)
                   if any(m in l for m in ERROR_MARKERS) and l.strip()), None)
    if not reason:
        reason = next((l.strip() for l in reversed(tail) if l.strip()),
                      f"exit status {code}")
    raise RuntimeError(f"{label} failed: {reason}")


def find_binary(llama_dir: Path, name):
    """One llama.cpp binary, whichever layout this generator produced."""
    bindir = Path(llama_dir) / "build" / "bin"
    for candidate in (bindir / f"{name}.exe", bindir / "Release" / f"{name}.exe",
                      bindir / name, bindir / "Release" / name):
        if candidate.exists():
            return candidate
    return None


def binaries(llama_dir: Path):
    """(llama-quantize, llama-imatrix) if both exist, else None."""
    quantize = find_binary(llama_dir, "llama-quantize")
    imatrix = find_binary(llama_dir, "llama-imatrix")
    return (quantize, imatrix) if quantize and imatrix else None


# =====================================================
# BUILD CONFIGURATION
# =====================================================
def _total_ram_gb():
    from ..sysprobe import _memory_gb
    return _memory_gb()[0]


def build_jobs():
    """How many compiler processes this machine can actually feed.

    Compiling ggml-cuda's mmq template instances costs roughly 1.5-2 GB per
    nvcc process, so the job count is bounded by RAM, not by core count — this
    box has 24 logical cores but ~16 GB, and -j24 would OOM-kill the compiler
    mid-build. Roughly one job per 2 GB, leaving headroom for the OS.
    """
    return os.getenv("BUILD_JOBS") or str(
        max(2, min(os.cpu_count() or 4, int(_total_ram_gb() // 2))))


def msvc_env_bat():
    """vcvars64.bat for the newest installed Visual Studio, or None.

    The Visual Studio cmake generator locates MSVC by itself; Ninja does not —
    it needs cl.exe already on PATH. Ninja is worth the extra step because it
    parallelises at FILE level, while the VS generator drives MSBuild, whose
    /m flag parallelises across PROJECTS — and ggml-cuda is a single project,
    so its ~100 .cu files would compile one at a time with every other core idle.
    """
    if sys.platform != "win32":
        return None
    vswhere = (Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
               / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if not vswhere.exists():
        return None
    try:
        install = subprocess.run(
            [str(vswhere), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-property", "installationPath"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip().splitlines()
    except Exception:
        return None
    if not install:
        return None
    bat = Path(install[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return bat if bat.exists() else None


def _interpreter_scripts():
    """The bin/Scripts directory of the interpreter running us."""
    return Path(sys.executable).parent


def find_tool(name):
    """Locate a build tool on PATH, or beside our own interpreter.

    The second place matters and is easy to miss. `pip install ninja` (which is
    what the package's [build] extra does) puts ninja.exe in the environment's
    Scripts directory — and when AgentQuantix is installed as an isolated tool
    via uv or pipx, that directory is deliberately NOT on PATH; only the `aqx`
    entry point is exposed. So the extra installed a build tool that nothing
    could then find, and every build silently fell back to the slow generator.
    """
    if found := shutil.which(name):
        return found
    return shutil.which(name, path=str(_interpreter_scripts()))


def build_env():
    """Environment for build subprocesses, with our own Scripts dir on PATH.

    Detecting a pip-installed ninja is only half the fix: cmake looks it up on
    PATH itself, so `-G Ninja` would fail with "CMAKE_MAKE_PROGRAM not found"
    even though we just proved the binary exists. Prepending the directory
    fixes the tool we found being usable by the tool we hand it to.
    """
    env = os.environ.copy()
    scripts = str(_interpreter_scripts())
    if scripts not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = scripts + os.pathsep + env.get("PATH", "")
    return env


def has_cuda_toolkit():
    """Can this machine actually COMPILE CUDA.

    nvcc, not nvidia-smi: a container or VM frequently has the driver and a
    visible GPU while lacking the toolkit, and it is nvcc that the build needs.
    CUDA_PATH covers a Windows install that has not been put on PATH.
    """
    if find_tool("nvcc"):
        return True
    for variable in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(variable)
        if root and (Path(root) / "bin").exists():
            return True
    return False


def cuda_arch_from_gpu():
    """This GPU's compute capability as a CMAKE_CUDA_ARCHITECTURES value."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.split("\n")[0].strip()
        return out.replace(".", "") or None          # "8.9" -> "89"
    except Exception:
        return None


def _cmake_cache_text():
    cache = config.UPSTREAM_LLAMA / "build" / "CMakeCache.txt"
    return (cache.read_text(encoding="utf-8", errors="replace")
            if cache.exists() else "")


def upstream_generator():
    """The cmake generator the user's own llama.cpp build used, if any."""
    match = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.+)$", _cmake_cache_text(), re.M)
    return match.group(1).strip() if match else None


def build_settings():
    """Cmake defines for a fork build, copied from the existing upstream build.

    CMAKE_CUDA_ARCHITECTURES matters enormously and is easy to miss: left
    unset, llama.cpp generates code for every CUDA arch it supports, and each
    of the ~100 ggml-cuda mmq template-instance files is compiled once PER
    arch. Pinning it to this GPU alone (sm_89 on the RTX 4060) cuts that by
    roughly 8x. If the upstream cache has no value, it is probed from the GPU.
    """
    text = _cmake_cache_text()
    if not text:
        print("No existing CMakeCache.txt to copy build settings from - "
              "detecting from this machine.")

    # GGML_CUDA must be DETECTED, not defaulted on. A fresh VM with no GPU and
    # no CUDA toolkit fails the configure step outright if this is forced ON,
    # which is the least helpful possible way to discover you are on a CPU box.
    cuda_default = "ON" if has_cuda_toolkit() else "OFF"

    defines = []
    for key, default in (("GGML_CUDA", cuda_default), ("GGML_NATIVE", "OFF")):
        match = re.search(rf"^{key}:BOOL=(\w+)$", text, re.M)
        defines.append(f"-D{key}={match.group(1) if match else default}")

    # Only meaningful when CUDA is actually being compiled. Warning that "the
    # build will target every supported architecture and take far longer" on a
    # CPU-only machine is both alarming and false — there are no CUDA sources
    # in that build at all.
    cuda_on = any(d == "-DGGML_CUDA=ON" for d in defines)
    if cuda_on:
        match = re.search(r"^CMAKE_CUDA_ARCHITECTURES:\w+=(.+)$", text, re.M)
        arch = match.group(1).strip() if match else cuda_arch_from_gpu()
        if arch:
            defines.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
        else:
            print("WARNING: could not determine this GPU's CUDA arch - the "
                  "build will target every supported architecture and take "
                  "far longer.")
    return defines


def build_is_stale(llama_dir: Path, generator, defines):
    """True if an existing build/ was configured differently than we now want.

    Changing generator (or CUDA arch) cannot be done in place — a tree
    configured without an arch pin has to be discarded, or we would silently
    resume the very build that takes forever.
    """
    cache = llama_dir / "build" / "CMakeCache.txt"
    if not cache.exists():
        return False
    text = cache.read_text(encoding="utf-8", errors="replace")

    match = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.+)$", text, re.M)
    if generator and (not match or match.group(1).strip() != generator):
        return True

    for define in defines:
        key, _, value = define[2:].partition("=")
        match = re.search(rf"^{key}:\w+=(.*)$", text, re.M)
        if not match or match.group(1).strip() != value:
            return True
    return False


# =====================================================
# THE BUILD
# =====================================================
def compile_tools(llama_dir: Path, work_root: Path):
    """Configure + build llama-quantize and llama-imatrix in `llama_dir`."""
    defines = build_settings()
    vcvars = msvc_env_bat()
    jobs = build_jobs()

    # Ninja needs a compiler on PATH. On Windows only vcvars64.bat provides
    # that, so both are required; everywhere else cc/g++ is already there and
    # Ninja alone is enough. The old condition demanded vcvars on every
    # platform, which quietly denied Linux the faster generator entirely.
    have_ninja = bool(find_tool("ninja"))
    can_use_ninja = have_ninja and (vcvars is not None
                                    or sys.platform != "win32")

    if can_use_ninja:
        generator = "Ninja"
        defines.append("-DCMAKE_BUILD_TYPE=Release")
    else:
        generator = upstream_generator()
        missing = ("ninja" if not have_ninja
                   else "the MSVC build environment")
        print(f"{missing} is unavailable - falling back to "
              f"{generator or 'the default generator'}. Installing ninja "
              "makes this build several times faster.")

    if build_is_stale(llama_dir, generator, defines):
        print("Existing build/ was configured differently - reconfiguring.")
        shutil.rmtree(llama_dir / "build", ignore_errors=True)

    configure = ["cmake", "-B", llama_dir / "build", "-S", llama_dir]
    if generator:
        configure += ["-G", generator]
    configure += defines

    # Only the two tools the pipeline uses — building the full tree (server,
    # tests, examples) would roughly triple the compile time. Even so,
    # ggml-cuda is a long compile: expect minutes, and a wall of harmless
    # "variable declared but never referenced" warnings as the mmq template
    # instances go by. That output means it is working, not stuck.
    compile_cmd = ["cmake", "--build", llama_dir / "build", "--config", "Release",
                   "--target", "llama-quantize", "llama-imatrix", "-j", jobs]

    print(f"\nBuilding llama-quantize + llama-imatrix ({generator}, {jobs} jobs). "
          "This takes a few minutes.\n", flush=True)
    started = time.time()

    # The batch-file detour exists for ONE reason: on Windows, Ninja needs
    # cl.exe on PATH and only vcvars64.bat puts it there, so configure and
    # compile have to run inside a single cmd.exe that sources it first.
    #
    # It was gated on the generator alone, which meant Linux — where Ninja is
    # the right choice and cc is already on PATH — took the Windows path and
    # died on `FileNotFoundError: 'cmd'`. Gate on the thing that actually
    # requires it: having a vcvars to source.
    if generator == "Ninja" and vcvars is not None:
        script = work_root / "_build.bat"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(
            "@echo off\r\n"
            f'call "{vcvars}" >nul || exit /b 1\r\n'
            + " ".join(f'"{c}"' for c in map(str, configure)) + " || exit /b 1\r\n"
            + " ".join(f'"{c}"' for c in map(str, compile_cmd)) + " || exit /b 1\r\n",
            encoding="utf-8")
        try:
            run(["cmd", "/c", str(script)])
        finally:
            script.unlink(missing_ok=True)
    else:
        run(configure)
        run(compile_cmd)

    built = binaries(llama_dir)
    if built is None:
        raise RuntimeError("Build finished but the binaries are still missing - "
                           "check the cmake output above.")
    feasibility.record("fork_build", minutes=round((time.time() - started) / 60, 1),
                       dir=str(llama_dir))
    return built


def _slug(repo, ref):
    """Directory name for one fork checkout. Stable, so it can be reused."""
    return re.sub(r"[^A-Za-z0-9]+", "-", f"{repo}-{ref}").strip("-").lower()


def ensure_fork(repo, ref):
    """Clone (or refresh) and build a fork. Returns (llama_dir, quantize, imatrix).

    `repo` is "owner/name"; `ref` is a branch, a tag, or a "pull/N/head" refspec
    for an open upstream PR. Idempotent: an existing checkout with usable
    binaries is returned untouched, which is what makes fork reuse across
    models free.
    """
    llama_dir = config.FORKS_DIR / _slug(repo, ref)
    url = repo if repo.startswith("http") else f"https://github.com/{repo}.git"

    if not llama_dir.exists():
        llama_dir.parent.mkdir(parents=True, exist_ok=True)
        if ref.startswith("pull/"):
            # A PR head is not a branch, so it cannot be cloned directly —
            # clone the default branch shallowly, then fetch the PR ref into a
            # local branch and check that out.
            run(["git", "clone", "--depth", "1", url, llama_dir])
            run(["git", "-C", llama_dir, "fetch", "--depth", "1", "origin",
                 f"{ref}:aqx-pr"])
            run(["git", "-C", llama_dir, "checkout", "aqx-pr"])
        else:
            run(["git", "clone", "--depth", "1", "--branch", ref, url, llama_dir])
    elif binaries(llama_dir) is None:
        # Only refresh when we are going to rebuild anyway: these branches are
        # actively developed and may be force-pushed, but skipping the fetch
        # when the binaries already exist keeps repeat runs instant. This
        # directory is owned exclusively by the agent, so a hard reset is safe.
        try:
            if ref.startswith("pull/"):
                run(["git", "-C", llama_dir, "fetch", "--force", "--depth", "1",
                     "origin", f"{ref}:aqx-pr"])
                run(["git", "-C", llama_dir, "reset", "--hard", "aqx-pr"])
            else:
                run(["git", "-C", llama_dir, "fetch", "--force", "--depth", "1",
                     "origin", ref])
                run(["git", "-C", llama_dir, "reset", "--hard", "FETCH_HEAD"])
        except subprocess.CalledProcessError:
            print("Could not refresh the fork checkout - building what is there.")

    if (built := binaries(llama_dir)) is not None:
        return llama_dir, *built
    return llama_dir, *compile_tools(llama_dir, llama_dir.parent)


UPSTREAM_LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"


def ensure_upstream(clone=True):
    """The main llama.cpp checkout, cloned and built if it is not there yet.

    On a machine that already has one this is free. On a bare VM it is the
    whole setup: clone, configure, compile. Shallow, because the pipeline only
    ever needs the working tree — nobody bisects llama.cpp from here, and the
    full history is a large download for no benefit.
    """
    llama_dir = config.UPSTREAM_LLAMA

    if not llama_dir.exists():
        if not clone:
            raise RuntimeError(
                f"No llama.cpp checkout at {llama_dir}. Run `aqx bootstrap`, "
                "or set AQX_WORK_ROOT to wherever one already lives.")
        if not shutil.which("git"):
            raise RuntimeError("git is not installed, so llama.cpp cannot be "
                               "cloned. Install git and try again.")
        print(f"Cloning llama.cpp into {llama_dir}...")
        llama_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", UPSTREAM_LLAMA_REPO, llama_dir])

    if (built := binaries(llama_dir)) is not None:
        return llama_dir, *built
    return llama_dir, *compile_tools(llama_dir, llama_dir)


def ensure_tools(fork=None):
    """The binaries to run with. A fork lead if given, else the local checkout.

    Returns (llama_dir, llama_quantize, llama_imatrix). The checkout is cloned
    and/or built as needed — a one-off cost the user would otherwise hit as a
    confusing "file not found" mid-run.
    """
    if fork:
        return ensure_fork(fork["repo"], fork["ref"])
    return ensure_upstream()


def cleanup_fork(llama_dir: Path):
    """Remove a fork checkout and its build tree.

    KEEP_FORK=1 skips this — a rebuild costs far more than the disk, and when
    several trending models share an architecture the next run wants it.
    """
    if os.getenv("KEEP_FORK") == "1":
        print(f"KEEP_FORK=1 - leaving {llama_dir} in place.")
        return
    if config.FORKS_DIR not in llama_dir.parents:
        return              # never touch the user's own checkout

    def force_writable(func, path, _exc):
        """git marks .git/objects/pack/*.idx and *.pack read-only, and Windows
        refuses to unlink a read-only file — so rmtree dies with WinError 5
        partway through. Clear the flag and retry.

        The signature suits both rmtree error hooks: 3.12's onexc passes the
        exception, 3.11's onerror passes an exc_info triple. Neither is used.
        """
        os.chmod(path, stat.S_IWRITE)
        func(path)

    # `onexc` only exists from 3.12; on 3.10 and 3.11 it is `onerror`, and
    # passing the wrong one raises TypeError. The package supports 3.10 — which
    # is what Ubuntu 22.04 ships — so this cannot assume the newer spelling.
    hook = ({"onexc": force_writable} if sys.version_info >= (3, 12)
            else {"onerror": force_writable})

    try:
        shutil.rmtree(llama_dir, **hook)
        print(f"Removed fork checkout {llama_dir}.")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Could not remove {llama_dir}: {e}")
