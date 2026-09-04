"""What machine are we on, and what can it actually do?

Step 2 of the agent. The session moves between machines, so nothing here is
hard-coded: every number is measured at run time and the whole probe is
returned as one plain dict that the feasibility model and the report both read.

The probe is cheap (well under a second) EXCEPT for the disk throughput
measurement, which writes and reads a real file. That result is cached in
state/ keyed by the volume, because a laptop's SSD does not get faster between
runs and the pipeline should not pay for the measurement on every invocation.
"""

from pathlib import Path
import json
import os
import platform
import shutil
import subprocess
import sys
import time

from . import config, transfer


# =====================================================
# MEMORY
# =====================================================
def _memory_gb():
    """(total, available) physical RAM in GB.

    Available matters more than total: a quantize run competing with a browser
    and an IDE has far less to work with than the spec sheet suggests, and the
    imatrix strategy is chosen from what is actually free.
    """
    if sys.platform == "win32":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return (status.ullTotalPhys / 1024 ** 3, status.ullAvailPhys / 1024 ** 3)

    try:
        meminfo = dict(
            (parts[0].rstrip(":"), int(parts[1]))
            for line in Path("/proc/meminfo").read_text().splitlines()
            if len(parts := line.split()) >= 2
        )
        # MemAvailable is the kernel's own estimate of what a new workload can
        # get without swapping — much better than MemFree, which excludes the
        # reclaimable page cache.
        return (meminfo["MemTotal"] / 1024 ** 2,
                meminfo.get("MemAvailable", meminfo["MemFree"]) / 1024 ** 2)
    except Exception:
        total = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                 / 1024 ** 3)
        return (total, total * 0.5)


# =====================================================
# CPU
# =====================================================
def _cpu():
    name = platform.processor() or platform.machine()
    physical = None

    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "$p=Get-CimInstance Win32_Processor;"
                 "$p.Name + '|' + ($p.NumberOfCores | Measure-Object -Sum).Sum"],
                capture_output=True, text=True, timeout=20, check=True,
            ).stdout.strip()
            label, _, cores = out.partition("|")
            name = label.strip() or name
            physical = int(cores) if cores.strip().isdigit() else None
        except Exception:
            pass
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass

    return {"name": name.strip(),
            "logical_cores": os.cpu_count() or 1,
            "physical_cores": physical}


# =====================================================
# GPU
# =====================================================
def _gpu():
    """Every CUDA GPU nvidia-smi can see, with FREE VRAM as well as total.

    Free is what decides how many layers llama-imatrix can offload. On a laptop
    the desktop compositor alone can be holding a gigabyte, so using the total
    would over-offload and push the run into an out-of-memory abort partway
    through the forward pass.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
    except Exception:
        return []

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, total_mib, free_mib, compute_cap = parts[:4]
        gpus.append({
            "name": name,
            "vram_total_gb": float(total_mib) / 1024,
            "vram_free_gb": float(free_mib) / 1024,
            # "8.9" -> "89", the CMAKE_CUDA_ARCHITECTURES form.
            "compute_cap": compute_cap,
            "cuda_arch": compute_cap.replace(".", ""),
        })
    return gpus


# =====================================================
# DISK
# =====================================================
def _disk_free_gb(path: Path):
    """Free space on the volume holding `path`, creating it if needed.

    Reported for the WORK volume, not the system drive: on a machine where INF
    sits on a second disk, the C: figure would be irrelevant and dangerously
    optimistic.
    """
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1024 ** 3


def _disk_throughput_gbs(path: Path, size_mb=256, force=False):
    """Sequential write+read throughput in GB/s on the work volume.

    Quantizing is overwhelmingly a streaming-IO job — llama-quantize reads the
    whole BF16 and writes a smaller file, tensor by tensor — so this single
    number carries most of the time estimate. Measured with a real file rather
    than assumed, since an external USB drive and an NVMe differ by 20x and the
    session moves between machines.

    Cached per volume: an SSD does not change speed between runs, and paying
    512 MB of IO on every `aqx research` would be absurd.
    """
    cache_file = config.STATE_DIR / "disk-throughput.json"
    key = str(Path(path).resolve().anchor or path)

    if not force and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text()).get(key)
            if cached:
                return cached["gbs"], True
        except Exception:
            pass

    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".aqx-diskprobe.bin"
    # Incompressible-ish payload so a compressing filesystem cannot flatter the
    # result. One buffer reused across writes to keep RAM flat.
    chunk = os.urandom(4 * 1024 * 1024)
    chunks = max(1, (size_mb * 1024 * 1024) // len(chunk))

    try:
        started = time.perf_counter()
        with probe.open("wb") as handle:
            for _ in range(chunks):
                handle.write(chunk)
            # Without this the "write" time is really just time-to-page-cache,
            # which on a 16 GB box measures RAM bandwidth, not the disk.
            handle.flush()
            os.fsync(handle.fileno())
        write_s = time.perf_counter() - started

        started = time.perf_counter()
        with probe.open("rb") as handle:
            while handle.read(len(chunk)):
                pass
        read_s = time.perf_counter() - started

        written_gb = chunks * len(chunk) / 1024 ** 3
        # The pipeline reads and writes in roughly equal measure, so the
        # harmonic mean of the two rates is the honest single number.
        gbs = 2 * written_gb / max(write_s + read_s, 1e-6)
    finally:
        probe.unlink(missing_ok=True)

    try:
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    except Exception:
        cache = {}
    cache[key] = {"gbs": round(gbs, 3), "measured_at": time.time(),
                  "size_mb": size_mb}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=2))
    return gbs, False


# =====================================================
# TOOLCHAIN
# =====================================================
def _llama_build():
    """Is there a usable llama.cpp build here, and how was it configured?

    A missing llama-quantize is not fatal — the pipeline can build one — but it
    turns a two-hour job into a two-hour-plus-compile job, and the report
    should say so before anything is approved.
    """
    build = {"root": str(config.UPSTREAM_LLAMA),
             "present": config.UPSTREAM_LLAMA.exists(),
             "binaries": {}, "generator": None, "cuda": None, "commit": None}
    if not build["present"]:
        return build

    bindir = config.UPSTREAM_LLAMA / "build" / "bin"
    for tool in ("llama-quantize", "llama-imatrix", "llama-cli"):
        for candidate in (bindir / f"{tool}.exe", bindir / "Release" / f"{tool}.exe",
                          bindir / tool, bindir / "Release" / tool):
            if candidate.exists():
                build["binaries"][tool] = str(candidate)
                break

    cache = config.UPSTREAM_LLAMA / "build" / "CMakeCache.txt"
    if cache.exists():
        text = cache.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("CMAKE_GENERATOR:INTERNAL="):
                build["generator"] = line.split("=", 1)[1].strip()
            elif line.startswith("GGML_CUDA:BOOL="):
                build["cuda"] = line.split("=", 1)[1].strip().upper() == "ON"

    try:
        build["commit"] = subprocess.run(
            ["git", "-C", str(config.UPSTREAM_LLAMA), "log", "-1", "--format=%h %cs"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
    except Exception:
        pass
    return build


def _msvc_present():
    """vcvars64.bat for the newest installed Visual Studio, or None.

    Only needed when a fork has to be BUILT. The Ninja generator will not find
    cl.exe by itself, so this is the difference between a fork build working
    unattended and failing ten seconds in.
    """
    if sys.platform != "win32":
        return shutil.which("cc") or shutil.which("gcc")
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
    return str(bat) if bat.exists() else None


def _hub_backends():
    """Which large-file transfer backends are importable, and is xet switched off.

    hf_xet or hf_transfer is REQUIRED for any file over huggingface_hub's 50 GB
    plain-HTTP cap. Worth knowing before a 75 GB BF16 is approved, not after.
    """
    backends = {}
    for module in ("hf_xet", "hf_transfer"):
        try:
            __import__(module)
            backends[module] = True
        except ImportError:
            backends[module] = False
    backends["xet_disabled"] = (
        os.environ.get("HF_HUB_DISABLE_XET", "").lower() in ("1", "true", "yes"))
    # An oversized file can only be fetched if a backend is present AND, in the
    # xet case, not disabled by the environment.
    backends["can_exceed_http_limit"] = bool(
        backends["hf_transfer"]
        or (backends["hf_xet"] and not backends["xet_disabled"]))
    return backends


# =====================================================
# THE PROBE
# =====================================================
def probe(measure_disk=True, force_disk=False):
    """One dict describing this machine. Safe to call repeatedly."""
    total_ram, avail_ram = _memory_gb()
    gpus = _gpu()

    info = {
        "hostname": platform.node(),
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "cpu": _cpu(),
        "ram_total_gb": round(total_ram, 2),
        "ram_available_gb": round(avail_ram, 2),
        "gpus": gpus,
        "vram_total_gb": round(sum(g["vram_total_gb"] for g in gpus), 2),
        "vram_free_gb": round(sum(g["vram_free_gb"] for g in gpus), 2),
        "work_root": str(config.INF_ROOT),
        "disk_free_gb": round(_disk_free_gb(config.INF_ROOT), 1),
        "llama": _llama_build(),
        "msvc": _msvc_present(),
        "hub_backends": _hub_backends(),
        "hf_token": bool(config.TOKEN),
        "xet_policy": transfer.summary(),
    }

    if measure_disk:
        gbs, cached = _disk_throughput_gbs(config.TEMP_DIR, force=force_disk)
        info["disk_gbs"] = round(gbs, 2)
        info["disk_gbs_cached"] = cached
    else:
        info["disk_gbs"] = None
        info["disk_gbs_cached"] = None

    # "Fast memory" is the pool a forward pass can touch without hitting disk.
    # It is what decides whether the imatrix runs on the BF16 or on a quant,
    # and it is the single most consequential number in the whole probe.
    info["fast_memory_gb"] = round(avail_ram + info["vram_free_gb"], 2)
    return info


def summary(info):
    """One-screen human rendering of probe(), for the report header."""
    gpu = (", ".join(f"{g['name']} {g['vram_total_gb']:.0f} GB (cc {g['compute_cap']})"
                     for g in info["gpus"]) or "none detected")
    cores = info["cpu"]["logical_cores"]
    physical = info["cpu"]["physical_cores"]
    core_text = f"{cores} logical" + (f" / {physical} physical" if physical else "")
    llama = info["llama"]
    llama_text = ("not found" if not llama["present"]
                  else f"{llama.get('commit') or 'unknown'}"
                       f"{' CUDA' if llama.get('cuda') else ' CPU-only'}"
                       f"{'' if llama['binaries'] else ' (NOT BUILT)'}")
    disk = (f"{info['disk_gbs']} GB/s" if info["disk_gbs"] else "not measured")
    return "\n".join([
        f"host       {info['hostname']}  ({info['platform']}, py{info['python']})",
        f"cpu        {info['cpu']['name']}  [{core_text}]",
        f"ram        {info['ram_total_gb']} GB total, "
        f"{info['ram_available_gb']} GB available",
        f"gpu        {gpu}  ({info['vram_free_gb']} GB VRAM free)",
        f"fast mem   {info['fast_memory_gb']} GB  (available RAM + free VRAM)",
        f"disk       {info['disk_free_gb']} GB free at {info['work_root']}"
        f"  @ {disk}",
        f"llama.cpp  {llama_text}",
        f"transfer   {info['xet_policy']}",
        f"hf token   {'present' if info['hf_token'] else 'MISSING'}",
    ])
