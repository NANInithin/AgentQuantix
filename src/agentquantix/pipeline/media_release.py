"""Independent pinned native media build, conversion, resume and publication."""
from __future__ import annotations

from dataclasses import fields
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import urllib.parse

from .. import archsupport, config, media, sysprobe
from ..media_catalog import CONDITION_FLAGS, ENGINE_REPO, ENGINE_REVISION, ROLE_FLAGS, load_catalog
from . import build as build_mod


def run_dir(plan_id):
    if not re.fullmatch(r"[0-9a-f]{64}", plan_id):
        raise media.MediaValidationError("invalid media run id")
    return config.TEMP_DIR / "aqx-media-runs" / plan_id


def execute(command, log_path, *, cwd=None, timeout=3600):
    command = [str(arg) for arg in command]
    started = time.monotonic()
    record = {"argv": command, "cwd": str(cwd) if cwd else None,
              "engine_revision": load_catalog()["revision"]}
    try:
        result = subprocess.run(command, cwd=cwd, env=build_mod.build_env(),
                                capture_output=True, text=True, errors="replace", timeout=timeout)
        record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    except (OSError, subprocess.TimeoutExpired) as error:
        record.update(stdout=str(getattr(error, "stdout", "") or ""),
                      stderr=str(getattr(error, "stderr", "") or error), returncode=None)
        raise media.MediaValidationError(f"native command failed: {error}; log: {log_path}") from error
    finally:
        record["duration_seconds"] = time.monotonic() - started
        media.atomic_json(log_path, record)
    diagnostics = record["stdout"] + "\n" + record["stderr"]
    # Word boundaries also match directory names such as Documents/INF/temp.
    # A non-finite value must be a standalone token, not part of a file path.
    if result.returncode or re.search(
        r"(?<![\w/\\.\-])(?:nan|[+-]?inf(?:inity)?)(?![\w/\\.\-])"
        r"|\b(?:assertion failed|assert failed)\b", diagnostics, re.I
    ):
        raise media.MediaValidationError(f"native command failed validation; log: {log_path}")
    return record


def build_identity(compute_backend, compiler=None, flags=()):
    choices = sysprobe.compute_backends()
    if compute_backend == "auto":
        compute_backend = next((b for b in ("cuda", "metal", "hip", "vulkan") if choices.get(b)), "cpu")
    if compute_backend not in choices or not choices[compute_backend]:
        raise media.MediaValidationError(f"compute toolchain not detected: {compute_backend}")
    compiler = compiler or os.getenv("CXX") or shutil.which("c++") or sysprobe._msvc_present() or "cmake-default"
    compiler_stamp = {}
    if Path(compiler).is_file():
        compiler_stamp = {"path": compiler, "sha256": media.file_hash(compiler)}
        if Path(compiler).suffix.lower() == ".bat":
            candidates = sorted((Path(compiler).parents[2] / "Tools/MSVC").glob("*/bin/Hostx64/x64/cl.exe"))
            if candidates:
                compiler_stamp["native_compiler"] = str(candidates[-1])
                compiler_stamp["native_compiler_sha256"] = media.file_hash(candidates[-1])
    identity = {"engine_revision": ENGINE_REVISION, "platform": platform.platform(),
                "machine": platform.machine(), "compute_backend": compute_backend,
                "compiler": compiler_stamp or compiler, "flags": [*cmake_flags(compute_backend), *flags],
                "cmake": shutil.which("cmake"), "generator": os.getenv("CMAKE_GENERATOR")}
    return media.canonical_hash(identity), identity


def cmake_flags(compute_backend):
    native_flags = {"cuda": "SD_CUDA", "hip": "SD_HIPBLAS", "vulkan": "SD_VULKAN",
                    "metal": "SD_METAL", "opencl": "SD_OPENCL", "sycl": "SD_SYCL", "musa": "SD_MUSA"}
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DSD_BUILD_EXAMPLES=ON", "-DSD_WEBM=ON", "-DSD_WEBP=ON",
            "-DSD_USE_SYSTEM_GGML=OFF", "-DSD_USE_UPSTREAM_GGML=OFF"] + [
                f"-D{value}={'ON' if name == compute_backend else 'OFF'}" for name, value in native_flags.items()]
    if compute_backend == "cuda" and (architecture := build_mod.cuda_arch_from_gpu()):
        flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={architecture}")
    return flags


def ensure_backend(compute_backend="auto"):
    from .. import runtime_control
    if handle := runtime_control.ensure_runtime("stable-diffusion.cpp", compute_backend=compute_backend):
        binary, = runtime_control.require_binaries(handle, "sd-cli")
        return binary, {**handle.manifest.get("provenance", {}), "engine_revision": handle.revision,
                        "compute_backend": handle.manifest.get("build_fingerprint", {}).get("compute_backend", "unknown"),
                        "runtime_snapshot": handle.snapshot()}
    key, provenance = build_identity(compute_backend)
    root = config.TEMP_DIR / "aqx-media-backends" / key
    source, build = root / "source", root / "build"
    binary = build / "bin" / ("Release/sd-cli.exe" if os.name == "nt" else "sd-cli")
    if os.name == "nt" and not binary.exists():
        binary = build / "bin/sd-cli.exe"
    if not source.exists():
        root.mkdir(parents=True, exist_ok=True)
        execute(["git", "clone", "--depth", "1", "--no-checkout", f"https://github.com/{ENGINE_REPO}.git", source], root / "clone.json")
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True)
    actual = result.stdout.strip() if result.returncode == 0 else ""
    if actual != ENGINE_REVISION or not (source / "CMakeLists.txt").exists():
        if actual != ENGINE_REVISION:
            execute(["git", "-C", source, "fetch", "--depth", "1", "origin", ENGINE_REVISION], root / "fetch.json")
        execute(["git", "-C", source, "checkout", "--detach", ENGINE_REVISION], root / "checkout.json")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise media.MediaValidationError("cached engine source is modified")
    execute(["git", "-C", source, "submodule", "update", "--init", "--recursive", "ggml",
             "thirdparty/libwebp", "thirdparty/libwebm"], root / "submodules.json")
    submodules = subprocess.check_output(["git", "-C", str(source), "submodule", "status", "--recursive"], text=True)
    provenance["submodules"] = submodules.strip().splitlines()
    flags = provenance["flags"]
    if not binary.is_file():
        execute(["cmake", "-S", source, "-B", build, *flags], root / "configure.json")
        execute(["cmake", "--build", build, "--config", "Release", "--target", "sd-cli", "--parallel", build_mod.build_jobs()],
                root / "build.json", timeout=7200)
        candidates = [build / "bin/sd-cli", build / "bin/sd-cli.exe", build / "bin/Release/sd-cli.exe"]
        binary = next((p for p in candidates if p.is_file()), binary)
    if not binary.is_file():
        raise media.MediaValidationError("sd-cli build produced no executable")
    provenance.update(binary_sha256=media.file_hash(binary), cmake_flags=flags,
                      cmake_cache=(build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace"))
    media.atomic_json(root / "provenance.json", provenance)
    return binary, provenance


def converter_types(*, build=False, binary=None, compute_backend="auto"):
    """Probe conversion on a tiny real tensor; help examples are not a type table."""
    revision = load_catalog()["revision"]
    fallback = {"types": load_catalog()["source_types"], "probed": False,
                "basis": "pinned native type enum; each type still requires an executable probe",
                "engine_revision": revision}
    if binary is None and not build:
        pointer = config.STATE_DIR / "media-converter-types.json"
        try:
            cached = json.loads(pointer.read_text(encoding="utf-8"))
            if (cached.get("probe_version") == 2 and cached["engine_revision"] == revision and Path(cached["binary"]).is_file()
                    and media.file_hash(cached["binary"]) == cached["binary_sha256"]):
                return cached
        except (OSError, KeyError, ValueError):
            pass
        return fallback
    provenance = {}
    if binary is None:
        binary, provenance = ensure_backend(compute_backend)
        revision = load_catalog()["revision"]
        fallback["types"] = load_catalog()["source_types"]
        fallback["engine_revision"] = revision
    root = config.TEMP_DIR / "aqx-media-probes" / media.file_hash(binary)
    cached = root / "types.json"
    if cached.exists():
        previous = json.loads(cached.read_text(encoding="utf-8"))
        if previous.get("probe_version") == 2 and previous.get("engine_revision") == revision:
            return previous
    root.mkdir(parents=True, exist_ok=True)
    import struct
    header = json.dumps({"probe.weight": {"dtype": "F32", "shape": [256, 256], "data_offsets": [0, 256 * 256 * 4]}}).encode()
    header += b" " * (-len(header) % 8)
    source = root / "probe.safetensors"
    source.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.125) * (256 * 256))
    execute([binary, "--help"], root / "help.json")
    accepted, rejected = [], {}
    for quant in fallback["types"]:
        output = root / f"{quant}.gguf"
        try:
            execute([binary, "-M", "convert", "-m", source, "-o", output, "--type", quant], root / f"{quant}.json", timeout=60)
            validate_gguf(output)
            accepted.append(quant)
        except media.MediaValidationError as error:
            rejected[quant] = str(error)
        finally:
            output.unlink(missing_ok=True)
    result = {"types": accepted, "rejected": rejected, "probed": True,
              "probe_version": 2,
              "basis": "native conversion of 256x256 F32 tensor", "engine_revision": revision,
              "binary": str(binary), "binary_sha256": media.file_hash(binary), "build_provenance": provenance}
    media.atomic_json(cached, result)
    media.atomic_json(config.STATE_DIR / "media-converter-types.json", result)
    return result


def validate_gguf(path):
    try:
        metadata = media.local_gguf_metadata(path)
        if not metadata["tensors"]:
            raise ValueError("GGUF contains no tensors")
        for tensor in metadata["tensors"]:
            if any(d <= 0 for d in tensor["shape"]) or tensor["offset"] + metadata["header_bytes"] >= Path(path).stat().st_size:
                raise ValueError("GGUF tensor data missing or invalid dimensions")
    except Exception as error:
        raise media.MediaValidationError(f"invalid GGUF {path}: {error}") from error


def tensor_precisions(path):
    from collections import Counter
    if Path(path).suffix.lower() == ".gguf":
        codes = {code: name for name, code in load_catalog()["type_codes"].items()}
        return dict(Counter(codes.get(t["type"], f"native_type_{t['type']}") for t in media.local_gguf_metadata(path)["tensors"]))
    if Path(path).suffix.lower() in (".safetensors", ".sft"):
        import struct
        with Path(path).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 2 <= size <= 32 * 1024 * 1024:
                raise media.MediaValidationError("invalid safetensors metadata length")
            header = json.loads(stream.read(size))
        if not isinstance(header, dict):
            raise media.MediaValidationError("invalid safetensors metadata")
        precisions = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list) or not tensor.get("dtype"):
                raise media.MediaValidationError("safetensors tensor metadata is incomplete")
            begin, end = tensor["data_offsets"]
            if begin < 0 or end <= begin or end + 8 + size > Path(path).stat().st_size:
                raise media.MediaValidationError("safetensors tensor data is truncated")
            precisions.append(tensor["dtype"].lower())
        if not precisions:
            raise media.MediaValidationError("source contains no tensors")
        return dict(Counter(precisions))
    return {}


def find_forks(repo_id, metadata, use_cache=True):
    cache_path = config.STATE_DIR / "media-fork-leads.json"
    key = media.canonical_hash({"repo": repo_id, "configs": metadata.get("configs", {})})
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    entry = cache.get(key)
    if use_cache and entry and time.time() - entry["at"] < (86400 if entry["leads"] else 3600):
        return entry["leads"]
    needles = [re.sub(r"[^a-zA-Z0-9]", "", repo_id.split("/")[-1])]
    leads = []
    query = f"{repo_id.split('/')[0]} stable-diffusion.cpp in:name fork:true"
    result = archsupport._github_json(f"{archsupport.GITHUB_API}/search/repositories?q={urllib.parse.quote(query)}&per_page=10") or {}
    for repo in result.get("items", []):
        for branch in archsupport._branches_matching(repo.get("full_name", ""), [n.lower() for n in needles]):
            leads.append({"kind": "publisher-fork", "status": "lead", "repo": repo["full_name"],
                          "ref": branch, "url": repo.get("html_url"), "executable": False})
    for kind, qualifier in (("upstream-pr", "is:pr is:open"), ("upstream-issue", "is:issue")):
        query = f"repo:{ENGINE_REPO} {qualifier} {needles[0]}"
        result = archsupport._github_json(f"{archsupport.GITHUB_API}/search/issues?q={urllib.parse.quote(query)}&per_page=5") or {}
        leads += [{"kind": kind, "status": "lead", "repo": ENGINE_REPO, "url": item.get("html_url"),
                   "ref": f"pull/{item['number']}/head" if kind == "upstream-pr" else None,
                   "title": item.get("title"), "executable": False} for item in result.get("items", [])]
    cache[key] = {"at": time.time(), "leads": leads}
    media.atomic_json(cache_path, cache)
    return leads


def _component_dict(data, quant):
    allowed = {f.name for f in fields(media.MediaComponent)}
    return media.MediaComponent(**{k: v for k, v in data.items() if k in allowed})


def materialize_component(data, root, binary, journal, *, preserve_failed=False, retain_source=False):
    from huggingface_hub import hf_hub_download
    if data.get("precision") in ("f16", "bf16", "f32") and data.get("baseline_source"):
        data = {**data, **data["baseline_source"]}
    component = _component_dict(data, data.get("precision"))
    component.converter_revision = load_catalog()["revision"]
    output = root / media.relative_path(component.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    signature = media.canonical_hash({"source_revision": component.source_revision,
        "source_sha256": component.source_sha256, "source_path": component.source_path,
        "engine_revision": component.converter_revision, "binary_sha256": media.file_hash(binary),
        "precision": component.precision, "tensor_rules": component.tensor_rules,
        "output": component.output_path})
    previous = journal.get(component.output_path, {})
    if (previous.get("signature") == signature and output.is_file()
            and output.stat().st_size == previous.get("bytes")
            and media.file_hash(output) == previous.get("sha256")):
        component.bytes, component.sha256 = previous["bytes"], previous["sha256"]
        component.tensor_precisions = previous.get("tensor_precisions", {})
        component.precision = previous.get("precision", component.precision)
        return component
    source_dir = root / "sources" / media.canonical_hash({"repo": component.source_repo, "revision": component.source_revision})
    source = Path(hf_hub_download(component.source_repo, component.source_path,
                                 revision=component.source_revision, token=config.TOKEN, local_dir=source_dir))
    before = media.file_hash(source)
    source_precisions = tensor_precisions(source)
    if (data.get("quantizable") and data.get("is_baseline")
            and source_precisions and not set(source_precisions) <= {"f16", "bf16", "f32"}):
        raise media.MediaValidationError("high-precision baseline requires original floating-point weights; dequantizing a low-bit source is insufficient")
    if (not data.get("quantizable") and component.precision not in ("source", "original")
            and source_precisions and set(source_precisions) != {component.precision.lower()}):
        raise media.MediaValidationError(f"retained precision differs from the source metadata: {component.role}")
    if component.source_sha256 and before != component.source_sha256:
        raise media.MediaValidationError(f"source checksum mismatch: {component.source_path}")
    validate_tensor_contract(source, data.get("tensor_assertions", {}))
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    command = [binary, "-M", "convert", "-m", source, "-o", partial, "--type", component.precision]
    if component.tensor_rules:
        command += ["--tensor-type-rules", component.tensor_rules]
    try:
        if data.get("quantizable") and source.suffix.lower() != ".gguf":
            execute(command, root / "logs" / f"{signature}.json")
            validate_gguf(partial)
        elif data.get("quantizable") and component.precision != "orig":
            execute(command, root / "logs" / f"{signature}.json")
            validate_gguf(partial)
        else:
            shutil.copyfile(source, partial)
            if source.suffix.lower() == ".gguf":
                validate_gguf(partial)
        if not partial.is_file() or partial.stat().st_size == 0:
            raise media.MediaValidationError("empty converted component")
        if media.file_hash(source) != before:
            raise media.MediaValidationError("source changed during conversion")
        partial.replace(output)
        component.bytes, component.sha256 = output.stat().st_size, media.file_hash(output)
        component.tensor_precisions = tensor_precisions(output)
        if not data.get("quantizable") and component.precision == "source" and component.tensor_precisions:
            component.precision = next(iter(component.tensor_precisions)) if len(component.tensor_precisions) == 1 else "mixed (see tensor_precisions)"
        component.source_sha256 = before
        journal[component.output_path] = {"signature": signature, "bytes": component.bytes,
                                          "sha256": component.sha256, "source_sha256": before,
                                          "tensor_precisions": component.tensor_precisions,
                                          "precision": component.precision,
                                          "command": [str(c) for c in command], "engine_revision": component.converter_revision}
        media.atomic_json(root / "journal.json", journal)
        if not retain_source:
            source.unlink()
        return component
    except BaseException:
        if not preserve_failed:
            partial.unlink(missing_ok=True)
        raise


def validate_tensor_contract(path, assertions):
    if not assertions:
        return
    if Path(path).suffix.lower() == ".gguf":
        names = [t["name"] for t in media.local_gguf_metadata(path)["tensors"]]
    else:
        import struct
        with Path(path).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if size > 32 * 1024 * 1024:
                raise media.MediaValidationError("invalid safetensors header size")
            names = list(json.loads(stream.read(size)))
    validate_tensor_names(names, assertions)


def validate_tensor_names(names, assertions):
    if assertions.get("layer_count"):
        layers = {int(m.group(1)) for n in names if (m := re.search(assertions["layer_pattern"], n))}
        if layers != set(range(assertions["layer_count"])):
            raise media.MediaValidationError("text encoder layer layout differs from the native contract")
    for pattern in assertions.get("required_patterns", []):
        if not any(re.search(pattern, name) for name in names):
            raise media.MediaValidationError(f"required tensor contract not satisfied: {pattern}")
    for pattern in assertions.get("forbidden_patterns", []):
        if any(re.search(pattern, name) for name in names):
            raise media.MediaValidationError(f"forbidden tensor in this encoder variant: {pattern}")


def runtime_command(bundle, binary, output):
    recipe = bundle.runtime_recipe
    settings = recipe["settings"]
    command = [str(binary), "-M", recipe["mode"], "-o", str(output), "-p", recipe["prompt"],
               "-W", str(settings["width"]), "-H", str(settings["height"]),
               "--steps", str(settings["steps"]), "--seed", str(settings["seed"]),
               "--rng", "cpu", "--sampling-method", settings["sampler"],
               "--scheduler", settings["scheduler"], "--cfg-scale", str(settings.get("cfg_scale", 7.0)), "--offload-to-cpu"]
    for role, name in recipe["components"].items():
        flag = ROLE_FLAGS.get(role)
        if flag:
            command += [flag, media.relative_path(name)]
    for role, value in recipe.get("conditioning", {}).items():
        if role not in CONDITION_FLAGS:
            raise media.MediaValidationError(f"unknown conditioning input: {role}")
        for name in value if isinstance(value, list) else [value]:
            command += [CONDITION_FLAGS[role], media.relative_path(name)]
    if recipe["mode"] == "vid_gen":
        command += ["--video-frames", str(settings["frames"]), "--fps", str(settings["fps"])]
    return command


def record_review(run_id, quant, reviewer, accepted, *, notes="", watched_video=False, listened_audio=False):
    root = run_dir(run_id)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    quant = next((q for q in plan["quants"] if q.casefold() == quant.casefold()), quant)
    candidate = json.loads((root / "candidates" / media.relative_path(quant) / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in candidate.get("quality_results", {}).get("outputs", {}).items():
        if media.file_hash(root / "candidates" / quant / media.relative_path(name)) != expected:
            raise media.MediaValidationError("rendered review artifact changed since native validation")
    if not reviewer.strip():
        raise media.MediaValidationError("a human reviewer name is required")
    task = candidate["task"]
    if "video" in task and not watched_video:
        raise media.MediaValidationError("video review must include playback and extracted frames")
    if "audio-video" in task and not listened_audio:
        raise media.MediaValidationError("joint review must include listening to the audio")
    review = {"accepted": bool(accepted), "reviewer": reviewer, "notes": notes,
              "manifest_sha256": media.canonical_hash(candidate), "quant": quant,
              "watched_video": watched_video, "listened_audio": listened_audio,
              "reviewed_at": time.time()}
    media.atomic_json(root / "reviews" / f"{quant}.json", review)
    return review


def publish_bundle(bundle, root, target_repo, review, api=None):
    from huggingface_hub import CommitOperationAdd, HfApi
    from .voice_release import _remote_files
    problems = bundle.problems(root)
    if problems:
        raise media.MediaValidationError("; ".join(problems))
    if not bundle.quality_results.get("passed") or not review.get("accepted"):
        raise media.MediaValidationError("automated quality and human review must pass before publication")
    if not bundle.quality_results.get("native_inference_verified"):
        raise media.MediaValidationError("candidate has no successful native inference evidence")
    if review.get("manifest_sha256") != media.canonical_hash(bundle.manifest()):
        raise media.MediaValidationError("review does not apply to these exact artifacts and quality results")
    if any(c.license_results.get("redistribution") != "permitted" for c in bundle.components):
        raise media.MediaValidationError("a component lacks redistribution permission")
    api = api or HfApi(token=config.TOKEN)
    root = Path(root)
    media.atomic_json(root / "manifest.json", bundle.manifest())
    media.atomic_json(root / "quality.json", bundle.quality_results)
    media.atomic_json(root / "review.json", review)
    shutil.copyfile(media.FIXTURES / "fixtures.json", root / "fixtures.json")
    (root / "README.md").write_text(render_card(bundle), encoding="utf-8")
    shutil.copyfile(media.FIXTURES / "LICENSE.md", root / "FIXTURE_LICENSE.md")
    names = [c.output_path for c in bundle.components] + ["manifest.json", "quality.json", "review.json", "fixtures.json", "FIXTURE_LICENSE.md", "README.md"]
    api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)
    remote = _remote_files(api, target_repo)
    operations = []
    for name in names:
        path = root / name
        actual = remote.get(name, {})
        if actual.get("bytes") == path.stat().st_size and actual.get("sha256") == media.file_hash(path):
            continue
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)))
    if operations:
        api.create_commit(repo_id=target_repo, repo_type="model", operations=operations,
                          commit_message=f"Publish verified {bundle.family} {bundle.quant} media bundle")
    remote = _remote_files(api, target_repo)
    for name in names:
        actual, path = remote.get(name, {}), root / name
        if actual.get("bytes") != path.stat().st_size or actual.get("sha256") != media.file_hash(path):
            # Git blobs lack LFS hashes; download those small files for strict verification.
            from huggingface_hub import hf_hub_download
            if not actual or actual.get("bytes") != path.stat().st_size:
                raise media.MediaValidationError(f"remote verification failed: {name}")
            downloaded = hf_hub_download(target_repo, name, token=config.TOKEN, force_download=True)
            if media.file_hash(downloaded) != media.file_hash(path):
                raise media.MediaValidationError(f"remote checksum mismatch: {name}")
    return {"published": True, "verified": True, "repo": target_repo, "members": names}


def load_bundle(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.pop("schema_version", None)
    data.pop("engine", None)
    data["components"] = [media.MediaComponent(**c) for c in data["components"]]
    return media.MediaBundle(**data)


def publish_run(run_id, quant, api=None):
    root = run_dir(run_id)
    quant = media.relative_path(quant)
    if "/" in quant:
        raise media.MediaValidationError("invalid quant policy")
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    media.validate_plan(plan)
    quant = next((q for q in plan["quants"] if q.casefold() == quant.casefold()), quant)
    if quant not in plan["quants"]:
        raise media.MediaValidationError("candidate was not in the approved plan")
    bundle = load_bundle(root / "candidates" / quant / "manifest.json")
    if bundle.plan_id != run_id:
        raise media.MediaValidationError("candidate does not belong to this approved run")
    review = json.loads((root / "reviews" / f"{quant}.json").read_text(encoding="utf-8"))
    return publish_bundle(bundle, root / "candidates" / quant, plan["target_repositories"][quant], review, api)


def render_card(bundle):
    import shlex
    recipe = runtime_command(bundle, "sd-cli", "sample.avi" if "video" in bundle.task else "sample.png")
    lines = ["---", "library_name: stable-diffusion.cpp", f"pipeline_tag: {bundle.task}", "---", "",
             f"# {bundle.family} — {bundle.quant} component policy", "",
             f"Engine: `stable-diffusion.cpp` at `{bundle.engine_revision}`.", "",
             "The quant label describes selected components. Companion precisions are retained as listed.", "",
             "| Role | File | Format | Actual precision | License |", "|---|---|---|---|---|"]
    lines += [f"| {c.role} | {c.output_path} | {c.format} | {c.precision} | {c.license_results.get('license')} |" for c in bundle.components]
    lines += ["", "Run from the downloaded bundle directory:", "", "```sh", shlex.join(recipe), "```", "",
              "All model files and conditioning assets referenced by this command are in the manifest.", "",
              "SafeTensors companions retain their recorded source precision rather than being forced to the candidate quant.", "",
              "Tensor precision overrides:", "", *[f"- `{c.role}`: `{c.tensor_rules}`" for c in bundle.components if c.tensor_rules], "",
              "## Validation", "", "```json", json.dumps(bundle.quality_results, indent=2), "```", "",
              "## Provenance and limitations", "", f"Approved plan: `{bundle.plan_id}`.",
              "See manifest.json for per-component source revisions, tensor overrides, build provenance, and checksums.",
              "Only the recorded task, settings, and compute backend were evaluated. Quality metrics do not establish factuality or suitability for sensitive uses.",
              "Use reference media only with the necessary rights and consent.", ""]
    lines += ["Source components:", "", *[f"- `{c.role}`: `{c.source_repo}` at `{c.source_revision}`, `{c.source_path}`." for c in bundle.components], ""]
    return "\n".join(lines)


def validate_candidate(bundle, binary, candidate_root, fixtures, baselines, profile):
    from .. import media_quality
    signature = media.canonical_hash({"components": [c.sha256 for c in bundle.components],
        "recipe": bundle.runtime_recipe, "fixtures": fixtures, "profile": profile, "binary": media.file_hash(binary),
        "baselines": {k: media.file_hash(v) for k, v in baselines.items()}})
    quality_file = candidate_root / "quality.json"
    try:
        cached = json.loads(quality_file.read_text(encoding="utf-8"))
        if (cached.get("cache_signature") == signature and cached.get("native_inference_verified")
                and all((candidate_root / name).is_file() and media.file_hash(candidate_root / name) == digest
                        for name, digest in cached.get("outputs", {}).items())
                and cached.get("outputs")):
            return cached
    except (OSError, ValueError):
        pass
    results, outputs, duration, primary_duration = [], {}, 0.0, 0.0
    original_prompt = bundle.runtime_recipe["prompt"]
    extension = ".avi" if bundle.runtime_recipe["mode"] == "vid_gen" else ".png"
    for index, prompt in enumerate(fixtures["prompts"]):
        name = "sample" if index == 0 else "sample-" + prompt["id"]
        output = candidate_root / (name + extension)
        repeat = candidate_root / (name + "-repeat" + extension)
        bundle.runtime_recipe["prompt"] = prompt["text"]
        execution = execute(runtime_command(bundle, binary, output.name), candidate_root / "logs" / f"runtime-{prompt['id']}.json",
                            cwd=candidate_root, timeout=profile.get("timeout_seconds", 3600))
        repeated = execute(runtime_command(bundle, binary, repeat.name), candidate_root / "logs" / f"repeat-{prompt['id']}.json",
                           cwd=candidate_root, timeout=profile.get("timeout_seconds", 3600))
        duration += execution["duration_seconds"] + repeated["duration_seconds"]
        primary_duration += execution["duration_seconds"]
        recipe = {**bundle.runtime_recipe, "repeat_output": str(repeat)}
        quality = media_quality.assess(output, recipe, baselines.get(prompt["id"]), profile,
                                        candidate_root / "evaluations" / prompt["id"])
        quality.update(fixture_id=prompt["id"], native_inference_verified=True,
                       runtime_command=execution["argv"])
        results.append(quality)
        outputs[output.name], outputs[repeat.name] = media.file_hash(output), media.file_hash(repeat)
    bundle.runtime_recipe["prompt"] = original_prompt
    quality = {"cache_signature": signature, "outputs": outputs, "fixtures": results,
               "passed": all(r["passed"] for r in results), "structural_passed": all(r["structural_passed"] for r in results),
               "native_inference_verified": True, "runtime_seconds": duration,
               "runtime_command": results[0]["runtime_command"], "settings": bundle.runtime_recipe["settings"],
               "engine_revision": bundle.engine_revision,
               "compute_backend": bundle.build_provenance["compute_backend"],
               "unavailable_metrics": sorted({m for r in results for m in r["unavailable_metrics"]})}
    quality["rejected_metrics"] = [dict(metric, fixture_id=r["fixture_id"]) for r in results for metric in r["rejected_metrics"]]
    count = len(results)
    frames = bundle.runtime_recipe["settings"].get("frames", 1) if extension == ".avi" else 1
    quality["measurements"] = {"basis": "native subprocess wall time, including model load; repeats excluded",
        "bundle_disk_gb": sum(c.bytes for c in bundle.components) / 1024**3,
        "seconds_per_image": primary_duration / count if extension == ".png" else None,
        "seconds_per_video_frame": primary_duration / (count * frames) if extension == ".avi" else None,
        "processing_frames_per_second": count * frames / primary_duration if primary_duration else None,
        "generated_output_fps": bundle.runtime_recipe["settings"].get("fps") if extension == ".avi" else None,
        "host_ram_peak_gb": None, "vram_peak_gb": None, "model_load_seconds": None,
        "first_frame_seconds": None, "unmeasured": ["peak RAM", "peak VRAM", "model load phase", "first frame latency"]}
    media.atomic_json(quality_file, quality)
    return quality


def run_release(plan, *, user_approved=False, publish=False, preserve_failed=False):
    if user_approved is not True:
        raise media.MediaValidationError("start_release requires approval of the exact media plan")
    media.validate_plan(plan, check_runtime=False)
    from .. import runtime_control
    runtime_control.ensure_runtime(
        "stable-diffusion.cpp", compute_backend=plan["compute_backend"],
        frozen=plan.get("runtime_snapshot"),
        revision={"revision": plan["engine_revision"], "repository": ENGINE_REPO})
    media.validate_plan(plan)
    root = run_dir(plan["plan_id"])
    root.mkdir(parents=True, exist_ok=True)
    media.atomic_json(root / "plan.json", plan)
    binary, provenance = ensure_backend(plan["compute_backend"])
    types = converter_types(binary=binary)
    unsupported = [q for q in plan["quants"] if q not in types["types"]]
    try:
        journal = json.loads((root / "journal.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        journal = {}
    settings = {"width": 256, "height": 256, "steps": 4, "seed": 42,
                "sampler": "euler", "scheduler": "discrete", "frames": 5, "fps": 24, **plan["settings"]}
    fixtures = json.loads((media.FIXTURES / "fixtures.json").read_text(encoding="utf-8"))
    result = {"run_id": plan["plan_id"], "candidates": [
        {"quant": q, "published": False, "rejected": types.get("rejected", {}).get(q, "native converter rejected this policy")}
        for q in unsupported], "build_provenance": provenance}
    baselines = {}
    prepared = {}
    has_float_baseline = not any(Path(c["source_path"]).suffix.lower() == ".gguf"
                                 and (c.get("quantizable") or c["role"] in ("diffusion_model", "unet", "llm", "clip_l", "clip_g", "t5xxl"))
                                 and not c.get("baseline_source") for c in plan["components"])
    policies = list(dict.fromkeys([*(["f16"] if has_float_baseline else []), *[q for q in plan["quants"] if q not in unsupported]]))
    if len(unsupported) == len(plan["quants"]):
        media.atomic_json(root / "result.json", result)
        return result
    for original in plan["components"]:
        for index, quant in enumerate(policies):
            item = media.expand_policy([original], quant, types["types"])[0]
            item["is_baseline"] = has_float_baseline and quant == "f16"
            relative = item["output_path"]
            item["output_path"] = (f"candidates/{quant}/{relative}" if item.get("quantizable") else f"shared/{relative}")
            component = materialize_component(item, root, binary, journal, preserve_failed=preserve_failed,
                                              retain_source=bool(item.get("quantizable") and index < len(policies) - 1))
            destination = root / "candidates" / quant / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not item.get("quantizable") and not destination.exists():
                try:
                    os.link(root / item["output_path"], destination)
                except OSError:
                    shutil.copyfile(root / item["output_path"], destination)
            component.output_path = relative
            prepared.setdefault(quant, []).append(component)
    # Candidate artifacts stay local until one atomic publication; peak estimates
    # explicitly include retained candidates and shared companions.
    for quant in policies:
        candidate_root = root / "candidates" / quant
        candidate_root.mkdir(parents=True, exist_ok=True)
        components = prepared[quant]
        for role, names in plan["conditioning"].items():
            for name in names if isinstance(names, list) else [names]:
                source = media.FIXTURES / Path(media.relative_path(name)).name
                if not source.is_file():
                    raise media.MediaValidationError(f"unresolved conditioning asset: {name}")
                destination = candidate_root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                components.append(media.MediaComponent("config", "agentquantix/fixtures", media.file_hash(source),
                    source.name, name, source.suffix.lstrip("."), "original", sha256=media.file_hash(source),
                    bytes=source.stat().st_size, license_results={"license": "cc0-1.0", "redistribution": "permitted"}))
        recipe = {"mode": "vid_gen" if "video" in plan["task"] else "img_gen",
                  "is_baseline": has_float_baseline and quant == "f16",
                  "requires_audio": "audio-video" in plan["task"],
                  "prompt": fixtures["prompts"][0]["text"], "settings": settings,
                  "components": {c.role: c.output_path for c in components if ROLE_FLAGS.get(c.role)},
                  "conditioning": plan["conditioning"]}
        bundle = media.MediaBundle(plan["family"], plan["task"], quant, components, recipe,
                                   engine_revision=plan["engine_revision"], build_provenance=provenance,
                                   plan_id=plan["plan_id"], required_roles=plan["required_roles"])
        try:
            if problems := bundle.problems(candidate_root):
                raise media.MediaValidationError("; ".join(problems))
            quality = validate_candidate(bundle, binary, candidate_root, fixtures, baselines, plan["quality_profile"])
            bundle.quality_results = quality
            media.atomic_json(candidate_root / "manifest.json", bundle.manifest())
            media.atomic_json(candidate_root / "quality.json", quality)
            if quant == "f16" and has_float_baseline:
                if not quality["structural_passed"]:
                    raise media.MediaValidationError("high-precision baseline failed structural validation")
                extension = ".avi" if "video" in plan["task"] else ".png"
                baselines = {prompt["id"]: candidate_root / (("sample" if i == 0 else "sample-" + prompt["id"]) + extension)
                             for i, prompt in enumerate(fixtures["prompts"])}
            entry = {"quant": quant, "quality": quality, "published": False,
                     "status": "reviewable" if quality["passed"] else "quality-rejected" if quality["rejected_metrics"] else "structural-only",
                     "manifest": str(candidate_root / "manifest.json")}
            if not quality["passed"]:
                entry["publication_blockers"] = [*plan.get("publication_blockers", []),
                    "automated quality approval is incomplete or rejected"]
            if publish and quant in plan["quants"] and quality["passed"]:
                try:
                    review = json.loads((root / "reviews" / f"{quant}.json").read_text(encoding="utf-8"))
                except OSError:
                    review = {}
                # A quant per repository keeps each root manifest and card atomic.
                if review.get("accepted"):
                    entry.update(publish_bundle(bundle, candidate_root, plan["target_repositories"][quant], review))
                else:
                    entry["review_required"] = True
            result["candidates"].append(entry)
        except media.MediaValidationError as error:
            result["candidates"].append({"quant": quant, "published": False, "rejected": str(error)})
            if quant == "f16" and has_float_baseline:
                break
    media.atomic_json(root / "result.json", result)
    return result
