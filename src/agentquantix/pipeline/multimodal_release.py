"""Native conversion, multimodal evaluation and atomic publication."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download

from .. import config, multimodal
from ..runtime_control import require_binaries
from .native_common import (_command, _free_port, _hash, _record_stage, _server as _shared_server,
                            _slug, _valid_stage, _write)

FIXTURES = multimodal.FIXTURES
RUNTIME_TARGETS = ("llama-server", "llama-mtmd-cli", "llama-quantize")
CASE_FILE = FIXTURES / "cases.json"
_IMAGE_CASES = json.loads(CASE_FILE.read_text(encoding="utf-8"))


@contextmanager
def _server(binary, model, root, arguments):
    with _shared_server(binary, model, root, arguments, free_port_fn=_free_port,
                        open_url=urlopen, spawn=subprocess.Popen,
                        monotonic=time.monotonic, sleep=time.sleep) as base_url:
        yield base_url


def _fixture_path(value):
    return (FIXTURES / value).resolve()


def _chat_request(base_url, prompt, *, image=None, audio=None):
    content = []
    if image is not None:
        mime = "image/png" if Path(image).suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(Path(image).read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    if audio is not None:
        encoded = base64.b64encode(Path(audio).read_bytes()).decode("ascii")
        content.append({"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}})
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"messages": [{"role": "user", "content": content}],
                       "temperature": 0, "seed": 42, "max_tokens": 96}).encode()
    request = Request(f"{base_url}/v1/chat/completions", body,
                      {"Content-Type": "application/json"})
    with urlopen(request, timeout=900) as response:
        result = json.load(response)
    try:
        answer = result["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError) as error:
        raise RuntimeError("llama-server chat response is missing message content") from error
    if isinstance(answer, list):
        answer = "".join(str(part.get("text", "")) for part in answer if isinstance(part, dict))
    return str(answer)


def _video_support(binary):
    if not shutil.which("ffmpeg"):
        return False
    try:
        result = subprocess.run([str(binary), "--help"], capture_output=True, text=True,
                                errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--video" in result.stdout + result.stderr


def _video_answer(binary, text_model, mmproj, root):
    case = _IMAGE_CASES["video"]
    argv = [binary, "-m", text_model, "--mmproj", mmproj, "--video",
            _fixture_path(case["file"]), "--video-fps", "1",
            "--video-timestamp-interval", "1000", "--temp", "0", "--seed", "42",
            "-n", "96", "--no-mmproj-offload", "-p", case["prompt"]]
    started = time.monotonic()
    result = subprocess.run([str(value) for value in argv], cwd=root, capture_output=True,
                            text=True, errors="replace", timeout=1800)
    if result.returncode:
        raise RuntimeError(f"llama-mtmd-cli failed ({result.returncode}): {result.stderr}")
    return result.stdout.strip(), time.monotonic() - started


def _run_reference_case(model, processor, image_path, prompt):
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=96)
    prompt_length = inputs["input_ids"].shape[-1]
    return processor.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()


def _reference(plan, source, journal, journal_path):
    cached = journal.get("reference")
    if cached:
        return cached
    if "vision" not in plan["contract"].get("claimed_modalities", []):
        reference = {"available": False, "reason": "vision modality is absent"}
        journal["reference"] = reference
        _write(journal_path, journal)
        return reference
    try:
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as ModelClass
        processor = AutoProcessor.from_pretrained(source)
        try:
            model = ModelClass.from_pretrained(source, torch_dtype=torch.float32).to("cpu").eval()
        except (ImportError, KeyError, OSError, ValueError) as first_error:
            if ModelClass.__name__ != "AutoModelForImageTextToText":
                raise
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                source, torch_dtype=torch.float32).to("cpu").eval()
            del first_error
        answers = {}
        for case in _IMAGE_CASES["vqa"]:
            answers[case["id"]] = _run_reference_case(
                model, processor, _fixture_path(case["image"]), case["prompt"])
        ocr = _IMAGE_CASES["ocr"]
        answers["ocr"] = _run_reference_case(model, processor, _fixture_path(ocr["image"]),
                                             ocr["prompt"])
        hits = [multimodal.answer_hits(answers[case["id"]], case["expected"])
                for case in _IMAGE_CASES["vqa"]]
        reference = {"available": True, "answers": answers,
                     "vqa_accuracy": sum(hits) / len(hits)}
        del model, processor
    except Exception as error:
        reference = {"available": False,
                     "reason": f"{type(error).__name__}: {error}"}
    journal["reference"] = reference
    _write(journal_path, journal)
    return reference


def _actual_modalities(metadata):
    result = []
    if bool(metadata.get("clip.has_vision_encoder")):
        result.append("vision")
    if bool(metadata.get("clip.has_audio_encoder")):
        result.append("audio")
    return result


def _field_value(reader, key):
    field = reader.get_field(key)
    if field is None:
        return None
    value = field.contents()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [item.item() if hasattr(item, "item") else item for item in value]
    return value.item() if hasattr(value, "item") else value


def read_projector_metadata(runtime_source, projector_path):
    gguf_path = Path(runtime_source) / "gguf-py"
    package = gguf_path / "gguf"
    if not (package / "__init__.py").is_file():
        raise RuntimeError(f"runtime checkout does not contain gguf-py: {gguf_path}")
    package_modules = {
        name: module for name, module in sys.modules.items()
        if name == "gguf" or name.startswith("gguf.")
    }
    for name in package_modules:
        sys.modules.pop(name, None)
    try:
        spec = importlib.util.spec_from_file_location(
            "gguf", package / "__init__.py", submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["gguf"] = module
        spec.loader.exec_module(module)
        reader = module.GGUFReader(str(projector_path))
        keys = (
            "general.type", "clip.has_vision_encoder", "clip.has_audio_encoder",
            "clip.projector_type", "clip.vision.projector_type", "clip.audio.projector_type",
            "clip.vision.image_size", "clip.vision.preproc_image_size",
            "clip.vision.image_mean", "clip.vision.image_std", "clip.vision.patch_size",
            "clip.audio.num_mel_bins",
        )
        metadata = {key: _field_value(reader, key) for key in keys}
    finally:
        for name in [key for key in sys.modules if key == "gguf" or key.startswith("gguf.")]:
            sys.modules.pop(name, None)
        sys.modules.update(package_modules)
    if metadata.get("general.type") != "mmproj":
        raise RuntimeError(
            f"projector GGUF general.type must be 'mmproj', got {metadata.get('general.type')!r}")
    return metadata


def _convert(plan, source, handle, root, journal, journal_path):
    name = plan["model"].rsplit("/", 1)[-1]
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    text_path = models / f"{name}-F16.gguf"
    mmproj_path = models / f"mmproj-{name}-F16.gguf"
    if text_path.resolve() == mmproj_path.resolve():
        raise RuntimeError("text and projector output paths must be distinct")
    assert text_path.resolve() != mmproj_path.resolve()
    converter = handle.source / "convert_hf_to_gguf.py"
    env = os.environ.copy()
    gguf_py = str(handle.source / "gguf-py")
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (gguf_py, env.get("PYTHONPATH", "")) if item)
    text_key, projector_key = "text_f16", "projector_f16"
    if _valid_stage(journal, text_key):
        text_path = Path(journal[text_key]["path"])
    else:
        _command([sys.executable, converter, source, "--outtype", "f16",
                  "--outfile", text_path], env=env)
        if not text_path.is_file():
            raise RuntimeError(f"text conversion produced no F16 GGUF: {text_path}")
        _record_stage(journal_path, journal, text_key, text_path)
    if _valid_stage(journal, projector_key):
        mmproj_path = Path(journal[projector_key]["path"])
    else:
        try:
            _command([sys.executable, converter, source, "--outtype", "f16",
                      "--outfile", mmproj_path, "--mmproj"], env=env)
        except Exception as error:
            raise RuntimeError(f"F16 projector conversion failed: {error}") from error
        if not mmproj_path.is_file():
            raise RuntimeError(f"F16 projector conversion produced no file: {mmproj_path}")
        _record_stage(journal_path, journal, projector_key, mmproj_path)
    if text_path.resolve() == mmproj_path.resolve():
        raise RuntimeError("text and projector output paths must be distinct")
    metadata_key = "projector_metadata"
    if journal.get(metadata_key) and journal[metadata_key].get("sha256") == _hash(mmproj_path):
        metadata = journal[metadata_key]["value"]
    else:
        metadata = read_projector_metadata(handle.source, mmproj_path)
        journal[metadata_key] = {"sha256": _hash(mmproj_path), "value": metadata}
        _write(journal_path, journal)
    actual = _actual_modalities(metadata)
    claimed = plan["contract"].get("claimed_modalities", [])
    missing = sorted(set(claimed) - set(actual))
    if missing:
        reason = "projector metadata does not cover claimed modalities: " + ", ".join(missing)
        journal["blocker"] = reason
        _write(journal_path, journal)
        raise RuntimeError(reason)
    return text_path, mmproj_path, metadata, actual


def _quantize(binary, source, root, journal, journal_path, quant):
    key = f"quant:{quant}"
    if _valid_stage(journal, key):
        return Path(journal[key]["path"])
    output = root / "models" / f"{Path(source).stem.replace('-F16', '')}-{quant}.gguf"
    _command([binary, source, output, quant])
    if not output.is_file():
        raise RuntimeError(f"llama-quantize produced no file for {quant}")
    _record_stage(journal_path, journal, key, output)
    return output


def _evaluate(plan, text_model, mmproj, metadata, modalities, binaries, root, video_available):
    started_all = time.monotonic()
    metrics = {"vqa_accuracy": None, "ocr_cer": None, "ocr_wer": None,
               "video_order_ok": None, "asr_wer": None, "asr_cer": None}
    answers = {"vqa": [], "ocr": None, "video": None, "asr": None}
    timings = {}
    arguments = ["--mmproj", mmproj, "--jinja", "--no-webui", "-c", "4096",
                 "--no-mmproj-offload"]
    with _server(binaries["llama-server"], text_model, root, arguments) as base_url:
        if "vision" in modalities:
            vqa_cases = _IMAGE_CASES["vqa"]
            hits = []
            for case in vqa_cases:
                start = time.monotonic()
                answer = _chat_request(base_url, case["prompt"], image=_fixture_path(case["image"]))
                seconds = time.monotonic() - start
                hit = multimodal.answer_hits(answer, case["expected"])
                answers["vqa"].append({"id": case["id"], "answer": answer,
                                       "hit": hit, "seconds": seconds})
                hits.append(hit)
            metrics["vqa_accuracy"] = sum(hits) / len(hits)
            case = _IMAGE_CASES["ocr"]
            start = time.monotonic()
            answer = _chat_request(base_url, case["prompt"], image=_fixture_path(case["image"]))
            seconds = time.monotonic() - start
            answers["ocr"] = {"answer": answer, "seconds": seconds}
            metrics["ocr_cer"] = multimodal.cer(answer, case["expected"])
            metrics["ocr_wer"] = multimodal.wer(answer, case["expected"])
            timings["ocr"] = seconds
        if "audio" in modalities:
            case = _IMAGE_CASES["asr"]
            audio = _fixture_path(case["audio"])
            transcript_data = json.loads(_fixture_path(case["transcript_source"]).read_text(
                encoding="utf-8"))
            fixture = next((entry for entry in transcript_data.get("fixtures", [])
                            if entry.get("id") == case["fixture_id"]), None)
            if fixture is None:
                raise RuntimeError(f"ASR transcript fixture is missing: {case['fixture_id']}")
            transcript = fixture["transcript"]
            start = time.monotonic()
            answer = _chat_request(base_url, case["prompt"], audio=audio)
            seconds = time.monotonic() - start
            cleaned = multimodal.strip_markup(answer)
            answers["asr"] = {"answer": answer, "cleaned_answer": cleaned, "seconds": seconds}
            metrics["asr_wer"] = multimodal.wer(cleaned, transcript)
            metrics["asr_cer"] = multimodal.cer(cleaned, transcript)
            timings["asr"] = seconds
    if "vision" in modalities and video_available:
        start = time.monotonic()
        answer, seconds = _video_answer(
            binaries["llama-mtmd-cli"], text_model, mmproj, root)
        case = _IMAGE_CASES["video"]
        answers["video"] = {"answer": answer, "seconds": seconds}
        metrics["video_order_ok"] = multimodal.order_ok(answer, case["first"], case["second"])
        timings["video"] = time.monotonic() - start
    elif "vision" in modalities:
        answers["video"] = {"status": "skipped: runtime lacks video"}
    return {"metrics": metrics, "answers": answers, "request_seconds": timings,
            "total_seconds": time.monotonic() - started_all,
            "projector_metadata": metadata, "modalities": modalities,
            "mmproj_offload": False}


def _quality_gate(quality, modalities, video_available, *, baseline=None, reference=None):
    values = quality.get("metrics", quality)
    gates = []
    if "vision" in modalities:
        gates.append(("vqa_accuracy", values.get("vqa_accuracy") is not None))
        gates.append(("ocr_cer", values.get("ocr_cer") is not None))
        if video_available:
            gates.append(("video_order_ok", values.get("video_order_ok") is not None))
    if "audio" in modalities:
        gates.append(("asr_wer", values.get("asr_wer") is not None))
    applicable = [name for name, present in gates if present]
    if not applicable:
        return False, ["no applicable multimodal quality gates"]
    failures = []
    if baseline is None:
        if "vision" in modalities:
            if values.get("vqa_accuracy") is None or values["vqa_accuracy"] < 0.8:
                failures.append("vqa_accuracy below 0.8")
            if values.get("ocr_cer") is None or values["ocr_cer"] > 0.05:
                failures.append("ocr_cer above 0.05")
            if video_available and values.get("video_order_ok") is not True:
                failures.append("video order gate failed")
            if reference and reference.get("available") and values.get("vqa_accuracy", 0) < \
                    reference["vqa_accuracy"] - 0.2 - 1e-12:
                failures.append("vqa_accuracy is more than 0.2 below Transformers reference")
        if "audio" in modalities and (values.get("asr_wer") is None or values["asr_wer"] > 0.10):
            failures.append("asr_wer above 0.10")
    else:
        base = baseline.get("metrics", baseline)
        if "vision" in modalities:
            if values.get("vqa_accuracy") is None or values["vqa_accuracy"] < \
                    base.get("vqa_accuracy", 0) - 0.2 - 1e-12:
                failures.append("vqa_accuracy regressed by more than 0.2 from F16")
            if values.get("ocr_cer") is None or values["ocr_cer"] > \
                    base.get("ocr_cer", 0) + 0.05 + 1e-12:
                failures.append("ocr_cer regressed by more than 0.05 from F16")
            if video_available and base.get("video_order_ok") is True and \
                    values.get("video_order_ok") is not True:
                failures.append("video order regressed from F16")
        if "audio" in modalities and (values.get("asr_wer") is None or values["asr_wer"] > \
                base.get("asr_wer", 0) + 0.05 + 1e-12):
            failures.append("asr_wer regressed by more than 0.05 from F16")
    return not failures, failures


def _snapshot_patterns(plan):
    allow = ["*.json", "*.txt", "*.model", "*.safetensors", "tokenizer*", "vocab*",
             "merges*", "spiece*", "sentencepiece*", "preprocessor_config.json",
             "processor_config.json", "chat_template*", "*.jinja"]
    if not any(name.lower().endswith(".safetensors") for name in plan.get("files", [])):
        allow.append("*.bin")
    ignore = ["onnx/*", "onnx/**", "openvino/*", "openvino/**", "*.h5", "**/*.h5",
              "*.msgpack", "**/*.msgpack", "*.ot", "**/*.ot"]
    return allow, ignore


def run_release(plan, *, user_approved, publish=False):
    if user_approved is not True:
        raise ValueError("multimodal run requires explicit approval of the exact plan")
    if plan.get("status") != "ready" or plan.get("blockers"):
        raise ValueError("multimodal plan is blocked: " + "; ".join(plan.get("blockers", [])))
    root = config.TEMP_DIR / "aqx-multimodal-runs" / _slug(plan["model"])
    root.mkdir(parents=True, exist_ok=True)
    run_path, journal_path = root / "run.json", root / "journal.json"
    saved = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.is_file() else {}
    from .. import runtime_control
    handle = runtime_control.ensure_runtime(
        "llama.cpp", frozen=saved.get("runtime_snapshot") or plan.get("runtime_snapshot"),
        required_targets=RUNTIME_TARGETS)
    if handle is None:
        raise RuntimeError("no execution runtime session for llama.cpp")
    binaries = dict(zip(RUNTIME_TARGETS, require_binaries(handle, *RUNTIME_TARGETS)))
    run = {"plan": plan, "runtime_snapshot": handle.snapshot(),
           "created": saved.get("created") or time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    _write(run_path, run)
    allow, ignore = _snapshot_patterns(plan)
    source = Path(snapshot_download(
        plan["model"], revision=plan.get("source_revision"), token=config.TOKEN,
        allow_patterns=allow, ignore_patterns=ignore))
    reference = _reference(plan, source, journal, journal_path)
    text_f16, mmproj, metadata, modalities = _convert(
        plan, source, handle, root, journal, journal_path)
    video_available = "vision" in modalities and _video_support(binaries["llama-mtmd-cli"])
    candidates = []
    baseline = None
    quant_list = list(dict.fromkeys(["F16", *plan.get("quants", [])]))
    for quant in quant_list:
        model = text_f16 if quant == "F16" else _quantize(
            binaries["llama-quantize"], text_f16, root, journal, journal_path, quant)
        key = f"evaluation:{quant}"
        cached = journal.get(key, {})
        if cached.get("sha256") == _hash(model) and cached.get("quality"):
            item = cached
            quality = item["quality"]
            passed = item["status"] == "passed"
        else:
            quality = _evaluate(plan, model, mmproj, metadata, modalities, binaries, root,
                                video_available)
            passed, failures = _quality_gate(
                quality, modalities, video_available,
                reference=reference if quant == "F16" else None,
                baseline=baseline if quant != "F16" else None)
            quality["gate_failures"] = failures
            quality["status"] = "passed" if passed else "quality-rejected"
            item = {"quant": quant, "path": str(model), "sha256": _hash(model),
                    "bytes": model.stat().st_size, "quality": quality,
                    "status": quality["status"]}
        candidates.append(item)
        journal[f"evaluation:{quant}"] = item
        _write(journal_path, journal)
        if quant == "F16":
            baseline = quality
            if not passed:
                break
    result = {"run_id": root.name, "plan": plan, "runtime_snapshot": handle.snapshot(),
              "projector": {"path": str(mmproj), "sha256": _hash(mmproj),
                            "bytes": mmproj.stat().st_size, "metadata": metadata},
              "modalities": modalities,
              "video": ("available" if video_available else
                        "skipped: runtime lacks video" if "vision" in modalities else "not applicable"),
              "candidates": candidates, "published": False, "reference": reference}
    _write(root / "result.json", result)
    if publish and not any(item["status"] == "passed" for item in candidates):
        raise ValueError("no multimodal text quant passed the automated gates")
    if publish and any(item["status"] == "passed" for item in candidates):
        result["publication"] = publish_bundle(root, plan, result, api=None)
        result["published"] = True
        _write(root / "result.json", result)
    return result


def publish_bundle(root, plan, result, api=None):
    from huggingface_hub import CommitOperationAdd, HfApi
    from .voice_release import _remote_files
    from .. import config as aqx_config

    api = api or HfApi(token=aqx_config.TOKEN)
    root = Path(root)
    publication = root / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    members = []
    name = plan["model"].rsplit("/", 1)[-1]
    for item in result["candidates"]:
        if item["status"] != "passed":
            continue
        path = Path(item["path"])
        destination = publication / "models" / f"{name}-{item['quant']}.gguf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        members.append({"path": destination.relative_to(publication).as_posix(),
                        "quant": item["quant"], "bytes": destination.stat().st_size,
                        "sha256": _hash(destination)})
    projector = Path(result["projector"]["path"])
    projector_dest = publication / "models" / f"mmproj-{name}-F16.gguf"
    projector_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(projector, projector_dest)
    members.append({"path": projector_dest.relative_to(publication).as_posix(),
                    "quant": "F16", "role": "projector", "bytes": projector_dest.stat().st_size,
                    "sha256": _hash(projector_dest)})
    snapshot = result.get("runtime_snapshot", {})
    bundle = multimodal.MultimodalBundle(
        "llama.cpp", name, plan["model"], plan.get("source_revision"),
        snapshot.get("revision"), members, plan["contract"], result["modalities"],
        result["projector"]["metadata"],
        runtime_commands=multimodal.runtime_commands(
            f"./models/{name}-F16.gguf", f"./models/mmproj-{name}-F16.gguf",
            result["modalities"], video=result["video"] == "available"),
        quality_results={item["quant"]: item["quality"] for item in result["candidates"]
                         if item["status"] == "passed"},
        quality_profile=plan.get("quality_profile", {}),
        build_provenance=snapshot.get("provenance", {}),
        fixtures=plan.get("fixtures", {}), source_license=plan.get("license"),
        target_repo=plan["target_repo"])
    manifest = bundle.manifest()
    manifest["video_available"] = result["video"] == "available"
    card_path = publication / "README.md"
    card_path.write_text(multimodal.render_card(manifest), encoding="utf-8")
    manifest_path = publication / "manifest.json"
    manifest["published_files"] = [
        {"path": path.relative_to(publication).as_posix(), "bytes": path.stat().st_size,
         "sha256": _hash(path)}
        for path in sorted(publication.rglob("*"))
        if path.is_file() and path != manifest_path
        and path.relative_to(publication).as_posix() not in {member["path"] for member in members}]
    _write(manifest_path, manifest)
    operations = [CommitOperationAdd(path_in_repo=path.relative_to(publication).as_posix(),
                                     path_or_fileobj=str(path))
                  for path in sorted(publication.rglob("*")) if path.is_file()]
    api.create_repo(repo_id=plan["target_repo"], repo_type="model", exist_ok=True)
    api.create_commit(repo_id=plan["target_repo"], repo_type="model", operations=operations,
                      commit_message="Publish multimodal bundle with measured quality gates")
    remote = _remote_files(api, plan["target_repo"])
    mismatches = []
    for member in [*members, *manifest["published_files"],
                   {"path": "manifest.json", "bytes": manifest_path.stat().st_size,
                    "sha256": _hash(manifest_path)}]:
        actual = remote.get(member["path"])
        if actual is None or actual.get("bytes") != member["bytes"] or \
                actual.get("sha256") != member["sha256"]:
            mismatches.append(member["path"])
    if mismatches:
        raise ValueError("remote verification mismatch: " + ", ".join(mismatches))
    return {"repo_id": plan["target_repo"], "verified": True, "members": members}


def publish_run(slug, quant):
    root = config.TEMP_DIR / "aqx-multimodal-runs" / slug
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    if not any(item["quant"].casefold() == quant.casefold() and item["status"] == "passed"
               for item in result["candidates"]):
        raise ValueError(f"{quant} is not a passing multimodal quant in {slug}")
    result["publication"] = publish_bundle(root, result["plan"], result)
    result["published"] = True
    _write(root / "result.json", result)
    return result["publication"]


def verify_manifest(manifest, root):
    root = Path(root)
    problems = []
    for member in [*manifest.get("members", []), *manifest.get("published_files", [])]:
        path = root / member["path"]
        if not path.is_file():
            problems.append(f"member missing: {member['path']}")
        elif path.stat().st_size != member["bytes"]:
            problems.append(f"member size mismatch: {member['path']}")
        elif _hash(path) != member["sha256"]:
            problems.append(f"member checksum mismatch: {member['path']}")
    return problems
