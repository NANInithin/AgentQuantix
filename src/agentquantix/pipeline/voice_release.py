"""End-to-end TTS and ASR release pipelines.

The publication unit is a verified :class:`VoiceBundle`.  TTS uses the cached
llama.cpp checkout; ASR has a separate whisper.cpp checkout, build, converter,
model format, quantizer, and acceptance corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .. import config, voice
from . import build as build_mod, sanity


@dataclass
class VoiceReleaseOptions:
    quants: list[str] = field(default_factory=list)
    language: str | None = None
    speaker: Path | None = None
    publish: bool = True
    max_roundtrip_wer: float = 0.35
    max_wer_regression: float = 0.05
    runtime_timeout: int = 900
    pocket_language: str = "english"
    human_review: Path | None = None


def _slug(repo_id: str) -> str:
    return repo_id.replace("/", "--").replace("\\", "--")


def work_dir(repo_id: str) -> Path:
    return config.TEMP_DIR / "voice" / _slug(repo_id)


def _git_revision(directory: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=20,
        ).stdout.strip()
    except Exception:
        return None


def ensure_llama_tts() -> tuple[Path, Path, Path]:
    """Return the llama.cpp checkout, llama-tts, and llama-quantize."""
    llama_dir, quantize, _ = build_mod.ensure_tools()
    runtime = build_mod.find_binary(llama_dir, "llama-tts")
    if runtime is None:
        build_mod.run(["cmake", "--build", llama_dir / "build", "--config",
                       "Release", "--target", "llama-tts", "-j",
                       build_mod.build_jobs()])
        runtime = build_mod.find_binary(llama_dir, "llama-tts")
    if runtime is None:
        raise RuntimeError("llama.cpp build completed without llama-tts")
    return llama_dir, runtime, quantize


def _cmake_build(directory: Path, targets: tuple[str, ...]) -> None:
    defines = build_mod.build_settings()
    build_mod.run(["cmake", "-S", directory, "-B", directory / "build",
                   *defines])
    for target in targets:
        build_mod.run(["cmake", "--build", directory / "build", "--config",
                       "Release", "--target", target, "-j",
                       build_mod.build_jobs()])


def ensure_whisper_tools() -> tuple[Path, Path, Path]:
    """Build/cache whisper.cpp independently of llama.cpp."""
    directory = config.UPSTREAM_WHISPER
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        build_mod.run(["git", "clone", "https://github.com/ggml-org/whisper.cpp.git",
                       directory])
    runtime = build_mod.find_binary(directory, "whisper-cli")
    quantize = build_mod.find_binary(directory, "quantize")
    if runtime is None or quantize is None:
        _cmake_build(directory, ("whisper-cli", "quantize"))
        runtime = build_mod.find_binary(directory, "whisper-cli")
        quantize = build_mod.find_binary(directory, "quantize")
    if runtime is None or quantize is None:
        raise RuntimeError("whisper.cpp build did not produce whisper-cli and quantize")
    return directory, runtime, quantize


def _snapshot(repo_id: str, destination: Path) -> tuple[Path, str | None]:
    from huggingface_hub import HfApi, snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    revision = HfApi(token=config.TOKEN).model_info(repo_id).sha
    snapshot_download(repo_id=repo_id, revision=revision,
                      local_dir=str(destination), token=config.TOKEN)
    return destination, revision


def _converter_requirements() -> list[str]:
    missing = []
    for package in ("torch", "transformers", "safetensors"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing


def _run(command: list, label: str) -> None:
    try:
        build_mod.run_verbose(command, label=label)
    except build_mod.CommandFailed as error:
        raise voice.VoiceValidationError(str(error)) from error


def convert_tts_source(repo_id: str, options: VoiceReleaseOptions
                       ) -> tuple[voice.VoiceBackend, Path, Path, str | None,
                                  Path, Path, Path]:
    """Convert one TTS source to a BF16 primary and Q8_0 mmproj."""
    backend = voice.backend_for(repo_id, track=voice.TTS)
    if backend is None:
        raise voice.VoiceValidationError(f"no TTS backend for {repo_id}")
    if missing := _converter_requirements():
        raise RuntimeError("TTS conversion requires " + ", ".join(missing)
                           + "; install AgentQuantix[convert]")
    llama_dir, runtime, quantizer = ensure_llama_tts()
    root = work_dir(repo_id)
    source, revision = _snapshot(repo_id, root / "source")
    model_source = source
    if backend.family == "pocket-tts":
        model_source = source / "languages" / options.pocket_language
        if not model_source.is_dir():
            raise voice.VoiceValidationError(
                f"Pocket TTS language directory is missing: {model_source}")
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    base = models / f"{repo_id.split('/')[-1]}-BF16.gguf"
    mmproj = models / f"mmproj-{repo_id.split('/')[-1]}-Q8_0.gguf"
    converter = llama_dir / "convert_hf_to_gguf.py"
    if not base.exists():
        _run([sys.executable, converter, model_source, "--outfile", base,
              "--outtype", "bf16"], f"convert {backend.family} primary")
    if not mmproj.exists():
        _run([sys.executable, converter, model_source, "--outfile", mmproj,
              "--outtype", "q8_0", "--mmproj"],
             f"convert {backend.family} mmproj")
        prefixed = mmproj.with_name("mmproj-" + mmproj.name)
        if not mmproj.exists() and prefixed.exists():
            prefixed.replace(mmproj)
    if not base.is_file() or not mmproj.is_file():
        raise voice.VoiceValidationError(
            "llama.cpp conversion did not produce both primary and mmproj GGUFs")
    return backend, base, mmproj, revision, runtime, quantizer, llama_dir


def quantize_tts(base: Path, quantizer: Path, quant: str) -> Path:
    quant = quant.upper()
    if quant not in voice.TTS_QUANTS:
        raise voice.VoiceValidationError(
            f"voice-safe TTS quants are {', '.join(voice.TTS_QUANTS)}")
    output = base.with_name(base.name.replace("-BF16.gguf", f"-{quant}.gguf"))
    if not output.exists():
        _run([quantizer, base, output, quant], f"quantize TTS {quant}")
    return output


def _standard_whisper_name(repo_id: str) -> str | None:
    if not repo_id.casefold().startswith("openai/whisper-"):
        return None
    return repo_id.split("/", 1)[1].removeprefix("whisper-")


def _ensure_openai_whisper_assets() -> Path:
    directory = config.VOICE_BACKENDS_DIR / "openai-whisper"
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        build_mod.run(["git", "clone", "https://github.com/openai/whisper.git",
                       directory])
    return directory


def prepare_whisper_source(repo_id: str) -> tuple[voice.VoiceBackend, Path,
                                                   str | None, Path, Path, Path]:
    """Download a standard model or convert a custom HF Whisper checkpoint."""
    backend = voice.backend_for(repo_id, track=voice.ASR)
    if backend is None:
        raise voice.VoiceValidationError(f"no Whisper backend for {repo_id}")
    whisper_dir, runtime, quantizer = ensure_whisper_tools()
    root = work_dir(repo_id)
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    name = _standard_whisper_name(repo_id)
    revision = None
    if name:
        from huggingface_hub import HfApi, hf_hub_download
        filename = f"ggml-{name}.bin"
        revision = HfApi(token=config.TOKEN).model_info(
            "ggerganov/whisper.cpp").sha
        cached = Path(hf_hub_download(
            repo_id="ggerganov/whisper.cpp", filename=filename,
            revision=revision,
            token=config.TOKEN))
        base = models / filename
        if not base.exists():
            shutil.copy2(cached, base)
    else:
        if missing := _converter_requirements():
            raise RuntimeError("Whisper conversion requires " + ", ".join(missing))
        source, revision = _snapshot(repo_id, root / "source")
        assets = _ensure_openai_whisper_assets()
        base = models / f"ggml-{repo_id.split('/')[-1]}.bin"
        if not base.exists():
            _run([sys.executable, whisper_dir / "models" / "convert-h5-to-ggml.py",
                  source, assets, models], "convert Whisper checkpoint")
        candidates = list(models.glob("ggml-*.bin"))
        if not base.exists() and len(candidates) == 1:
            candidates[0].replace(base)
    if not base.is_file() or base.stat().st_size < 1024:
        raise voice.VoiceValidationError("Whisper conversion produced no valid model")
    return backend, base, revision, runtime, quantizer, whisper_dir


def quantize_whisper(base: Path, quantizer: Path, quant: str) -> Path:
    quant = quant.casefold()
    if quant not in voice.WHISPER_QUANTS:
        raise voice.VoiceValidationError(
            f"whisper.cpp quants are {', '.join(voice.WHISPER_QUANTS)}")
    output = base.with_name(base.stem + f"-{quant}.bin")
    if not output.exists():
        _run([quantizer, base, output, quant], f"quantize Whisper {quant}")
    return output


def load_fixtures(track: str, directory: Path | None = None) -> list[dict]:
    directory = directory or config.VOICE_FIXTURES_DIR
    path = directory / f"{track}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise voice.VoiceValidationError(f"cannot load voice fixtures {path}: {error}") \
            from error
    fixtures = data.get("fixtures") if isinstance(data, dict) else None
    if not isinstance(fixtures, list) or not fixtures:
        raise voice.VoiceValidationError(f"voice fixture pack is empty: {path}")
    return fixtures


def _tiny_whisper() -> tuple[Path, Path]:
    _, runtime, _ = ensure_whisper_tools()
    from huggingface_hub import hf_hub_download
    model = Path(hf_hub_download(
        repo_id="ggerganov/whisper.cpp", filename="ggml-tiny.bin",
        token=config.TOKEN))
    return runtime, model


def score_tts(bundle: voice.VoiceBundle, runtime: Path, options: VoiceReleaseOptions,
              fixtures: list[dict] | None = None,
              asr_runtime: Path | None = None,
              asr_model: Path | None = None) -> dict:
    fixtures = fixtures or load_fixtures(voice.TTS)
    fixtures = [fixture for fixture in fixtures
                if (fixture.get("language") in bundle.backend.languages
                    or "multilingual" in bundle.backend.languages)
                and (not fixture.get("speaker_conditioned") or options.speaker)]
    if not fixtures:
        raise voice.VoiceValidationError(
            "no TTS fixtures match this backend, language, and speaker setup")
    if asr_runtime is None or asr_model is None:
        asr_runtime, asr_model = _tiny_whisper()
    quality_dir = bundle.primary.path.parent / "quality" / (bundle.quant or "base")
    quality_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, fixture in enumerate(fixtures):
        output = quality_dir / f"{index:02d}-{fixture['id']}.wav"
        output.unlink(missing_ok=True)
        generated = sanity.validate_tts_runtime(
            runtime, bundle, fixture["text"], output,
            language=fixture.get("language") or options.language,
            speaker=options.speaker, timeout=options.runtime_timeout)
        prefix = quality_dir / f"{index:02d}-{fixture['id']}-asr"
        transcribed = sanity.validate_asr_runtime(
            asr_runtime, asr_model, output, fixture["text"], prefix,
            language=fixture.get("language"), timeout=options.runtime_timeout)
        generated.update({"id": fixture["id"], "text": fixture["text"],
                          "transcript": transcribed["transcript"],
                          "roundtrip_wer": transcribed["wer"]})
        results.append(generated)
    aggregate = {
        "fixtures": len(results),
        "roundtrip_wer": round(sum(r["roundtrip_wer"] for r in results)
                                / len(results), 6),
        "silence_ratio": round(max(r["silence_ratio"] for r in results), 6),
        "clipping_ratio": round(max(r["clipping_ratio"] for r in results), 6),
        "real_time_factor": round(sum(r["real_time_factor"] for r in results)
                                  / len(results), 4),
        "frames_per_second": round(sum(r["frames_per_second"] for r in results)
                                   / len(results), 2),
        "first_audio_seconds": round(sum(r["first_output_seconds"] for r in results)
                                     / len(results), 4),
        "minutes_per_audio_hour": round(
            sum(r["minutes_per_audio_hour"] for r in results) / len(results), 2),
    }
    aggregate["passed"] = voice.quality_passed(
        aggregate, options.max_roundtrip_wer)
    aggregate["results"] = results
    return aggregate


def score_asr(model: Path, runtime: Path, options: VoiceReleaseOptions,
              fixtures: list[dict] | None = None) -> dict:
    fixtures = fixtures or load_fixtures(voice.ASR)
    results = []
    for index, fixture in enumerate(fixtures):
        audio = config.VOICE_FIXTURES_DIR / fixture["audio"]
        prefix = model.parent / "quality" / model.stem / f"{index:02d}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        result = sanity.validate_asr_runtime(
            runtime, model, audio, fixture["transcript"], prefix,
            language=fixture.get("language"), vad=fixture.get("vad", False),
            timeout=options.runtime_timeout)
        result["id"] = fixture["id"]
        results.append(result)
    return {
        "fixtures": len(results),
        "wer": round(sum(result["wer"] for result in results) / len(results), 6),
        "real_time_factor": round(sum(result["real_time_factor"] for result in results)
                                  / len(results), 4),
        "audio_seconds_per_second": round(
            sum(result["audio_seconds_per_second"] for result in results)
            / len(results), 4),
        "results": results,
    }


def feasibility(bundle: voice.VoiceBundle, quality: dict | None = None,
                ram_gb: float | None = None, vram_gb: float | None = None) -> dict:
    quality = quality or {}
    total = sum(member.path.stat().st_size for member in bundle.members)
    return {
        "track": bundle.backend.track,
        "bundle_disk_bytes": total,
        "bundle_disk_gb": round(total / 1024 ** 3, 3),
        "ram_gb": ram_gb,
        "vram_gb": vram_gb,
        "real_time_factor": quality.get("real_time_factor"),
        "frames_per_second": quality.get("frames_per_second"),
        "audio_seconds_per_second": quality.get("audio_seconds_per_second"),
        "first_audio_seconds": quality.get("first_audio_seconds"),
        "minutes_per_audio_hour": quality.get("minutes_per_audio_hour"),
    }


def _remote_files(api, repo_id: str) -> dict[str, dict]:
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception:
        return {}
    result = {}
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        if isinstance(lfs, dict):
            digest = lfs.get("sha256") or lfs.get("oid")
            lfs_size = lfs.get("size")
        else:
            digest = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
            lfs_size = getattr(lfs, "size", None)
        if isinstance(digest, str):
            digest = digest.removeprefix("sha256:")
        result[sibling.rfilename] = {
            "bytes": getattr(sibling, "size", None) or lfs_size,
            "sha256": digest}
    return result


def publish_bundle(bundle: voice.VoiceBundle, target_repo: str,
                   license_name: str | None = None, api=None) -> dict:
    """Upload missing/mismatched members in one Hub commit, then verify."""
    from huggingface_hub import CommitOperationAdd, HfApi

    if problems := bundle.problems():
        raise voice.VoiceValidationError("refusing incomplete bundle: "
                                         + "; ".join(problems))
    api = api or HfApi(token=config.TOKEN)
    api.create_repo(repo_id=target_repo, repo_type="model", exist_ok=True)
    if license_name is None and bundle.source_repo:
        try:
            source_info = api.model_info(bundle.source_repo)
            card_data = getattr(source_info, "card_data", None)
            license_name = (getattr(card_data, "license", None)
                            or (card_data.get("license")
                                if isinstance(card_data, dict) else None))
        except Exception:
            license_name = None
    stage = bundle.primary.path.parent / "publication" / (bundle.quant or "base")
    stage.mkdir(parents=True, exist_ok=True)
    manifest_path = voice.write_manifest(bundle, stage / "bundle.json")
    quality_path = stage / "quality.json"
    quality_path.write_text(json.dumps(bundle.quality or {}, indent=2,
                                       sort_keys=True) + "\n", encoding="utf-8")
    card_path = stage / "README.md"
    card_path.write_text(voice.render_model_card(bundle, target_repo, license_name),
                         encoding="utf-8")
    manifest = voice.load_manifest(manifest_path)
    remote = _remote_files(api, target_repo)
    operations = []
    for member in bundle.members:
        expected = member.manifest()
        actual = remote.get(member.remote_path)
        if (actual and actual.get("bytes") == expected["bytes"]
                and (not actual.get("sha256")
                     or actual["sha256"] == expected["sha256"])):
            continue
        operations.append(CommitOperationAdd(
            path_in_repo=member.remote_path, path_or_fileobj=str(member.path)))
    for local, remote_path in ((manifest_path, "bundle.json"),
                               (quality_path, "quality.json"),
                               (card_path, "README.md")):
        operations.append(CommitOperationAdd(
            path_in_repo=remote_path, path_or_fileobj=str(local)))
    if operations:
        api.create_commit(repo_id=target_repo, repo_type="model",
                          operations=operations,
                          commit_message=(f"Publish {bundle.backend.family} "
                                          f"{bundle.quant or 'base'} bundle"))
    remote = _remote_files(api, target_repo)
    if problems := voice.remote_bundle_problems(manifest, remote):
        raise voice.VoiceValidationError("published bundle failed verification: "
                                         + "; ".join(problems))
    return {"repo_id": target_repo, "uploaded": len(operations),
            "members": [member.remote_path for member in bundle.members],
            "verified": True}


def _review_for(path: Path, quant: str) -> dict | None:
    candidate = path / f"{quant}.json" if path.is_dir() else path
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    reviews = data.get("reviews") if isinstance(data, dict) else None
    entries = reviews if isinstance(reviews, list) else [data]
    return next((entry for entry in entries
                 if isinstance(entry, dict) and entry.get("quant") == quant), None)


def _aggregate_bundles(bundles: list[voice.VoiceBundle]) -> voice.VoiceBundle:
    """One atomic publication containing every accepted quant variant."""
    if not bundles:
        raise voice.VoiceValidationError("cannot publish an empty voice bundle")
    first = bundles[0]
    primary_members = [bundle.primary for bundle in bundles]
    companion_by_path = {}
    for bundle in bundles:
        for member in bundle.companions:
            companion_by_path[member.remote_path] = member
    quality = {"quants": {bundle.quant: bundle.quality for bundle in bundles}}
    return voice.VoiceBundle(
        backend=first.backend,
        primary=primary_members[0],
        companions=tuple([*primary_members[1:], *companion_by_path.values()]),
        source_repo=first.source_repo,
        source_revision=first.source_revision,
        configuration=first.configuration,
        quality=quality,
    )


def run_tts_release(repo_id: str, target_repo: str,
                    options: VoiceReleaseOptions | None = None) -> dict:
    options = options or VoiceReleaseOptions()
    backend, base, mmproj, revision, runtime, quantizer, llama_dir = \
        convert_tts_source(repo_id, options)
    quants = options.quants or list(backend.supported_quants)
    results = {}
    accepted = []
    for quant in quants:
        primary = quantize_tts(base, quantizer, quant)
        bundle = voice.VoiceBundle(
            backend=backend,
            primary=voice.BundleMember(primary, "primary", quant=quant),
            companions=(voice.BundleMember(mmproj, "mmproj", quant="Q8_0"),),
            source_repo=repo_id, source_revision=revision, quant=quant,
            configuration={"language": options.language,
                           "speaker_reference": backend.speaker_reference,
                           "llama_cpp_revision": _git_revision(llama_dir)},
        )
        quality = score_tts(bundle, runtime, options)
        bundle = voice.VoiceBundle(**{**bundle.__dict__, "quality": quality})
        entry = {"quality": quality, "feasibility": feasibility(bundle, quality)}
        if not quality["passed"]:
            entry["published"] = False
            entry["reason"] = "automated quality gate failed"
        elif options.human_review is None:
            entry["published"] = False
            entry["reason"] = "human listening review required"
        else:
            review = _review_for(Path(options.human_review), quant)
            if not review or not review.get("accepted"):
                entry["published"] = False
                entry["reason"] = "human listening review did not accept this quant"
            else:
                accepted.append(bundle)
                entry["published"] = False
                entry["reason"] = "accepted and waiting for atomic bundle publication"
        results[quant] = entry
    if accepted and options.publish:
        publication = publish_bundle(_aggregate_bundles(accepted), target_repo)
        for bundle in accepted:
            results[bundle.quant]["publication"] = publication
            results[bundle.quant]["published"] = True
            results[bundle.quant].pop("reason", None)
    return {"backend": backend.id, "source": repo_id, "target": target_repo,
            "results": results}


def run_asr_release(repo_id: str, target_repo: str,
                    options: VoiceReleaseOptions | None = None) -> dict:
    options = options or VoiceReleaseOptions()
    backend, base, revision, runtime, quantizer, whisper_dir = \
        prepare_whisper_source(repo_id)
    baseline = score_asr(base, runtime, options)
    results = {}
    accepted = []
    for quant in options.quants or list(backend.supported_quants):
        model = quantize_whisper(base, quantizer, quant)
        quality = score_asr(model, runtime, options)
        quality["regression"] = voice.asr_regression(
            quality["wer"], baseline["wer"], options.max_wer_regression)
        bundle = voice.VoiceBundle(
            backend=backend,
            primary=voice.BundleMember(model, "asr-model", quant=quant),
            source_repo=repo_id, source_revision=revision, quant=quant,
            configuration={"sample_rate": 16_000,
                           "whisper_cpp_revision": _git_revision(whisper_dir)},
            quality=quality,
        )
        entry = {"quality": quality, "feasibility": feasibility(bundle, quality),
                 "published": False}
        if quality["regression"]["passed"]:
            accepted.append(bundle)
            entry["reason"] = "accepted and waiting for atomic bundle publication"
        elif not quality["regression"]["passed"]:
            entry["reason"] = "WER regression gate failed"
        results[quant] = entry
    if accepted and options.publish:
        publication = publish_bundle(_aggregate_bundles(accepted), target_repo)
        for bundle in accepted:
            results[bundle.quant]["publication"] = publication
            results[bundle.quant]["published"] = True
            results[bundle.quant].pop("reason", None)
    return {"backend": backend.id, "source": repo_id, "target": target_repo,
            "baseline": baseline, "results": results}
