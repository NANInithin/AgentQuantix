"""Contracts and measured quality gates for native multimodal releases."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
import unicodedata
from pathlib import Path

TEXT_QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
QUALITY_PROFILE = {
    "f16": {"vqa_accuracy": 0.8, "ocr_cer": 0.05, "asr_wer": 0.10},
    "quant": {"vqa_accuracy_regression": 0.2, "ocr_cer_regression": 0.05,
              "asr_wer_regression": 0.05},
}
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "multimodal"


def source_contract(repo_id, files, read_json, *, license=None):
    names = set(files)
    contract = {"architectures": [], "model_type": None, "license": license,
                "chat_template": False, "claimed_modalities": []}
    blockers = []
    model = read_json("config.json") if "config.json" in names else {}
    architectures = list(model.get("architectures") or [])
    contract.update(architectures=architectures, model_type=model.get("model_type"))
    claimed = []
    if "vision_config" in model:
        claimed.append("vision")
    if "audio_config" in model or any(
            any(marker in str(arch).casefold() for marker in ("audio", "asr"))
            for arch in architectures):
        claimed.append("audio")
    contract["claimed_modalities"] = claimed
    chat_template = any(
        name.startswith("chat_template") or name.endswith(".jinja") for name in names)
    if "tokenizer_config.json" in names:
        try:
            chat_template = chat_template or bool(read_json("tokenizer_config.json").get("chat_template"))
        except (OSError, TypeError, ValueError):
            pass
    contract["chat_template"] = chat_template
    if not claimed:
        blockers.append("no vision or audio encoder declared; not a multimodal-understanding source")
    return {"repo_id": repo_id, "contract": contract, "blockers": blockers}


def _normalise(text):
    return " ".join(str(text).split())


def _distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def cer(hyp, ref):
    hypothesis, reference = _normalise(hyp), _normalise(ref)
    return _distance(hypothesis, reference) / max(len(reference), 1)


def _words(text):
    lowered = "".join(char for char in str(text).casefold()
                      if not unicodedata.category(char).startswith("P"))
    return _normalise(lowered).split()


def wer(hyp, ref):
    hypothesis, reference = _words(hyp), _words(ref)
    return _distance(hypothesis, reference) / max(len(reference), 1)


def _synonym_index(text, synonyms):
    positions = []
    for word in synonyms:
        match = re.search(r"(?<!\w)" + re.escape(str(word)) + r"(?!\w)", text, re.I)
        if match:
            positions.append(match.start())
    return min(positions) if positions else None


def answer_hits(answer, expected):
    return all(_synonym_index(str(answer), group) is not None for group in expected)


def order_ok(answer, first, second):
    text = str(answer)
    left, right = _synonym_index(text, first), _synonym_index(text, second)
    return left is not None and right is not None and left < right


def strip_markup(text):
    value = str(text)
    markers = list(re.finditer(r"<[A-Za-z][\w:.-]*_text>", value))
    if markers:
        value = value[markers[-1].end():]
    return re.sub(r"<[^>]+>", "", value).strip()


@dataclass
class MultimodalBundle:
    engine: str
    family: str
    source_repo: str
    source_revision: str | None
    engine_revision: str | None
    members: list
    contract: dict
    modalities: list
    projector_metadata: dict
    runtime_commands: list = field(default_factory=list)
    quality_results: dict = field(default_factory=dict)
    quality_profile: dict = field(default_factory=dict)
    build_provenance: dict = field(default_factory=dict)
    fixtures: dict = field(default_factory=dict)
    source_license: str | None = None
    target_repo: str | None = None

    def manifest(self):
        return {"schema_version": 1, "pipeline": "multimodal", "engine": self.engine,
                "family": self.family,
                "source": {"repo_id": self.source_repo, "revision": self.source_revision},
                "engine_revision": self.engine_revision,
                "build_provenance": self.build_provenance, "members": self.members,
                "contract": self.contract, "modalities": self.modalities,
                "projector_metadata": self.projector_metadata,
                "runtime_commands": self.runtime_commands,
                "quality_results": self.quality_results,
                "quality_profile": self.quality_profile, "fixtures": self.fixtures,
                "source_license": self.source_license, "target_repo": self.target_repo}


def runtime_commands(model_path, mmproj_path, modalities, *, video=True):
    commands = [f"llama-server -m {model_path} --mmproj {mmproj_path} --jinja --no-webui -c 4096"]
    if "vision" in modalities:
        commands.append(
            "curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' "
            "-d '{\"messages\":[{\"role\":\"user\",\"content\":["
            "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,...\"}},"
            "{\"type\":\"text\",\"text\":\"What is in this image?\"}]}],"
            "\"temperature\":0,\"seed\":42,\"max_tokens\":96}'")
    if "audio" in modalities:
        commands.append(
            "curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' "
            "-d '{\"messages\":[{\"role\":\"user\",\"content\":["
            "{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"...\",\"format\":\"wav\"}},"
            "{\"type\":\"text\",\"text\":\"Transcribe the audio.\"}]}],"
            "\"temperature\":0,\"seed\":42,\"max_tokens\":96}'")
    if video and "vision" in modalities:
        commands.append(
            f"llama-mtmd-cli -m {model_path} --mmproj {mmproj_path} --video video.mp4 "
            "--video-fps 1 --video-timestamp-interval 1000 "
            "--temp 0 --seed 42 -n 96 --no-mmproj-offload -p 'What happens first and what happens next?'")
    return commands


def render_card(bundle):
    manifest = bundle.manifest() if isinstance(bundle, MultimodalBundle) else bundle
    metadata = manifest.get("projector_metadata", {})
    modalities = manifest.get("modalities", [])
    types = {"vision": metadata.get("clip.vision.projector_type"),
             "audio": metadata.get("clip.audio.projector_type"),
             "general": metadata.get("clip.projector_type")}
    modality_text = ", ".join(
        f"{modality} (`{types.get(modality) or types.get('general') or 'unknown'}`)"
        for modality in modalities) or "none"
    commands = runtime_commands(
        "./model.gguf", "./mmproj-model-F16.gguf", modalities,
        video=manifest.get("video_available", True))
    image = {"image_size": metadata.get("clip.vision.image_size"),
             "preproc_image_size": metadata.get("clip.vision.preproc_image_size"),
             "image_mean": metadata.get("clip.vision.image_mean"),
             "image_std": metadata.get("clip.vision.image_std"),
             "patch_size": metadata.get("clip.vision.patch_size")}
    table = []
    for quant, result in manifest.get("quality_results", {}).items():
        table.append(f"| {quant} | {result.get('status', 'unknown')} | "
                     f"{json.dumps(result.get('metrics', {}), sort_keys=True)} |")
    evidence = ("| Quant | Status | Metrics |\n|---|---|---|\n" + "\n".join(table)
                if table else "No native quality results are recorded.")
    return (f"# {manifest.get('family', 'Multimodal model')} — llama.cpp\n\n"
            f"- Source: `{manifest.get('source', {}).get('repo_id', 'unknown')}`\n"
            f"- Source revision: `{manifest.get('source', {}).get('revision')}`\n"
            f"- Runtime revision: `{manifest.get('engine_revision')}`\n"
            f"- Build provenance: `{json.dumps(manifest.get('build_provenance', {}), sort_keys=True)}`\n"
            f"- License: `{manifest.get('source_license') or 'not recorded'}`\n"
            f"- Modalities: {modality_text}\n"
            f"- Projector: F16 only; projector quantization is unaccepted.\n"
            f"- Vision image size: `{image.get('image_size')}`; preprocessing size: "
            f"`{image.get('preproc_image_size')}`, mean `{image.get('image_mean')}`, "
            f"std `{image.get('image_std')}`, patch `{image.get('patch_size')}`.\n\n"
            "Use the source chat template with `--jinja`:\n\n```sh\n" + "\n".join(commands) +
            "\n```\n\nVideo is CLI-only and requires ffmpeg. Audio input is experimental upstream; "
            "this bundle accepts ASR-style transcription only. No audio QA is accepted.\n\n"
            "## Evidence\n\n" + evidence + "\n")


def plan_release(repo_id, *, task="multimodal-understanding", engine="llama.cpp",
                 target_repo=None, quants=None):
    from huggingface_hub import HfApi, hf_hub_download
    from . import config
    from .tasks.registry import MULTIMODAL, task_spec

    spec = task_spec(task)
    if spec.pipeline != MULTIMODAL or engine not in spec.engines:
        raise ValueError(f"{task} does not use multimodal engine {engine}")
    info = HfApi(token=config.TOKEN).model_info(repo_id, files_metadata=True)
    files = [item.rfilename for item in info.siblings or []]

    def read_json(name):
        return json.loads(Path(hf_hub_download(repo_id, name, revision=info.sha,
                                               token=config.TOKEN)).read_text(encoding="utf-8"))

    license_name = getattr(getattr(info, "card_data", None), "license", None)
    source = source_contract(repo_id, files, read_json, license=license_name)
    try:
        from .archsupport import supported_quants
        available = [q for q in supported_quants()
                     if q in {"F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0"}]
    except Exception:
        available = []
    selected = list(quants if quants is not None else TEXT_QUANTS)
    allowed = set(available or ("F16", *TEXT_QUANTS))
    invalid = [q for q in selected if q not in allowed]
    blockers = list(source["blockers"])
    if invalid:
        blockers.append(f"unsupported quant(s): {', '.join(invalid)}")
    engine_revision, runtime_snapshot = None, None
    try:
        from .runtimes import RuntimeManager
        handle = RuntimeManager().inspect(engine, verify=False)
        if handle:
            engine_revision = handle.revision
            runtime_snapshot = handle.snapshot()
    except Exception:
        pass
    fixture_hashes = {}
    fixture_root = Path(__file__).resolve().parent / "fixtures"
    for directory in (FIXTURES, fixture_root / "retrieval", fixture_root / "voice"):
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    fixture_hashes[str(path.relative_to(fixture_root))] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
    return {"model": repo_id, "status": "blocked" if blockers else "ready",
            "blockers": blockers, "task": task, "engine": engine, "files": files,
            "engine_revision": engine_revision, "runtime_snapshot": runtime_snapshot,
            "source_revision": getattr(info, "sha", None),
            "source_bytes": sum(getattr(item, "size", 0) or 0 for item in info.siblings or []),
            "gated": bool(getattr(info, "gated", False)), "license": license_name,
            "architectures": source["contract"]["architectures"],
            "contract": source["contract"], "quants": selected, "available_quants": available,
            "projector_precision": "F16",
            "runtime_targets": ["llama-server", "llama-mtmd-cli", "llama-quantize"],
            "target_repo": target_repo or f"{config.namespace()}/{repo_id.rsplit('/', 1)[-1]}-GGUF",
            "quality_profile": QUALITY_PROFILE, "fixtures": fixture_hashes,
            "publication": spec.publication, "review": "none",
            "converter": "convert_hf_to_gguf.py"}
