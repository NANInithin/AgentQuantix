"""Contracts for native retrieval releases and their measured quality gates."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

TEXT_QUANTS = ("F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0")
CLIP_QUANTS = ("f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0")
CLIP_QUANT_TYPES = {"q4_0": 2, "q4_1": 3, "q5_0": 6, "q5_1": 7, "q8_0": 8}
QUALITY_PROFILE = {
    "embeddings": {"f16_reference_cosine": 0.999, "quant_f16_cosine": 0.98,
                   "metric_regression": 0.05, "norm_tolerance": 1e-3},
    "reranking": {"f16_logit_error": 0.1, "metric_regression": 0.05},
    "clip": {"f16_image_cosine": 0.999, "f16_text_cosine": 0.999,
             "quant_f16_cosine": 0.95},
}


def source_contract(repo_id, files, read_json):
    """Extract native preprocessing and pooling facts without guessing."""
    names = set(files)
    contract, blockers = {}, []
    if "config.json" in names:
        model = read_json("config.json")
        contract["architectures"] = model.get("architectures", [])
        if any(str(arch).endswith("ForSequenceClassification")
               for arch in contract["architectures"]):
            num_labels = model.get("num_labels")
            if num_labels is None:
                id2label = model.get("id2label")
                num_labels = len(id2label) if isinstance(id2label, dict) else 2
            contract["num_labels"] = num_labels
            contract["reranker"] = num_labels == 1
    if "preprocessor_config.json" in names:
        preprocessing = read_json("preprocessor_config.json")
        contract["image_preprocessing"] = {
            key: preprocessing.get(key) for key in ("image_mean", "image_std", "size",
                                                      "crop_size", "do_center_crop",
                                                      "resample", "do_resize")
            if key in preprocessing
        }
    if "modules.json" in names:
        modules = read_json("modules.json")
        classes = [str(item.get("type", "")).rsplit(".", 1)[-1]
                   for item in modules if isinstance(item, dict)]
        contract["module_directories"] = list(dict.fromkeys(
            str(item["path"]).strip("/") for item in modules
            if isinstance(item, dict) and item.get("path")
            and ".." not in str(item["path"]).split("/")))
        pooling = None
        if "Pooling" in classes:
            if "1_Pooling/config.json" in names:
                settings = read_json("1_Pooling/config.json")
                supported_modes = (("pooling_mode_cls_token", "cls"),
                                   ("pooling_mode_mean_tokens", "mean"),
                                   ("pooling_mode_lasttoken", "last"))
                enabled = [value for key, value in supported_modes if settings.get(key)]
                unsupported = ("pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens",
                               "pooling_mode_weightedmean_tokens")
                if len(enabled) == 1 and not any(settings.get(key) for key in unsupported):
                    pooling = enabled[0]
            if pooling is None:
                blockers.append("unsupported pooling")
        contract["pooling"] = pooling
        contract["normalize"] = "Normalize" in classes
        if "Normalize" not in classes:
            contract["normalize"] = False
        if "config_sentence_transformers.json" in names:
            st = read_json("config_sentence_transformers.json")
            contract["prompts"] = st.get("prompts", {})
        if "sentence_bert_config.json" in names:
            contract["max_seq_length"] = read_json("sentence_bert_config.json").get("max_seq_length")
        if not classes or "Pooling" not in classes:
            blockers.append("unsupported pooling")
    if "CLIPModel" in contract.get("architectures", []) and not contract.get("image_preprocessing"):
        blockers.append("missing CLIP preprocessing contract")
    if not contract.get("architectures") and "modules.json" not in names and \
            "preprocessor_config.json" not in names:
        blockers.append("no sentence-transformers pooling contract; the native pooling cannot be verified against a reference")
    if "modules.json" not in names and "preprocessor_config.json" not in names and \
            contract.get("num_labels") is None:
        blockers.append("no sentence-transformers pooling contract; the native pooling cannot be verified against a reference")
    return {"repo_id": repo_id, "contract": contract, "blockers": list(dict.fromkeys(blockers))}


def cosine(a, b):
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("cosine inputs must have the same nonzero dimension")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(y) ** 2 for y in b))
    return dot / (na * nb) if na and nb else 0.0


def recall_at_k(ranked, relevant, k):
    relevant = set(relevant)
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(rankings):
    values = []
    for ranked, relevant in rankings:
        relevant = set(relevant)
        values.append(next((1 / (index + 1) for index, item in enumerate(ranked)
                            if item in relevant), 0.0))
    return sum(values) / len(values) if values else 0.0


def ndcg_at_k(ranked, gains, k):
    def dcg(items):
        return sum((2 ** gains.get(item, 0) - 1) / math.log2(index + 2)
                   for index, item in enumerate(items[:k]))
    ideal = sorted(gains, key=gains.get, reverse=True)
    best = dcg(ideal)
    return dcg(ranked) / best if best else 0.0


def top1_agreement(left, right):
    return sum(len(a) > 0 and len(b) > 0 and a[0] == b[0] for a, b in zip(left, right)) / max(
        len(left), 1)


@dataclass
class RetrievalBundle:
    engine: str
    task: str
    family: str
    architecture: str
    source_repo: str
    source_revision: str | None
    engine_revision: str | None
    members: list
    contract: dict
    runtime_commands: list = field(default_factory=list)
    quality_results: dict = field(default_factory=dict)
    build_provenance: dict = field(default_factory=dict)
    quality_profile: dict = field(default_factory=dict)
    fixtures: dict = field(default_factory=dict)
    source_license: str | None = None
    target_repo: str | None = None

    def manifest(self):
        return {"schema_version": 1, "pipeline": "retrieval", "engine": self.engine,
                "task": self.task, "family": self.family, "architecture": self.architecture,
                "source": {"repo_id": self.source_repo, "revision": self.source_revision},
                "engine_revision": self.engine_revision, "build_provenance": self.build_provenance,
                "members": self.members, "contract": self.contract,
                "runtime_commands": self.runtime_commands, "quality_results": self.quality_results,
                "quality_profile": self.quality_profile, "fixtures": self.fixtures,
                "source_license": self.source_license, "target_repo": self.target_repo}


def runtime_commands(engine, task, contract, model_path):
    if engine == "clip.cpp":
        return [f"extract -m {model_path} -t 4 --image image.png --text 'a red circle'",
                f"zsl -m {model_path} -t 4 --image image.png --text 'a red circle' "
                "--text 'a blue square'"]
    if task == "reranking":
        return [f"llama-server --reranking -m {model_path} --host 127.0.0.1 --port 8080",
                "curl -s http://127.0.0.1:8080/v1/rerank -H 'Content-Type: application/json' "
                "-d '{\"query\":\"query\",\"documents\":[\"document\"]}'"]
    pooling = contract.get("pooling") or "mean"
    return [f"llama-embedding -m {model_path} -p 'query text' --pooling {pooling} "
            "--embd-normalize 2 --embd-output-format array"]


def _literal_prompt_prefix(template):
    value = str(template)
    for marker in ("{query}", "{document}", "{text}"):
        if marker in value:
            return value.split(marker, 1)[0]
    return value


def render_card(bundle):
    manifest = bundle.manifest() if isinstance(bundle, RetrievalBundle) else bundle
    metrics = manifest.get("quality_results", {})
    engine, task = manifest["engine"], manifest["task"]
    target_repo = manifest.get("target_repo") or "TARGET_REPO"
    members = manifest.get("members", [])
    model_path = f"./retrieval-model/{members[0]['path']}" if members else "./model.gguf"
    download = "\n".join(
        f"hf download {target_repo} {member['path']} --local-dir ./retrieval-model"
        for member in members)
    contract = manifest.get("contract", {})
    example = "\n".join(runtime_commands(engine, task, contract, model_path))
    prompts = contract.get("prompts") or {}
    prompt_text = "\n".join(
        f"- {role.title()} prompt prefix to prepend: "
        f"`{_literal_prompt_prefix(prompts[role])}`"
        for role in ("query", "document") if role in prompts and prompts[role] is not None)
    if prompt_text:
        prompt_text = "\n\nPrompt templates:\n\n" + prompt_text + "\n"
    probes = [entry.get("tokenizer_parity_probe") for entry in metrics.values()
              if isinstance(entry, dict) and entry.get("tokenizer_parity_probe")]
    probe_text = ""
    if probes:
        probe_text = "\n\nTokenizer-parity probe: " + ", ".join(
            f"normalized cosine {probe.get('normalized_cosine', 0):.4f}, "
            f"punctuated/capitalized cosine {probe.get('raw_cosine', 0):.4f}"
            for probe in probes)
    source = manifest.get("source", {})
    member_text = "\n".join(
        f"- `{member['path']}` ({member['quant']}, {member['bytes']} bytes, "
        f"SHA-256 `{member['sha256']}`)"
        for member in members)
    license_text = manifest.get("source_license") or "unspecified"
    reranking_note = ("\n\nThe endpoint's `relevance_score` is a raw, unbounded logit, "
                      "not a probability." if task == "reranking" else "")
    return (f"# {manifest.get('family', 'Retrieval model')} — {engine}\n\n"
            f"- Task: `{task}`\n- Source: `{source.get('repo_id', 'unknown')}` "
            f"(revision `{source.get('revision', 'unknown')}`)\n"
            f"- Engine revision: `{manifest.get('engine_revision', 'unknown')}`\n"
            f"- Architecture: `{manifest.get('architecture', 'unknown')}`\n"
            f"- Source model license: `{license_text}`\n\n"
            f"## Bundle files\n\n{member_text}\n\n"
            f"## Usage\n\n```sh\n{download}\n{example}\n```\n{prompt_text}{reranking_note}\n\n"
            f"## Measured quality\n\n```json\n{json.dumps(metrics, indent=2)}\n```\n\n"
            "CLIP text inputs are lowercased, stripped of punctuation, and whitespace-collapsed; "
            "the tokenizer-parity measurement above documents the raw-text limitation. llama.cpp "
            "reranking uses `/v1/rerank`; `--pooling rank` is broken in the recorded llama.cpp revision."
            f"{probe_text}\n\nModel redistribution terms are distinct from the runtime code license; "
            "check the source model license before redistribution.\n")


def plan_release(repo_id, *, task, engine, target_repo=None, quants=None):
    """Resolve source metadata and contract; execution rechecks runtime support."""
    from huggingface_hub import HfApi, hf_hub_download
    from . import config
    from .tasks.registry import task_spec

    spec = task_spec(task)
    if spec.pipeline != "retrieval" or engine not in spec.engines:
        raise ValueError(f"{task} does not use retrieval engine {engine}")
    info = HfApi(token=config.TOKEN).model_info(repo_id, files_metadata=True)
    files = [item.rfilename for item in info.siblings or []]

    def read_json(name):
        return json.loads(Path(hf_hub_download(repo_id, name, revision=info.sha,
                                               token=config.TOKEN)).read_text(encoding="utf-8"))

    source = source_contract(repo_id, files, read_json)
    if engine == "clip.cpp":
        available = list(CLIP_QUANTS)
    else:
        try:
            from .archsupport import supported_quants
            installed = supported_quants()
            available = [quant for quant in installed if not quant.startswith(("IQ", "TQ"))
                         and quant in {"F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_0",
                                       "Q4_K_M", "Q4_0"}]
        except Exception:
            available = []
    selected = list(quants or (CLIP_QUANTS if engine == "clip.cpp" else TEXT_QUANTS))
    allowed = set(CLIP_QUANTS if engine == "clip.cpp" else (available or TEXT_QUANTS))
    invalid = [quant for quant in selected if quant not in allowed]
    blockers = list(source["blockers"])
    if invalid:
        blockers.append(f"unsupported quant(s): {', '.join(invalid)}")
    if task == "reranking" and source["contract"].get("num_labels") != 1:
        blockers.append("reranker must have one classification label")
    engine_revision = None
    architecture_support = "not-applicable" if engine == "clip.cpp" else "unknown"
    try:
        from .runtimes import RuntimeManager
        handle = RuntimeManager().inspect(engine, verify=False)
        if handle:
            engine_revision = handle.revision
        if engine == "llama.cpp":
            if handle:
                from .archsupport import supported_architectures
                supported = supported_architectures(handle.source)
                architectures = source["contract"].get("architectures", [])
                architecture_support = {
                    "supported": bool(set(architectures) & set(supported)),
                    "known": sorted(set(architectures) & set(supported))}
                if architectures and not architecture_support["supported"]:
                    blockers.append("unsupported architecture by installed llama.cpp converter")
    except Exception:
        if engine == "llama.cpp":
            architecture_support = "unknown"
    fixture_root = Path(__file__).resolve().parent / "fixtures" / "retrieval"
    fixtures = {}
    if fixture_root.is_dir():
        for path in sorted(fixture_root.rglob("*")):
            if path.is_file():
                fixtures[str(path.relative_to(fixture_root))] = hashlib.sha256(
                    path.read_bytes()).hexdigest()
    return {"model": repo_id, "status": "blocked" if blockers else "ready",
            "blockers": blockers, "task": task, "engine": engine,
            "files": files,
            "engine_revision": engine_revision, "architecture_support": architecture_support,
            "source_revision": getattr(info, "sha", None),
            "source_bytes": sum(getattr(item, "size", 0) or 0 for item in info.siblings or []),
            "gated": bool(getattr(info, "gated", False)),
            "license": (getattr(getattr(info, "card_data", None), "license", None)),
            "architectures": source["contract"].get("architectures", []),
            "converter": "models/convert_hf_to_gguf.py" if engine == "clip.cpp" else "convert_hf_to_gguf.py",
            "contract": source["contract"], "quants": selected, "available_quants": available,
            "target_repo": target_repo or f"{config.namespace()}/{repo_id.rsplit('/', 1)[-1]}-GGUF",
            "quality_profile": QUALITY_PROFILE["clip" if engine == "clip.cpp" else task],
            "fixtures": fixtures, "publication": spec.publication, "review": "none",
            "imatrix": "not used; only retrieval-supported quant types are offered"}
