"""Modality-neutral capabilities, media bundle graphs, and read-only planning."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fnmatch
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import urllib.request

from . import config
from .media_catalog import (ENGINE_REVISION, ROLE_FLAGS, catalog_runtime, load_catalog, selected_catalog_path)

FIXTURES = Path(__file__).parent / "fixtures/media"
MEDIA_TASKS = {"text-to-image", "image-to-image", "image-editing", "text-to-video",
               "image-to-video", "text-to-audio-video", "image-to-audio-video",
               "first-last-to-audio-video", "reference-to-audio-video"}
MEDIA_TASKS.update({"first-last-to-video", "reference-to-video"})


class MediaValidationError(ValueError):
    """A source, artifact, approval, or release contract is incomplete."""


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or ":" in value or value in (".", "")):
        raise MediaValidationError(f"unsafe bundle path: {value}")
    return path.as_posix()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def fixture_hashes():
    return {p.name: file_hash(p) for p in sorted(FIXTURES.iterdir()) if p.is_file()}


@dataclass(frozen=True)
class RuntimeCapability:
    id: str
    engine: str
    family: str
    tasks: list
    source_matchers: list
    architecture_matchers: list
    load_formats: list
    converter: str
    quantizer: str
    runtime: str
    component_contract: list
    quant_policies: dict
    conditioning_inputs: list
    output_contract: dict
    license_policy: str
    evidence: list
    backend_revision: str
    load: bool = False
    convert: str = "unresolved"
    quantize: str = "runtime-probe-required"
    status: str = "runtime-only"


@dataclass
class MediaComponent:
    role: str
    source_repo: str
    source_revision: str
    source_path: str
    output_path: str
    format: str
    precision: str
    sha256: str = ""
    bytes: int = 0
    source_bytes: int = 0
    source_sha256: str = ""
    required: bool = True
    quantized: bool = False
    converter_revision: str = ENGINE_REVISION
    dependencies: list[str] = field(default_factory=list)
    tensor_rules: str = ""
    tensor_precisions: dict = field(default_factory=dict)
    license_results: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)


@dataclass
class MediaBundle:
    family: str
    task: str
    quant: str
    components: list[MediaComponent]
    runtime_recipe: dict
    engine_revision: str = ENGINE_REVISION
    build_provenance: dict = field(default_factory=dict)
    quality_results: dict = field(default_factory=dict)
    license_results: dict = field(default_factory=dict)
    plan_id: str = ""
    required_roles: list[str] = field(default_factory=list)

    def manifest(self):
        return {"schema_version": 1, "engine": "stable-diffusion.cpp", **asdict(self)}

    def problems(self, root=None):
        problems, seen = [], set()
        roles = {c.role for c in self.components}
        missing_roles = set(self.required_roles) - roles
        if missing_roles:
            problems.append("missing required roles: " + ", ".join(sorted(missing_roles)))
        if "audio-video" in self.task and "audio_vae" not in roles:
            problems.append("audio-video bundle requires audio_vae")
        conditioning = self.runtime_recipe.get("conditioning", {})
        if (self.task.startswith("image-to") or self.task == "image-editing") and not ({"init_image", "ref_image"} & set(conditioning)):
            problems.append("image-conditioned task has no image input")
        if self.task.startswith("first-last") and not {"init_image", "end_image"} <= set(conditioning):
            problems.append("first/last-frame task needs both conditioning frames")
        if self.task.startswith("reference-to") and not ({"ref_image", "ref_video", "ref_audio"} & set(conditioning)):
            problems.append("reference task has no reference inputs")
        for component in self.components:
            if component.role not in ROLE_FLAGS and component.required:
                problems.append(f"unknown required component role: {component.role}")
            try:
                name = relative_path(component.output_path)
            except MediaValidationError as error:
                problems.append(str(error))
                continue
            if name in seen:
                problems.append(f"duplicate output path: {name}")
            seen.add(name)
            if not re.fullmatch(r"[0-9a-f]{40,64}", component.source_revision):
                problems.append(f"unpinned source revision: {component.role}")
            if not set(component.dependencies) <= roles:
                problems.append(f"missing dependency for {component.role}")
            if root is not None:
                path = Path(root) / name
                if not path.is_file() or path.stat().st_size != component.bytes or component.bytes <= 0:
                    problems.append(f"missing or wrong size: {name}")
                elif not component.sha256 or file_hash(path) != component.sha256:
                    problems.append(f"checksum mismatch: {name}")
        dependencies = {c.role: set(c.dependencies) for c in self.components}
        while dependencies:
            ready = {role for role, deps in dependencies.items() if not deps}
            if not ready:
                problems.append("component graph has a dependency cycle")
                break
            dependencies = {r: ds - ready for r, ds in dependencies.items() if r not in ready}
        expected_revision = self.build_provenance.get("revision", self.build_provenance.get("engine_revision"))
        if (not re.fullmatch(r"[0-9a-f]{40,64}", self.engine_revision)
                or (expected_revision and self.engine_revision != expected_revision)):
            problems.append("engine revision mismatch")
        if self.task not in MEDIA_TASKS:
            problems.append(f"unsupported media task: {self.task}")
        for name in self.runtime_recipe.get("components", {}).values():
            if name not in seen:
                problems.append(f"recipe references a file outside the bundle: {name}")
        for name in self.runtime_recipe.get("conditioning", {}).values():
            for item in name if isinstance(name, list) else [name]:
                if item not in seen:
                    problems.append(f"conditioning file outside the bundle: {item}")
        return problems


def capabilities():
    return [RuntimeCapability(**entry) for entry in load_catalog()["capabilities"]]


def looks_like_media(repo_id, metadata=None):
    metadata = metadata or {}
    tag = metadata.get("pipeline_tag", "")
    if tag in MEDIA_TASKS:
        return True
    if metadata.get("library_name") in ("diffusers", "stable-diffusion.cpp"):
        return True
    if set(metadata.get("tags", [])) & {"diffusers", "stable-diffusion", "text-to-image", "text-to-video"}:
        return True
    return bool(re.search(r"diffusion|flux|wan[\d.-]|minimax.?h3|qwen.?image|ltx.?video|sdxl",
                          repo_id, re.I))


def _json_url(url, token=None):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read(4 * 1024 * 1024))


def tensor_names_from_prefix(data, suffix, details=False):
    stream = io.BytesIO(data) if isinstance(data, bytes) else data
    consumed = 0

    def read(count):
        nonlocal consumed
        if consumed + count > 32 * 1024 * 1024:
            raise MediaValidationError("tensor metadata exceeds the 32 MiB inspection limit")
        result = stream.read(count)
        consumed += len(result)
        if len(result) != count:
            raise EOFError("tensor metadata exceeds the header range")
        return result

    def number(fmt):
        return struct.unpack("<" + fmt, read(struct.calcsize("<" + fmt)))[0]

    def string():
        size = number("Q")
        if size > 32 * 1024 * 1024:
            raise MediaValidationError("unreasonable tensor metadata string")
        return read(size).decode("utf-8")

    if suffix in (".safetensors", ".sft"):
        size = number("Q")
        if size > 32 * 1024 * 1024:
            raise MediaValidationError("unreasonable safetensors header")
        return list(json.loads(read(size)))
    if suffix != ".gguf" or read(4) != b"GGUF" or number("I") not in (2, 3):
        raise MediaValidationError("unsupported tensor metadata format")
    tensors, metadata = number("Q"), number("Q")
    if max(tensors, metadata) > 2_000_000:
        raise MediaValidationError("unreasonable GGUF metadata count")

    def value(kind):
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
        if kind in formats:
            number(formats[kind])
        elif kind == 8:
            string()
        elif kind == 9:
            item_type, count = number("I"), number("Q")
            if item_type == 9 or count > 2_000_000:
                raise MediaValidationError("invalid GGUF metadata array")
            for _ in range(count):
                value(item_type)
        else:
            raise MediaValidationError("unknown GGUF metadata type")

    for _ in range(metadata):
        string()
        value(number("I"))
    names, descriptions = [], []
    for _ in range(tensors):
        names.append(string())
        dims = number("I")
        if dims > 8:
            raise MediaValidationError("invalid GGUF tensor dimensions")
        shape = [number("Q") for _ in range(dims)]
        kind, offset = number("I"), number("Q")
        descriptions.append({"name": names[-1], "shape": shape, "type": kind, "offset": offset})
    return {"tensors": descriptions, "metadata_count": metadata, "header_bytes": consumed} if details else names


def local_gguf_metadata(path):
    with Path(path).open("rb") as stream:
        return tensor_names_from_prefix(stream, ".gguf", details=True)


def inspect_tensor_names(repo_id, revision, name):
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(repo_id, name, revision=revision)
    headers = {"Range": "bytes=0-33554431"}
    if config.TOKEN:
        headers["Authorization"] = f"Bearer {config.TOKEN}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return tensor_names_from_prefix(response, Path(name).suffix.lower())


def inspect_repository(repo_id, api=None):
    from huggingface_hub import HfApi, hf_hub_url
    api = api or HfApi(token=config.TOKEN)
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as error:
        raise MediaValidationError(f"{repo_id}: cannot inspect the Hub repository ({type(error).__name__}: {error})") from error
    if not re.fullmatch(r"[0-9a-f]{40,64}", info.sha or ""):
        raise MediaValidationError("Hub did not return an immutable source revision")
    files = {}
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        digest = (lfs.get("sha256") or lfs.get("oid")) if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files[sibling.rfilename] = {"bytes": getattr(sibling, "size", 0) or 0, "sha256": digest}
    configs = {}
    for name in files:
        if name in ("config.json", "model_index.json", "aqx-media-source.json") or name.endswith("/config.json"):
            if files[name]["bytes"] > 4 * 1024 * 1024:
                continue
            configs[name] = _json_url(hf_hub_url(repo_id, name, revision=info.sha), config.TOKEN)
    card = getattr(info, "card_data", None)
    card = card.to_dict() if hasattr(card, "to_dict") else card or {}
    return {"repo_id": repo_id, "revision": info.sha, "files": files, "configs": configs,
            "pipeline_tag": getattr(info, "pipeline_tag", None),
            "library_name": getattr(info, "library_name", None), "tags": list(info.tags or []),
            "license": card.get("license"), "card_data": card}


def redistribution_result(license_name, evidence=None):
    # Deliberately narrow. Nonstandard/community licenses require explicit
    # reviewed evidence rather than treating an unfamiliar string as consent.
    permitted = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "cc-by-4.0"}
    value = str(license_name or "unknown").lower()
    return {"license": value, "redistribution": "permitted" if value in permitted else "review-required",
            "evidence": evidence or "source model-card license metadata"}


def expand_policy(components, quant, supported):
    quant = next((q for q in supported if q.casefold() == quant.casefold()), quant)
    # Explicitly report the native type; no llama.cpp mixed-type alias is implied.
    if quant not in supported:
        raise MediaValidationError(f"{quant} is not supported by this sd-cli converter")
    expanded = []
    for data in components:
        item = dict(data)
        if item.get("quantizable"):
            item["precision"] = quant
            item["quantized"] = quant not in ("f16", "bf16", "f32", "orig")
            item["format"] = "gguf"
        else:
            item["quantized"] = False
        expanded.append(item)
    return expanded


def _matches(pattern, value):
    return fnmatch.fnmatchcase(value.casefold(), pattern.casefold())


def _pipeline_family(metadata):
    # Diffusers class names are not abbreviations of the native loader enum.
    pipeline = metadata.get("configs", {}).get("model_index.json", {}).get("_class_name")
    return {"StableDiffusionXLPipeline": "sdxl",
            "StableDiffusionXLInpaintPipeline": "sdxl_inpaint",
            "StableDiffusionXLInstructPix2PixPipeline": "sdxl_pix2pix"}.get(pipeline)


def _configured_vae(metadata):
    """Use the pipeline's declared VAE, preserving its default source weights."""
    configs = metadata.get("configs", {})
    declaration = configs.get("model_index.json", {}).get("vae")
    if declaration != ["diffusers", "AutoencoderKL"]:
        return None
    if configs.get("vae/config.json", {}).get("_class_name") != "AutoencoderKL":
        return None
    path = "vae/diffusion_pytorch_model.safetensors"
    return path if path in metadata["files"] else None


def derive_source_contract(capability, metadata, task=None, api=None):
    """Resolve only unambiguous native-example components after family evidence.

    File names help map roles; architecture or publisher evidence must already
    have selected the capability. Ambiguous source layouts remain questions.
    """
    recipes = capability["component_contract"]
    if task:
        recipes = [r for r in recipes if r["task"] == task]
    snapshots = {metadata["repo_id"]: metadata}
    for recipe in recipes:
        members = []
        for spec in recipe["components"]:
            role = spec["role"]
            example_name = spec["example_path"].replace("\\", "/").split("/")[-1]
            stem = re.sub(r"[-_.](?:q\d[^.]*|fp\d+|bf\d+|f\d+)$", "", Path(example_name).stem, flags=re.I)
            candidates = [name for name in metadata["files"] if name == example_name or Path(name).name == example_name]
            if not candidates:
                candidates = [name for name in metadata["files"] if name.lower().endswith((".safetensors", ".gguf", ".ckpt", ".sft"))
                              and Path(name).stem.casefold() == stem.casefold()]
            # Native examples may use an optional replacement VAE. A declared
            # Diffusers VAE is a supported source too; do not choose vae_1_0,
            # fp16 variants, or other similarly named directories by guessing.
            if not candidates and role == "vae" and (vae := _configured_vae(metadata)):
                candidates = [vae]
            if len(candidates) != 1:
                alternatives = []
                for option in spec.get("source_options", []):
                    try:
                        if option["repo"] not in snapshots:
                            snapshots[option["repo"]] = inspect_repository(option["repo"], api)
                        snapshot = snapshots[option["repo"]]
                        if option["path"] in snapshot["files"]:
                            alternatives.append((option["repo"], option["path"]))
                    except (OSError, ValueError):
                        continue
                if len(set(alternatives)) != 1:
                    break
                source_repo, source_path = alternatives[0]
            else:
                source_repo, source_path = metadata["repo_id"], candidates[0]
            retained = role in ("vae", "audio_vae", "llm_vision", "config", "tokenizer", "scheduler")
            precision = "source"
            if retained:
                match = re.search(r"(?:fp|bf|f)(16|32)", source_path, re.I)
                precision = ("f" + match.group(1)) if match else "source"
            members.append({"role": role, "source_repo": source_repo, "source_path": source_path, "quantizable": not retained,
                            "precision": precision, "format": Path(source_path).suffix.lstrip("."),
                            "tensor_rules": r".*(?:norm|bias|embed).* =f32".replace(" ", "") if not retained else "",
                            "tensor_assertions": capability["output_contract"].get("component_assertions", {}).get(role, {})})
        else:
            return {"components": members, "required_roles": [m["role"] for m in members],
                    "task": recipe["task"], "evidence": {"native_recipe": recipe["evidence"],
                    "source_metadata_sha256": canonical_hash(metadata.get("configs", {}))}}
    return {}


def plan_release(repo_id, *, target_repo=None, quants=None, family=None,
                 task=None, components=None, settings=None, compute_backend="auto",
                 sequential=False, hunt_forks=True, metadata=None, api=None, contract=None,
                 quality_profile=None, conditioning=None):
    """Inspect metadata only. No weights, builds, repositories, or uploads."""
    from .pipeline import media_release
    metadata = metadata or inspect_repository(repo_id, api)
    source_contract = contract or metadata.get("configs", {}).get("aqx-media-source.json", {})
    architecture_text = re.sub(r"[^a-z0-9]", "", json.dumps(metadata.get("configs", {})).lower())
    pipeline_family = _pipeline_family(metadata)
    matches = []
    for capability in load_catalog()["capabilities"]:
        if family and capability["family"] != family:
            continue
        matched = (any(_matches(p, repo_id) for p in capability["source_matchers"])
                   or any(a.casefold() in architecture_text for a in capability["architecture_matchers"])
                   or capability["family"] == pipeline_family)
        if matched or (family and looks_like_media(repo_id, metadata)):
            matches.append(capability)
    if len(matches) > 1:
        exact = [m for m in matches if any(_matches(p, repo_id) for p in m["source_matchers"])]
        if len(exact) == 1:
            matches = exact
    blockers, publication_blockers, questions = [], [], []
    capability = matches[0] if len(matches) == 1 else None
    if not capability:
        blockers.append("family unresolved" if not matches else "ambiguous family")
        questions.append({"field": "family", "choices": [m["family"] for m in (matches or load_catalog()["capabilities"])],
                          "reason": "architecture and pipeline evidence must identify one native family"})
    if capability and not looks_like_media(repo_id, metadata):
        blockers.append("repository metadata does not establish a media model")
    contract = source_contract or (derive_source_contract(capability, metadata, task, api) if capability else {})
    if components and not contract and capability and capability["component_contract"]:
        recipe = next((r for r in capability["component_contract"] if not task or r["task"] == task),
                      capability["component_contract"][0])
        contract = {"components": components, "task": task or recipe["task"],
                    "required_roles": [s["role"] for s in recipe["components"]],
                    "evidence": {"native_recipe": recipe["evidence"],
                                 "source_metadata_sha256": canonical_hash(metadata.get("configs", {}))}}
    if quality_profile:
        contract = {**contract, "quality_profile": quality_profile}
    if conditioning:
        contract = {**contract, "conditioning": conditioning}
    members = components or contract.get("components", [])
    if not members:
        questions.append({"field": "components", "reason": "select exact files with --component role=path (or provide a source contract); ambiguous layouts are not guessed",
                          "available_weight_files": [name for name in metadata["files"] if name.endswith((".gguf", ".safetensors", ".ckpt", ".pth", ".sft"))][:200],
                          "native_recipes": capability["component_contract"] if capability else []})
        blockers.append("component mapping unresolved; no filename-only inference")
    chosen_task = task or contract.get("task") or metadata.get("pipeline_tag") or "text-to-image"
    if chosen_task not in MEDIA_TASKS:
        blockers.append(f"unsupported task: {chosen_task}")
    native = media_release.converter_types(build=False)
    selected = [next((q for q in native["types"] if q.casefold() == requested.casefold()), requested)
                for requested in (quants or native["types"])]
    if not selected:
        blockers.append("converter types unavailable; build/probe the pinned sd-cli first")
    resolved = []
    snapshots = {repo_id: metadata}
    for raw in members:
        item = dict(raw)
        role = item.get("role")
        if role not in ROLE_FLAGS:
            blockers.append(f"unknown component role: {role}")
            continue
        source_repo = item.get("source_repo", repo_id)
        if source_repo not in snapshots:
            snapshots[source_repo] = inspect_repository(source_repo, api)
        snapshot = snapshots[source_repo]
        name = item.get("source_path", "")
        try:
            relative_path(name)
        except MediaValidationError as error:
            blockers.append(str(error))
            continue
        if name not in snapshot["files"]:
            blockers.append(f"missing {role}: {source_repo}/{name}")
            continue
        if item.get("source_revision") and item["source_revision"] != snapshot["revision"]:
            blockers.append(f"source revision changed: {source_repo}")
        license_result = redistribution_result(snapshot.get("license"))
        if license_result["redistribution"] != "permitted":
            publication_blockers.append(f"redistribution review required for {role}: {license_result['license']}")
        if role in ("controlnet", "upscaler") and not item.get("explicitly_requested"):
            blockers.append(f"{role} requires an explicit user request")
        if item.get("quantizable") and role in ("vae", "audio_vae", "tokenizer", "config", "scheduler"):
            blockers.append(f"{role} must retain its documented source precision")
        source_format = Path(name).suffix.lower()
        if item.get("quantizable") and source_format not in (".safetensors", ".gguf", ".ckpt", ".pth", ".pt", ".sft"):
            blockers.append(f"no verified native conversion contract for {source_format}: {role}")
        assertions = capability["output_contract"].get("component_assertions", {}).get(role, {}) if capability else {}
        if any(c.get("role") == "llm_vision" for c in members) and capability:
            encoder_assertions = capability["output_contract"].get("component_assertions", {}).get("llm", {})
            if role == "llm":
                assertions = {**assertions, "required_patterns": []}
            elif role == "llm_vision" and encoder_assertions:
                assertions = {"required_patterns": encoder_assertions.get("required_patterns", [])}
        if assertions and not item.get("tensor_assertions"):
            item["tensor_assertions"] = assertions
        if item.get("tensor_assertions"):
            try:
                names = inspect_tensor_names(source_repo, snapshot["revision"], name)
                media_release.validate_tensor_names(names, item["tensor_assertions"])
                item["tensor_metadata_sha256"] = canonical_hash(names)
            except (OSError, ValueError, EOFError, struct.error) as error:
                blockers.append(f"component tensor contract not verified for {role}: {error}")
        if baseline_source := item.get("baseline_source"):
            baseline_repo = baseline_source.get("source_repo", source_repo)
            if baseline_repo not in snapshots:
                snapshots[baseline_repo] = inspect_repository(baseline_repo, api)
            baseline_snapshot = snapshots[baseline_repo]
            baseline_path = baseline_source.get("source_path")
            baseline_file = baseline_snapshot["files"].get(baseline_path)
            if not baseline_file:
                blockers.append(f"missing high-precision baseline for {role}")
            else:
                item["baseline_source"] = {"source_repo": baseline_repo, "source_path": baseline_path,
                    "source_revision": baseline_snapshot["revision"], "source_sha256": baseline_file.get("sha256") or "",
                    "source_bytes": baseline_file["bytes"]}
        output = item.get("output_path") or f"components/{role}{'.gguf' if item.get('quantizable') else Path(name).suffix}"
        if output in {"manifest.json", "quality.json", "review.json", "fixtures.json", "README.md", "FIXTURE_LICENSE.md"}:
            blockers.append(f"component output conflicts with release metadata: {output}")
        try:
            relative_path(output)
        except MediaValidationError as error:
            blockers.append(str(error))
            continue
        resolved.append({**item, "role": role, "source_repo": source_repo,
                         "source_revision": snapshot["revision"], "source_path": name,
                         "output_path": output, "source_bytes": snapshot["files"][name]["bytes"],
                         "format": item.get("format") or source_format.lstrip("."),
                         "precision": item.get("precision") or "source",
                         "source_sha256": snapshot["files"][name].get("sha256") or "",
                         "license_results": license_result})
    if len({c["output_path"] for c in resolved}) != len(resolved):
        blockers.append("duplicate component output paths")
    required = contract.get("required_roles", [])
    if not required:
        blockers.append("required component roles must be established by a source contract")
    missing = set(required) - {c["role"] for c in resolved}
    if missing:
        blockers.append("missing required roles: " + ", ".join(sorted(missing)))
    if not any(c["role"] in ("diffusion_model", "unet") for c in resolved):
        blockers.append("missing primary diffusion model")
    if "audio-video" in chosen_task and "audio_vae" not in {c["role"] for c in resolved}:
        blockers.append("audio-video recipe requires audio_vae")
    if capability:
        conditions = contract.get("conditioning", {})
        unsupported = set(conditions) - set(capability["conditioning_inputs"])
        if unsupported:
            blockers.append("conditioning lacks backend evidence: " + ", ".join(sorted(unsupported)))
        for pair in capability["output_contract"].get("conditioning_exclusions", []):
            if set(pair) <= set(conditions):
                blockers.append("incompatible conditioning inputs: " + ", ".join(pair))
    if not contract.get("evidence"):
        blockers.append("source contract needs architecture/component metadata evidence")
    if not contract.get("quality_profile"):
        publication_blockers.append("family/task calibrated quality profile and pinned evaluator required")
    elif (contract["quality_profile"].get("family") != (capability or {}).get("family")
          or contract["quality_profile"].get("task") != chosen_task):
        blockers.append("quality profile is not calibrated for this family and task")
    for quant in selected:
        if quant not in native["types"]:
            blockers.append(f"unsupported native quant type: {quant}")
    if (chosen_task.startswith(("image-to", "first-last", "reference-to")) or chosen_task == "image-editing") and not contract.get("conditioning"):
        blockers.append("selected task requires explicit conditioning fixtures")
    size = sum(c["source_bytes"] for c in resolved)
    size += sum(c.get("baseline_source", {}).get("source_bytes", 0) for c in resolved)
    largest = max((c["source_bytes"] for c in resolved), default=0)
    estimates = {"confidence": "conservative heuristic", "basis": "source file sizes; no measured local baseline",
                 "source_download_gb": size / 1024**3,
                 "peak_conversion_disk_gb": (size * (4 + 4 * len(selected)) + largest * (2 if sequential else 4)) / 1024**3,
                 "bundle_size_gb": {q: size * 4 / 1024**3 for q in selected},
                 "host_ram_gb": size * 6 / 1024**3 + 2, "peak_vram_gb": 0 if compute_backend == "cpu" else size * 4 / 1024**3,
                 "model_load_seconds": None, "first_image_seconds": None, "first_frame_seconds": None,
                 "seconds_per_image": None, "seconds_per_frame": None, "render_seconds": None,
                 "processing_fps": None, "offload": "--offload-to-cpu", "transfer_overhead_seconds": None,
                 "audio_decode_peak_gb": None}
    generation_settings = {"width": 256, "height": 256, "steps": 4, "seed": 42,
                           "sampler": "euler", "scheduler": "discrete", "cfg_scale": 7.0,
                           "frames": 5, "fps": 24, **contract.get("settings", {}), **(settings or {})}
    for key in ("width", "height", "steps", "frames", "fps"):
        if not isinstance(generation_settings[key], (int, float)) or generation_settings[key] <= 0:
            blockers.append(f"generation setting must be positive: {key}")
    if not isinstance(generation_settings["seed"], int) or generation_settings["seed"] < 0:
        blockers.append("a fixed nonnegative seed is required for comparable outputs")
    prebuilt = resolved and all(Path(c["source_path"]).suffix == ".gguf" or not c.get("quantizable") for c in resolved)
    state = "blocked" if blockers else "prebuilt" if prebuilt else "convertible"
    if not resolved and capability and capability["load"]:
        state = "runtime-only"
    plan = {"schema_version": 1, "repo_id": repo_id, "target_repo": target_repo or f"{config.namespace()}/{repo_id.split('/')[-1]}-GGUF",
            "engine": "stable-diffusion.cpp", "engine_revision": load_catalog()["revision"],
            "capability_manifest_sha256": file_hash(selected_catalog_path()),
            "fixture_sha256": fixture_hashes(),
            "family": capability["family"] if capability else None,
            "selection_evidence": capability["evidence"] if capability else [],
            "support": {"load": bool(capability and capability["load"]),
                        "convert": state == "convertible", "quantize": native["probed"]},
            "state": state, "blockers": blockers, "questions": questions,
            "publication_blockers": publication_blockers,
            "task": chosen_task, "components": resolved, "quants": selected,
            "converter_types": native, "settings": generation_settings,
            "conditioning": contract.get("conditioning", {}), "required_roles": required,
            "quality_profile": contract.get("quality_profile", {}),
            "contract_evidence": contract.get("evidence", {}), "compute_backend": compute_backend,
            "sequential": sequential, "estimates": estimates,
            "status": "experimental; native acceptance and human review required",
            "fork_leads": []}
    fixture_index = json.loads((FIXTURES / "fixtures.json").read_text(encoding="utf-8"))
    plan["validation"] = {"fixture_ids": [p["id"] for p in fixture_index["prompts"]],
                          "generations_per_policy": len(fixture_index["prompts"]) * 2,
                          "repeats_per_prompt": 2,
                          "baseline": "f16 from original floating-point weights when available",
                          "human_review_required_for_publication": True}
    plan["target_repositories"] = {q: f"{plan['target_repo']}-{q.upper()}" for q in selected}
    if blockers and hunt_forks:
        plan["fork_leads"] = media_release.find_forks(repo_id, metadata)
    from .runtime_control import frozen_runtime
    snapshot = frozen_runtime("stable-diffusion.cpp")
    if not snapshot and (installed := catalog_runtime()):
        snapshot = installed.snapshot()
    if snapshot:
        plan["runtime_snapshot"] = snapshot
    plan["plan_id"] = canonical_hash(plan)
    return plan


def validate_plan(plan, *, check_runtime=True):
    original = dict(plan)
    plan_id = original.pop("plan_id", None)
    if not plan_id or canonical_hash(original) != plan_id:
        raise MediaValidationError("approved media plan was modified; inspect and approve a fresh plan")
    if plan.get("fixture_sha256") != fixture_hashes():
        raise MediaValidationError("media fixtures changed since approval")
    if plan.get("blockers") or plan.get("state") not in ("convertible", "prebuilt"):
        raise MediaValidationError("media plan is blocked: " + "; ".join(plan.get("blockers", [])))
    if check_runtime and (plan.get("engine_revision") != load_catalog()["revision"]
                          or plan.get("capability_manifest_sha256") != file_hash(selected_catalog_path())):
        raise MediaValidationError("engine/catalog changed since approval")
    return plan
