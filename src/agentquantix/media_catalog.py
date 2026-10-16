"""Deterministic extraction of the native engine's offline capability catalog.

Extraction reports loader families, separate conversion evidence, and literal
documentation recipes. An example is never promoted to a validated source
contract: repository inspection and native inference remain required.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

ENGINE_REPO = "leejet/stable-diffusion.cpp"
ENGINE_REVISION = "8cd731e7b4667eb8f40a153315b9a7125d09a58d"
CATALOG_PATH = Path(__file__).parent / "fixtures/stable_diffusion_cpp_capabilities.json"
ROLE_FLAGS = {
    "diffusion_model": "--diffusion-model", "unet": "-m",
    "text_encoder": "--clip_l", "text_encoder_2": "--clip_g",
    "llm": "--llm", "llm_vision": "--llm_vision", "vae": "--vae",
    "audio_vae": "--audio-vae", "clip_l": "--clip_l", "clip_g": "--clip_g",
    "t5xxl": "--t5xxl", "controlnet": "--control-net", "upscaler": "--upscale-model",
    "tokenizer": None, "config": None, "scheduler": None,
}
CONDITION_FLAGS = {"init_image": "--init-img", "end_image": "--end-img",
                   "ref_image": "--ref-image", "ref_audio": "--ref-audio",
                   "ref_video": "--ref-video", "ref_video_audio": "--ref-video-audio"}


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _flat(value):
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def generate_catalog(checkout: Path, expected_revision=ENGINE_REVISION) -> dict:
    checkout = Path(checkout)
    revision = _git(checkout, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise ValueError(f"expected engine {expected_revision}, found {revision}")
    if _git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("capability extraction requires an unmodified engine checkout")
    paths = ["src/model.h", "src/model_loader.cpp", "src/stable-diffusion.cpp", "src/model/te/llm.hpp",
             "include/stable-diffusion.h", "examples/common/common.cpp",
             "examples/cli/main.cpp", "CMakeLists.txt", ".gitmodules"]
    paths += [p.relative_to(checkout).as_posix() for p in sorted((checkout / "docs").glob("*.md"))]
    texts = {p: (checkout / p).read_text(encoding="utf-8") for p in paths}
    evidence = {p: hashlib.sha256(value.encode()).hexdigest() for p, value in texts.items()}
    enum = texts["src/model.h"].split("enum SDVersion {", 1)[1].split("};", 1)[0]
    families = re.findall(r"\bVERSION_(\w+)\s*,", enum)
    header = re.sub(r"//[^\n]*", "", texts["include/stable-diffusion.h"])
    codes = {name.lower(): int(code) for name, code in re.findall(r"\bSD_TYPE_(\w+)\s*=\s*(\d+)", header) if name != "COUNT"}
    types = [(t.replace("_k", "_K") if re.fullmatch(r"q[2-8]_k", t) else t)
             for t in codes if not re.fullmatch(r"i\d+|f64", t)]
    documents = []
    for path in paths:
        if not path.startswith("docs/"):
            continue
        text = texts[path]
        recipes = []
        for match in re.finditer(r"(?:\./|\.\\)?(?:bin[/\\](?:Release[/\\])?)?sd-cli(?:\.exe)?\s+([^\n`]+)", text):
            command = match.group(1).strip()
            if re.search(r"-M\s+(?:convert|metadata)", command):
                continue
            components = {}
            for role, flag in ROLE_FLAGS.items():
                if flag and (item := re.search(r"(?:^|\s)" + re.escape(flag) + r"\s+(\S+)", command)):
                    components.setdefault(flag, {"role": role, "flag": flag,
                                                "example_path": item.group(1)})
            if not components:
                continue
            for component in components.values():
                example = Path(component["example_path"].replace("\\", "/")).name
                stem = re.sub(r"[-_.](?:q\d[^.]*|fp\d+|bf\d+|f\d+)$", "", Path(example).stem, flags=re.I).casefold()
                options = []
                for link in re.finditer(r"https://huggingface\.co/([\w.-]+/[\w.-]+)/(?:blob|resolve|tree)/[\w.-]+/([^\s`)>]+)", text):
                    source_path = link.group(2).rstrip(".,")
                    source_stem = re.sub(r"[-_.](?:q\d[^.]*|fp\d+|bf\d+|f\d+)$", "", Path(source_path).stem, flags=re.I).casefold()
                    if source_stem == stem:
                        options.append({"repo": link.group(1), "path": source_path})
                component["source_options"] = list({(o["repo"], o["path"]): o for o in options}.values())
            video = bool(re.search(r"-M\s+vid_gen|--video-frames", command))
            conditions = [key for key, flag in CONDITION_FLAGS.items() if flag in command]
            if re.search(r"(?:^|\s)-i\s", command):
                conditions.append("init_image")
            if re.search(r"(?:^|\s)-r\s", command):
                conditions.append("ref_image")
            task = "text-to-video" if video else "text-to-image"
            if "--audio-vae" in command:
                task = "text-to-audio-video"
            if "init_image" in conditions:
                task = task.replace("text-to", "image-to")
            elif "ref_image" in conditions and "audio-video" in task:
                task = "reference-to-audio-video"
            elif "ref_image" in conditions and not video:
                task = "image-editing"
            recipes.append({"task": task, "mode": "vid_gen" if video else "img_gen",
                            "components": list(components.values()), "conditioning": sorted(set(conditions)),
                            "evidence": {"path": path, "line": text[:match.start()].count("\n") + 1},
                            "status": "experimental"})
        variants = []
        for recipe in recipes:
            if "audio-video" in recipe["task"] and "Omitting `--audio-vae`" in text:
                variants.append({**recipe, "task": recipe["task"].replace("audio-video", "video"),
                                 "components": [c for c in recipe["components"] if c["role"] != "audio_vae"]})
            if recipe["task"] == "text-to-audio-video" and "FL2VA" in text and "--end-img" in text:
                variants.append({**recipe, "task": "first-last-to-audio-video", "conditioning": ["init_image", "end_image"]})
        recipes += variants
        documents.append({"path": path, "key": _flat(Path(path).stem), "recipes": recipes,
                          "conditioning": [k for k, f in CONDITION_FLAGS.items() if f in text]})
    capabilities = []
    for family in families:
        if family == "COUNT":
            continue
        key = _flat(family)
        matches = [d for d in documents if d["key"] == key or
                   (len(d["key"]) >= 4 and key.startswith(d["key"]))]
        if re.match(r"sd[12xl]", key):
            matches += [d for d in documents if d["key"] == "sd"]
        # Function groups in the native header provide shared family evidence.
        for group in re.finditer(r"static inline bool sd_version_is_(\w+)\([^)]*\)\s*\{(.*?)\n\}", texts["src/model.h"], re.S):
            if f"VERSION_{family}" in group.group(2):
                group_key = _flat(group.group(1))
                matches += [d for d in documents if d["key"] == group_key]
        matches = list({d["path"]: d for d in matches}.values())
        recipes = [r for d in matches for r in d["recipes"]]
        sources = sorted({repo.rstrip(".,") for d in matches if d["key"] == key for repo in re.findall(
            r"https://huggingface\.co/([\w.-]+/[\w.-]+)", texts[d["path"]])})
        assertions = {}
        exclusions = []
        for document in matches:
            text = texts[document["path"]]
            if "Ref2VA cannot be combined" in text:
                exclusions = [["ref_image", "init_image"], ["ref_image", "end_image"],
                              ["ref_audio", "init_image"], ["ref_audio", "end_image"],
                              ["ref_video", "init_image"], ["ref_video", "end_image"]]
            truncated = re.search(r"truncated to (\d+)\s+language layers", text)
            if truncated:
                assertions["llm"] = {"layer_pattern": r"(?:language_model\.)?(?:model\.)?layers\.(\d+)\.",
                                      "layer_count": int(truncated.group(1)),
                                      "forbidden_patterns": [r"^(?:(?:model|language_model)\.)*norm\.weight$"],
                                      "required_patterns": [r"deepstack_merger_list\.0\.", r"deepstack_merger_list\.1\.", r"deepstack_merger_list\.2\."],
                                      "evidence": document["path"]}
        capabilities.append({
            "id": f"stable-diffusion.cpp:{family.lower()}", "engine": "stable-diffusion.cpp",
            "family": family.lower(), "tasks": sorted({r["task"] for r in recipes}),
            "source_matchers": sources, "architecture_matchers": [family.lower().replace("_", "") + "pipeline"],
            "load_formats": ["gguf", "safetensors", "ckpt"],
            "load": f"VERSION_{family}" in texts["src/model_loader.cpp"],
            "convert": "experimental" if recipes else "unresolved", "quantize": "runtime-probe-required",
            "converter": "sd-cli -M convert", "quantizer": "sd-cli -M convert", "runtime": "sd-cli",
            "component_contract": recipes, "quant_policies": {"source_candidates": types},
            "conditioning_inputs": sorted({c for d in matches for c in d["conditioning"]} | {c for r in recipes for c in r["conditioning"]}),
            "output_contract": {"component_assertions": assertions, "conditioning_exclusions": exclusions}, "license_policy": "per-component-redistribution-required",
            "evidence": ["src/model.h", "src/model_loader.cpp", *[d["path"] for d in matches]],
            "backend_revision": revision, "status": "experimental" if recipes else "runtime-only",
        })
    return {"schema_version": 1, "engine": "stable-diffusion.cpp", "repository": ENGINE_REPO,
            "revision": revision, "extracted_at": _git(checkout, "show", "-s", "--format=%cI", "HEAD"),
            "timestamp_basis": "upstream commit time (deterministic regeneration)",
            "source_types": types, "type_codes": codes, "evidence_sha256": evidence, "capabilities": capabilities}


def load_catalog(path=CATALOG_PATH):
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    if catalog.get("revision") != ENGINE_REVISION or catalog.get("schema_version") != 1:
        raise ValueError("media capability catalog does not match the pinned engine")
    return catalog


def sync_catalog(checkout, output=CATALOG_PATH, check=False):
    generated = generate_catalog(Path(checkout))
    content = json.dumps(generated, indent=2, sort_keys=True) + "\n"
    output = Path(output)
    if check:
        if output.read_text(encoding="utf-8") != content:
            raise ValueError("media capability manifest drift; regenerate and review the engine contracts")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    return generated
