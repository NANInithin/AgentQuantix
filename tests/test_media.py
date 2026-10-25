import pytest

from agentquantix import config, media, media_quality
from agentquantix.media_catalog import ENGINE_REVISION, load_catalog
from agentquantix.pipeline import media_release


@pytest.fixture
def source():
    return {"repo_id": "org/Synthetic", "revision": "a" * 40,
            "pipeline_tag": "text-to-image", "library_name": "diffusers", "tags": [],
            "configs": {"model_index.json": {"_class_name": "FluxPipeline"}}, "license": "apache-2.0",
            "files": {"model.safetensors": {"bytes": 8192, "sha256": "b" * 64},
                      "vae.safetensors": {"bytes": 2048, "sha256": "c" * 64}}}


@pytest.fixture
def contract():
    return {"task": "text-to-image", "required_roles": ["diffusion_model", "vae"],
            "evidence": {"native_recipe": "docs/flux.md", "architecture": "FluxPipeline"},
            "components": [{"role": "diffusion_model", "source_path": "model.safetensors", "quantizable": True},
                           {"role": "vae", "source_path": "vae.safetensors", "precision": "f16", "quantizable": False}],
            "quality_profile": {"family": "flux", "task": "text-to-image", "calibration_evidence": "fixture corpus v1"}}


@pytest.fixture(autouse=True)
def isolate_media_state(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "work")


def plan(source, contract, **kwargs):
    return media.plan_release("org/Synthetic", family="flux", metadata=source, contract=contract,
                              quants=["q4_0"], hunt_forks=False, **kwargs)


def test_catalog_is_pinned_and_covers_all_extracted_loader_families():
    catalog = load_catalog()
    assert catalog["revision"] == ENGINE_REVISION
    assert len(catalog["capabilities"]) >= 56
    assert len({c["id"] for c in catalog["capabilities"]}) == len(catalog["capabilities"])
    assert catalog["evidence_sha256"]["src/model_loader.cpp"]
    for capability in media.capabilities():
        assert capability.backend_revision == ENGINE_REVISION
        assert capability.status in ("experimental", "runtime-only")
        assert capability.load != capability.convert


def test_sdxl_standard_repository_plans_without_manual_overrides(source):
    source["repo_id"] = "stabilityai/stable-diffusion-xl-base-1.0"
    source["configs"] = {
        "model_index.json": {"_class_name": "StableDiffusionXLPipeline",
                             "vae": ["diffusers", "AutoencoderKL"]},
        "vae/config.json": {"_class_name": "AutoencoderKL"},
    }
    names = ["sd_xl_base_1.0.safetensors", "sd_xl_base_1.0_0.9vae.safetensors",
             "sd_xl_offset_example-lora_1.0.safetensors",
             "vae/diffusion_pytorch_model.safetensors",
             "vae/diffusion_pytorch_model.fp16.safetensors",
             "vae_1_0/diffusion_pytorch_model.safetensors",
             "unet/diffusion_pytorch_model.safetensors"]
    source["files"] = dict.fromkeys(names, {"bytes": 8192, "sha256": "b" * 64})
    result = media.plan_release(source["repo_id"], metadata=source, hunt_forks=False,
                                target_repo="user/sdxl-GGUF")
    assert not result["blockers"]
    assert result["family"] == "sdxl"
    assert {c["role"]: c["source_path"] for c in result["components"]} == {
        "unet": "sd_xl_base_1.0.safetensors",
        "vae": "vae/diffusion_pytorch_model.safetensors"}
    assert not result["components"][1]["quantizable"]
    assert result["publication_blockers"]  # Planning does not waive quality gates.


@pytest.mark.parametrize("declaration,vae_config", [
    (["diffusers", "OtherVAE"], {"_class_name": "AutoencoderKL"}),
    (["diffusers", "AutoencoderKL"], {}),
])
def test_vae_resolution_requires_component_metadata(source, declaration, vae_config):
    source["configs"] = {"model_index.json": {"vae": declaration},
                         "vae/config.json": vae_config}
    source["files"]["vae/diffusion_pytorch_model.safetensors"] = {"bytes": 8192}
    assert media._configured_vae(source) is None


def test_blocked_media_run_does_not_prompt_or_build(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agentquantix import media_cli
    plan_path = tmp_path / "plan.json"
    media.atomic_json(plan_path, {"blockers": ["family unresolved"]})
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("blocked plan prompted"))
    monkeypatch.setattr(media_release, "run_release", lambda *args, **kwargs: pytest.fail("blocked plan ran"))
    assert media_cli.command(SimpleNamespace(media_command="run", plan=str(plan_path),
                                             output=None, dry_run=False)) == 1


def test_plan_freezes_sources_components_licenses_and_targets(source, contract):
    result = plan(source, contract)
    assert not result["blockers"]
    assert result["state"] == "convertible"
    assert all(c["source_revision"] == source["revision"] for c in result["components"])
    assert result["target_repositories"]["q4_0"].endswith("-Q4_0")
    assert result["support"]["quantize"] is False
    assert media.validate_plan(result) is result
    result["quants"].append("q8_0")
    with pytest.raises(media.MediaValidationError, match="modified"):
        media.validate_plan(result)


@pytest.mark.parametrize("mutation,reason", [
    (lambda s, c: c["components"][1].update(source_path="missing"), "missing vae"),
    (lambda s, c: c["components"][1].update(quantizable=True), "must retain"),
    (lambda s, c: c["components"][1].update(output_path="../escape"), "unsafe bundle"),
    (lambda s, c: c["components"][1].update(role="mystery"), "unknown component"),
    (lambda s, c: c["quality_profile"].update(task="text-to-video"), "not calibrated"),
])
def test_incomplete_plans_fail_closed(source, contract, mutation, reason):
    mutation(source, contract)
    result = plan(source, contract)
    assert result["blockers"]
    assert reason in "; ".join(result["blockers"])
    with pytest.raises(media.MediaValidationError):
        media.validate_plan(result)


def test_matching_names_do_not_establish_media_support(source, contract):
    source.update(pipeline_tag="text-generation", library_name="transformers")
    source["configs"] = {"config.json": {"architectures": ["LlamaForCausalLM"]}}
    result = media.plan_release("someorg/FluxPipeline", metadata=source, contract=contract, hunt_forks=False)
    assert result["blockers"]


def test_local_manual_testing_does_not_require_publication_clearance(source, contract):
    source["license"] = "unknown"
    contract["quality_profile"] = {}
    result = plan(source, contract)
    assert not result["blockers"]
    assert any("redistribution" in p for p in result["publication_blockers"])
    assert any("calibrated" in p for p in result["publication_blockers"])
    media.validate_plan(result)


def test_policy_preserves_companions_and_reports_actual_types():
    components = [{"role": "diffusion_model", "quantizable": True, "tensor_rules": "norm=f32"},
                  {"role": "audio_vae", "quantizable": False, "precision": "f32", "format": "safetensors"}]
    result = media.expand_policy(components, "q4_k", ["q4_k"])
    assert result[0]["tensor_rules"] == "norm=f32"
    assert result[1]["precision"] == "f32" and result[1]["format"] == "safetensors"
    with pytest.raises(media.MediaValidationError):
        media.expand_policy(components, "Q4_K_M", ["q4_k"])
    assert media.expand_policy(components, "Q4_K", ["q4_K"])[0]["precision"] == "q4_K"


def test_native_tensor_headers_do_not_depend_on_llama_gguf_type_enums(tmp_path):
    import struct
    name = b"model.weight"
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    data += struct.pack("<Q", len(name)) + name + struct.pack("<IQQIQ", 2, 256, 256, 42, 0)
    data += b"\0" * (-len(data) % 32) + b"x" * 128
    path = tmp_path / "native-q2_0.gguf"
    path.write_bytes(data)
    media_release.validate_gguf(path)
    assert media.tensor_names_from_prefix(data, ".gguf") == ["model.weight"]
    assert media_release.tensor_precisions(path) == {"q2_0": 1}


def test_native_text_encoder_contract_rejects_wrong_layers_and_final_norm():
    assertions = {"layer_pattern": r"layers\.(\d+)\.", "layer_count": 2,
                  "required_patterns": [r"deepstack_merger_list\.0\."],
                  "forbidden_patterns": [r"^model\.norm\.weight$"]}
    correct = ["layers.0.weight", "layers.1.weight", "visual.deepstack_merger_list.0.weight"]
    media_release.validate_tensor_names(correct, assertions)
    for bad in [correct + ["layers.2.weight"], correct + ["model.norm.weight"], correct[:2]]:
        with pytest.raises(media.MediaValidationError):
            media_release.validate_tensor_names(bad, assertions)


@pytest.mark.parametrize("value", ["../x", "/absolute", "C:/x", "a\\b", "."])
def test_bundle_paths_cannot_escape(value):
    with pytest.raises(media.MediaValidationError):
        media.relative_path(value)


def test_approval_precedes_build_or_download(monkeypatch):
    monkeypatch.setattr(media_release, "ensure_backend", lambda *args: pytest.fail("unapproved build"))
    with pytest.raises(media.MediaValidationError, match="approval"):
        media_release.run_release({}, user_approved=False)


def test_graph_checks_hashes_dependencies_and_external_recipe_files(tmp_path):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"model")
    component = media.MediaComponent("diffusion_model", "org/model", "a" * 40, "source", "model.gguf", "gguf", "f16",
                                     sha256=media.file_hash(artifact), bytes=5)
    bundle = media.MediaBundle("flux", "text-to-image", "f16", [component], {"components": {"diffusion_model": "outside.gguf"}})
    assert any("outside" in p for p in bundle.problems(tmp_path))
    bundle.runtime_recipe["components"]["diffusion_model"] = "model.gguf"
    assert not bundle.problems(tmp_path)
    artifact.write_bytes(b"other")
    assert any("checksum" in p for p in bundle.problems(tmp_path))
    component.dependencies = ["diffusion_model"]
    assert any("cycle" in p for p in bundle.problems(tmp_path))


def test_image_validation_rejects_blank_wrong_size_and_transparency(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "out.png"
    Image.new("RGB", (64, 64), "black").save(path)
    with pytest.raises(media.MediaValidationError, match="blank"):
        media_quality.inspect_image(path, {"width": 64, "height": 64})
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(path)
    with pytest.raises(media.MediaValidationError, match="transparent"):
        media_quality.inspect_image(path, {"width": 64, "height": 64})
    with pytest.raises(media.MediaValidationError, match="dimensions"):
        media_quality.inspect_image(media.FIXTURES / "reference.png", {"width": 32, "height": 64})
    assert media_quality.inspect_image(media.FIXTURES / "reference.png", {"width": 64, "height": 64})["channels"] == 3


def test_missing_heavy_evaluator_is_disclosed_and_blocks_quality(tmp_path):
    pytest.importorskip("PIL")
    recipe = {"mode": "img_gen", "settings": {"width": 64, "height": 64}, "components": {}, "prompt": "shapes"}
    result = media_quality.assess(media.FIXTURES / "reference.png", recipe, None, {}, tmp_path)
    assert result["structural_passed"]
    assert not result["passed"] and "prompt_alignment" in result["unavailable_metrics"]


def test_readonly_plan_never_builds_or_downloads_weights(source, contract, monkeypatch):
    monkeypatch.setattr(media_release, "ensure_backend", lambda *args: pytest.fail("read-only plan built backend"))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *args, **kwargs: pytest.fail("downloaded weights"))
    assert not plan(source, contract)["blockers"]


def test_media_cli_and_shared_tools_are_exposed():
    from agentquantix import cli
    from agentquantix.agent import tools
    assert cli.build_parser().parse_args(["media", "run", "org/model", "--sequential"]).sequential
    names = {t["name"] for t in tools.TOOLS}
    assert {"plan_media_release", "start_media_release", "record_media_review", "publish_media_release"} <= names
    assert "media" in tools.call_json("plan_quantization", {"models": ["org/minimax-h3"]})


def test_renamed_diffusers_models_cannot_enter_text_assessment(monkeypatch):
    from agentquantix import hub, research
    candidate = hub.Candidate("org/innocent-name", 0, pipeline_tag="text-to-image")
    monkeypatch.setattr(hub, "one", lambda name: candidate)
    monkeypatch.setattr(hub, "enrich", lambda *args, **kwargs: pytest.fail("entered text converter planning"))
    with pytest.raises(ValueError, match="media plan"):
        research.assess_one(candidate.repo_id)
