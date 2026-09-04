"""The candidate filter, and the quant-name parser the verifier depends on.

The FP8 case is the one that matters most. Large MoEs are now routinely
TRAINED and released in FP8 by their authors, so treating the tag as a
quantization marker threw away GLM-5.3, DeepSeek-V4 and friends — precisely
the models most worth being first to publish.
"""

from agentquantix import hub


def _model(repo_id, tags=(), pipeline_tag="text-generation", **kwargs):
    return hub.Candidate(repo_id=repo_id, rank=1, tags=list(tags),
                         pipeline_tag=pipeline_tag, **kwargs)


# =====================================================
# WHAT SURVIVES
# =====================================================
def test_an_original_text_model_is_kept():
    kept = hub.filter_candidates([_model("org/Model-7B", ["transformers"])])
    assert len(kept) == 1


def test_fp8_native_releases_are_kept():
    """REGRESSION. FP8 is how large MoEs now ship from their authors."""
    models = [_model("zai-org/GLM-5.3", ["transformers", "fp8"]),
              _model("deepseek-ai/DeepSeek-V4", ["transformers",
                                                 "compressed-tensors"])]
    assert len(hub.filter_candidates(models)) == 2


def test_image_text_models_are_kept():
    kept = hub.filter_candidates(
        [_model("org/VLM-8B", ["transformers"], "image-text-to-text")])
    assert len(kept) == 1


# =====================================================
# WHAT IS DROPPED
# =====================================================
def test_non_text_pipelines_are_dropped():
    for tag in ("text-to-speech", "image-to-video", "time-series-forecasting",
                "automatic-speech-recognition"):
        model = _model("org/Thing", ["transformers"], tag)
        hub.filter_candidates([model])
        assert model.dropped, f"{tag} should have been dropped"


def test_quantized_reuploads_are_dropped():
    for tags in (["gguf"], ["awq"], ["gptq"], ["exl2"], ["mlx"],
                 ["bitsandbytes"], ["quantized"]):
        model = _model("someone/Model-7B", ["transformers"] + tags)
        hub.filter_candidates([model])
        assert model.dropped, f"{tags} should have been dropped"


def test_typed_base_model_relations_are_dropped():
    for relation in ("finetune", "merge", "adapter", "quantized"):
        model = _model("someone/Model",
                       ["transformers", f"base_model:{relation}:org/Original"])
        hub.filter_candidates([model])
        assert model.dropped, f"base_model:{relation}: should be dropped"


def test_a_plain_base_model_citation_is_not_a_disqualifier():
    """Plenty of original releases cite an ancestor without being derivatives."""
    model = _model("org/Model", ["transformers", "base_model:org/Ancestor"])
    assert len(hub.filter_candidates([model])) == 1


def test_giveaway_names_are_dropped():
    for name in ("Model-7B-GGUF", "Model-7B-4bit", "Model-abliterated",
                 "Model-AWQ"):
        model = _model(f"someone/{name}", ["transformers"])
        hub.filter_candidates([model])
        assert model.dropped, f"{name} should have been dropped"


def test_the_org_name_is_not_matched_against_name_hints():
    """An org called 'gguf-org' should not disqualify its original releases."""
    model = _model("gguf-org/Original-Model", ["transformers"])
    assert len(hub.filter_candidates([model])) == 1


def test_private_repos_are_dropped():
    model = _model("org/Secret", ["transformers"])
    model.private = True
    hub.filter_candidates([model])
    assert model.dropped == "private"


def test_dropped_models_are_reported_not_hidden():
    """A model dropped by a rule the user disagrees with must stay visible."""
    models = [_model("org/Keep", ["transformers"]),
              _model("org/TTS", ["transformers"], "text-to-speech")]
    kept = hub.filter_candidates(models)
    assert len(kept) == 1
    assert models[1].dropped and models[1] not in kept


# =====================================================
# QUANT NAMES
# =====================================================
def test_quant_names_are_parsed_from_filenames():
    cases = {
        "Model-Q4_K_M.gguf": "Q4_K_M",
        "Model-IQ3_XXS.gguf": "IQ3_XXS",
        "Model-BF16.gguf": "BF16",
        "Model-MXFP4_MOE.gguf": "MXFP4_MOE",
        "Model-TQ1_0.gguf": "TQ1_0",
        "Model-7B-Q8_0.gguf": "Q8_0",
    }
    for filename, expected in cases.items():
        assert hub.quant_of(filename) == expected, filename


def test_non_quant_files_parse_to_nothing():
    for filename in ("README.md", ".gitattributes", "mmproj-Model-F16.gguf"):
        assert hub.quant_of(filename) in (None, "F16")


def test_target_repo_follows_the_namespace(monkeypatch):
    monkeypatch.setenv("AQX_NAMESPACE", "someone")
    assert _model("org/Model-7B").target_repo == "someone/Model-7B-GGUF"
