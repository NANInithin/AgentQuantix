"""Offline contracts and orchestration checks for the retrieval release pipeline."""
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentquantix import actions, cli, config, retrieval, runs
from agentquantix.pipeline import retrieval_release as release


def _read_json(name):
    return {
        "modules.json": [{"type": "sentence_transformers.models.Pooling"},
                         {"type": "sentence_transformers.models.Normalize"}],
        "1_Pooling/config.json": {"pooling_mode_cls_token": True,
                                  "pooling_mode_mean_tokens": False},
        "config.json": {"architectures": ["BertForSequenceClassification"], "num_labels": 1},
        "preprocessor_config.json": {"image_mean": [0.5] * 3, "image_std": [0.5] * 3,
                                     "size": 224, "crop_size": 224},
    }[name]


def test_source_contract_extracts_pooling_reranker_and_clip_preprocessing():
    result = retrieval.source_contract(
        "org/model", ["modules.json", "1_Pooling/config.json", "config.json",
                      "preprocessor_config.json"], _read_json)
    assert result["contract"]["pooling"] == "cls"
    assert result["contract"]["normalize"] is True
    assert result["contract"]["reranker"] is True
    clip = retrieval.source_contract(
        "org/clip", ["config.json", "preprocessor_config.json"],
        lambda name: {"config.json": {"architectures": ["CLIPModel"]},
                      "preprocessor_config.json": _read_json("preprocessor_config.json")}[name])
    assert clip["contract"]["image_preprocessing"]["size"] == 224
    assert not clip["blockers"]


def test_missing_sentence_transformer_contract_has_exact_blocker():
    result = retrieval.source_contract("org/unknown", ["config.json"],
                                       lambda _: {"architectures": ["BertModel"]})
    assert result["blockers"] == [
        "no sentence-transformers pooling contract; the native pooling cannot be verified against a reference"]


def test_plan_includes_revision_size_license_contract_and_fixture_hashes(monkeypatch, tmp_path):
    import huggingface_hub

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"architectures": ["BertModel"]}), encoding="utf-8")
    modules_file = tmp_path / "modules.json"
    modules_file.write_text(json.dumps([{"type": "sentence_transformers.models.Pooling"}]),
                            encoding="utf-8")
    pooling_file = tmp_path / "pooling.json"
    pooling_file.write_text(json.dumps({"pooling_mode_mean_tokens": True}), encoding="utf-8")
    files = {"config.json": config_file, "modules.json": modules_file,
             "1_Pooling/config.json": pooling_file}
    siblings = [SimpleNamespace(rfilename=name, size=size)
                for name, size in (("config.json", 40), ("modules.json", 50),
                                   ("1_Pooling/config.json", 30))]
    info = SimpleNamespace(sha="a" * 40, siblings=siblings, gated=False,
                           card_data=SimpleNamespace(license="apache-2.0"))
    metadata_requests = []

    class Api:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id, *, files_metadata):
            metadata_requests.append(files_metadata)
            return info

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, name, **kwargs: str(files[name]))
    monkeypatch.setattr("agentquantix.archsupport.supported_quants",
                        lambda: ["F16", "Q8_0"])
    monkeypatch.setattr("agentquantix.runtimes.RuntimeManager.inspect",
                        lambda *args, **kwargs: None)
    plan = retrieval.plan_release("org/encoder", task="embeddings", engine="llama.cpp",
                                  quants=["F16"])
    assert metadata_requests == [True]
    assert plan["source_bytes"] == 120
    assert plan["source_revision"] == "a" * 40
    assert plan["license"] == "apache-2.0"
    assert plan["contract"]["pooling"] == "mean"
    assert plan["quants"] == ["F16"] and plan["available_quants"] == ["F16", "Q8_0"]
    assert plan["target_repo"] and plan["quality_profile"] and plan["fixtures"]
    assert plan["publication"] == (
        "one atomic bundle commit of passing quants after native inference parity and retrieval-quality gates")
    assert plan["review"] == "none"


@pytest.mark.parametrize("settings", [
    {"pooling_mode_max_tokens": True},
    {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": True},
])
def test_unsupported_pooling_is_blocked_without_guessing(settings):
    result = retrieval.source_contract(
        "org/model", ["modules.json", "1_Pooling/config.json"],
        lambda name: ([{"type": "sentence_transformers.models.Pooling"}]
                      if name == "modules.json" else settings))
    assert result["blockers"] == ["unsupported pooling"]


def test_metrics_match_hand_calculated_values():
    assert retrieval.cosine([1, 0], [1, 0]) == 1
    assert retrieval.recall_at_k(["a", "b"], {"b", "c"}, 1) == 0
    assert retrieval.mrr([(["x", "a"], {"a"})]) == 0.5
    expected = (1 + 3 / 1.584962500721156) / (3 + 1 / 1.584962500721156)
    assert retrieval.ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2) == pytest.approx(expected)
    assert retrieval.top1_agreement([["a"], ["b"]], [["a"], ["c"]]) == 0.5


def test_clip_text_normalization_and_rank_correlation():
    assert release._normalise_clip(" A RED circle, on white! ") == "a red circle on white"
    assert release._spearman([1, 2, 3], [1, 2, 3]) == 1
    assert release._spearman([1, 2, 3], [3, 2, 1]) == -1
    assert release._zsl_label("initializing model\nred circle = 0.2\nblue square = 0.8") == \
        "blue square"


def test_journal_skips_only_files_with_matching_hash(tmp_path):
    artifact = tmp_path / "artifact.gguf"
    artifact.write_bytes(b"f16 payload")
    record = {"conversion": {"path": str(artifact), "sha256": release._hash(artifact)}}
    assert release._valid_stage(record, "conversion")
    artifact.write_bytes(b"changed")
    assert not release._valid_stage(record, "conversion")


def test_clip_converter_output_uses_its_output_directory_name(tmp_path, monkeypatch):
    runtime = tmp_path / "clip.cpp"
    (runtime / "models").mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    root = tmp_path / "run"

    class Handle:
        source = runtime

    def convert(argv, **kwargs):
        output_dir = Path(argv[-1])
        (output_dir / f"{output_dir.name}_ggml-model-f16.gguf").write_bytes(b"f16")

    monkeypatch.setattr(release, "_command", convert)
    output = release._convert(
        {"engine": "clip.cpp"}, source, Handle(), root, {}, root / "journal.json")
    assert output.name == "converted_ggml-model-f16.gguf"
    assert output.read_bytes() == b"f16"


def test_authored_fixtures_are_cc0_and_stable_224px_pngs():
    fixture_root = Path(retrieval.__file__).resolve().parent / "fixtures" / "retrieval"
    corpus = json.loads((fixture_root / "corpus.json").read_text(encoding="utf-8"))
    zsl = json.loads((fixture_root / "zsl.json").read_text(encoding="utf-8"))
    assert len(corpus["documents"]) == 16 and len(corpus["queries"]) == 8
    assert all(query["relevance"] for query in corpus["queries"])
    assert 4 <= len(zsl["cases"]) <= 6
    assert "CC0 1.0 Universal" in (fixture_root / "LICENSE.md").read_text(encoding="utf-8")
    for case in zsl["cases"]:
        image = (fixture_root / case["image"]).read_bytes()
        assert image[:8] == b"\x89PNG\r\n\x1a\n"
        assert struct.unpack(">II", image[16:24]) == (224, 224)


def test_reference_missing_is_structural_only(monkeypatch, tmp_path):
    monkeypatch.setattr(release, "_references", lambda *args: None)
    result = release._evaluate({"engine": "llama.cpp", "task": "embeddings"},
                               tmp_path / "model.gguf", {}, tmp_path, tmp_path)
    assert result == {
        "structural_only": True,
        "blocker": "reference libraries not installed: pip install agentquantix[retrieval]"}


def test_references_return_structural_only_when_optional_import_is_missing(monkeypatch, tmp_path):
    import builtins

    original = builtins.__import__

    def missing_reference(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_reference)
    assert release._references({"engine": "llama.cpp", "task": "embeddings"},
                              tmp_path) is None


def test_stubbed_run_resumes_with_frozen_runtime_and_approved_plan(monkeypatch, tmp_path):
    from agentquantix import runtime_control

    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    frozen_inputs = []

    class Handle:
        revision = "a" * 40

        def snapshot(self):
            return {"revision": self.revision, "source": str(tmp_path / "runtime"),
                    "provenance": {"binary_sha256": {"llama-embedding": "b" * 64}}}

    handle = Handle()

    def ensure_runtime(engine, *, required_targets, frozen=None):
        frozen_inputs.append(frozen)
        assert engine == "llama.cpp"
        assert {"llama-embedding", "llama-server", "llama-quantize"} <= set(required_targets)
        return handle

    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(runtime_control, "ensure_runtime", ensure_runtime)
    monkeypatch.setattr(release, "require_binaries",
                        lambda runtime, *names: ["/native/" + name for name in names])
    monkeypatch.setattr(release, "snapshot_download",
                        lambda *args, **kwargs: str(source))

    def convert(plan, source, runtime, root, journal, journal_path):
        output = root / "model-f16.gguf"
        output.write_bytes(b"f16")
        return output

    monkeypatch.setattr(release, "_convert", convert)
    monkeypatch.setattr(release, "_evaluate", lambda *args: {
        "reference_cosine_min": 1.0, "dimension": 2, "reference_dimension": 2,
        "vectors_finite": True, "vectors_unit_norm": True, "mrr": 1.0,
        "ndcg_at_5": 1.0})
    plan = {"model": "org/embed", "task": "embeddings", "engine": "llama.cpp",
            "status": "ready", "blockers": [], "source_revision": "c" * 40,
            "quants": ["F16"], "quality_profile": {}}
    first = release.run_release(plan, user_approved=True)
    second = release.run_release(plan, user_approved=True)
    assert first["candidates"][0]["status"] == "passed"
    assert second["candidates"][0]["status"] == "passed"
    assert frozen_inputs[0] is None
    assert frozen_inputs[1]["revision"] == handle.revision


def test_rerank_server_is_always_terminated(monkeypatch, tmp_path):
    class Process:
        def __init__(self, *args, **kwargs):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            raise AssertionError("server should terminate gracefully")

    process = Process()
    monkeypatch.setattr(release.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(release, "_free_port", lambda: 12345)

    class Healthy:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    calls = 0

    def request(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Healthy()
        raise OSError("request failed")

    monkeypatch.setattr(release, "urlopen", request)
    with pytest.raises(OSError, match="request failed"):
        release._rerank("llama-server", "model.gguf", "query", ["doc"], tmp_path)
    assert process.terminated


def test_local_manifest_verifies_member_hash_and_size(tmp_path):
    member = tmp_path / "model.gguf"
    member.write_bytes(b"known model")
    manifest = {"members": [{"path": member.name, "bytes": member.stat().st_size,
                             "sha256": release._hash(member)}]}
    assert release.verify_manifest(manifest, tmp_path) == []
    member.write_bytes(b"changed")
    assert "member size mismatch: model.gguf" in release.verify_manifest(manifest, tmp_path)


def test_remote_publication_requires_exact_sha256_and_size():
    members = [{"path": "model.gguf", "bytes": 10, "sha256": "a" * 64}]
    assert release._remote_member_mismatches(
        members, {"model.gguf": {"bytes": 10, "sha256": "a" * 64}}) == []
    assert release._remote_member_mismatches(
        members, {"model.gguf": {"bytes": 10, "sha256": None}}) == [
            "remote verification mismatch: model.gguf"]


def test_remote_publication_downloads_git_files_without_lfs_hash(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("card\n", encoding="utf-8")
    members = [{"path": "README.md", "bytes": path.stat().st_size,
                "sha256": release._hash(path)}]
    assert release._remote_member_mismatches(
        members, {"README.md": {"bytes": path.stat().st_size, "sha256": None}},
        download=lambda _: path) == []


def test_card_contains_commands_metrics_and_license_distinction():
    card = retrieval.render_card(retrieval.RetrievalBundle(
        engine="clip.cpp", task="zero-shot-image-classification", family="clip",
        architecture="CLIPModel", source_repo="org/clip", source_revision="abc",
        engine_revision="def", members=[], contract={},
        quality_results={"f16": {"accuracy": 1.0,
                                 "tokenizer_parity_probe": {
                                     "normalized_cosine": 1.0, "raw_cosine": 0.9036}}}))
    assert "extract -m ./model.gguf" in card and "zsl -m ./model.gguf" in card
    assert "0.9036" in card and "runtime code license" in card


def test_cli_parser_accepts_retrieval_run_contract():
    args = cli.build_parser().parse_args([
        "run", "org/embed", "--task", "embeddings", "--target", "owner/embed",
        "--quant", "Q8_0", "--publish", "--yes"])
    assert args.task == "embeddings"
    assert args.target == "owner/embed" and args.quant == ["Q8_0"]
    assert args.publish is True


@pytest.mark.parametrize("reviewer", ["Reviewer", ""])
def test_review_rejects_retrieval_with_exact_automated_gate_message(
        tmp_path, monkeypatch, reviewer):
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    run_dir = runs.retrieval_root() / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "plan": {"model": "org/demo", "task": "embeddings", "engine": "llama.cpp"}}),
        encoding="utf-8")
    with pytest.raises(ValueError, match="^retrieval releases are gated by automated parity and retrieval metrics; there is no human review$"):
        actions.review("retrieval:demo", "Q8_0", accepted=True, reviewer=reviewer)
