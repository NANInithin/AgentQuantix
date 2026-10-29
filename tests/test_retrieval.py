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


def _plan_for_reranker(monkeypatch, tmp_path, model_config):
    import huggingface_hub

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(model_config), encoding="utf-8")
    info = SimpleNamespace(
        sha="a" * 40,
        siblings=[SimpleNamespace(rfilename="config.json", size=config_file.stat().st_size)],
        gated=False,
        card_data=SimpleNamespace(license="mit"))

    class Api:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id, *, files_metadata):
            return info

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, name, **kwargs: str(config_file))
    monkeypatch.setattr("agentquantix.archsupport.supported_quants", lambda: ["F16"])
    monkeypatch.setattr("agentquantix.runtimes.RuntimeManager.inspect",
                        lambda *args, **kwargs: None)
    return retrieval.plan_release("BAAI/bge-reranker-base", task="reranking",
                                  engine="llama.cpp", quants=["F16"])


def test_id2label_only_config_produces_ready_reranker_plan(monkeypatch, tmp_path):
    plan = _plan_for_reranker(monkeypatch, tmp_path, {
        "architectures": ["XLMRobertaForSequenceClassification"],
        "id2label": {"0": "LABEL_0"}})

    assert plan["status"] == "ready"
    assert plan["contract"]["num_labels"] == 1
    assert plan["contract"]["reranker"] is True
    assert not plan["blockers"]


def test_two_label_id2label_remains_blocked_for_reranking(monkeypatch, tmp_path):
    plan = _plan_for_reranker(monkeypatch, tmp_path, {
        "architectures": ["XLMRobertaForSequenceClassification"],
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"}})

    assert plan["status"] == "blocked"
    assert plan["contract"]["num_labels"] == 2
    assert plan["contract"]["reranker"] is False
    assert "reranker must have one classification label" in plan["blockers"]


def test_plan_includes_revision_size_license_contract_and_fixture_hashes(monkeypatch, tmp_path):
    import huggingface_hub

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"architectures": ["BertModel"]}), encoding="utf-8")
    modules_file = tmp_path / "modules.json"
    modules_file.write_text(json.dumps([
        {"type": "sentence_transformers.models.Transformer", "path": "0_Transformer"},
        {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}]),
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
    assert plan["files"] == ["config.json", "modules.json", "1_Pooling/config.json"]
    assert plan["source_bytes"] == 120
    assert plan["source_revision"] == "a" * 40
    assert plan["license"] == "apache-2.0"
    assert plan["contract"]["pooling"] == "mean"
    assert plan["quants"] == ["F16"] and plan["available_quants"] == ["F16", "Q8_0"]
    assert plan["target_repo"] and plan["quality_profile"] and plan["fixtures"]
    assert plan["publication"] == (
        "one atomic bundle commit of passing quants after native inference parity and retrieval-quality gates")
    assert plan["review"] == "none"
    assert plan["contract"]["module_directories"] == ["0_Transformer", "1_Pooling"]


def test_clip_plan_records_installed_runtime_revision(monkeypatch):
    import huggingface_hub

    info = SimpleNamespace(
        sha="a" * 40, siblings=[SimpleNamespace(rfilename="config.json", size=10)],
        gated=False, card_data=SimpleNamespace(license="apache-2.0"))

    class Api:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id, *, files_metadata):
            return info

    inspect_calls = []

    def inspect(self, engine, verify=False):
        inspect_calls.append((engine, verify))
        return SimpleNamespace(revision="d" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(retrieval, "source_contract", lambda *args: {
        "contract": {"architectures": ["CLIPModel"]}, "blockers": []})
    monkeypatch.setattr("agentquantix.runtimes.RuntimeManager.inspect", inspect)
    plan = retrieval.plan_release("org/clip", task="zero-shot-image-classification",
                                  engine="clip.cpp", quants=["F16"])
    assert plan["engine_revision"] == "d" * 40
    assert inspect_calls == [("clip.cpp", False)]


def test_snapshot_patterns_use_safetensors_when_available_and_exclude_optional_formats():
    plan = {"files": ["model.safetensors", "pytorch_model.bin"],
            "contract": {"module_directories": ["0_Transformer", "1_Pooling"]}}
    allow, ignore = release._snapshot_patterns(plan)
    assert "*.safetensors" in allow and "*.bin" not in allow
    assert "1_Pooling/*" in allow and "0_Transformer/*" in allow
    assert {"onnx/**", "openvino/**", "*.h5", "**/*.h5", "*.msgpack",
            "**/*.msgpack", "*.ot", "**/*.ot"} <= set(ignore)
    fallback, _ = release._snapshot_patterns(
        {"files": ["pytorch_model.bin"], "contract": {"module_directories": []}})
    assert "*.bin" in fallback


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
    import numpy as np

    assert retrieval.cosine([1, 0], [1, 0]) == 1
    assert retrieval.cosine(np.array([1, 0]), np.array([1, 0])) == 1
    assert retrieval.recall_at_k(["a", "b"], {"b", "c"}, 1) == 0
    assert retrieval.recall_at_k(np.array(["a", "b"]), np.array(["b", "c"]), 1) == 0
    assert retrieval.mrr([(["x", "a"], {"a"})]) == 0.5
    assert retrieval.mrr([(np.array(["x", "a"]), np.array(["a"]))]) == 0.5
    expected = (1 + 3 / 1.584962500721156) / (3 + 1 / 1.584962500721156)
    assert retrieval.ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2) == pytest.approx(expected)
    assert retrieval.ndcg_at_k(np.array(["b", "a"]), {"a": 2, "b": 1}, 2) == \
        pytest.approx(expected)
    assert retrieval.top1_agreement([["a"], ["b"]], [["a"], ["c"]]) == 0.5
    assert retrieval.top1_agreement(np.array([["a"], ["b"]]),
                                    np.array([["a"], ["c"]])) == 0.5


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


def test_reference_missing_is_structural_only(tmp_path):
    result = release._evaluate({"engine": "llama.cpp", "task": "embeddings"},
                               tmp_path / "model.gguf", {}, tmp_path, None)
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
                    "provenance": {"binary_sha256": {"llama-server": "b" * 64}}}

    handle = Handle()

    def ensure_runtime(engine, *, required_targets, frozen=None):
        frozen_inputs.append(frozen)
        assert engine == "llama.cpp"
        assert set(required_targets) == {"llama-server", "llama-quantize"}
        return handle

    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(runtime_control, "ensure_runtime", ensure_runtime)
    monkeypatch.setattr(release, "require_binaries",
                        lambda runtime, *names: ["/native/" + name for name in names])
    download_requests = []

    def snapshot(*args, **kwargs):
        download_requests.append(kwargs)
        return str(source)

    monkeypatch.setattr(release, "snapshot_download", snapshot)

    def convert(plan, source, runtime, root, journal, journal_path):
        output = root / "model-f16.gguf"
        output.write_bytes(b"f16")
        return output

    monkeypatch.setattr(release, "_convert", convert)

    def quantize(plan, f16, binaries, root, journal, journal_path, quant):
        output = root / f"model-{quant}.gguf"
        output.write_bytes(quant.encode())
        return output

    monkeypatch.setattr(release, "_quantize", quantize)
    reference_calls, prepared_references, evaluation_references = [], [], []

    def reference_factory(plan, source):
        reference = object()
        reference_calls.append(reference)
        return reference

    def prepare_reference(plan, reference):
        prepared = {"source_reference": reference}
        prepared_references.append(prepared)
        return prepared

    def evaluate(plan, model, binaries, root, references):
        evaluation_references.append(references)
        return {"reference_cosine_min": 1.0, "dimension": 2, "reference_dimension": 2,
                "vectors_finite": True, "vectors_unit_norm": True, "mrr": 1.0,
                "ndcg_at_5": 1.0, "document_vectors": [[1.0, 0.0]],
                "query_vectors": [[1.0, 0.0]]}

    monkeypatch.setattr(release, "_references", reference_factory)
    monkeypatch.setattr(release, "_reference_outputs", prepare_reference)
    monkeypatch.setattr(release, "_evaluate", evaluate)
    plan = {"model": "org/embed", "task": "embeddings", "engine": "llama.cpp",
            "status": "ready", "blockers": [], "source_revision": "c" * 40,
            "quants": ["F16", "Q8_0"], "quality_profile": {},
            "files": ["config.json", "model.safetensors", "pytorch_model.bin",
                      "onnx/model.onnx", "openvino/config.json", "model.h5",
                      "model.msgpack", "model.ot"],
            "contract": {"module_directories": ["0_Transformer"]}}
    first = release.run_release(plan, user_approved=True)
    second = release.run_release(plan, user_approved=True)
    assert all(item["status"] == "passed" for item in first["candidates"])
    assert all(item["status"] == "passed" for item in second["candidates"])
    assert frozen_inputs[0] is None
    assert frozen_inputs[1]["revision"] == handle.revision
    assert len(reference_calls) == len(prepared_references) == 2
    assert evaluation_references[0] is evaluation_references[1]
    assert evaluation_references[2] is evaluation_references[3]
    assert evaluation_references[0] is not evaluation_references[2]
    assert len(download_requests) == 2
    for request in download_requests:
        assert "*.safetensors" in request["allow_patterns"]
        assert "*.bin" not in request["allow_patterns"]
        assert "0_Transformer/*" in request["allow_patterns"]
        assert "onnx/**" in request["ignore_patterns"]


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
        with release._rerank("llama-server", "model.gguf", tmp_path) as score:
            score("query", ["doc"])
    assert process.terminated


def test_rerank_quality_uses_one_server_for_all_fixture_queries(monkeypatch, tmp_path):
    from contextlib import contextmanager

    starts, requests = [], []

    @contextmanager
    def server(binary, model, root):
        starts.append((binary, model, root))

        def score(query, documents):
            requests.append((query, len(documents)))
            return [{"index": index, "relevance_score": float(len(documents) - index)}
                    for index in range(len(documents))]

        yield score

    monkeypatch.setattr(release, "_rerank", server)
    corpus = json.loads((Path(retrieval.__file__).resolve().parent /
                         "fixtures/retrieval/corpus.json").read_text(encoding="utf-8"))
    references = {"reference_scores": [
        [float(len(corpus["documents"]) - index) for index in range(len(corpus["documents"]))]
        for _ in corpus["queries"]]}
    quality = release._reranking_quality(
        Path("model.gguf"), Path("llama-server"), references)
    assert len(starts) == 1
    assert len(requests) == len(corpus["queries"])
    assert quality["scores"]
    assert "full_ranking_identical" in quality
    assert "reference_relevant_order_agreement" in quality
    assert "reference_ndcg_at_5" in quality


def test_embedder_uses_embeddings_endpoint_and_configured_server_options(monkeypatch, tmp_path):
    class Process:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            raise AssertionError("server should terminate gracefully")

    class Response:
        status = 200

        def __init__(self, body=b'{"data":[{"embedding":[0.6,0.8]}]}'):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.body

    process = Process()
    started, requests = [], []

    def start(argv, **kwargs):
        started.append((argv, kwargs))
        return process

    def open_request(url, **kwargs):
        if isinstance(url, str):
            assert url.endswith("/health")
            return Response()
        requests.append(url)
        return Response()

    monkeypatch.setattr(release.subprocess, "Popen", start)
    monkeypatch.setattr(release, "_free_port", lambda: 12345)
    monkeypatch.setattr(release, "urlopen", open_request)

    with release._embedder("/native/llama-server", "model.gguf", tmp_path, "cls", 64) as embed:
        assert embed("query text") == [0.6, 0.8]

    assert started[0][0] == [
        "/native/llama-server", "-m", "model.gguf", "--embedding", "--pooling", "cls",
        "-c", "64", "-b", "64", "-ub", "64", "--host", "127.0.0.1", "--port", "12345"]
    assert started[0][1]["cwd"] == tmp_path
    assert requests[0].full_url == "http://127.0.0.1:12345/v1/embeddings"
    assert json.loads(requests[0].data) == {
        "input": ["query text"], "embd_normalize": 2}
    assert process.terminated


def test_embedding_quality_uses_one_server_for_all_fixture_texts(monkeypatch, tmp_path):
    from contextlib import contextmanager

    started, texts = [], []

    @contextmanager
    def embedder(binary, model, root, pooling, ctx):
        started.append((binary, model, root, pooling, ctx))
        yield lambda text: texts.append(text) or [1.0, 0.0]

    class Matrix(list):
        @property
        def shape(self):
            return len(self), len(self[0])

        def tolist(self):
            return [list(row) for row in self]

    monkeypatch.setattr(release, "_embedder", embedder)
    corpus = json.loads((release.FIXTURES / "corpus.json").read_text(encoding="utf-8"))
    documents = Matrix([[1.0, 0.0] for _ in corpus["documents"]])
    queries = Matrix([[1.0, 0.0] for _ in corpus["queries"]])
    quality = release._embedding_quality(
        {"contract": {"pooling": "cls", "max_seq_length": 128,
                      "prompts": {"document": "D: ", "query": "Q: "}}},
        Path("model.gguf"),
        {"reference_document_vectors": documents, "reference_query_vectors": queries},
        Path("/native/llama-server"), tmp_path)

    assert started == [
        (Path("/native/llama-server"), Path("model.gguf"), tmp_path, "cls", 128)]
    assert len(texts) == len(corpus["documents"]) + len(corpus["queries"])
    assert texts[0].startswith("D: ")
    assert texts[-1].startswith("Q: ")
    assert quality["dimension"] == 2


@pytest.mark.parametrize("body", [
    b"not json",
    b"{}",
    b'{"data":[]}',
    b'{"data":[{}]}',
    b'{"data":[{"embedding":[1,"nan"]}]}',
    json.dumps({"data": [{"embedding": [10 ** 400]}]}).encode(),
])
def test_embedder_rejects_malformed_response_and_terminates_server(monkeypatch, tmp_path, body):
    class Process:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            raise AssertionError("server should terminate gracefully")

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return body

    process = Process()
    monkeypatch.setattr(release.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(release, "_free_port", lambda: 12345)
    monkeypatch.setattr(release, "urlopen", lambda *args, **kwargs: Response())

    with pytest.raises(RuntimeError, match="llama-server embeddings response"):
        with release._embedder("llama-server", "model.gguf", tmp_path, "mean", 512) as embed:
            embed("text")
    assert process.terminated


def _reranking_quality_for_scores(monkeypatch, tmp_path, relevance, reference_scores, scores):
    from contextlib import contextmanager

    documents = [{"id": f"doc-{index}", "text": f"document {index}"} for index in range(len(scores))]
    corpus = {"documents": documents,
              "queries": [{"query": "query", "relevance": relevance}]}
    (tmp_path / "corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    monkeypatch.setattr(release, "FIXTURES", tmp_path)

    @contextmanager
    def rerank(binary, model, root):
        yield lambda query, texts: [
            {"index": index, "relevance_score": value} for index, value in enumerate(scores)]

    monkeypatch.setattr(release, "_rerank", rerank)
    return release._reranking_quality(
        Path("model.gguf"), Path("/native/llama-server"),
        {"reference_scores": [reference_scores]})


def test_reranker_f16_gate_ignores_irrelevant_tail_reordering(monkeypatch, tmp_path):
    quality = _reranking_quality_for_scores(
        monkeypatch, tmp_path, {"doc-0": 2, "doc-1": 1},
        [10.0, 9.0, -10.190, -10.191],
        [10.0, 9.0, -10.191, -10.190])

    assert quality["full_ranking_identical"] is False
    assert quality["max_reference_logit_error"] <= 0.1
    assert quality["reference_relevant_order_agreement"] == 1.0
    assert quality["ndcg_at_5"] >= quality["reference_ndcg_at_5"] - 1e-6
    assert release._reranking_f16_passes(quality)


def test_reranker_f16_gate_rejects_relevant_top_k_swap(monkeypatch, tmp_path):
    quality = _reranking_quality_for_scores(
        monkeypatch, tmp_path, {"doc-0": 1, "doc-1": 1},
        [10.01, 10.00, -10.190, -10.191],
        [10.00, 10.01, -10.191, -10.190])

    assert quality["max_reference_logit_error"] <= 0.1
    assert quality["ndcg_at_5"] == pytest.approx(quality["reference_ndcg_at_5"])
    assert quality["reference_relevant_order_agreement"] == 0.0
    assert not release._reranking_f16_passes(quality)


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


def test_embedding_card_uses_server_api_and_preserves_prompt_prefixes():
    card = retrieval.render_card({
        "engine": "llama.cpp",
        "task": "embeddings",
        "family": "embedding model",
        "source": {"repo_id": "org/model", "revision": "a" * 40},
        "engine_revision": "b" * 40,
        "members": [{"path": "model-F16.gguf", "quant": "F16", "bytes": 1,
                     "sha256": "c" * 64}],
        "contract": {
            "pooling": "cls",
            "max_seq_length": 128,
            "prompts": {"query": "query: {query}", "document": "passage: {document}"}},
        "quality_results": {},
        "source_license": "mit",
    })

    assert ("llama-server -m ./retrieval-model/model-F16.gguf --embedding --pooling cls "
            "-c 128 -b 128 -ub 128 --host 127.0.0.1 --port 8080") in card
    assert "http://127.0.0.1:8080/v1/embeddings" in card
    assert '{"input":["query text"],"embd_normalize":2}' in card
    assert "Query prompt prefix to prepend: `query: `" in card
    assert "Document prompt prefix to prepend: `passage: `" in card


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


def test_reranking_card_describes_raw_unbounded_logits():
    card = retrieval.render_card(retrieval.RetrievalBundle(
        engine="llama.cpp", task="reranking", family="bge-reranker",
        architecture="XLMRobertaForSequenceClassification", source_repo="org/reranker",
        source_revision="abc", engine_revision="def", members=[], contract={}))
    assert "`relevance_score` is a raw, unbounded logit, not a probability." in card


def test_publish_bundle_uses_shared_commands_prompts_and_model_quant_names(
        monkeypatch, tmp_path):
    from agentquantix.pipeline import voice_release

    class Api:
        operations = []

        def create_repo(self, **kwargs):
            pass

        def create_commit(self, *, operations, **kwargs):
            self.operations = operations

    api = Api()

    def remote_files(api, repo_id):
        return {
            operation.path_in_repo: {
                "bytes": Path(operation.path_or_fileobj).stat().st_size,
                "sha256": release._hash(operation.path_or_fileobj)}
            for operation in api.operations}

    monkeypatch.setattr(voice_release, "_remote_files", remote_files)
    root = tmp_path / "run"
    root.mkdir()
    model = root / "converted_ggml-model-f16.gguf"
    model.write_bytes(b"quantized model")
    (root / "run.json").write_text(json.dumps({
        "runtime_snapshot": {"revision": "b" * 40}}), encoding="utf-8")
    plan = {"engine": "llama.cpp", "task": "embeddings",
            "model": "BAAI/bge-small-en-v1.5", "target_repo": "owner/bge-small-en-v1.5",
            "source_revision": "a" * 40, "architectures": ["BertModel"],
            "contract": {"pooling": "mean",
                         "prompts": {"query": "query: {query}", "document": "passage: "}},
            "quality_profile": {}, "fixtures": {}, "license": "apache-2.0"}
    candidate = {"status": "passed", "path": str(model), "quant": "Q8_0",
                 "sha256": release._hash(model), "bytes": model.stat().st_size,
                 "quality": {"reference_cosine_min": 1.0}}
    result = release.publish_bundle(root, plan, [candidate], api=api)
    manifest = json.loads((root / "publication/manifest.json").read_text(encoding="utf-8"))
    card = (root / "publication/README.md").read_text(encoding="utf-8")
    expected_member = "bge-small-en-v1.5-Q8_0.gguf"
    assert result["verified"]
    assert manifest["members"][0]["path"] == expected_member
    assert (root / "publication" / expected_member).is_file()
    assert manifest["runtime_commands"] == retrieval.runtime_commands(
        "llama.cpp", "embeddings", plan["contract"],
        f"./retrieval-model/{expected_member}")
    assert "Query prompt prefix to prepend: `query: `" in card
    assert "Document prompt prefix to prepend: `passage: `" in card


def test_cli_parser_accepts_retrieval_run_contract():
    args = cli.build_parser().parse_args([
        "run", "org/embed", "--task", "embeddings", "--target", "owner/embed",
        "--quant", "Q8_0", "--publish", "--yes"])
    assert args.task == "embeddings"
    assert args.target == "owner/embed" and args.quant == ["Q8_0"]
    assert args.publish is True


def test_saved_plan_routing_uses_retrieval_pipeline_registry(monkeypatch, tmp_path):
    from agentquantix.agent import tools

    task_name = "future-retrieval-task"
    task_spec = SimpleNamespace(pipeline="retrieval")
    monkeypatch.setitem(cli.TASKS, task_name, task_spec)
    monkeypatch.setitem(tools.TASKS, task_name, task_spec)
    plan = {"task": task_name, "engine": "future-engine"}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(cli, "_check_applicable", lambda *args: None)
    monkeypatch.setattr(cli, "_run_retrieval",
                        lambda args, route=None, saved=None: ("retrieval", saved))
    assert cli.cmd_run(cli.build_parser().parse_args(
        ["run", "--plan", str(plan_path), "--yes"])) == ("retrieval", plan)

    monkeypatch.setattr(tools.actions, "start_retrieval",
                        lambda approved, **kwargs: (approved, kwargs))
    started, options = tools._start({"user_approved": True, "plan": plan})
    assert started is plan
    assert options == {"user_approved": True, "publish": False}


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
