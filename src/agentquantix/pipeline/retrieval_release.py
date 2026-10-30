"""Native conversion, parity evaluation and atomic publication for retrieval models."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download

from .. import config
from ..retrieval import (CLIP_QUANT_TYPES, RetrievalBundle, cosine, mrr, ndcg_at_k,
                         recall_at_k, render_card, runtime_commands, top1_agreement)
from ..runtime_control import require_binaries
from .native_common import (_command, _free_port, _hash, _record_stage, _server as _shared_server,
                            _slug, _valid_stage, _write)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "retrieval"


def _convert(plan, source, handle, root, journal, journal_path):
    engine = plan["engine"]
    output = root / ("model-f16.gguf" if engine == "llama.cpp" else "model_ggml-model-f16.gguf")
    if _valid_stage(journal, "conversion"):
        return Path(journal["conversion"]["path"])
    converter = handle.source / (
        "models/convert_hf_to_gguf.py" if engine == "clip.cpp" else "convert_hf_to_gguf.py")
    env = os.environ.copy()
    if engine == "llama.cpp":
        gguf_py = str(handle.source / "gguf-py")
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (gguf_py, env.get("PYTHONPATH", "")) if item)
    if engine == "clip.cpp":
        output_dir = root / "converted"
        output_dir.mkdir(parents=True, exist_ok=True)
        _command([sys.executable, converter, "-m", source, "-o", output_dir], env=env)
        generated = output_dir / f"{output_dir.name.replace('ggml_', '')}_ggml-model-f16.gguf"
        if not generated.is_file():
            raise RuntimeError(f"clip.cpp converter did not produce expected file {generated}")
        output = generated
    else:
        _command([sys.executable, converter, source, "--outtype", "f16",
                  "--outfile", output], env=env)
    if not output.is_file():
        raise RuntimeError(f"converter produced no F16 GGUF: {output}")
    _record_stage(journal_path, journal, "conversion", output)
    return output


def _quantize(plan, f16, binaries, root, journal, journal_path, quant):
    engine = plan["engine"]
    key = f"quant:{quant}"
    if _valid_stage(journal, key):
        return Path(journal[key]["path"])
    output = root / f"model-{quant}.gguf"
    if engine == "clip.cpp":
        _command([binaries["quantize"], f16, output, CLIP_QUANT_TYPES[quant]])
    else:
        _command([binaries["llama-quantize"], f16, output, quant])
    if not output.is_file():
        raise RuntimeError(f"quantizer produced no file for {quant}")
    _record_stage(journal_path, journal, key, output)
    return output


def _template(text, template):
    value = str(template)
    if any(marker in value for marker in ("{text}", "{query}", "{document}")):
        return value.replace("{text}", text).replace("{query}", text).replace("{document}", text)
    return value + text


def _embedding_quality(plan, model, ref, binary, root):
    corpus = json.loads((FIXTURES / "corpus.json").read_text(encoding="utf-8"))
    pooling = plan["contract"]["pooling"]
    prompts = plan["contract"].get("prompts", {})
    qprompt, dprompt = prompts.get("query", ""), prompts.get("document", "")
    docs = corpus["documents"]
    queries = corpus["queries"]
    ctx = plan["contract"].get("max_seq_length") or 512
    with _embedder(binary, model, root, pooling, ctx) as embed:
        native_docs = [embed(_template(item["text"], dprompt)) for item in docs]
        native_queries = [embed(_template(item["query"], qprompt)) for item in queries]
    reference_docs = ref["reference_document_vectors"]
    reference_queries = ref["reference_query_vectors"]
    ranked, reference_ranked, gains = [], [], []
    for qi, query in enumerate(queries):
        scores = [cosine(native_queries[qi], vector) for vector in native_docs]
        ref_scores = [cosine(reference_queries[qi], vector) for vector in reference_docs]
        ranked.append([docs[i]["id"] for i in sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)])
        reference_ranked.append([docs[i]["id"] for i in sorted(
            range(len(docs)), key=lambda i: ref_scores[i], reverse=True)])
        gains.append(query["relevance"])
    return {"dimension": len(native_docs[0]), "reference_dimension": int(reference_docs.shape[1]),
            "reference_cosine_min": min(cosine(a, b) for a, b in
                                        zip(native_docs + native_queries,
                                            list(reference_docs) + list(reference_queries))),
            "recall_at_1": sum(recall_at_k(r, g, 1) for r, g in zip(ranked, gains)) / len(gains),
            "mrr": mrr([(rank, gain) for rank, gain in zip(ranked, gains)]),
            "ndcg_at_5": sum(ndcg_at_k(r, g, 5) for r, g in zip(ranked, gains)) / len(gains),
            "ranked": ranked, "reference_ranked": reference_ranked, "gains": gains,
            "document_vectors": [list(map(float, row)) for row in native_docs],
            "query_vectors": [list(map(float, row)) for row in native_queries],
            "reference_document_vectors": reference_docs.tolist(),
            "reference_query_vectors": reference_queries.tolist(),
            "vectors_finite": all(math.isfinite(float(x)) for row in (*native_docs, *native_queries) for x in row),
            "vectors_unit_norm": all(abs(math.sqrt(sum(float(x) ** 2 for x in row)) - 1) <= 1e-3
                                     for row in (*native_docs, *native_queries))}


@contextmanager
def _server(binary, model, root, arguments):
    with _shared_server(binary, model, root, arguments, free_port_fn=_free_port,
                        open_url=urlopen, spawn=subprocess.Popen,
                        monotonic=time.monotonic, sleep=time.sleep) as base_url:
        yield base_url


@contextmanager
def _rerank(binary, model, root):
    with _server(binary, model, root, ["--reranking"]) as base_url:
        def score(query, documents):
            body = json.dumps({"query": query, "documents": documents}).encode()
            request = Request(f"{base_url}/v1/rerank", body,
                              {"Content-Type": "application/json"})
            with urlopen(request, timeout=120) as response:
                return json.load(response)["results"]

        yield score


@contextmanager
def _embedder(binary, model, root, pooling, ctx):
    arguments = ["--embedding", "--pooling", pooling, "-c", ctx, "-b", ctx, "-ub", ctx]
    with _server(binary, model, root, arguments) as base_url:
        def embed(text):
            body = json.dumps({"input": [text], "embd_normalize": 2}).encode()
            request = Request(f"{base_url}/v1/embeddings", body,
                              {"Content-Type": "application/json"})
            with urlopen(request, timeout=120) as response:
                try:
                    result = json.load(response)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError("llama-server embeddings response is not valid JSON") from exc
            try:
                vector = result["data"][0]["embedding"]
            except (IndexError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    "llama-server embeddings response is missing data[0].embedding") from exc
            if not isinstance(vector, list) or not vector:
                raise RuntimeError("llama-server embeddings response has no embedding vector")
            try:
                values = [float(value) for value in vector]
            except (OverflowError, TypeError, ValueError) as exc:
                raise RuntimeError("llama-server embeddings response has an invalid vector") from exc
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("llama-server embeddings response contains non-finite values")
            return values

        yield embed


def _normalise_clip(text):
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def _zsl_label(output):
    scores = []
    for line in output.splitlines():
        if " = " not in line:
            continue
        label, score = line.rsplit(" = ", 1)
        try:
            scores.append((label, float(score)))
        except ValueError:
            continue
    if not scores:
        raise RuntimeError("clip.cpp zsl output did not contain label scores")
    return max(scores, key=lambda item: item[1])[0]


def _clip_quality(plan, model, binaries, root, ref):
    import numpy as np
    cases = json.loads((FIXTURES / "zsl.json").read_text(encoding="utf-8"))
    image_cosines, text_cosines, correct, zsl_agreement = [], [], 0, 0
    native_images, native_texts = [], []
    for index, case in enumerate(cases["cases"]):
        image_path = FIXTURES / case["image"]
        labels = [_normalise_clip(item) for item in case["labels"]]
        work = Path(tempfile.mkdtemp(prefix="clip-", dir=root))
        _command([binaries["extract"], "-m", model, "-t", "4", "-v", "0",
                  "--image", image_path, *sum((["--text", text] for text in labels), [])], cwd=work)
        image_vec = np.load(work / f"img_vec_{image_path.name}.npy").reshape(-1)
        text_vecs = [np.load(work / f"text_vec_{i}.npy").reshape(-1) for i in range(len(labels))]
        scores = [cosine(image_vec, vector) for vector in text_vecs]
        correct += int(max(range(len(scores)), key=scores.__getitem__) == case["correct"])
        zsl_output = _command([binaries["zsl"], "-m", model, "-t", "4", "-v", "0",
                               "--image", image_path,
                               *sum((["--text", text] for text in labels), [])], cwd=work)
        zsl_label = _zsl_label(zsl_output)
        zsl_agreement += int(zsl_label == labels[max(range(len(scores)), key=scores.__getitem__)])
        hf_image = np.asarray(ref["image_vectors"][index])
        hf_text = np.asarray(ref["text_vectors"][index])
        reference_scores = [cosine(hf_image, vector) for vector in hf_text]
        image_cosines.append(cosine(image_vec, hf_image))
        text_cosines.extend(cosine(native, target) for native, target in zip(text_vecs, hf_text))
        native_images.append(image_vec.tolist())
        native_texts.append([vector.tolist() for vector in text_vecs])
    probe = cases["tokenizer_parity_probe"]
    normalised_probe = _normalise_clip(probe)
    probe_work = Path(tempfile.mkdtemp(prefix="clip-probe-", dir=root))
    _command([binaries["extract"], "-m", model, "-t", "4", "-v", "0",
              "--text", normalised_probe, "--text", probe], cwd=probe_work)
    native_normalised = np.load(probe_work / "text_vec_0.npy").reshape(-1)
    native_raw = np.load(probe_work / "text_vec_1.npy").reshape(-1)
    probe_features = np.asarray(ref["probe_features"])
    return {"image_cosine_min": min(image_cosines), "text_cosine_min": min(text_cosines),
            "accuracy": correct / len(cases["cases"]),
            "reference_accuracy": ref["reference_accuracy"],
            "zsl_agreement": zsl_agreement / len(cases["cases"]),
            "tokenizer_parity_probe": {
                "normalized_text": normalised_probe,
                "normalized_cosine": cosine(native_normalised, probe_features[0]),
                "raw_text": probe,
                "raw_cosine": cosine(native_raw, probe_features[1])},
            "image_vectors": native_images, "text_vectors": native_texts}


def _reranking_quality(model, server, reference):
    corpus = json.loads((FIXTURES / "corpus.json").read_text(encoding="utf-8"))
    docs, queries = corpus["documents"], corpus["queries"]
    rankings, ref_rankings, gains, scores, reference_scores = [], [], [], [], []
    relevant_counts = []
    with _rerank(server, model, server.parent) as score:
        for query_index, item in enumerate(queries):
            document_texts = [doc["text"] for doc in docs]
            result = score(item["query"], document_texts)
            row = [0.0] * len(document_texts)
            for entry in result:
                row[entry["index"]] = float(entry["relevance_score"])
            reference_row = reference["reference_scores"][query_index]
            rankings.append([docs[index]["id"] for index in sorted(
                range(len(docs)), key=lambda index: row[index], reverse=True)])
            ref_rankings.append([docs[index]["id"] for index in sorted(
                range(len(docs)), key=lambda index: reference_row[index], reverse=True)])
            scores.append(row)
            reference_scores.append(list(map(float, reference_row)))
            gain = item["relevance"]
            gains.append(gain)
            relevant_counts.append(max(1, sum(value > 0 for value in gain.values())))
    ndcg = sum(ndcg_at_k(rank, gain, 5) for rank, gain in zip(rankings, gains)) / len(gains)
    reference_ndcg = sum(ndcg_at_k(rank, gain, 5)
                         for rank, gain in zip(ref_rankings, gains)) / len(gains)
    relevant_order_agreement = sum(
        ranking[:count] == reference_ranking[:count]
        for ranking, reference_ranking, count in zip(rankings, ref_rankings, relevant_counts)
    ) / len(gains)
    return {"ranked": rankings, "reference_ranked": ref_rankings, "scores": scores,
            "reference_scores": reference_scores, "gains": gains,
            "max_reference_logit_error": max(abs(a - b) for left, right in
                                             zip(scores, reference_scores)
                                             for a, b in zip(left, right)),
            "full_ranking_identical": rankings == ref_rankings,
            "reference_relevant_order_agreement": relevant_order_agreement,
            "mrr": mrr(list(zip(rankings, gains))),
            "ndcg_at_5": ndcg, "reference_ndcg_at_5": reference_ndcg}


def _reranking_f16_passes(quality):
    return (quality["max_reference_logit_error"] <= 0.1
            and quality["reference_relevant_order_agreement"] == 1.0
            and quality["ndcg_at_5"] >= quality["reference_ndcg_at_5"] - 1e-6)


def _spearman(left, right):
    size = len(left)
    left_order = sorted(range(size), key=left.__getitem__)
    right_order = sorted(range(size), key=right.__getitem__)
    left_ranks = {index: rank for rank, index in enumerate(left_order)}
    right_ranks = {index: rank for rank, index in enumerate(right_order)}
    squared = sum((left_ranks[index] - right_ranks[index]) ** 2 for index in range(size))
    return 1 - 6 * squared / (size * (size ** 2 - 1))


def _references(plan, source):
    try:
        if plan["engine"] == "clip.cpp":
            import torch
            from transformers import CLIPModel
            model = CLIPModel.from_pretrained(str(source), local_files_only=True).eval()
            return {"source": source, "model": model}
        if plan["task"] == "reranking":
            import torch
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(str(source), device="cpu")
            return {"cross_encoder": model, "identity": torch.nn.Identity()}
        from sentence_transformers import SentenceTransformer
        return {"sentence_transformer": SentenceTransformer(str(source), device="cpu")}
    except ImportError:
        return None


def _reference_outputs(plan, reference):
    try:
        if plan["engine"] == "clip.cpp":
            import torch
            from PIL import Image
            from transformers import CLIPProcessor

            cases = json.loads((FIXTURES / "zsl.json").read_text(encoding="utf-8"))
            processor = CLIPProcessor.from_pretrained(str(reference["source"]),
                                                      local_files_only=True)
            image_vectors, text_vectors, correct = [], [], 0
            for case in cases["cases"]:
                labels = [_normalise_clip(label) for label in case["labels"]]
                image = Image.open(FIXTURES / case["image"]).convert("RGB")
                inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
                with torch.inference_mode():
                    image_features = reference["model"].get_image_features(
                        pixel_values=inputs["pixel_values"])
                    text_features = reference["model"].get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"))
                image_vector = image_features[0].cpu().numpy()
                text_vectors_for_case = text_features.cpu().numpy()
                scores = [cosine(image_vector, vector) for vector in text_vectors_for_case]
                correct += int(max(range(len(scores)), key=scores.__getitem__) == case["correct"])
                image_vectors.append(image_vector.tolist())
                text_vectors.append(text_vectors_for_case.tolist())
            probe = cases["tokenizer_parity_probe"]
            normalised_probe = _normalise_clip(probe)
            inputs = processor(text=[normalised_probe, probe], return_tensors="pt", padding=True)
            with torch.inference_mode():
                probe_features = reference["model"].get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask")).cpu().numpy()
            return {"image_vectors": image_vectors, "text_vectors": text_vectors,
                    "reference_accuracy": correct / len(cases["cases"]),
                    "probe_features": probe_features.tolist()}

        corpus = json.loads((FIXTURES / "corpus.json").read_text(encoding="utf-8"))
        docs, queries = corpus["documents"], corpus["queries"]
        if plan["task"] == "embeddings":
            prompts = plan.get("contract", {}).get("prompts", {})
            document_texts = [_template(item["text"], prompts.get("document", ""))
                              for item in docs]
            query_texts = [_template(item["query"], prompts.get("query", ""))
                           for item in queries]
            model = reference["sentence_transformer"]
            reference_docs = model.encode(document_texts, normalize_embeddings=True,
                                          convert_to_numpy=True)
            reference_queries = model.encode(query_texts, normalize_embeddings=True,
                                              convert_to_numpy=True)
            return {"reference_document_vectors": reference_docs,
                    "reference_query_vectors": reference_queries}

        pairs = [(item["query"], doc["text"]) for item in queries for doc in docs]
        logits = reference["cross_encoder"].predict(
            pairs, activation_fn=reference["identity"], convert_to_numpy=True,
            show_progress_bar=False)
        return {"reference_scores": logits.reshape(len(queries), len(docs)).tolist()}
    except ImportError:
        return None


def _evaluate(plan, model, binaries, root, references):
    if references is None:
        return {"structural_only": True,
                "blocker": "reference libraries not installed: pip install agentquantix[retrieval]"}
    if plan["engine"] == "clip.cpp":
        return _clip_quality(plan, model, binaries, root, references)
    if plan["task"] == "embeddings":
        return _embedding_quality(plan, model, references, binaries["llama-server"], root)
    return _reranking_quality(model, binaries["llama-server"], references)


def _snapshot_patterns(plan):
    patterns = ["*.json", "*.txt", "*.model", "*.safetensors", "tokenizer*",
                "vocab*", "merges*", "spiece*", "sentencepiece*"]
    modules = (plan.get("contract") or {}).get("module_directories") or []
    for directory in dict.fromkeys(("1_Pooling", *modules)):
        patterns.append(f"{directory.rstrip('/')}/*")
    if not any(name.lower().endswith(".safetensors") for name in plan.get("files", [])):
        patterns.append("*.bin")
    ignored = ["onnx/*", "onnx/**", "openvino/*", "openvino/**", "*.h5",
               "**/*.h5", "*.msgpack", "**/*.msgpack", "*.ot", "**/*.ot"]
    return patterns, ignored


def run_release(plan, *, user_approved, publish=False):
    if user_approved is not True:
        raise ValueError("retrieval run requires explicit user approval")
    if plan.get("status") != "ready" or plan.get("blockers"):
        raise ValueError("retrieval plan is blocked")
    root = config.TEMP_DIR / "aqx-retrieval-runs" / _slug(plan["model"])
    root.mkdir(parents=True, exist_ok=True)
    run_path, journal_path = root / "run.json", root / "journal.json"
    saved = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    frozen = saved.get("runtime_snapshot")
    required = (("extract", "zsl", "quantize") if plan["engine"] == "clip.cpp"
                else ("llama-server", "llama-quantize"))
    from .. import runtime_control
    handle = runtime_control.ensure_runtime(plan["engine"], required_targets=required,
                                            frozen=frozen)
    if handle is None:
        raise RuntimeError(f"no execution runtime session for {plan['engine']}")
    binaries = dict(zip(required, require_binaries(handle, *required)))
    allow_patterns, ignore_patterns = _snapshot_patterns(plan)
    source = Path(snapshot_download(
        plan["model"], revision=plan.get("source_revision"), token=config.TOKEN,
        allow_patterns=allow_patterns, ignore_patterns=ignore_patterns))
    reference = _references(plan, source)
    references = _reference_outputs(plan, reference) if reference is not None else None
    journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.is_file() else {}
    run = {"plan": plan, "runtime_snapshot": handle.snapshot(),
           "created": saved.get("created") or time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    _write(run_path, run)
    f16 = _convert(plan, source, handle, root, journal, journal_path)
    quants = plan["quants"]
    if (plan["engine"] == "clip.cpp" and "f16" not in quants) or \
            (plan["engine"] == "llama.cpp" and "F16" not in quants):
        quants = [("f16" if plan["engine"] == "clip.cpp" else "F16"), *quants]
    baseline = None
    candidates = []
    for quant in quants:
        model = f16 if quant.casefold() == "f16" else \
            _quantize(plan, f16, binaries, root, journal, journal_path, quant)
        quality = _evaluate(plan, model, binaries, root, references)
        if quant.casefold() == "f16":
            baseline = quality
            if plan["engine"] == "clip.cpp" and not quality.get("structural_only"):
                passed = (quality["image_cosine_min"] >= 0.999
                          and quality["text_cosine_min"] >= 0.999
                          and quality["accuracy"] == quality["reference_accuracy"])
            elif plan["task"] == "embeddings" and not quality.get("structural_only"):
                passed = (quality["reference_cosine_min"] >= 0.999
                          and quality["dimension"] == quality["reference_dimension"]
                          and quality["vectors_finite"] and quality["vectors_unit_norm"])
            elif plan["task"] == "reranking" and not quality.get("structural_only"):
                passed = _reranking_f16_passes(quality)
            else:
                passed = False
            passed = passed and not quality.get("blocker")
        elif quality.get("structural_only") or baseline is None:
            passed = False
        elif plan["engine"] == "clip.cpp":
            image_cosines = [cosine(left, right) for left, right in
                             zip(quality["image_vectors"], baseline["image_vectors"])]
            text_cosines = [cosine(left, right) for rows, base_rows in
                            zip(quality["text_vectors"], baseline["text_vectors"])
                            for left, right in zip(rows, base_rows)]
            passed = (min(image_cosines) >= 0.95 and min(text_cosines) >= 0.95
                      and quality["accuracy"] >= baseline["accuracy"])
            quality["image_cosine_vs_f16_min"] = min(image_cosines)
            quality["text_cosine_vs_f16_min"] = min(text_cosines)
        elif plan["task"] == "embeddings":
            similarities = [cosine(left, right) for left, right in
                            zip(quality["document_vectors"] + quality["query_vectors"],
                                baseline["document_vectors"] + baseline["query_vectors"])]
            passed = (min(similarities) >= 0.98
                      and quality["dimension"] == baseline["dimension"]
                      and quality["vectors_finite"] and quality["vectors_unit_norm"]
                      and quality["mrr"] >= baseline["mrr"] - 0.05
                      and quality["ndcg_at_5"] >= baseline["ndcg_at_5"] - 0.05)
            quality["f16_cosine_min"] = min(similarities)
        else:
            correlations = [_spearman(left, right) for left, right in
                            zip(quality["scores"], baseline["scores"])]
            quality["spearman_per_query"] = correlations
            passed = (top1_agreement(quality["ranked"], baseline["ranked"]) == 1.0
                      and quality["ndcg_at_5"] >= baseline["ndcg_at_5"] - 0.05)
        candidates.append({"quant": quant, "path": str(model), "sha256": _hash(model),
                           "bytes": model.stat().st_size, "quality": quality,
                           "status": "passed" if passed else "quality-rejected"})
        if quant.casefold() == "f16" and not passed:
            break
    _write(root / "quality.json", {"profile": plan.get("quality_profile"),
                                   "quants": {item["quant"]: item["quality"]
                                              for item in candidates}})
    result = {"run_id": root.name, "plan": plan, "runtime_snapshot": handle.snapshot(),
              "candidates": candidates, "quality": {"baseline": baseline}, "published": False}
    _write(root / "result.json", result)
    if publish and candidates and all(item["status"] != "passed" for item in candidates):
        raise ValueError("no retrieval quant passed the automated gates")
    if publish and any(item["status"] == "passed" for item in candidates):
        publish_bundle(root, plan, candidates)
        result["published"] = True
        _write(root / "result.json", result)
    return result


def publish_bundle(root, plan, candidates, api=None):
    from huggingface_hub import CommitOperationAdd, HfApi
    from ..pipeline.voice_release import _remote_files
    api = api or HfApi(token=config.TOKEN)
    root = Path(root)
    passing = [item for item in candidates if item["status"] == "passed"]
    if not passing:
        raise ValueError("no passing quant is publishable")
    publication_root = root / "publication"
    publication_root.mkdir(parents=True, exist_ok=True)
    members = []
    model_name = plan.get("model", "retrieval-model").rsplit("/", 1)[-1]
    for item in passing:
        path = Path(item["path"])
        destination = f"{model_name}-{item['quant'].upper()}.gguf"
        shutil.copy2(path, publication_root / destination)
        members.append({"path": destination, "quant": item["quant"],
                        "sha256": item["sha256"], "bytes": item["bytes"]})
    fixture_root = publication_root / "fixtures"
    fixture_root.mkdir(exist_ok=True)
    for fixture in FIXTURES.iterdir():
        if fixture.is_file():
            shutil.copy2(fixture, fixture_root / fixture.name)
    saved_run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    snapshot = saved_run.get("runtime_snapshot", {})
    model_path = f"./retrieval-model/{members[0]['path']}"
    commands = runtime_commands(plan["engine"], plan["task"], plan["contract"], model_path)
    public_quality = {
        item["quant"]: {key: value for key, value in item["quality"].items()
                        if "vectors" not in key}
        for item in passing
    }
    bundle = RetrievalBundle(plan["engine"], plan["task"], plan["model"], "",
                             plan["model"], plan.get("source_revision"),
                             snapshot.get("revision"), members, plan["contract"],
                             runtime_commands=commands,
                             quality_results=public_quality,
                             quality_profile=plan.get("quality_profile", {}),
                             fixtures=plan.get("fixtures", {}),
                             source_license=plan.get("license"),
                             target_repo=plan.get("target_repo"))
    bundle.family = plan.get("model", "").rsplit("/", 1)[-1]
    bundle.architecture = ", ".join(plan.get("architectures", []))
    bundle.build_provenance = snapshot.get("provenance", {})
    manifest = bundle.manifest()
    manifest["pipeline"] = "retrieval"
    manifest_path = root / "publication" / "manifest.json"
    quality_path = manifest_path.parent / "quality.json"
    _write(quality_path, manifest["quality_results"])
    card_path = manifest_path.parent / "README.md"
    card_path.write_text(render_card(bundle), encoding="utf-8")
    license_path = manifest_path.parent / "FIXTURE_LICENSE.md"
    shutil.copyfile(FIXTURES / "LICENSE.md", license_path)
    member_paths = {item["path"] for item in members}
    manifest["published_files"] = [
        {"path": path.relative_to(publication_root).as_posix(),
         "bytes": path.stat().st_size, "sha256": _hash(path)}
        for path in sorted(publication_root.rglob("*"))
        if path.is_file() and path.relative_to(publication_root).as_posix() not in member_paths
        and path != manifest_path]
    _write(manifest_path, manifest)
    operations = []
    for path in sorted(publication_root.rglob("*")):
        if path.is_file():
            operations.append(CommitOperationAdd(
                path_in_repo=path.relative_to(publication_root).as_posix(),
                path_or_fileobj=str(path)))
    api.create_repo(repo_id=plan["target_repo"], repo_type="model", exist_ok=True)
    api.create_commit(repo_id=plan["target_repo"], repo_type="model", operations=operations,
                      commit_message="Publish retrieval bundle with measured quality gates")
    remote = _remote_files(api, plan["target_repo"])
    from huggingface_hub import hf_hub_download
    manifest_file = {"path": "manifest.json", "bytes": manifest_path.stat().st_size,
                     "sha256": _hash(manifest_path)}
    problems = _remote_member_mismatches(
        [*members, *manifest["published_files"], manifest_file], remote,
        download=lambda name: hf_hub_download(
            plan["target_repo"], name, repo_type="model", token=config.TOKEN))
    if problems:
        raise ValueError("; ".join(problems))
    return {"repo_id": plan["target_repo"], "verified": True, "members": members}


def _remote_member_mismatches(members, remote, download=None):
    problems = []
    for member in members:
        actual = remote.get(member["path"])
        if actual is None:
            problems.append(f"remote verification mismatch: {member['path']}")
            continue
        size, digest = actual.get("bytes"), actual.get("sha256")
        if (size is None or digest is None) and download is not None:
            try:
                path = Path(download(member["path"]))
                size, digest = path.stat().st_size, _hash(path)
            except Exception:
                size, digest = None, None
        if size != member["bytes"] or digest != member["sha256"]:
            problems.append(f"remote verification mismatch: {member['path']}")
    return problems


def publish_run(slug, quant):
    root = config.TEMP_DIR / "aqx-retrieval-runs" / slug
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    candidates = [item for item in result["candidates"]
                 if item["quant"].casefold() == quant.casefold() and item["status"] == "passed"]
    if not candidates:
        raise ValueError(f"{quant} is not a passing retrieval quant in {slug}")
    passing = [item for item in result["candidates"] if item["status"] == "passed"]
    return publish_bundle(root, result["plan"], passing)


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
