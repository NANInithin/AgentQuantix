# AgentQuantix 0.5.3 retrieval status

0.5.3 adds retrieval pipelines for text embeddings, cross-encoder reranking,
CLIP image features, and zero-shot image classification. The package version
remains unchanged. Native acceptance is specific to the model, revision,
precision, platform, and backend recorded here; it is not a general support
claim for every compatible model.

## Support matrix

| Task | Engine | Native route | Native acceptance |
|---|---|---|---|
| `embeddings` | llama.cpp | `llama-server --embedding --pooling <pooling>` and `POST /v1/embeddings` | `TODO(native-llama)` |
| `reranking` | llama.cpp | `llama-server --reranking` and `POST /v1/rerank` | `TODO(native-llama)` |
| `image-feature-extraction` | clip.cpp | `extract`; `quantize` for supported precisions | Linux CPU accepted for `openai/clip-vit-base-patch32` below |
| `zero-shot-image-classification` | clip.cpp | `extract`, `zsl`, and `quantize` | Linux CPU accepted for `openai/clip-vit-base-patch32` below |

Task metadata routes through the common CLI; use `--task` when an explicit
choice is needed. Install the optional reference-evaluation dependencies with
`pip install "agentquantix[retrieval]"`. The run inventory can be filtered with
`aqx runs --pipeline retrieval`.

llama.cpp embeddings use the server endpoint rather than `llama-embedding`.
At the pinned revision, `llama-embedding --pooling rank` produces broken
output, and the CLI's asynchronous logger can truncate stdout nondeterministically.
Reranking `relevance_score` values from `/v1/rerank` are raw, unbounded logits,
not probabilities.

CLIP text prompts are normalized before encoding: lowercase and remove
punctuation. This matters for tokenizer parity; the normalized probe below
closely matches the reference, while the original capitalized, punctuated
sentence does not.

## Quality gates

Thresholds below are implemented in
`src/agentquantix/pipeline/retrieval_release.py`. All vector cosines are
minimums over the evaluated fixture vectors.

### Embeddings

| Candidate | Required gates |
|---|---|
| F16 vs reference | Minimum vector cosine `>= 0.999`; equal embedding dimensions; every vector finite; every vector unit-normalized within `1e-3`. |
| Quant vs F16 | Minimum vector cosine `>= 0.98`; equal dimensions; finite, unit-normalized vectors within `1e-3`; MRR and NDCG@5 each no more than `0.05` below F16. |

### Reranking

| Candidate | Required gates |
|---|---|
| F16 vs reference | Maximum absolute logit error `<= 0.1`; relevant-order agreement `== 1.0`; NDCG@5 `>= reference_ndcg_at_5 - 1e-6`. For each query, the compared top-k contains `max(1, count(relevance > 0))` documents and must match the reference IDs in order. |
| Quant vs F16 | Top-1 ranking agreement `== 1.0` across queries; NDCG@5 `>= F16_ndcg_at_5 - 0.05`. |

`full_ranking_identical` is informational only; differences among irrelevant
tail documents do not fail the F16 gate.

### CLIP features and zero-shot classification

| Candidate | Required gates |
|---|---|
| F16 vs reference | Minimum image cosine `>= 0.999`; minimum text cosine `>= 0.999`; classification accuracy equals reference accuracy. |
| Quant vs F16 | Minimum image and text cosines each `>= 0.95`; classification accuracy is no lower than F16. |

The tokenizer parity probe is reported as evidence but is not an independent
pass/fail gate.

## Fixtures and license

`src/agentquantix/fixtures/retrieval/` contains:

- `corpus.json`: 16 text documents and 8 queries with graded relevance labels.
- `zsl.json`: five zero-shot image-classification cases and the tokenizer
  parity probe.
- `red-circle.png`, `blue-square.png`, `green-triangle.png`,
  `yellow-star.png`, and `black-stripes.png`.

These fixtures were authored for AgentQuantix, contain no third-party material,
and are dedicated to the public domain under **CC0 1.0 Universal**, as stated
in `src/agentquantix/fixtures/retrieval/LICENSE.md`.

## Native environment and revisions

The recorded clip.cpp acceptance ran on **Linux x86_64, 8 vCPU, AVX2, CPU
only**. No GPU or Metal support is claimed.

| Engine/component | Revision |
|---|---|
| llama.cpp | `8642541b255bfb5dc39cd351a178be09013a55d6` |
| clip.cpp | `913458d5d1c9238380a0b0826cd1e71c8828a82d` |
| clip.cpp ggml submodule | `c3ae31e5a090a6259c674b18983de53ac4538aa6` |

## clip.cpp native acceptance

`openai/clip-vit-base-patch32` passed native acceptance on the environment
above for all six listed precisions. Cosines are minimums against the
Transformers reference and, where available, against F16. The tokenizer probe
columns show cosine against the corresponding reference feature for the
normalized lowercase/no-punctuation text and the original raw text,
respectively. F16 has no separate F16 comparison.

| Quant | Status | Size (bytes) | Accuracy | Image cosine (vs F16 / reference) | Text cosine (vs F16 / reference) | Tokenizer probe cosine (normalized / raw) |
|---|---:|---:|---:|---:|---:|---:|
| F16 | passed | 303,810,624 | 1.000000 | — / 0.999986 | — / 1.000000 | 1.000000 / 0.924113 |
| Q8_0 | passed | 164,391,776 | 1.000000 | 0.999434 / 0.999369 | 0.999877 / 0.999877 | 0.999798 / 0.923460 |
| Q5_1 | passed | 117,918,816 | 1.000000 | 0.992262 / 0.991848 | 0.997323 / 0.997324 | 0.996678 / 0.917773 |
| Q5_0 | passed | 108,624,224 | 1.000000 | 0.988064 / 0.988132 | 0.996029 / 0.996028 | 0.994315 / 0.934778 |
| Q4_1 | passed | 99,329,632 | 1.000000 | 0.969955 / 0.969959 | 0.992733 / 0.992726 | 0.979412 / 0.909819 |
| Q4_0 | passed | 90,035,040 | 1.000000 | 0.965852 / 0.965486 | 0.987453 / 0.987454 | 0.971714 / 0.945996 |

Evidence was transcribed from `/home/ubuntu/spike/acceptance/clip-results.json`
and the run record at
`/home/ubuntu/.agentquantix/temp/aqx-retrieval-runs/openai-clip-vit-base-patch32/result.json`.

## TODO(native-llama): native acceptance pending

The llama.cpp native rerun is pending. Fill in these tables from the native
acceptance results; do not infer values from offline tests or feasibility
spikes.

### BAAI/bge-small-en-v1.5 — embeddings

| Quant | Status | Size (bytes) | Reference cosine min | F16 cosine min | MRR | NDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| F16 | TODO | TODO | TODO | — | TODO | TODO |
| Q8_0 | TODO | TODO | TODO | TODO | TODO |
| Q4_K_M | TODO | TODO | TODO | TODO | TODO | TODO |

### sentence-transformers/all-MiniLM-L6-v2 — embeddings

| Quant | Status | Size (bytes) | Reference cosine min | F16 cosine min | MRR | NDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| F16 | TODO | TODO | TODO | — | TODO | TODO |
| Q8_0 | TODO | TODO | TODO | TODO | TODO | TODO |
| Q4_K_M | TODO | TODO | TODO | TODO | TODO | TODO |

### BAAI/bge-reranker-base — reranking

| Quant | Status | Size (bytes) | Max logit error vs reference | Relevant-order agreement | NDCG@5 / reference |
|---|---|---:|---:|---:|---:|
| F16 | TODO | TODO | TODO | TODO | TODO |
| Q8_0 | TODO | TODO | TODO | TODO | TODO |
