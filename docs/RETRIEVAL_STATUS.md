# AgentQuantix 0.5.3 retrieval status

0.5.3 adds retrieval pipelines for text embeddings, cross-encoder reranking,
CLIP image features, and zero-shot image classification. The package version
remains unchanged. Native acceptance is specific to the model, revision,
precision, platform, and backend recorded here; it is not a general support
claim for every compatible model.

## Support matrix

| Task | Engine | Native route | Native acceptance |
|---|---|---|---|
| `embeddings` | llama.cpp | `llama-server --embedding --pooling <pooling>` and `POST /v1/embeddings` | Linux CPU accepted for `BAAI/bge-small-en-v1.5` and `sentence-transformers/all-MiniLM-L6-v2`; MiniLM Q4_0 rejected and excluded from bundle |
| `reranking` | llama.cpp | `llama-server --reranking` and `POST /v1/rerank` | Linux CPU accepted for `BAAI/bge-reranker-base` |
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

## llama.cpp native acceptance

All three native runs used Linux x86_64, 8 vCPU, AVX2, CPU only, llama.cpp
`8642541b255bfb5dc39cd351a178be09013a55d6`, and the llama-server-only runtime.
The quant set was F16, Q8_0, Q6_K, Q5_K_M, Q5_0, Q4_K_M, and Q4_0.

| Model | Wall time |
|---|---:|
| BAAI/bge-small-en-v1.5 | 1m18s |
| sentence-transformers/all-MiniLM-L6-v2 | 1m06s |
| BAAI/bge-reranker-base | 4m50s |

Wall times include conversion, all quantizations, and evaluation.

### BAAI/bge-small-en-v1.5 — embeddings

| Quant | Status | Bytes | Reference cosine min | F16 cosine min | MRR | NDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| F16 | passed | 67582720 | 0.9999985068259786 | — | 1.0 | 1.0 |
| Q8_0 | passed | 36688256 | 0.9997885152392768 | 0.9997921058233357 | 1.0 | 1.0 |
| Q6_K | passed | 34974080 | 0.9996722646001461 | 0.9996818750315805 | 1.0 | 1.0 |
| Q5_K_M | passed | 30356864 | 0.9976152473237103 | 0.9976170319358425 | 1.0 | 1.0 |
| Q5_0 | passed | 28725632 | 0.9951458894992765 | 0.9951580758966088 | 1.0 | 1.0 |
| Q4_K_M | passed | 29085056 | 0.9964719945950782 | 0.9964343483149408 | 1.0 | 1.0 |
| Q4_0 | passed | 26071424 | 0.9830456833654644 | 0.983006166048699 | 1.0 | 1.0 |

### sentence-transformers/all-MiniLM-L6-v2 — embeddings

| Quant | Status | Bytes | Reference cosine min | F16 cosine min | MRR | NDCG@5 |
|---|---|---:|---:|---:|---:|---:|
| F16 | passed | 46223712 | 0.99999885660977 | — | 1.0 | 0.9915972986637638 |
| Q8_0 | passed | 25282528 | 0.9995415373188302 | 0.9995487322207873 | 1.0 | 0.9915972986637638 |
| Q6_K | passed | 24425440 | 0.9992733966319769 | 0.9992881156908507 | 1.0 | 0.9915972986637638 |
| Q5_K_M | passed | 21992416 | 0.9945727530524924 | 0.9945649172131832 | 1.0 | 0.9915972986637638 |
| Q5_0 | passed | 21301216 | 0.9907101474366855 | 0.9907202619266919 | 1.0 | 0.9915972986637638 |
| Q4_K_M | passed | 21273568 | 0.9905151301741749 | 0.9905221328767098 | 1.0 | 0.9915972986637638 |
| Q4_0 | quality-rejected | 19974112 | 0.9642918098749264 | 0.9641978964623991 | 1.0 | 0.9931059945699483 |

MiniLM Q4_0 was correctly excluded from the bundle because its F16 cosine
`0.9641978964623991` is below the `0.98` quant gate.

### BAAI/bge-reranker-base — reranking

| Quant | Status | Bytes | Max logit error vs reference | Relevant-order agreement vs reference | NDCG@5 / reference NDCG@5 | Min per-query Spearman (quants only) |
|---|---|---:|---:|---:|---:|---:|
| F16 | passed | 563949824 | 0.07009141147136688 | 1.0 | 0.9491768952476267 / 0.9491768952476267 | — |
| Q8_0 | passed | 303769152 | 0.3460383415222168 | 0.875 | 0.9446694494122083 / 0.9491768952476267 | 0.9029411764705882 |
| Q6_K | passed | 236555840 | 1.3730783462524414 | 1.0 | 0.9491768952476267 / 0.9491768952476267 | 0.8058823529411765 |
| Q5_K_M | passed | 227547200 | 2.3893677592277527 | 0.875 | 0.9446694494122083 / 0.9491768952476267 | 0.8294117647058823 |
| Q5_0 | passed | 225197120 | 2.131668448448181 | 0.875 | 0.9446694494122083 / 0.9491768952476267 | 0.5205882352941176 |
| Q4_K_M | passed | 219068480 | 1.9095258712768555 | 0.875 | 0.9446694494122083 / 0.9491768952476267 | 0.7794117647058824 |
| Q4_0 | passed | 214506560 | 4.1329076290130615 | 0.75 | 0.9401620035767899 / 0.9491768952476267 | 0.4529411764705882 |

Reranker quant gates are ranking-based: top-1 agreement and NDCG within `0.05`
of F16, not logit-based. Low-bit quants can drift substantially in raw logits
while preserving top-1 and NDCG acceptance. Q4_0 has maximum reference logit
error `4.1329076290130615` and minimum per-query Spearman `0.4529411764705882`.
Users who need calibrated scores should prefer F16 or Q8_0.

The native results are preserved outside the repository at
`/home/ubuntu/spike/acceptance/bge-results.json`,
`/home/ubuntu/spike/acceptance/minilm-results.json`, and
`/home/ubuntu/spike/acceptance/reranker-results.json`.
