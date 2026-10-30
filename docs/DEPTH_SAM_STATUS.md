# Depth and SAM status

## Native depth support

0.5.8 adds two depth-anything.cpp families. The runtime is pinned to
`14f7461d1f704761a038ac9f50dbde8fdb7275e2` and is MIT-licensed. The weights
are licensed separately:

| Family | Source | Revision | Weights | Semantics |
|---|---|---|---|---|
| `da2-metric-hypersim-small` | [Depth-Anything-V2-Metric-Hypersim-Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small) | `3bc65d4e14a6786a61acec16453c50e12bf5f338` | Apache-2.0 | Metric depth in metres; indoor Hypersim training; maximum depth 20 m |
| `da3-small` | [DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) | `e08cab65ca0ec38e7826075418411ab90cab4da3` | Apache-2.0 | Relative depth with camera pose and intrinsics |

DA2 relative Small remains routed to vision.cpp. The depth-anything.cpp
converter produces F32. F16, Q8_0 and Q4_K are produced by the pinned runtime's
native quantizer from that F32 GGUF; provenance and the runtime card identify
the producer for each precision. A precision is bundled only if it passes its
family gates, and a failed F32 baseline rejects every precision.

Acceptance uses ten runtime-fetched COCO val2017 images. This is a small-sample
parity check, not a broad quality evaluation:

| Image IDs | COCO license ID | License |
|---|---:|---|
| 785, 872, 885, 1000, 1268, 2473, 5001 | 4 | Attribution License (CC BY 2.0) |
| 274219, 427077, 560880 | 7 | No known copyright restrictions |

The DA2 metric baseline is compared unaligned in metres with Spearman >= 0.99,
AbsRel <= 0.05, and median absolute error <= 5% of reference median depth.
Other DA2 precisions use the same thresholds against F32. DA3 is compared in
the model frame: depth Spearman >= 0.99 and scale-shift AbsRel <= 0.05 using a
5th-percentile floor; camera rotation <= 1 degree, translation L2 <= 1e-3 in
model-normalized units, and intrinsics mean relative error <= 0.01. Other DA3
precisions use the same thresholds against F32. DA3 confidence output is not
evaluated.

Native Linux CPU acceptance used runtime revision
`14f7461d1f704761a038ac9f50dbde8fdb7275e2`, with
`-DDA_HAS_AVX512F=OFF`. DA2 metric results on the ten-image sample:

| Precision | Producer | Status | Spearman | AbsRel | Median absolute error |
|---|---|---|---:|---:|---:|
| F32 | converter | pass | 0.99327 vs reference | 0.02876 vs reference | 0.03373 m (0.73% of reference median) |
| F16 | native-quantizer | pass | 0.9999998 vs F32 | 0.00021 vs F32 | 0.00062 m vs F32 |
| Q8_0 | native-quantizer | pass | 0.99962 vs F32 | 0.00718 vs F32 | 0.01498 m vs F32 |
| Q4_K | native-quantizer | pass | 0.99530 vs F32 | 0.04238 vs F32 | 0.13407 m vs F32 |

DA3 native results on the same ten-image sample, compared in the model frame:

| Precision | Producer | Status | Depth Spearman vs baseline | Scale-shift AbsRel | Camera rotation | Translation L2 | Intrinsics relative error |
|---|---|---|---:|---:|---:|---:|---:|
| F32 | converter | pass | 0.999988 vs reference | 0.000741 vs reference | 0.0055 deg vs reference | 4.07e-6 vs reference | 0.000197 vs reference |
| F16 | native-quantizer | pass | 0.9999999 vs F32 | 0.000094 vs F32 | 0.0060 deg vs F32 | 6.62e-5 vs F32 | 0.000071 vs F32 |
| Q8_0 | native-quantizer | reject | 0.999926 vs F32 | 0.00303 vs F32 | 0.2232 deg vs F32 | 0.001162 vs F32 | 0.001926 vs F32 |
| Q4_K | native-quantizer | reject | 0.998858 vs F32 | 0.01468 vs F32 | 1.4315 deg vs F32 | 0.01530 vs F32 | 0.008866 vs F32 |

F32 and F16 pass all depth and camera gates. Q8_0 fails the translation bar;
Q4_K fails rotation and translation. The reference took 11.11 s and peaked at
1,061,644 KiB RSS. Native per-image timings, conversion and quantization
timings, and memory measurements are in the result JSON. Acceptance used
`--no-publish`, so no bundle was published.

## SAM3 remains blocked

`mask-generation` does not accept sam3cpp. The blocker is:

> official facebook/sam3 weights are gated under the SAM License; no ungated
> source with verifiable conversion provenance; sam3cpp not accepted

The official `facebook/sam3` weights require gated access under the SAM
License. The audited ungated MLX-layout source did not provide independently
verifiable conversion provenance. No SAM3 source or native acceptance is
approved by this milestone.
