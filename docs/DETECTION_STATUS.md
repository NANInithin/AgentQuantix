# Detection task status

## Implemented routes

| Task | Catalog family | Engine | Status |
|---|---|---|---|
| Object detection | `rfdetr-nano` | `rf-detr.cpp` | Rejected on this sample; no precision passes |
| Instance segmentation | `rfdetr-seg-nano` | `rf-detr.cpp` | F32 and F16 pass |
| Keypoint detection | `yolo26n-pose` | `ultralytics-ggml` | F32, F16 and Q8_0 pass |
| Image classification | `yolo26n-cls` | `ultralytics-ggml` | F32, F16 and Q8_0 pass narrow parity |

Plans route by family, not engine-list order, and use `<engine>:<family>` run
targets. Source weights are downloaded from catalog URLs and SHA-256/size
checked before either converter or reference model loading. RF-DETR uses its
restricted checkpoint loader; `--trust-checkpoint` is never passed.

## Evaluation contract

Evaluation uses ten COCO val2017 images with license ids 4, 7 or 8, downloaded
at run time with pinned SHA-256 values. The small annotation subset is
attributed under CC BY 4.0; per-image license and URL details are in
[`fixtures/detection/LICENSE.md`](../src/agentquantix/fixtures/detection/LICENSE.md).
Results are a small-sample estimate, not broad model validation.
The `rfdetr-nano` family remains in the catalog so the run-time quality gate
can decide on each evaluation. It is not an accepted object-detection family
in 0.5.7 on this sample.

D(0.25) retains reference and native detections scoring at least 0.25, performs
class-agnostic greedy matching by descending box IoU, and accepts matches at
IoU 0.5. F32 gates require recall and precision at least 0.95, class agreement
at least 0.98, mean matched box IoU at least 0.90, and absolute GT mAP50 delta
at most 0.02. Segmentation adds matched mask IoU at least 0.85 and mask mAP50
delta at most 0.02. Pose adds mean matched keypoint error at most 2 pixels.
Lower precisions are compared with native F32 and must meet the same D(0.25)
bars and remain within 0.03 mAP50 (and mask mAP50) of F32. Classification
requires top-1 agreement at least 0.95 and mean top-5 overlap at least 0.90;
probability MAE is recorded.

## Spike evidence and limitations

The settled spike under `/home/ubuntu/spike/0.5.7/` reports:

| Family/task | Spike recommendation | Evidence |
|---|---|---|
| YOLO26n detection | Reject | F32 D(0.25) recall 0.8889, native precision 0.9333 |
| YOLO26n instance segmentation | Reject | Matched mask IoU 0.9013; mask mAP50 delta 0.0286 |
| YOLO26n pose | Accept for tested sample | F32 keypoint error 0.001 px |
| YOLO26n classification | Narrow parity only | Top-1 agreement 1.0 |
| YOLO26 semantic segmentation | Reject | Pixel agreement 0.9631; mIoU 0.7778 |
| YOLO26 depth | Reject reference parity | Spearman 0.9861; scale-shift AbsRel 0.1284 |
| YOLOv8s-World | Reject | Native precision 0.9024 |
| Ultralytics OBB | Not evaluated | No suitable permissively licensed aerial sample |
| RF-DETR nano detection | Accept after default-resolution follow-up | F32 recall/precision 0.9661/0.9828; mean IoU 0.9976 |
| RF-DETR seg-nano | Accept after default-resolution follow-up | F32 recall/precision 0.9658/0.9826; box/mask IoU 0.9883/0.9818 |

## Native acceptance

Linux CPU acceptance used runtime revisions
`59792c1d1b10a87172c1fdf6eceaa19262c1f11e` and
`98d0f381b832ef08a608b65c7dd78db066ed8b9a`, with `--no-publish --yes`.
Results use the ten pinned COCO val2017 images above. F32 is compared with the
PyTorch reference; lower precisions are compared with native F32.

| Family | Precision | Result | Key D(0.25)/quality evidence |
|---|---|---|---|
| RF-DETR nano | F32 | Reject | Recall 0.9478; precision 0.9909; class agreement 1.0; mean IoU 0.9967; mAP50 0.7514 vs reference 0.7498 |
| RF-DETR nano | F16, Q8_0 | Reject | Each matches F32 closely, but a lower precision cannot pass when its F32 baseline fails reference parity |
| RF-DETR nano | Q4_0, Q4_1, Q5_0, Q5_1 | Reject | Baseline failed; each also fails at least one lower-precision gate |
| RF-DETR seg-nano | F32, F16 | Pass | F32 recall/precision 0.9735/0.9910; box/mask IoU 0.9863/0.9895; mask mAP50 0.5441 vs 0.5566 reference |
| RF-DETR seg-nano | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | Reject | Fail at least one fixed lower-precision gate |
| YOLO26n pose | F32, F16, Q8_0 | Pass | F32 recall/precision 1.0/1.0; mean IoU 0.9993; keypoint error 0.521 px |
| YOLO26n classification | F32, F16, Q8_0 | Narrow parity pass | F32 top-1 1.0 and top-5 overlap 0.92; Q8_0 overlap 0.90 |

RF-DETR nano F32 misses the 0.95 reference-recall gate; its matched-box
quality and COCO mAP50 delta pass. The run records `baseline_failed: true`
with `ref_recall<0.95`; every precision is quality-rejected, the run status is
`rejected`, and no bundle is produced. Thus 0.5.7 has no accepted
object-detection family on this sample. The spike follow-up parity result in
the preceding table did not replace the release gate's evaluation; the native
run-time result is authoritative for release status. Classification is only
a ten-image reference-parity result, not a classification-quality evaluation.
YOLO26n pose's COCO mAP50 is low on this selected subset, so the result is
limited to the stated reference-parity gates.

Native results and per-run wall/RSS records are retained under
`/home/ubuntu/spike/acceptance/detect-*-results.json`; detailed native logs
and intermediate outputs remain under `~/.agentquantix/temp/aqx-detection-runs/`.
The first RF-DETR attempt is preserved with its original failed result and
logs; a missing timing directory and evaluator dependency were fixed before
the successful rerun. These runs do not publish bundles.

## Licensing

YOLO model weights and `ultralytics-ggml` are AGPL-3.0. RF-DETR nano and
seg-nano weights and the `rf-detr.cpp` engine are Apache-2.0. Model cards record
weight and engine licenses separately, the source URL/SHA, GGUF labels and
runtime/converter provenance. COCO annotations are CC BY 4.0; downloaded
images are not vendored.
