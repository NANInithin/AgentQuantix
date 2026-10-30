# Forecasting status

**State: blocked and not offered.** This status records a feasibility
investigation only; no model reached native inference or forecasting metrics.

## Licence and source

The `liodon-ai/timeseries.cpp` repository declares no licence at the settled
pin `70d7a436d7833cc41c09804823a2b99d159222ee` or at the audited HEAD. No
license file, README/pyproject licence field, or SPDX header was found. The
repository fetches a moving ggml master unless externally pinned. The audited
build used ggml `da8e2a54b919557b816b1c2beb432ccb766f7ce0`.

The Chronos T5 tiny and TimesFM 1.0 200M weights declare Apache-2.0, but that
does not license the engine source.

## Pinned feasibility results

| Model / precision | Result |
|---|---|
| Chronos T5 tiny F32 | Blocked at load: the pinned GGUF loader rejects a tensor name longer than 64 characters (`68 >= 64`). The local wrapper produced a 90-tensor, 33.6 MB GGUF. |
| Chronos T5 tiny F16 | Conversion completed through the spike wrapper; native inference was not attempted because of the same loader limit. |
| Chronos Q8_0 / Q4_K / Q4_K_M | Not attempted; there was no loadable F32 baseline and the converter does not produce these types. |
| TimesFM 1.0 200M F32 | Conversion blocked: the local PyTorch snapshot has no recognized `model_type` in `config.json`. No GGUF was produced. |
| TimesFM F16 or quantized types | Not attempted; no F32 baseline was produced. |

No reference or native forecasts, parity metrics, holdout MASE/WQL, precision
comparisons, or accepted precisions were produced. The reported conversion
timings and peak RSS do not constitute model acceptance.

## Evidence

The full feasibility report and JSON are in
`/home/ubuntu/spike/0.5.9/REPORT.md` and
`/home/ubuntu/spike/0.5.9/results/results.json`. Detailed failures are in
`/home/ubuntu/spike/0.5.9/logs/native-chronos-f32-etth1-w0.log`,
`/home/ubuntu/spike/0.5.9/logs/convert-timesfm-f32.log`, and
`/home/ubuntu/spike/0.5.9/logs/convert-timesfm-f32.time`.

The TaskSpec is deferred to `post-0.6.0` until the engine has a licence and the
pinned upstream can load Chronos and convert TimesFM. No capability route or
forecasting implementation is offered.
