# 0.5.2 early engine audit — 25 October 2026

The 0.5.2 exit evidence asks for audited SHA, build, API and source contracts
for every new engine, early enough to surface late blockers. This records what
was actually observed on the development machine. It is **not** acceptance: no
model was converted or executed by any new engine.

Machine: Windows, Visual Studio 18 (MSVC 14.51), CMake 4.4, CUDA 13.3 present,
no Ninja. Audit builds were CPU-only, Release, Visual Studio generator, in the
temporary tree `INF/temp/aqx-audits/` (not registered with the runtime manager).

## Resolved revisions

Per the approved policy: latest stable release, otherwise the default branch.

| Engine | Upstream | License | Selected revision | Basis |
|---|---|---|---|---|
| vision.cpp | Acly/vision.cpp | MIT | `b766b0acc30f6706f1bfe46769635e361172ce96` (v0.3.0) | stable release; `main` is ahead at `26a75291…` |
| ultralytics-ggml | Asher-1/ultralytics-ggml | AGPL-3.0 | `59792c1d1b10a87172c1fdf6eceaa19262c1f11e` | no release; `main` (designated original despite GitHub's fork flag) |
| rf-detr.cpp | localai-org/rf-detr.cpp | Apache-2.0 | `98d0f381b832ef08a608b65c7dd78db066ed8b9a` | no release; `main` |
| depth-anything.cpp | localai-org/depth-anything.cpp | MIT | `14f7461d1f704761a038ac9f50dbde8fdb7275e2` | no release; `master` |
| clip.cpp | monatis/clip.cpp | MIT | `913458d5d1c9238380a0b0826cd1e71c8828a82d` | no release; `main` |
| sam3cpp | ropoctl/sam3cpp | Apache-2.0 | `802f33e3c1d4b892063d069f56816b8b899abcb5` | no release; `master` |
| timeseries.cpp | liodon-ai/timeseries.cpp | **none declared** | `70d7a436d7833cc41c09804823a2b99d159222ee` | no release; `master` |

## Source and converter contracts

| Engine | ggml source | Converter | Converter output types | Notes |
|---|---|---|---|---|
| vision.cpp | submodule `cc98a9d` (vision branch) | `scripts/convert.py` | f32, f16 only | converter requires Python ≥ 3.12; isolate its environment |
| ultralytics-ggml | `cpp_ggml/third_party/ggml` `8599e0e` | `cpp_ggml/scripts/convert_yolo_to_gguf.py` (+ CLIP/MobileCLIP/YOLOE encoders) | f32, f16, q8_0 | open-vocabulary routes need encoder companions |
| rf-detr.cpp | `third_party/ggml` `e705c5f` | `scripts/convert_rfdetr_to_gguf.py` | f16, f32, q4_0, q4_1, q5_0, q5_1, q8_0 | default loads via the `rfdetr` package; `.pth` pickle fallback requires an explicit trust flag |
| depth-anything.cpp | `third_party/ggml` `eced84c` | `convert_da2/da3/mono/nested_to_gguf.py` | native quantizer: f16, q8_0, q4_k | DA2 converter takes `--max-depth`; nested/metric variants are separate converters |
| clip.cpp | submodule `c3ae31e` | `models/convert_hf_to_gguf.py` | f32, f16; native `quantize`: q4_0, q4_1, q5_0, q5_1, q8_0 | text-only / vision-only exports available |
| sam3cpp | vendored `third_party/ggml` | `tools/convert_mlx_sam3_to_gguf.py` | — | input is MLX-layout safetensors + index; official-source preparation is unestablished |
| timeseries.cpp | **FetchContent `ggml` master (moving)** | `convert/convert_chronos.py`, `convert_timesfm.py` | f32, f16, q8_0, q4_k, q4_k_m | README marks Chronos T5 tiny–large and TimesFM 1.0 200M as supported |

## Windows CPU build results

| Engine | Configure | Build | First blocking error |
|---|---|---|---|
| vision.cpp v0.3.0 | ok | **failed** | link: unresolved `visp::depthany_process_output` in `vision-cli` |
| ultralytics-ggml | ok | **failed** | `clock_gettime`/`CLOCK_MONOTONIC` (POSIX) in `common.cpp`; missing `<array>` include in `tracker.cpp` |
| rf-detr.cpp | ok | **failed** | GCC `__attribute__((format))` in `src/common.hpp` under MSVC |
| depth-anything.cpp | ok | **failed** | OpenMP array reduction in `icp.cpp`; `M_PI` without `_USE_MATH_DEFINES` |
| clip.cpp | **failed** | — | bundled ggml requires `cmake_minimum_required < 3.5`, rejected by CMake 4 |
| sam3cpp | **failed** | — | `find_package(ZLIB REQUIRED)`; no ZLIB on this machine |
| timeseries.cpp | ok (ggml pinned to `da8e2a5` via `FETCHCONTENT_SOURCE_DIR_GGML`) | **failed** | GCC-only `-Wextra` passed to MSVC |

## Confirmed upstream cautions

- **sam3cpp** forces `GGML_CUDA`, `GGML_VULKAN`, `GGML_BLAS` and `GGML_OPENMP`
  OFF in `CMakeLists.txt`; only Metal is optional. CUDA/Vulkan cannot be
  advertised from README text. ZLIB is a required installation dependency.
- **timeseries.cpp** declares no license and fetches a moving ggml branch.
  Builds must pin ggml through the lifecycle manager, and code redistribution
  terms must be resolved with the user before any release relies on it.
- **ultralytics-ggml** is AGPL-3.0; record that separately from model weights.
- **vision.cpp** has no low-bit conversion path; preserve floating precision.

## Consequences for the roadmap

None of the seven engines builds unmodified with MSVC on this machine. Each
milestone (0.5.3–0.5.9) must therefore either validate on Linux first, add a
reviewed, minimal portability patch applied by the runtime builder (recorded in
the build identity), or use an alternative compiler, before any Windows cell is
claimed. clip.cpp additionally needs `CMAKE_POLICY_VERSION_MINIMUM=3.5` or a
newer ggml; sam3cpp needs a ZLIB provider. These are scope risks to raise with
the user before each milestone starts, not reasons to drop an engine silently.
