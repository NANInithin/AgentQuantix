# Experimental media workflow (v0.5.0 development)

This implements the separate stable-diffusion.cpp track specified in
STABLE_DIFFUSION_CPP_RELEASE_PLAN.md. The source release remains 0.4.5 until
the complete native acceptance matrix and human MiniMax-H3 review pass.

```sh
pip install '.[media]'
aqx media backends
aqx media backends --build --compute-backend cpu
aqx media inspect org/model
aqx media plan org/model --quant q4_0 --sequential
aqx media run org/model --quant q4_0 --sequential
```

No quality-profile JSON is needed for a local manual test. With no calibrated
profile, the run reports `structural-only`, keeps its samples for your inspection,
and blocks publication. A plan can also report publication-only license blockers
without preventing local testing. Unresolved or ambiguous component layouts still
need an explicit selection, for example `--component vae=path/to/vae.safetensors`
or `--component clip_l=publisher/encoders::clip_l.safetensors`.

Use `--width`, `--height`, `--steps`, `--seed`, `--frames`, `--fps`, `--sampler`,
`--scheduler`, and `--cfg-scale` to choose model-appropriate settings; all are
frozen into the displayed plan. It also lists the number of generations per
policy, including repeats. Conditioned tests can use the packaged licensed
fixtures with `--init-image`, `--end-image`, `--ref-image`, or `--ref-audio`.

For a scored, reproducible release with a saved plan:

```sh
aqx media plan org/model --family flux --contract source-contract.json --profile quality-profile.json --quant q4_0 --sequential --output approved-plan.json
aqx media run org/model --plan approved-plan.json
aqx media review RUN_ID --accept q4_0 --reviewer 'Your name'
aqx media publish RUN_ID --quant q4_0
aqx media verify downloaded-bundle/manifest.json
```

Video reviews additionally require `--watched-video` (playback and extracted
frames); audio-video additionally requires `--listened-audio`. A rejected or
modified candidate cannot use an earlier accepted review. `run` builds and
scores locally by default. Publication is a separate explicit command.

The native type names are authoritative. For example, `q4_K` is a native type;
`Q4_K_M` must not silently borrow llama.cpp's meaning. The read-only offline
catalog reports unprobed type candidates. A successful native executable probe
overrides that fallback, and each real component must still convert and load.

## Source contracts

An exact publisher/architecture match and an unambiguous upstream example can
resolve component roles automatically. Other layouts return structured
questions. A source contract may live in the source repository as
`aqx-media-source.json`, or be supplied explicitly with `--contract`:

```json
{
  "task": "text-to-image",
  "required_roles": ["diffusion_model", "vae", "clip_l", "t5xxl"],
  "evidence": {
    "native_recipe": "docs/flux.md",
    "component_metadata": "Describe the exact inspected architecture/layout evidence"
  },
  "components": [
    {"role": "diffusion_model", "source_path": "flux1-dev.safetensors", "quantizable": true},
    {"role": "vae", "source_path": "ae.safetensors", "precision": "f16", "quantizable": false},
    {"role": "clip_l", "source_repo": "publisher/encoders", "source_path": "clip_l.safetensors", "quantizable": true},
    {"role": "t5xxl", "source_repo": "publisher/encoders", "source_path": "t5xxl_fp16.safetensors", "quantizable": true}
  ],
  "settings": {"width": 256, "height": 256, "steps": 4, "seed": 42, "sampler": "euler", "scheduler": "discrete"}
}
```

Paths and repository names in this example are illustrative, not a validated
or redistributable model recipe. The plan resolves each actual repository's
immutable revision, file presence, size, hash when available, and license.
Unknown/restricted licenses block publication pending review. Do not override
unknown licenses by guessing a permissive license.

For conditioned routes, `conditioning` maps native inputs such as `init_image`,
`end_image`, `ref_image`, and `ref_audio` to packaged fixture paths such as
`fixtures/reference.png`, `fixtures/last.png`, or `fixtures/reference.wav`.
Only known licensed fixtures are copied automatically. Unknown user-reference
assets require an explicit source contract and redistribution evidence.

MiniMax-H3 uses separate diffusion, truncated Qwen3-VL text encoder, video VAE,
and audio VAE components. Its generated contract records the 50-layer/no-final-
norm/three-DeepStack constraints. An arbitrary Qwen3-VL checkpoint does not
satisfy those constraints. FL2VA and Ref2VA are variant-specific; incompatible
conditioning combinations must remain blocked. Video-only is an explicit task
with no required audio VAE; it is never accepted as an audio-video release.

## Quality profiles

Structural image checks use the `media` extra. Video/audio validation also
requires `ffmpeg` and `ffprobe` on PATH. Heavy evaluators are optional to
install, but their absence is reported and blocks automated quality approval.

A quality profile names `family`, `task`, `calibration_evidence`, per-metric
minimum `thresholds`, and an `evaluator`. No universal thresholds are invented.
Required metrics are `perceptual_similarity`, `prompt_alignment`, `determinism`,
plus `conditioning_preservation` for conditioned recipes and
`temporal_consistency` for video. The profile must be calibrated for the exact
family/task. Near-threshold candidates and first releases require human review;
this implementation conservatively requires human review for every candidate.

`evaluator.command` is an argument array for a separately installed evaluator.
Its executable is pinned with `sha256`; script/file arguments are pinned by
absolute-path keys in `artifacts`. `name` and immutable `model_revision` identify
the evaluator and its weights. The command receives one extra argument: a JSON
request path containing candidate, baseline, prompt, conditioning, settings,
and model revision. It writes JSON to stdout:

```json
{"model_revision": "EXACT_EVALUATOR_MODEL_REVISION", "metrics": {"perceptual_similarity": 0.0, "prompt_alignment": 0.0, "determinism": 0.0}}
```

Numbers above illustrate the protocol only, not passing thresholds. The
`media-quality` extra provides optional Torch/Transformers helpers; it does not
silently select an evaluator, claim calibration, or replace unavailable metrics.

## Publication and reproducibility

Plans list the exact per-policy target repositories. Each has a complete
root manifest, all components, quality results, fixture metadata/license, review,
and a card submitted in one commit. Size-only remote matches are re-uploaded;
post-commit verification checks all hashes, downloading small non-LFS files
when a remote SHA-256 is unavailable. A card uses only the verified recipe and
actual component precisions.

Source shards are released only after successful conversion, hashing and an
atomic journal update. Retained companions are shared through hard links where
the filesystem supports them. Local candidates are retained for review until
publication; the disk estimate includes all candidates. Sequential operation
holds one source component at a time. An interrupted final artifact is never
trusted solely because its path exists.
