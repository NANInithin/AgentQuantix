# Release validation

The source package is 0.4.5. Media development is unreleased until all v0.5.0
acceptance criteria in STABLE_DIFFUSION_CPP_RELEASE_PLAN.md pass.

## Offline checks

```sh
python -m pytest -q
ruff check --select E9,F63,F7,F82,F401 .
python scripts/sync_adapters.py
python scripts/sync_media_capabilities.py /path/to/pinned/stable-diffusion.cpp --check
python -m build --outdir dist/0.4.5
python -m twine check dist/0.4.5/*
```

Always build into a version-specific directory. Existing `dist/0.2.0` artifacts
do not describe current source. Do not publish wheels selected by an unrestricted
glob from a workstation containing previous builds. CI starts from a clean checkout.

## Native acceptance

The default tests are offline. They are not evidence that native model inference
has passed. Run `scripts/native_acceptance.py` separately on both Linux and
Windows, using already available, licensed small local models:

```sh
python scripts/native_acceptance.py native-recipes.json --output native-results.json
```

The JSON spec contains `recipes`. Every recipe has `name`, `track`, `commands`
(arrays of arguments, no shell), `files` (absolute input/binary path to SHA-256),
`cwd`, `output`, and optionally `timeout_seconds`. Required tracks are `text`,
`llama-tts`, `audio.cpp-tts`, `audio.cpp-asr`, and `whisper-asr`.
Text recipes set `expected_text`; ASR recipes set `expected_transcript` and
`max_wer`. Each recipe should include conversion/quantization and inference.
An incomplete track matrix exits nonzero and lists missing tracks.

For v0.5.0, add `--require-media`. Required additional tracks are
`media-single-image`, `media-multi-image`, `media-edit`, `media-video`, and
`media-minimax-h3`. Media entries also require `runtime_recipe`, `baseline`, and
`quality_profile`, using the same pinned quality protocol as normal releases.
The MiniMax-H3 human review and a re-downloaded bundle/card-command run remain
separate required evidence. Never substitute mocked native executables for
this acceptance evidence.

## Publishing

The publishing workflow remains explicitly manual. It validates the requested
version against package source before uploading. Configuring PyPI trusted
publishing, creating Git tags/releases, and actual publication are external
release operations; editing source does not establish that those happened.
