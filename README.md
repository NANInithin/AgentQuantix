# AgentQuantix

Use one entry point for supported models:

```sh
aqx run org/model
```

AgentQuantix resolves the task and native engine from model metadata and
engine catalogs, shows the release plan, then runs only after approval. Use
`--task` or `--family` when a route is ambiguous or a repository is custom.

## What runs where

The evidence-backed platform scope is **Linux x86_64 CPU**. Acceptance applies
only to the named task, family, precision, and platform in the
[support matrix](docs/SUPPORT_MATRIX.md); a catalog entry alone is not an
acceptance claim.

## Install

For development and the media and detection helpers:

```sh
python -m pip install -e '.[dev,media,detection]'
```

Other optional extras include `voice`, `retrieval`, `multimodal`, `convert`,
and `media-quality`. Install the extras required by the task and evaluator you
use. Converter environments needed by a task are created automatically and
kept separate from the AgentQuantix environment.

The source package version is still **0.4.5**. The 0.6.0 notes are a draft,
not a release announcement.

## Approval and publication

- Approve the model before running it. `--yes` skips the final model
  confirmation only; it does not approve a runtime update.
- Pass `--no-publish` to keep a run local. `--publish` does not bypass task
  quality checks or required human review.
- TTS needs a listening review; video needs playback review; audio-video needs
  both. Use `aqx review` to record a required review, then `aqx publish` to
  publish the reviewed candidate.
- `aqx runs` lists resumable run IDs, and `aqx run --resume <id>` resumes the
  frozen plan and runtime.

See [USAGE.md](USAGE.md) for pipeline commands, [release validation](docs/RELEASE_VALIDATION.md)
for native acceptance requirements, and the [0.6.0 draft notes](docs/RELEASE_NOTES_0_6_0.md)
for unreleased milestone summaries.
