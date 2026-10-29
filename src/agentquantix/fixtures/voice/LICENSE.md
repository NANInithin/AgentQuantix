# Voice fixture licensing

The text prompts in `tts.json` were written for AgentQuantix and are dedicated
to the public domain under CC0 1.0.

`jfk.wav` is the sample distributed by the official whisper.cpp repository. It
is an excerpt from President John F. Kennedy's 1961 inaugural address, a work
of the United States federal government in the public domain. The expected
transcript is stored in `asr.json` so CI comparisons are deterministic.

The fixture is used only for conversion, audio-header, smoke-inference, and
word-error-rate tests. Larger quality corpora remain opt-in.

The synthetic separation mix, stems, and pitch-shifted VC reference are
generated from the public-domain JFK sample by
`scripts/generate_audio_task_fixtures.py`. The derived test fixtures are
dedicated to the public domain under CC0 1.0.
