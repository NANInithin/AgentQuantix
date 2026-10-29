import shutil
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentquantix import actions, cli, runtime_control, voice
from agentquantix.agent import tools
from agentquantix.pipeline import voice_release
from agentquantix.tasks import TASKS, route


def test_audio_cpp_task_tracks_are_derived_from_spec_tasks():
    mapping = {"sep": voice.SEP, "vc": voice.VC, "align": voice.ALIGN}
    for spec in voice.AUDIOCPP_SPECS:
        for task, track in mapping.items():
            if task in spec.get("tasks", []):
                backend = voice.backend_for(
                    f"audio.cpp:{spec['family']}", track=track)
                assert backend is not None
                assert backend.family == spec["family"]
                assert backend.runtime_task == task
                assert backend.track == track
    assert voice.backend_for("ResembleAI/chatterbox").track == voice.TTS
    assert voice.backend_for(
        "ResembleAI/chatterbox", track=voice.VC).runtime_task == "vc"


def test_audio_task_routing_and_catalog_priority():
    expected = [
        ("audio.cpp:bs_roformer", "audio-source-separation", voice.SEP),
        ("audio.cpp:meanvc2", "voice-conversion", voice.VC),
        ("audio.cpp:qwen3_forced_aligner", "forced-alignment", voice.ALIGN),
    ]
    for repo, task, track in expected:
        result = route(repo, task=task)
        assert result.ready and result.pipeline == "voice"
        assert result.engine == "audio.cpp"
        assert result.task == task
        assert voice.backend_for(repo, track=track)

    catalog = route("audio.cpp:bs_roformer",
                    metadata={"pipeline_tag": "audio-to-audio"})
    assert catalog.task == "audio-source-separation"
    ambiguous = route(
        "org/unknown", metadata={"pipeline_tag": "audio-to-audio", "tags": []})
    assert ambiguous.blocker == (
        "audio-to-audio is ambiguous; pass --task audio-source-separation "
        "or --task voice-conversion")
    for task, note in (
        ("audio-generation",
         "no permissively licensed audio.cpp generation package under the evaluation size limit was natively accepted"),
        ("audio-classification",
         "audio.cpp exposes no classification task at the pinned revision"),
        ("voice-activity-detection",
         "audio.cpp VAD models are bundled runtime assets, not releasable sources"),
    ):
        assert TASKS[task].milestone == "post-0.6.0"
        assert TASKS[task].note == note
        assert note in route("org/model", task=task).blocker


def test_new_audio_cpp_command_shapes(monkeypatch, tmp_path):
    monkeypatch.setattr(voice, "_audiocpp_threads", lambda: "4")
    sep = voice.backend_for("audio.cpp:bs_roformer", track=voice.SEP)
    vc = voice.backend_for("audio.cpp:meanvc2", track=voice.VC)
    align = voice.backend_for(
        "audio.cpp:qwen3_forced_aligner", track=voice.ALIGN)
    sep_command = voice.audiocpp_sep_command(
        "cli", sep, "model.gguf", "mix.wav", tmp_path / "stems")
    assert sep_command[:4] == ["cli", "--task", "sep", "--family"]
    assert ["--backend", "best"] == sep_command[
        sep_command.index("--backend"):sep_command.index("--backend") + 2]
    assert ["--threads", "4"] == sep_command[
        sep_command.index("--threads"):sep_command.index("--threads") + 2]
    assert "--metrics" in sep_command and "--seed" not in sep_command
    assert sep_command[sep_command.index("--out-dir") + 1] == str(tmp_path / "stems")

    vc_command = voice.audiocpp_vc_command(
        "cli", vc, "model.gguf", "in.wav", "ref.wav", "out.wav")
    assert vc_command[vc_command.index("--seed") + 1] == "42"
    assert "--voice-ref" in vc_command and "--metrics" in vc_command
    align_command = voice.audiocpp_align_command(
        "cli", align, "model.gguf", "in.wav", "hello world", "en", "words.json")
    assert align_command[align_command.index("--text") + 1] == "hello world"
    assert align_command[align_command.index("--language") + 1] == "en"
    assert align_command[align_command.index("--words-out") + 1] == "words.json"
    assert "--metrics" in align_command


def test_separation_stem_mapping_sisdr_and_cwd(tmp_path, monkeypatch):
    backend = voice.backend_for("audio.cpp:bs_roformer", track=voice.SEP)
    fixtures = voice_release.config.VOICE_FIXTURES_DIR / "separation"
    model = tmp_path / "Q8_0.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"model")
    audio_dir = tmp_path / "audio.cpp"
    out_dir = model.parent / "quality" / model.stem / "stems"
    seen = {}

    def fake_run(command, *, timeout, cwd):
        seen.update(command=command, cwd=cwd)
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixtures / "speech.wav", out_dir / "vocals.wav")
        shutil.copyfile(fixtures / "accompaniment.wav", out_dir / "drums.wav")
        return {"elapsed_seconds": 1.25, "stdout": ""}

    monkeypatch.setattr(voice, "run_checked", fake_run)
    result = voice_release.score_audiocpp_separation(
        backend, model, Path("cli"), voice_release.VoiceReleaseOptions(),
        audio_dir)
    assert seen["cwd"] == audio_dir
    assert seen["command"] == voice.audiocpp_sep_command(
        Path("cli"), backend, model, fixtures / "mixture.wav", out_dir)
    assert result["output_paths"]["vocal"].endswith("vocals.wav")
    assert len(result["output_paths"]["accompaniment"]) == 1
    assert result["vocal_sisdr_db"] > 40
    assert result["accompaniment_sisdr_db"] > 40
    assert result["elapsed_seconds"] == 1.25


def test_mono_audio_resamples_with_voice_helpers(tmp_path):
    source = tmp_path / "stereo.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(struct.pack("<8h", 1000, -1000, 2000, -2000,
                                    3000, -3000, 4000, -4000))
    samples, facts = voice_release._mono_audio(source, 16000)
    assert len(samples) == 8
    assert facts["sample_rate"] == 16000
    assert facts["source_sample_rate"] == 8000
    assert facts["finite"]


@pytest.mark.parametrize("improvement,passed", [(6.0, True), (5.999, False)])
def test_separation_baseline_gate_edge(improvement, passed):
    quality = {"vocal_improvement_db": improvement,
               "accompaniment_improvement_db": 6.0}
    result = voice_release._separation_quality_gate(quality, {}, True)
    assert result["passed"] is passed


@pytest.mark.parametrize("delta,passed", [(-1.0, True), (-1.001, False)])
def test_separation_precision_gate_edge(delta, passed):
    baseline = {"vocal_sisdr_db": 10.0, "accompaniment_sisdr_db": 8.0,
                "vocal_improvement_db": 6.0,
                "accompaniment_improvement_db": 6.0}
    candidate = {"vocal_sisdr_db": 10.0 + delta,
                 "accompaniment_sisdr_db": 8.0}
    assert voice_release._separation_quality_gate(
        candidate, baseline, False)["passed"] is passed


@pytest.mark.parametrize("wer,passed", [(0.35, True), (0.350001, False)])
def test_vc_baseline_gate_edge(wer, passed):
    assert voice_release._vc_quality_gate(
        {"roundtrip_wer": wer, "waveform_valid": True}, {}, True)[
            "passed"] is passed


@pytest.mark.parametrize("wer,passed", [(0.45, True), (0.450001, False)])
def test_vc_precision_gate_edge(wer, passed):
    result = voice_release._vc_quality_gate(
        {"roundtrip_wer": wer, "waveform_valid": True},
        {"roundtrip_wer": 0.35, "waveform_valid": True}, False)
    assert result["passed"] is passed


@pytest.mark.parametrize("valid,passed", [(True, True), (False, False)])
def test_vc_waveform_gate(valid, passed):
    result = voice_release._vc_quality_gate(
        {"roundtrip_wer": 0.1, "waveform_valid": valid}, {}, True)
    assert result["passed"] is passed


@pytest.mark.parametrize("delta,passed", [(0.05, True), (0.050001, False)])
def test_alignment_mean_boundary_gate_edge(delta, passed):
    baseline = {"words": [{"start": 1.0, "end": 2.0}], "valid": True}
    candidate = {"words": [{"start": 1.0 + delta, "end": 2.0 + delta}],
                 "valid": True}
    result = voice_release._alignment_quality_gate(
        candidate, baseline, False)
    assert result["passed"] is passed


@pytest.mark.parametrize("delta,passed", [(0.20, True), (0.200001, False)])
def test_alignment_max_boundary_gate_edge(delta, passed):
    baseline = {"words": [
        {"start": 1.0, "end": 1.2},
        {"start": 2.0, "end": 2.2},
        {"start": 3.0, "end": 3.2},
        {"start": 4.0, "end": 4.2},
    ], "valid": True}
    candidate = {"words": [
        {"start": 1.0, "end": 1.2 + delta},
        {"start": 2.0, "end": 2.2},
        {"start": 3.0, "end": 3.2},
        {"start": 4.0, "end": 4.2},
    ],
                 "valid": True}
    result = voice_release._alignment_quality_gate(
        candidate, baseline, False)
    assert result["passed"] is passed


def test_alignment_invalid_boundary_fails_gate():
    result = voice_release._alignment_quality_gate(
        {"words": [{"start": -0.1, "end": 2.0}], "valid": False},
        {"words": [{"start": 1.0, "end": 2.0}], "valid": True}, False)
    assert result["passed"] is False


def _fake_audiocpp_runtime(monkeypatch, tmp_path):
    audio_dir = tmp_path / "audio.cpp"
    audio_dir.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(
        voice_release, "ensure_audiocpp_tools",
        lambda: (audio_dir, Path("cli"), Path("converter")))
    monkeypatch.setattr(
        voice_release, "work_dir", lambda _repo: tmp_path / "work")
    monkeypatch.setattr(
        "agentquantix.runtime_control.prepare_voice_runtimes",
        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        voice_release, "_audiocpp_package_provenance",
        lambda *_args: ("audio-cpp/audio.cpp-gguf", "package-revision"))
    monkeypatch.setattr(
        voice_release, "feasibility", lambda *_args: {})
    return audio_dir, models


def test_generic_package_release_records_package_role_and_card(
        tmp_path, monkeypatch):
    audio_dir, models = _fake_audiocpp_runtime(monkeypatch, tmp_path)
    captured = []

    def install(_audio_dir, _backend, _repo, model_root, quant):
        model_root.mkdir(parents=True, exist_ok=True)
        path = model_root / f"{quant}.gguf"
        path.write_bytes(b"model")
        return path, (path,), "MeanVC2-GGUF"

    monkeypatch.setattr(voice_release, "install_audiocpp_package", install)
    monkeypatch.setattr(
        voice_release, "source_metadata",
        lambda *_args, **_kwargs: {"license": "Apache-2.0"})
    monkeypatch.setattr(
        voice_release, "feasibility",
        lambda bundle, _quality: captured.append(bundle) or {})
    monkeypatch.setattr(
        voice_release, "score_audiocpp_vc",
        lambda _backend, model, *_args: {
            "roundtrip_wer": 0.10 if "F32" in model.name else 0.15,
            "waveform_valid": True, "output_path": str(model),
            "elapsed_seconds": 1.0})
    result = voice_release.run_audiocpp_vc_release(
        "audio.cpp:meanvc2", "org/meanvc2",
        voice_release.VoiceReleaseOptions(
            quants=["F32", "Q4_K"], publish=False, family="meanvc2"))
    assert result["baseline_quant"] == "F32"
    assert all(item["quality"]["task_gate"]["passed"]
               for item in result["results"].values())
    bundle = captured[0]
    assert bundle.primary.role == "vc-model"
    assert bundle.source_repo == "audio-cpp/audio.cpp-gguf"
    assert bundle.source_revision == "package-revision"
    card = voice.render_model_card(bundle, "org/meanvc2")
    assert "prebuilt audio.cpp package; not converted by AgentQuantix" in card
    assert "MeanVC2-GGUF" in card and "package-revision" in card
    assert "speaker similarity not measured" in card
    assert audio_dir.is_dir()


def test_generic_conversion_release_records_source_and_alignment_role(
        tmp_path, monkeypatch):
    audio_dir, models = _fake_audiocpp_runtime(monkeypatch, tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    backend = voice.backend_for("org/aligner", track=voice.ALIGN,
                                family="qwen3_forced_aligner")
    monkeypatch.setattr(
        voice_release, "source_metadata",
        lambda *_args, **_kwargs: {"license": "Apache-2.0"})
    monkeypatch.setattr(
        voice_release, "prepare_audiocpp_source",
        lambda *_args: (backend, source, "source-revision", Path("cli"),
                        Path("converter"), audio_dir, models))

    def convert(_source, _backend, _converter, _audio_dir, _models, quant):
        output = models / f"{quant}.gguf"
        output.write_bytes(b"converted")
        return output

    captured = []
    monkeypatch.setattr(voice_release, "convert_audiocpp_model", convert)
    monkeypatch.setattr(
        voice_release, "score_audiocpp_align",
        lambda _backend, model, *_args: {
            "word_count": 22, "valid": True,
            "words": [{"start": 0.0, "end": 0.1}],
            "output_path": str(model), "elapsed_seconds": 0.5})
    monkeypatch.setattr(
        voice_release, "feasibility",
        lambda bundle, _quality: captured.append(bundle) or {})
    result = voice_release.run_audiocpp_align_release(
        "org/aligner", "org/aligner-release",
        voice_release.VoiceReleaseOptions(
            quants=["F16", "Q8_0"], publish=False,
            family="qwen3_forced_aligner"))
    assert result["baseline_quant"] == "F16"
    assert all(item["quality"]["task_gate"]["passed"]
               for item in result["results"].values())
    assert captured[0].primary.role == "alignment-model"
    assert captured[0].source_repo == "org/aligner"
    assert captured[0].source_revision == "source-revision"
    assert "no independent timing reference" in voice.render_model_card(
        captured[0], "org/aligner-release")


def test_publish_is_blocked_when_license_is_unknown(tmp_path, monkeypatch):
    _fake_audiocpp_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        voice_release, "source_metadata",
        lambda *_args, **_kwargs: {"license": None})
    monkeypatch.setattr(
        "agentquantix.runtime_control.prepare_voice_runtimes",
        lambda *_args, **_kwargs: pytest.fail("runtime prepared before license gate"))
    with pytest.raises(voice.VoiceValidationError,
                       match="^licence undetermined; cannot publish$"):
        voice_release.run_audiocpp_sep_release(
            "audio.cpp:bs_roformer", "org/separation",
            voice_release.VoiceReleaseOptions(
                quants=["Q8_0"], publish=True, family="bs_roformer"))


def test_audiocpp_asr_generic_refactor_preserves_release_shape(
        tmp_path, monkeypatch):
    _fake_audiocpp_runtime(monkeypatch, tmp_path)
    models = tmp_path / "models"

    def install(_audio_dir, _backend, _repo, model_root, quant):
        model_root.mkdir(parents=True, exist_ok=True)
        path = model_root / f"{quant}.gguf"
        path.write_bytes(b"model")
        return path, (path,), "Qwen3-ASR-GGUF"

    monkeypatch.setattr(voice_release, "install_audiocpp_package", install)
    monkeypatch.setattr(
        voice_release, "_audiocpp_package_provenance",
        lambda *_args: ("audio-cpp/audio.cpp-gguf", "rev"))
    monkeypatch.setattr(
        voice_release, "score_audiocpp_asr",
        lambda *_args: {"fixtures": 1, "wer": 0.0, "real_time_factor": 0.2,
                        "audio_seconds_per_second": 5.0, "results": []})
    result = voice_release.run_audiocpp_asr_release(
        "audio.cpp:qwen3_asr", "org/asr",
        voice_release.VoiceReleaseOptions(
            quants=["F16", "Q8_0"], publish=False, family="qwen3_asr"))
    assert result["baseline_quant"] == "F16"
    assert result["baseline"]["wer"] == 0.0
    assert result["results"]["F16"]["quality"]["regression"] == {
        "candidate_wer": 0.0, "baseline_wer": 0.0, "regression": 0.0,
        "max_regression": 0.05, "passed": True}
    assert "model_prepare_seconds" not in result["results"]["F16"]["quality"]
    assert result["results"]["F16"]["quality"]["fixtures"] == 1


def test_voice_plan_records_new_task_and_license(monkeypatch):
    monkeypatch.setattr(runtime_control, "selected_runtime", lambda _engine: None)
    monkeypatch.setattr(
        voice_release, "source_metadata",
        lambda *_args, **_kwargs: {
            "revision": "rev", "source_bytes": None, "source_files": None,
            "gated": False, "license": "Apache-2.0"})
    monkeypatch.setattr(voice_release, "audiocpp_source_plan", lambda *_args: [])
    plan = actions.voice_plan(
        "audio.cpp:meanvc2", task="voice-conversion", family="meanvc2",
        hunt_forks=False)
    assert plan["status"] == "ready"
    assert plan["task"] == "voice-conversion"
    assert plan["source_license"] == "Apache-2.0"
    assert "WER" in plan["quality_gate"]


def test_audio_cpp_catalog_revision_guard_only_blocks_new_tracks(monkeypatch):
    runtime = SimpleNamespace(
        revision="41c1f8ffa0d13a1b7e366e30ea5fe5ab8d169d48")
    monkeypatch.setattr(
        runtime_control, "selected_runtime", lambda _engine: runtime)
    plan = actions.voice_plan(
        "audio.cpp:meanvc2", task="voice-conversion", family="meanvc2",
        hunt_forks=False)
    assert plan["status"] == "blocked"
    assert plan["blocker"] == (
        "audio.cpp runtime 41c1f8ffa0d13a1b7e366e30ea5fe5ab8d169d48 does not "
        "match the vendored model-spec catalog v0.8.0; install with aqx runtime "
        "install audio.cpp --persistent --revision "
        "4af143229384fb6da3f373dc87de145ae954609b")
    assert actions._audio_cpp_catalog_blocker(voice.TTS) is None
    assert actions._audio_cpp_catalog_blocker(voice.ASR) is None


def test_audio_cpp_catalog_revision_guard_accepts_tag_and_sha(monkeypatch):
    class Runtime:
        def __init__(self, revision):
            self.revision = revision

    for revision in ("v0.8.0", voice_release.AUDIOCPP_REVISION_SHA):
        monkeypatch.setattr(
            runtime_control, "selected_runtime",
            lambda _engine, value=revision: Runtime(value))
        assert actions._audio_cpp_catalog_blocker(voice.SEP) is None


def test_voice_plan_run_and_cli_dispatch_new_tracks(monkeypatch):
    monkeypatch.setattr(actions.runs, "write_voice_record", lambda *_args: "run-id")
    captured = {}
    monkeypatch.setattr(
        voice_release, "run_audiocpp_align_release",
        lambda *_args: captured.update(release="align") or {"results": {}})
    plan = {"status": "ready", "model": "org/aligner",
            "target_repo": "org/aligner-release", "quants": ["F16"],
            "track": voice.ALIGN, "family": "qwen3_forced_aligner"}
    actions.run_voice(plan, publish=False)
    assert captured["release"] == "align"

    calls = []
    monkeypatch.setattr(
        actions, "voice_plan",
        lambda *_args, **_kwargs: {
            "status": "ready", "model": "audio.cpp:meanvc2",
            "target_repo": "org/meanvc2", "quants": ["F32"],
            "track": voice.VC, "family": "meanvc2"})
    monkeypatch.setattr(actions, "prepare_voice_runtimes", lambda *_args: {})
    monkeypatch.setattr(actions, "run_voice",
                        lambda plan, **_kwargs: calls.append(plan)
                        or {"results": {}})
    monkeypatch.setattr(cli, "_confirm", lambda *_args: True)
    monkeypatch.setattr(cli, "_json", lambda *_args: None)
    args = SimpleNamespace(
        target=None, quant=None, no_fork_hunt=True, language=None, speaker=None,
        dry_run=False, output=None, publish=False, yes=True, review=None,
        max_roundtrip_wer=0.35, max_wer_regression=0.05,
        pocket_language="english", task=None)
    cli._run_voice(args, "audio.cpp:meanvc2", family="meanvc2")
    assert calls and calls[0]["track"] == voice.VC


def test_agent_start_dispatches_a_catalog_voice_task(monkeypatch):
    calls = []
    monkeypatch.setattr(
        actions, "voice_plan",
        lambda *_args, **_kwargs: {
            "status": "ready", "model": "audio.cpp:meanvc2",
            "target_repo": "org/meanvc2", "quants": ["F32"],
            "track": voice.VC, "family": "meanvc2"})
    monkeypatch.setattr(actions, "prepare_voice_runtimes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(actions, "run_voice",
                        lambda plan, **_kwargs: calls.append(plan) or {"pipeline": "voice"})
    result = tools._start({
        "user_approved": True, "models": ["audio.cpp:meanvc2"],
        "family": "meanvc2", "publish": False})
    assert result["pipeline"] == "voice"
    assert calls[0]["track"] == voice.VC
