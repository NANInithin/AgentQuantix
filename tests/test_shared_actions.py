"""0.5.2: metadata routing, run ids and the shared CLI/agent actions."""
import json
from pathlib import Path

import pytest

from agentquantix import actions, cli, config, runs, voice
from agentquantix.pipeline.job import Job
from agentquantix.tasks import TASKS, capabilities, route


def _never(repo_id):
    raise AssertionError("catalog-declared source fetched Hub metadata")


# =====================================================
# routing
# =====================================================
def test_catalog_declared_voice_sources_route_without_the_network():
    tts = route("Qwen/Qwen3-TTS-12Hz-1.7B-Base", fetch=_never)
    assert (tts.pipeline, tts.task, tts.engine) == ("voice", "text-to-speech", "llama.cpp")
    asr = route("openai/whisper-small", fetch=_never)
    assert (asr.pipeline, asr.task, asr.engine) == ("voice", "automatic-speech-recognition", "whisper.cpp")
    audio = route("openbmb/VoxCPM2", fetch=_never)
    assert (audio.engine, audio.family) == ("audio.cpp", "voxcpm2")


def test_hub_task_metadata_routes_text_and_media():
    text = route("org/Some-LLM", metadata={"pipeline_tag": "text-generation", "tags": []})
    assert text.ready and text.pipeline == "text" and text.engine == "llama.cpp"
    image = route("org/innocent", metadata={"pipeline_tag": "text-to-image", "tags": ["diffusers"]})
    assert image.ready and image.pipeline == "media" and image.task == "text-to-image"


def test_retrieval_hub_routes_are_ready_and_clip_architecture_overrides_tag():
    embed = route("org/embed", metadata={"pipeline_tag": "feature-extraction",
                                         "library_name": "transformers"})
    rerank = route("org/reranker", metadata={"pipeline_tag": "text-ranking"})
    clip = route("org/clip", metadata={"pipeline_tag": "feature-extraction",
                                        "architectures": ["CLIPModel"]})
    assert (embed.pipeline, embed.task, embed.engine, embed.ready) == (
        "retrieval", "embeddings", "llama.cpp", True)
    assert (rerank.pipeline, rerank.task, rerank.engine, rerank.ready) == (
        "retrieval", "reranking", "llama.cpp", True)
    assert (clip.pipeline, clip.task, clip.engine, clip.ready) == (
        "retrieval", "image-feature-extraction", "clip.cpp", True)


@pytest.mark.parametrize("tag,task,milestone", [
    ("object-detection", "object-detection", "0.5.7"),
    ("time-series-forecasting", "time-series-forecasting", "post-0.6.0"),
])
def test_recognised_tasks_without_adapters_are_blocked_not_misrouted(tag, task, milestone):
    item = route("org/model", metadata={"pipeline_tag": tag, "library_name": "transformers"})
    assert item.task == task and item.pipeline is None and not item.ready
    if task == "object-detection":
        assert item.blocker == (
            "ultralytics-ggml YOLO26 detection, instance/semantic segmentation, depth and "
            "YOLO-World open-vocabulary did not meet native parity gates in 0.5.7; "
            "OBB was not evaluated")
    else:
        assert milestone in item.blocker and "not sent through the text-generation" in item.blocker


@pytest.mark.parametrize("metadata,task", [
    # Observed Hub metadata for BAAI/bge-reranker-v2-m3 and BAAI/bge-m3.
    ({"pipeline_tag": "text-classification", "library_name": "sentence-transformers",
      "architectures": ["XLMRobertaForSequenceClassification"]}, "reranking"),
    ({"pipeline_tag": None, "library_name": "sentence-transformers",
      "architectures": ["XLMRobertaModel"]}, "embeddings"),
])
def test_sentence_transformers_contracts_name_their_task(metadata, task):
    item = route("org/encoder", metadata=metadata)
    assert item.task == task and item.ready and item.pipeline == "retrieval"


@pytest.mark.parametrize("metadata", [
    {"pipeline_tag": "fill-mask", "library_name": "transformers", "architectures": ["BertForMaskedLM"]},
    {"pipeline_tag": None, "library_name": "transformers",
     "architectures": ["XLMRobertaForSequenceClassification"]},
])
def test_non_generative_transformers_do_not_default_to_text(metadata):
    item = route("org/encoder", metadata=metadata)
    assert item.pipeline is None and "--task" in item.blocker


def test_untagged_generative_checkpoints_still_route_to_text():
    item = route("org/llm", metadata={"library_name": "transformers",
                                      "architectures": ["LlamaForCausalLM"]})
    assert item.ready and item.pipeline == "text"


def test_retrieval_models_never_enter_text_assessment(monkeypatch):
    monkeypatch.setattr("agentquantix.tasks.router.hub_metadata",
                        lambda repo: {"pipeline_tag": "sentence-similarity", "tags": []})
    monkeypatch.setattr(actions, "assessments_for", lambda models: pytest.fail("entered text"))
    monkeypatch.setattr(actions, "retrieval_plan",
                        lambda model, **kwargs: {"model": model, "status": "ready"})
    result = actions.plan(["org/embedder"])
    assert result["text"] is None and result["releases"][0]["pipeline"] == "retrieval"


def test_conflicting_evidence_is_a_short_choice():
    item = route("org/stable-diffusion-chat",
                 metadata={"pipeline_tag": "text-generation", "tags": []})
    assert not item.ready and {c["pipeline"] for c in item.choices} == {"text", "media"}
    chosen = route("org/stable-diffusion-chat", task="text-generation",
                   metadata={"pipeline_tag": "text-generation"})
    assert chosen.ready and chosen.pipeline == "text"


def test_unknown_and_unimplemented_explicit_tasks():
    with pytest.raises(ValueError, match="unknown task"):
        route("org/x", task="teleportation")
    with pytest.raises(ValueError, match="unknown engine"):
        route("org/x", engine="nope.cpp")
    item = route("org/x", task="depth-estimation")
    assert not item.ready
    assert item.blocker == (
        "no vision.cpp family matches depth-estimation; "
        "depth-anything.cpp requires an exact catalog family")


def test_metadata_without_task_evidence_asks_for_a_task():
    item = route("org/mystery", metadata={"tags": [], "library_name": "timm"})
    assert not item.ready and "--task" in item.blocker


def test_offline_naming_falls_back_to_the_text_assessment():
    item = route("org/Demo")
    assert item.pipeline == "text" and any("offline" in r for r in item.reasons)


def test_task_registry_and_capability_cells():
    assert TASKS["text-to-speech"].pipeline == "voice"
    assert TASKS["embeddings"].implemented
    retrieval_cells = capabilities()
    assert {cell.task for cell in retrieval_cells if cell.engine == "llama.cpp"} >= {
        "embeddings", "reranking"}
    assert {cell.task for cell in retrieval_cells if cell.engine == "clip.cpp"} >= {
        "zero-shot-image-classification", "image-feature-extraction"}
    cells = capabilities(engine="audio.cpp")
    assert cells and all(cell.engine == "audio.cpp" for cell in cells)
    assert all(cell.acceptance == "pending" for cell in capabilities())


# =====================================================
# run ids
# =====================================================
@pytest.fixture
def recorded(tmp_path):
    Job("org/Demo", "Demo", "owner/Demo-GGUF", ["Q4_0"]).save()
    voice_dir = runs.voice_root() / "Qwen--Qwen3-TTS-12Hz-1.7B-Base"
    (voice_dir / "reviews").mkdir(parents=True)
    plan_id = "ab" * 32
    media_dir = runs.media_root() / plan_id
    media_dir.mkdir(parents=True)
    (media_dir / "plan.json").write_text(json.dumps({
        "repo_id": "org/flux", "task": "text-to-image", "quants": ["q4_0"],
        "target_repositories": {"q4_0": "owner/flux-Q4_0"}}), encoding="utf-8")
    return plan_id


def test_existing_records_of_every_pipeline_are_discovered(recorded):
    retrieval_dir = runs.retrieval_root() / "org-embedder"
    retrieval_dir.mkdir(parents=True)
    (retrieval_dir / "run.json").write_text(json.dumps({
        "plan": {"model": "org/embedder", "task": "embeddings", "engine": "llama.cpp"}}),
        encoding="utf-8")
    ids = {item.id for item in runs.discover()}
    assert {"text:Demo", "voice:Qwen--Qwen3-TTS-12Hz-1.7B-Base",
            f"media:{recorded}", "retrieval:org-embedder"} <= ids
    assert runs.resolve("retrieval:org-embedder").task == "embeddings"


def test_resolution_is_deterministic_and_migrates_voice_workspaces(recorded):
    assert runs.resolve("text:Demo").repo_id == "org/Demo"
    assert runs.resolve(recorded[:10]).id == f"media:{recorded}"
    voice_run = runs.resolve("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    assert voice_run.pipeline == "voice"
    record = json.loads((runs.voice_root() / "Qwen--Qwen3-TTS-12Hz-1.7B-Base" / "run.json")
                        .read_text(encoding="utf-8"))
    assert record["repo_id"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    with pytest.raises(ValueError, match="no recorded run"):
        runs.resolve("voice:missing")


def test_ambiguous_identifiers_return_choices(recorded):
    voice_dir = runs.voice_root() / "org--Demo"
    voice_dir.mkdir(parents=True)
    with pytest.raises(runs.AmbiguousRun) as error:
        runs.resolve("org/Demo")
    assert {item.id for item in error.value.matches} == {"text:Demo", "voice:org--Demo"}


# =====================================================
# review / publish gates
# =====================================================
def _create_tts_review_evidence(run_id, quant):
    root = Path(runs.resolve(run_id).location)
    quality = root / "models" / "candidate" / "quality" / quant
    quality.mkdir(parents=True, exist_ok=True)
    cache = quality / "quality-cache.json"
    cache.write_text('{"quality": {}}\n', encoding="utf-8")
    audio = quality / "00-short-en.wav"
    audio.write_bytes(b"generated fixture audio")
    resampled = quality / "00-short-en-16khz.wav"
    resampled.write_bytes(b"resampled fixture audio")
    return root, cache, audio, resampled


def test_tts_reviews_require_a_listening_attestation(recorded):
    run_id = "voice:Qwen--Qwen3-TTS-12Hz-1.7B-Base"
    with pytest.raises(ValueError, match="listened"):
        actions.review(run_id, "Q4_K_M", accepted=True, reviewer="Ana")
    with pytest.raises(ValueError, match="reviewer"):
        actions.review(run_id, "Q4_K_M", accepted=True, reviewer=" ", listened_audio=True)
    root, cache, audio, _ = _create_tts_review_evidence(run_id, "Q4_K_M")
    result = actions.review(run_id, "q4_k_m", accepted=True, reviewer="Ana", listened_audio=True)
    assert result["quant"] == "Q4_K_M" and result["accepted"]
    artifact = audio.relative_to(root).as_posix()
    assert result["artifacts"] == {artifact: voice.sha256(audio)}
    assert result["quality_cache_sha256"] == voice.sha256(cache)
    review = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert review["artifacts"] == result["artifacts"]
    assert review["quality_cache_sha256"] == result["quality_cache_sha256"]
    assert not any(path.endswith("-16khz.wav") for path in review["artifacts"])
    assert runs.resolve(run_id).details["reviews"] == ["Q4_K_M"]


def test_tts_review_requires_generated_audio(recorded):
    run_id = "voice:Qwen--Qwen3-TTS-12Hz-1.7B-Base"
    review_path = Path(runs.resolve(run_id).location) / "reviews" / "Q4_K_M.json"
    with pytest.raises(ValueError, match="no generated TTS audio"):
        actions.review(run_id, "Q4_K_M", accepted=True, reviewer="Ana",
                       listened_audio=True)
    assert not review_path.exists()


def test_text_runs_have_no_review_and_publish_during_the_run(recorded):
    with pytest.raises(ValueError, match="no human review"):
        actions.review("text:Demo", "Q4_0", accepted=True, reviewer="Ana")
    with pytest.raises(ValueError, match="--resume text:Demo"):
        actions.publish("text:Demo", "Q4_0", user_approved=True)


def test_media_reviews_and_publication_delegate_to_the_media_gates(recorded, monkeypatch):
    from agentquantix.pipeline import media_release
    calls = []
    monkeypatch.setattr(media_release, "record_review",
                        lambda *args, **kwargs: calls.append(("review", args, kwargs)) or {"accepted": True})
    monkeypatch.setattr(media_release, "publish_run",
                        lambda *args: calls.append(("publish", args)) or {"published": True})
    actions.review(f"media:{recorded}", "q4_0", accepted=True, reviewer="Ana", watched_video=False)
    with pytest.raises(ValueError, match="user_approved"):
        actions.publish(f"media:{recorded}", "q4_0", user_approved=False)
    assert actions.publish(f"media:{recorded}", "q4_0", user_approved=True)["published"]
    assert [call[0] for call in calls] == ["review", "publish"]
    assert calls[0][1][0] == recorded


def test_voice_publication_needs_an_accepted_review(recorded, monkeypatch):
    run_id = "voice:Qwen--Qwen3-TTS-12Hz-1.7B-Base"
    runs.write_voice_record("Qwen/Qwen3-TTS-12Hz-1.7B-Base", {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "target_repo": "owner/T-GGUF",
        "task": "text-to-speech", "engine": "llama.cpp"})
    _create_tts_review_evidence(run_id, "Q4_K_M")
    with pytest.raises(ValueError, match="no accepted listening review"):
        actions.publish(run_id, "Q4_K_M", user_approved=True)
    actions.review(run_id, "Q4_K_M", accepted=True, reviewer="Ana", listened_audio=True)
    _create_tts_review_evidence(run_id, "Q8_0")
    actions.review(run_id, "Q8_0", accepted=False, reviewer="Ana", listened_audio=True)
    seen = {}
    monkeypatch.setattr(actions, "voice_plan", lambda model, **kwargs: seen.update(kwargs) or {
        "model": model, "status": "ready", "track": "tts", "quants": kwargs["quants"],
        "target_repo": kwargs["target_repo"], "family": None, "speaker": None, "language": None})
    monkeypatch.setattr(actions, "run_voice", lambda plan, **kwargs: {"plan": plan, **kwargs})
    result = actions.publish(run_id, "q4_k_m", user_approved=True)
    assert seen["quants"] == ["Q4_K_M"] and result["publish"] is True


# =====================================================
# the CLI surface
# =====================================================
def test_modality_command_groups_are_removed_without_aliases():
    parser = cli.build_parser()
    commands = parser._subparsers._group_actions[0].choices
    assert {"inspect", "plan", "show", "run", "review", "verify", "publish", "card",
            "runs", "runtime"} <= set(commands)
    assert "voice" not in commands and "media" not in commands
    for argv in (["voice", "list"], ["media", "plan", "org/x"], ["media", "backends"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_options_for_another_pipeline_are_rejected(monkeypatch):
    monkeypatch.setattr(config, "migrate_legacy_state", lambda: [])
    assert cli.main(["run", "org/Demo", "--language", "en", "--dry-run"]) == 1
    assert cli.main(["run", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "--upload-workers", "2",
                     "--dry-run"]) == 1


def test_resume_uses_the_recorded_job(recorded, monkeypatch):
    seen = {}

    def execute(args, jobs, options, result=None, resumed=False):
        seen.update(jobs=jobs, resumed=resumed)
        return 0

    monkeypatch.setattr(cli, "_execute_text", execute)
    args = cli.build_parser().parse_args(["run", "--resume", "text:Demo", "--dry-run"])
    assert cli.cmd_run(args) == 0
    assert seen["resumed"] and seen["jobs"][0].target_repo == "owner/Demo-GGUF"
    with pytest.raises(ValueError, match="run id instead"):
        cli.cmd_run(cli.build_parser().parse_args(["run", "org/x", "--resume", "text:Demo"]))


def test_review_and_publish_commands_use_the_shared_actions(recorded, monkeypatch):
    calls = []
    monkeypatch.setattr(actions, "review", lambda *a, **k: calls.append(("review", a, k)) or {})
    monkeypatch.setattr(actions, "publish", lambda *a, **k: calls.append(("publish", a, k)) or {"published": True})
    args = cli.build_parser().parse_args(
        ["review", f"media:{recorded}", "--quant", "q4_0", "--reject", "--reviewer", "Ana",
         "--watched-video"])
    assert cli.cmd_review(args) == 0 and calls[0][2]["accepted"] is False
    args = cli.build_parser().parse_args(["publish", f"media:{recorded}", "--quant", "q4_0", "--yes"])
    assert cli.cmd_publish(args) == 0
    assert calls[1][1] == (f"media:{recorded}", "q4_0") and calls[1][2]["user_approved"] is True


def test_runtime_show_replaces_voice_list_and_media_backends(monkeypatch, capsys):
    from agentquantix.agent import runtime_tools
    monkeypatch.setattr(config, "migrate_legacy_state", lambda: [])
    monkeypatch.setattr(runtime_tools, "call", lambda name, arguments: {"engine": arguments["engine"]})
    assert cli.main(["runtime", "show", "audio.cpp"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert {"text-to-speech", "automatic-speech-recognition"} <= {t["id"] for t in shown["tasks"]}
    assert shown["capabilities"]
    assert cli.main(["runtime", "show", "stable-diffusion.cpp"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert "converter" in shown and shown["capabilities"]


def test_runs_command_lists_recorded_ids(recorded, capsys):
    detection_dir = runs.detection_root() / "detection-demo"
    detection_dir.mkdir(parents=True)
    (detection_dir / "run.json").write_text(json.dumps({
        "plan": {"model": "org/detection", "task": "image-classification",
                 "engine": "ultralytics-ggml"}}), encoding="utf-8")
    depth_dir = runs.vision_root() / "depth-anything.cpp--depth-demo"
    depth_dir.mkdir(parents=True)
    (depth_dir / "run.json").write_text(json.dumps({
        "plan": {"model": "org/depth", "task": "depth-estimation",
                 "engine": "depth-anything.cpp"}}), encoding="utf-8")
    assert cli.cmd_runs(cli.build_parser().parse_args(["runs"])) == 0
    out = capsys.readouterr().out
    assert "text:Demo" in out and f"media:{recorded}" in out
    assert "vision:detection-demo" in out and "vision:depth-anything.cpp--depth-demo" in out


@pytest.mark.parametrize("kind", [
    "text", "voice", "media", "retrieval", "multimodal",
    "vision.cpp", "detection", "depth",
])
def test_resumable_pipeline_ids_round_trip_through_resume(kind, recorded, monkeypatch):
    run_ids = {
        "text": "text:Demo",
        "voice": "voice:Qwen--Qwen3-TTS-12Hz-1.7B-Base",
        "media": f"media:{recorded}",
        "retrieval": "retrieval:resume-retrieval",
        "multimodal": "multimodal:resume-multimodal",
        "vision.cpp": "vision:resume-vision",
        "detection": "vision:resume-detection",
        "depth": "vision:resume-depth",
    }
    run_id = run_ids[kind]
    if kind == "retrieval":
        root = runs.retrieval_root() / "resume-retrieval"
        root.mkdir(parents=True)
        (root / "run.json").write_text(json.dumps({"plan": {
            "model": "org/retrieval", "task": "embeddings", "engine": "llama.cpp"}}))
    elif kind == "multimodal":
        root = runs.multimodal_root() / "resume-multimodal"
        root.mkdir(parents=True)
        (root / "run.json").write_text(json.dumps({"plan": {
            "model": "org/multimodal", "task": "multimodal-understanding",
            "engine": "llama.cpp"}}))
    elif kind in ("vision.cpp", "detection", "depth"):
        root = runs.detection_root() if kind == "detection" else runs.vision_root()
        directory = root / run_id.partition(":")[2]
        directory.mkdir(parents=True)
        engine = {"vision.cpp": "vision.cpp", "detection": "ultralytics-ggml",
                  "depth": "depth-anything.cpp"}[kind]
        (directory / "run.json").write_text(json.dumps({
            "plan": {"model": f"org/{kind}", "task": "image-classification",
                     "engine": engine},
            "runtime_snapshot": {"identity": "frozen"}}))
    vision_calls = []
    for name in ("_execute_text", "_run_voice", "_run_media", "_run_retrieval",
                 "_run_multimodal"):
        monkeypatch.setattr(cli, name, lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(cli, "_run_vision",
                        lambda _args, saved, **kwargs:
                        vision_calls.append((saved, kwargs)) or 0)
    monkeypatch.setattr(cli, "_check_applicable", lambda *_args: None)
    args = cli.build_parser().parse_args(["run", "--resume", run_id, "--dry-run"])
    assert cli.cmd_run(args) == 0
    if kind in ("vision.cpp", "detection", "depth"):
        assert vision_calls[0][1]["runtime_snapshot"] == {"identity": "frozen"}


def test_start_vision_resume_binds_the_saved_runtime_snapshot(monkeypatch):
    from agentquantix import runtime_control
    from agentquantix.pipeline import detection_release

    snapshot = {"identity": "frozen"}
    runtime = object()
    calls = {}
    monkeypatch.setattr(runtime_control, "ensure_runtime",
                        lambda *args, **kwargs: calls.update(kwargs) or runtime)
    monkeypatch.setattr(detection_release, "run_release",
                        lambda *_args, **_kwargs: {"run_id": "detection", "candidates": []})
    plan = {"engine": "ultralytics-ggml", "status": "ready", "blockers": [],
            "runtime_revision": "a" * 40, "runtime_repository": "Asher-1/ultralytics-ggml"}

    actions.start_vision(plan, user_approved=True, runtime_snapshot=snapshot)

    assert calls["frozen"] == snapshot


def test_unknown_resume_id_has_clean_cli_error(monkeypatch, capsys):
    monkeypatch.setattr(config, "migrate_legacy_state", lambda: [])
    assert cli.main(["run", "--resume", "vision:missing"]) == 1
    assert "no recorded run matches 'vision:missing'" in capsys.readouterr().out


def test_local_manifest_verification_detects_the_bundle_kind(tmp_path):
    from agentquantix import media
    manifest = tmp_path / "manifest.json"
    media.atomic_json(manifest, {"engine": "stable-diffusion.cpp", "schema_version": 1,
                                 "family": "flux", "task": "text-to-image", "quant": "q4_0",
                                 "components": [], "runtime_recipe": {},
                                 "required_roles": ["vae"]})
    result = actions.verify(str(manifest))
    assert result["kind"] == "media-bundle" and not result["valid"]
    assert "missing required roles: vae" in result["problems"]
