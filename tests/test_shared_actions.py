"""0.5.2: metadata routing, run ids and the shared CLI/agent actions."""
import json

import pytest

from agentquantix import actions, cli, config, runs
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
    ("time-series-forecasting", "time-series-forecasting", "0.5.9"),
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
def test_tts_reviews_require_a_listening_attestation(recorded):
    run_id = "voice:Qwen--Qwen3-TTS-12Hz-1.7B-Base"
    with pytest.raises(ValueError, match="listened"):
        actions.review(run_id, "Q4_K_M", accepted=True, reviewer="Ana")
    with pytest.raises(ValueError, match="reviewer"):
        actions.review(run_id, "Q4_K_M", accepted=True, reviewer=" ", listened_audio=True)
    result = actions.review(run_id, "q4_k_m", accepted=True, reviewer="Ana", listened_audio=True)
    assert result["quant"] == "Q4_K_M" and result["accepted"]
    assert runs.resolve(run_id).details["reviews"] == ["Q4_K_M"]


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
    with pytest.raises(ValueError, match="no accepted listening review"):
        actions.publish(run_id, "Q4_K_M", user_approved=True)
    actions.review(run_id, "Q4_K_M", accepted=True, reviewer="Ana", listened_audio=True)
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
    assert cli.cmd_runs(cli.build_parser().parse_args(["runs"])) == 0
    out = capsys.readouterr().out
    assert "text:Demo" in out and f"media:{recorded}" in out


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
