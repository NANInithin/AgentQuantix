"""The agent-facing surface: the MCP wire protocol and the approval gate.

The gate tests are the reason this file exists. The whole safety story of the
project is "it cannot start a six-hour job unasked", and that claim is only
worth anything if something checks it.
"""

import json

import pytest

from agentquantix import mcp_server, voice
from agentquantix.agent import prompt as prompt_mod, tools as tools_mod


# =====================================================
# THE TOOL REGISTRY
# =====================================================
def test_every_tool_has_a_usable_schema():
    for tool in tools_mod.TOOLS:
        assert tool["name"] and tool["description"]
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        # Required fields must actually be declared, or a model cannot call it.
        for field in schema.get("required", []):
            assert field in schema["properties"], f"{tool['name']}.{field}"


def test_only_explicit_start_tools_can_spend_hours():
    """Everything except the shared start/publish actions is speculative-safe."""
    destructive = [t["name"] for t in tools_mod.TOOLS
                   if "user_approved" in t["input_schema"].get("required", [])]
    assert destructive == ["start_release", "publish_release",
                           "prepare_runtime", "adopt_runtime", "rollback_runtime"]


def test_modality_specific_tools_are_gone():
    names = {t["name"] for t in tools_mod.TOOLS}
    for removed in ("plan_quantization", "start_quantization", "describe_candidate",
                    "plan_voice_release", "start_voice_release", "record_voice_review",
                    "inspect_media_model", "plan_media_release", "start_media_release",
                    "record_media_review", "publish_media_release", "verify_published"):
        assert removed not in names
        assert "error" in json.loads(tools_mod.call_json(removed, {}))


def test_unknown_tools_are_rejected():
    result = json.loads(tools_mod.call_json("no_such_tool", {}))
    assert "error" in result


def test_tool_errors_come_back_as_readable_content():
    """An exception must reach the model as something it can react to, not as
    a transport failure that ends the turn silently."""
    result = json.loads(tools_mod.call_json("record_review", {
        "run_id": "nothing", "quant": "Q4_K_M", "accepted": True, "reviewer": "me"}))
    assert "error" in result and isinstance(result["error"], str)


# =====================================================
# THE APPROVAL GATE
# =====================================================
def test_start_release_refuses_without_approval():
    result = json.loads(tools_mod.call_json(
        "start_release", {"models": ["org/Model"], "user_approved": False}))
    assert "error" in result
    assert "user_approved" in result["error"]


def test_start_release_refuses_when_approval_is_absent():
    result = json.loads(tools_mod.call_json(
        "start_release", {"models": ["org/Model"]}))
    assert "error" in result


def test_publish_release_refuses_without_approval():
    result = json.loads(tools_mod.call_json(
        "publish_release", {"run_id": "voice:x", "quant": "Q4_K_M", "user_approved": False}))
    assert "user_approved" in result["error"]


def test_the_agent_loop_confirms_every_execution_tool():
    from agentquantix.agent import loop
    assert loop.CONFIRM_BEFORE == {"start_release", "publish_release"}


def test_the_prompt_states_both_gates():
    text = prompt_mod.SYSTEM_PROMPT
    assert "never call start_release" in text
    assert "You never research on your own initiative." in text


def test_the_prompt_keeps_voice_out_of_the_text_pipeline():
    text = prompt_mod.SYSTEM_PROMPT
    assert "plan_release" in text and "start_release" in text
    assert "never route voice through the text pipeline" in text
    for removed in ("plan_quantization", "plan_voice_release", "start_media_release",
                    "describe_candidate"):
        assert removed not in text


def _no_text(monkeypatch, label):
    monkeypatch.setattr(
        tools_mod.actions, "assessments_for",
        lambda models: (_ for _ in ()).throw(AssertionError(f"{label} entered text assessment")))


def test_inspect_voice_model_uses_the_backend_registry(monkeypatch):
    _no_text(monkeypatch, "voice model")
    monkeypatch.setattr(
        tools_mod.actions.voice_release, "source_metadata",
        lambda model, **kwargs: {"revision": "abc", "source_bytes": 123,
                                 "gated": False})
    result = tools_mod.call(
        "inspect_model", {"model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"})
    assert result["route"]["pipeline"] == "voice"
    assert result["route"]["task"] == "text-to-speech"
    assert result["voice"]["backend"] == "llama-qwen3-tts"
    assert result["voice"]["runtime"] == "llama-tts"


def test_inspect_voxcpm_uses_audiocpp_not_text_assessment(monkeypatch):
    _no_text(monkeypatch, "VoxCPM2")
    monkeypatch.setattr(
        tools_mod.actions.voice_release, "source_metadata",
        lambda model, **kwargs: {"revision": "abc", "source_bytes": 123,
                                 "gated": False,
                                 "source_files": ["model.safetensors",
                                                  "audiovae.pth"]})
    result = tools_mod.call("inspect_model", {"model": "openbmb/VoxCPM2"})
    assert result["voice"]["backend"] == "audiocpp-voxcpm2-tts"
    assert result["voice"]["engine"] == "audio.cpp"
    assert result["voice"]["runtime"] == "audiocpp_cli"
    assert result["voice"]["quants"] == list(voice.AUDIOCPP_QUANTS)
    inputs = result["voice"]["required_source_inputs"]
    assert inputs[1]["status"] == "needs_preparation"
    assert inputs[1]["preparation"]["package"] == "voxcpm2_audiovae"


def test_voice_plan_uses_hub_preflight_for_catalog_matched_repo(monkeypatch):
    monkeypatch.setattr(
        tools_mod.actions.voice_release, "source_metadata",
        lambda model, **kwargs: (_ for _ in ()).throw(
            voice.VoiceValidationError("registered voice source is not accessible")))
    with pytest.raises(voice.VoiceValidationError, match="not accessible"):
        tools_mod.call("plan_release", {"models": ["Qwen/Qwen3-TTS-1.7B"]})


def test_unresolved_voice_plan_returns_cached_fork_hunt_evidence(monkeypatch):
    lead = {"backend": "audio.cpp", "repo": "org/audio.cpp",
            "ref": "model/new-voice", "actionable": False}
    monkeypatch.setattr(
        tools_mod.actions.voice_release, "find_backend_forks", lambda _repo: [lead])
    result = tools_mod.call("plan_release", {"models": ["org/NewVoice"]})
    plan = result["releases"][0]["plan"]
    assert result["releases"][0]["pipeline"] == "voice"
    assert plan["status"] == "blocked"
    assert plan["fork_hunt"] == "searched"
    assert plan["fork_leads"] == [lead]

    with pytest.raises(ValueError, match="voice release is blocked"):
        tools_mod.call("start_release", {
            "models": ["org/NewVoice"], "user_approved": True})


def test_inspect_likely_unknown_voice_model_does_not_use_text(monkeypatch):
    _no_text(monkeypatch, "likely voice model")
    monkeypatch.setattr(
        tools_mod.actions.voice_release, "find_backend_forks", lambda _repo: [])
    result = tools_mod.call("inspect_model", {"model": "org/NewVoice-TTS"})
    assert result["voice"]["status"] == "blocked"
    assert result["voice"]["fork_hunt"] == "searched"


def test_registered_voice_models_never_reach_the_text_pipeline(monkeypatch):
    _no_text(monkeypatch, "registered voice model")
    monkeypatch.setattr(tools_mod.actions.voice_release, "source_metadata",
                        lambda model, **kwargs: {"revision": "abc", "source_bytes": 1, "gated": False})
    result = tools_mod.call("plan_release", {"models": ["Qwen/Qwen3-TTS-12Hz-1.7B-Base"]})
    assert result["text"] is None and result["releases"][0]["pipeline"] == "voice"
    with pytest.raises(ValueError, match="different pipelines"):
        tools_mod.call("start_release", {
            "models": ["openai/whisper-small", "org/Some-LLM"], "user_approved": True})


def test_media_start_requires_the_exact_plan(monkeypatch):
    monkeypatch.setattr("agentquantix.tasks.router.hub_metadata",
                        lambda repo: {"pipeline_tag": "text-to-image", "tags": ["diffusers"]})
    with pytest.raises(ValueError, match="exact plan"):
        tools_mod.call("start_release", {"models": ["org/some-sdxl"], "user_approved": True})


def test_the_prompt_and_the_skill_cannot_drift():
    """REGRESSION. The skill and the prompt were maintained separately, drifted,
    and the agent confidently told the user a corrected number's old value."""
    markdown = prompt_mod.markdown()
    assert prompt_mod.SYSTEM_PROMPT in markdown


# =====================================================
# MCP WIRE PROTOCOL
# =====================================================
def _exchange(monkeypatch, messages):
    """Drive the server's handler and collect what it writes."""
    written = []
    monkeypatch.setattr(mcp_server, "_write", written.append)
    for message in messages:
        mcp_server.handle(message)
    return written


def test_initialize_advertises_tools_and_instructions(monkeypatch):
    sent = _exchange(monkeypatch, [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}])
    result = sent[0]["result"]
    assert result["capabilities"]["tools"] is not None
    assert result["serverInfo"]["name"] == "agentquantix"
    assert prompt_mod.SYSTEM_PROMPT in result["instructions"]


def test_server_version_matches_the_package(monkeypatch):
    """REGRESSION. __init__ said 0.1.0 while pyproject said 0.2.0, and this is
    the value clients are told over the wire."""
    import agentquantix
    sent = _exchange(monkeypatch, [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}])
    assert sent[0]["result"]["serverInfo"]["version"] == agentquantix.__version__


def test_tools_list_matches_the_registry(monkeypatch):
    sent = _exchange(monkeypatch, [
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])
    names = [t["name"] for t in sent[0]["result"]["tools"]]
    assert names == [t["name"] for t in tools_mod.TOOLS]
    # MCP spells it inputSchema, not input_schema.
    assert all("inputSchema" in t for t in sent[0]["result"]["tools"])


def test_notifications_get_no_response(monkeypatch):
    """A response to a notification desynchronises the client."""
    sent = _exchange(monkeypatch, [
        {"jsonrpc": "2.0", "method": "notifications/initialized"}])
    assert sent == []


def test_unknown_methods_return_an_error_not_a_crash(monkeypatch):
    sent = _exchange(monkeypatch, [
        {"jsonrpc": "2.0", "id": 9, "method": "does/not/exist"}])
    assert sent[0]["error"]["code"] == -32601


def test_a_failing_tool_call_is_flagged_as_an_error(monkeypatch):
    sent = _exchange(monkeypatch, [
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "start_release",
                    "arguments": {"models": ["x"], "user_approved": False}}}])
    assert sent[0]["result"]["isError"] is True
