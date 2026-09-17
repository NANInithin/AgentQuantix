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
    """Everything except the text and voice start tools is speculative-safe."""
    destructive = [t["name"] for t in tools_mod.TOOLS
                   if "user_approved" in t["input_schema"].get("required", [])]
    assert destructive == ["start_quantization", "start_voice_release"]


def test_unknown_tools_are_rejected():
    result = json.loads(tools_mod.call_json("no_such_tool", {}))
    assert "error" in result


def test_tool_errors_come_back_as_readable_content():
    """An exception must reach the model as something it can react to, not as
    a transport failure that ends the turn silently."""
    result = json.loads(tools_mod.call_json("describe_candidate",
                                            {"model": "nothing"}))
    assert "error" in result and isinstance(result["error"], str)


# =====================================================
# THE APPROVAL GATE
# =====================================================
def test_start_quantization_refuses_without_approval():
    result = json.loads(tools_mod.call_json(
        "start_quantization", {"models": ["org/Model"], "user_approved": False}))
    assert "error" in result
    assert "user_approved" in result["error"]


def test_start_quantization_refuses_when_approval_is_absent():
    result = json.loads(tools_mod.call_json(
        "start_quantization", {"models": ["org/Model"]}))
    assert "error" in result


def test_the_prompt_states_both_gates():
    text = prompt_mod.SYSTEM_PROMPT
    assert "never call start_quantization" in text
    assert "You never research on your own initiative." in text


def test_the_prompt_keeps_voice_out_of_the_text_pipeline():
    text = prompt_mod.SYSTEM_PROMPT
    assert "plan_voice_release" in text
    assert "start_voice_release" in text
    assert "never route voice through the text quantization tools" in text


def test_describe_voice_model_uses_the_backend_registry(monkeypatch):
    monkeypatch.setattr(
        tools_mod, "_assessments_for",
        lambda models: (_ for _ in ()).throw(
            AssertionError("voice model entered text assessment")))
    monkeypatch.setattr(
        tools_mod.voice_release, "source_metadata",
        lambda model, **kwargs: {"revision": "abc", "source_bytes": 123,
                                 "gated": False})
    result = tools_mod.call(
        "describe_candidate", {"model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"})
    assert result["voice"]["backend"] == "llama-qwen3-tts"
    assert result["voice"]["runtime"] == "llama-tts"


def test_describe_voxcpm_uses_audiocpp_not_text_assessment(monkeypatch):
    monkeypatch.setattr(
        tools_mod, "_assessments_for",
        lambda models: (_ for _ in ()).throw(
            AssertionError("VoxCPM2 entered text assessment")))
    monkeypatch.setattr(
        tools_mod.voice_release, "source_metadata",
        lambda model, **kwargs: {"revision": "abc", "source_bytes": 123,
                                 "gated": False,
                                 "source_files": ["model.safetensors",
                                                  "audiovae.pth"]})
    result = tools_mod.call("describe_candidate", {"model": "openbmb/VoxCPM2"})
    assert result["voice"]["backend"] == "audiocpp-voxcpm2-tts"
    assert result["voice"]["runtime"] == "audiocpp_cli"
    assert result["voice"]["quants"] == list(voice.AUDIOCPP_QUANTS)
    inputs = result["voice"]["required_source_inputs"]
    assert inputs[1]["status"] == "needs_preparation"
    assert inputs[1]["preparation"]["package"] == "voxcpm2_audiovae"


def test_voice_plan_uses_hub_preflight_for_catalog_matched_repo(monkeypatch):
    monkeypatch.setattr(
        tools_mod.voice_release, "source_metadata",
        lambda model, **kwargs: (_ for _ in ()).throw(
            voice.VoiceValidationError("registered voice source is not accessible")))
    with pytest.raises(voice.VoiceValidationError, match="not accessible"):
        tools_mod.call("plan_voice_release", {"model": "Qwen/Qwen3-TTS-1.7B"})


def test_unresolved_voice_plan_returns_cached_fork_hunt_evidence(monkeypatch):
    lead = {"backend": "audio.cpp", "repo": "org/audio.cpp",
            "ref": "model/new-voice", "actionable": False}
    monkeypatch.setattr(
        tools_mod.voice_release, "find_backend_forks", lambda _repo: [lead])
    result = tools_mod.call(
        "plan_voice_release", {"model": "org/NewVoice"})
    assert result["status"] == "blocked"
    assert result["fork_hunt"] == "searched"
    assert result["fork_leads"] == [lead]

    with pytest.raises(ValueError, match="voice release is blocked"):
        tools_mod.call("start_voice_release", {
            "model": "org/NewVoice", "user_approved": True})


def test_describe_likely_unknown_voice_model_does_not_use_text(monkeypatch):
    monkeypatch.setattr(
        tools_mod, "_assessments_for",
        lambda _models: (_ for _ in ()).throw(
            AssertionError("likely voice model entered text assessment")))
    monkeypatch.setattr(
        tools_mod.voice_release, "find_backend_forks", lambda _repo: [])
    result = tools_mod.call("describe_candidate", {"model": "org/NewVoice-TTS"})
    assert result["voice"]["status"] == "blocked"
    assert result["voice"]["fork_hunt"] == "searched"


def test_text_tools_reject_registered_voice_models():
    with pytest.raises(ValueError, match="plan_voice_release"):
        tools_mod.call("plan_quantization", {
            "models": ["Qwen/Qwen3-TTS-12Hz-1.7B-Base"]})
    with pytest.raises(ValueError, match="start_voice_release"):
        tools_mod.call("start_quantization", {
            "models": ["openai/whisper-small"], "user_approved": True})


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
         "params": {"name": "start_quantization",
                    "arguments": {"models": ["x"], "user_approved": False}}}])
    assert sent[0]["result"]["isError"] is True
