"""The agent-facing surface: the MCP wire protocol and the approval gate.

The gate tests are the reason this file exists. The whole safety story of the
project is "it cannot start a six-hour job unasked", and that claim is only
worth anything if something checks it.
"""

import json

from agentquantix import mcp_server
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


def test_only_one_tool_can_spend_hours():
    """Everything else must be safe to call speculatively."""
    destructive = [t["name"] for t in tools_mod.TOOLS
                   if "user_approved" in t["input_schema"].get("required", [])]
    assert destructive == ["start_quantization"]


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
