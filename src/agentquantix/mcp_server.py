"""AgentQuantix as an MCP server, over stdio, with no dependencies.

Claude Code, Codex, OpenCode and Kimi all speak MCP, so one server reaches all
of them. It is written directly against the JSON-RPC wire format rather than
against the `mcp` Python package on purpose: the whole point of this layer is
portability, and a server that needs a pip install in whichever environment
the harness happens to launch it from is not portable.

Protocol surface implemented: `initialize`, `tools/list`, `tools/call`, plus
the `notifications/initialized` and `ping` housekeeping. That is everything a
tool-only server needs; resources and prompts are deliberately absent because
this server has neither.

One rule that is easy to get wrong and fatal when you do: stdout carries the
protocol and NOTHING else. The pipeline prints progress constantly, so stdout
is redirected to stderr for the duration of every tool call — otherwise a
`llama-quantize` progress line would land mid-frame and desynchronise the
client.
"""

from __future__ import annotations

import contextlib
import json
import sys

from . import __version__
from .agent import prompt as prompt_mod, tools as tools_mod

PROTOCOL_VERSION = "2025-06-18"


def _write(message):
    """One JSON-RPC message, newline-delimited, flushed immediately."""
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _result(request_id, payload):
    _write({"jsonrpc": "2.0", "id": request_id, "result": payload})


def _error(request_id, code, message):
    _write({"jsonrpc": "2.0", "id": request_id,
            "error": {"code": code, "message": message}})


def _mcp_tools():
    """The shared tool registry, in MCP's schema shape."""
    return [{"name": tool["name"],
             "description": tool["description"],
             "inputSchema": tool["input_schema"]}
            for tool in tools_mod.TOOLS]


def handle(message):
    """One request. Returns nothing; responses are written as they are produced."""
    method = message.get("method")
    request_id = message.get("id")

    if method == "initialize":
        _result(request_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "agentquantix", "version": __version__},
            # Harnesses surface this to the model, which is how a plain MCP
            # client gets the same behaviour as the Claude Code skill.
            "instructions": prompt_mod.SYSTEM_PROMPT,
        })
        return

    if method in ("notifications/initialized", "notifications/cancelled"):
        return                                  # notifications get no response

    if method == "ping":
        _result(request_id, {})
        return

    if method == "tools/list":
        _result(request_id, {"tools": _mcp_tools()})
        return

    if method == "tools/call":
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}

        # The pipeline is chatty and every byte of that belongs on stderr.
        # Without this redirect a long-running tool call corrupts the stream.
        with contextlib.redirect_stdout(sys.stderr):
            text = tools_mod.call_json(name, arguments)

        failed = False
        try:
            failed = "error" in json.loads(text) and len(json.loads(text)) == 1
        except Exception:
            pass
        _result(request_id, {"content": [{"type": "text", "text": text}],
                             "isError": failed})
        return

    if request_id is not None:
        _error(request_id, -32601, f"method not found: {method}")


def main():
    """Read newline-delimited JSON-RPC from stdin until it closes."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _error(None, -32700, "parse error")
            continue
        try:
            handle(message)
        except Exception as e:
            # A crashed handler must not take the server down — the client
            # would see the pipe close mid-session with no explanation.
            if message.get("id") is not None:
                _error(message["id"], -32603, f"{type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
