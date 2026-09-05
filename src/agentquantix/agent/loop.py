"""The harness-free agent: an OpenAI-compatible tool-calling loop.

This is what makes AgentQuantix work with nothing but an API key. OpenRouter,
OpenAI, Groq, Together, a local vLLM — anything that speaks the
`/chat/completions` shape with `tools` — drives the same tools the MCP server
exposes, with the same system prompt, so the behaviour matches what you get
inside Claude Code.

Implemented with urllib rather than the `openai` package, for the same reason
the MCP server avoids the `mcp` package: this file has to run wherever the
user happens to be, and a dependency is a thing that can be missing.

The approval gate is enforced in two independent places, which is the point:
the model is told not to call `start_quantization` without approval, AND the
tool itself refuses without `user_approved`, AND this loop asks the human
directly before letting that call through. A confused model cannot start a
six-hour job by accident.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from . import prompt as prompt_mod, tools as tools_mod

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-opus-5"

# Tools that spend hours, write to the Hub or touch the disk in a way the user
# would want to have agreed to first. The loop confirms these with the human
# regardless of what the model believes it has been authorised to do.
CONFIRM_BEFORE = {"start_quantization"}

# Shown when `aqx agent` is started with no prompt. Printed locally rather
# than asked of the model: it is the same every time, it costs nothing, and
# it cannot come back wrong.
#
# There used to be a DEFAULT_TRIGGER here that made the first turn "research
# the trending models", so the session opened with a two-minute 100-model
# sweep nobody had asked for. That also contradicted the agent's own first
# rule — the user decides WHEN to research — by having the harness ask on the
# user's behalf before they had said anything.
WELCOME = """\
What would you like to do?

  Find work        "what's worth quantizing?"        researches trending models
                   "what can this machine handle?"   probes CPU/RAM/disk first

  A specific model "quantize ibm-granite/granite-4.2-3b"
                   "how long would Qwen/Qwen3-0.6B take?"
                   any Hub repo id works - it does not need to be trending

  Finish a repo    "verify NANI-Nithin/Qwen3-0.6B-GGUF"
                   "rewrite the card for Qwen3-0.6B"

Nothing is downloaded, built or published until you name a model and approve
it. Type your request, or 'quit'."""


def _api_key():
    """First key that is actually set, in preference order.

    OpenRouter first because it is the one that reaches every model with a
    single key, then the direct provider keys, so `aqx agent` works out of the
    box in whichever setup the user already has.
    """
    for name in ("OPENROUTER_API_KEY", "AQX_API_KEY", "OPENAI_API_KEY",
                 "ANTHROPIC_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY"):
        if value := os.getenv(name):
            return value, name
    return None, None


def _openai_tools():
    """The shared registry in the function-calling shape."""
    return [{"type": "function",
             "function": {"name": tool["name"],
                          "description": tool["description"],
                          "parameters": tool["input_schema"]}}
            for tool in tools_mod.TOOLS]


class ApiError(RuntimeError):
    """A non-2xx from the model API, with the body kept for diagnosis.

    Held as structured data rather than flattened into a string, because
    almost every failure here is something the user can fix in thirty seconds
    once told which thing to fix — and a raw JSON blob in a traceback tells
    them nothing.
    """

    def __init__(self, code, body):
        self.code = code
        self.body = body if isinstance(body, dict) else {}
        self.text = body if isinstance(body, str) else json.dumps(body)
        super().__init__(f"HTTP {code}")

    @property
    def message(self):
        error = self.body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or "")
        return str(error or self.text)[:800]

    @property
    def metadata(self):
        error = self.body.get("error")
        return (error.get("metadata") or {}) if isinstance(error, dict) else {}


def _post(url, payload, key, timeout=600):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            # OpenRouter uses these for attribution; harmless elsewhere.
            "HTTP-Referer": "https://github.com/AgentQuantix",
            "X-Title": "AgentQuantix",
        },
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = raw[:800]
        raise ApiError(e.code, body) from None
    except urllib.error.URLError as e:
        raise ApiError(0, f"could not reach {url}: {e.reason}") from None


def _diagnose(error: ApiError, model, base_url):
    """Turn an API failure into something actionable. Returns a list of lines.

    Every branch names the specific thing to change. A model that cannot be
    routed to and a key that is out of credit both surface as an unhelpful
    404/402 otherwise, and they have nothing to do with each other.
    """
    message = error.message
    lowered = message.lower()
    lines = []

    if error.code == 0:
        lines.append(f"Could not reach the API: {message}")
        lines.append("  Check the network, or --base-url if you set one.")
        return lines

    if error.code in (401, 403):
        lines.append("The API key was rejected.")
        lines.append("  Check the key is current and has not been revoked.")
        return lines

    if error.code == 402 or "credit" in lowered or "quota" in lowered:
        lines.append("The account is out of credit for this request.")
        lines.append("  Top up, or pick a cheaper model.")
        return lines

    if error.code == 429:
        lines.append("Rate limited.")
        lines.append("  Wait a moment, or use a model with more headroom.")
        return lines

    if error.code >= 500:
        lines.append(f"The provider returned {error.code}: {message}")
        lines.append("  Usually transient - retry, or try another model.")
        return lines

    # ---- 404 / 400: almost always the model, and almost always one of three
    # specific things. Each gets its own fix.
    if "data policy" in lowered or "training" in lowered:
        lines.append(f"'{model}' was filtered out by your account's privacy "
                     "settings, so there was nothing left to route to.")
        for reason in error.metadata.get("ineligibility_reasons") or []:
            lines.append(f"  reason: {reason.get('reason')}")
            if url := reason.get("configure_url"):
                lines.append(f"  fix:    {url}")
        lines.append("  Or choose a model that does not require that "
                     "permission - the default below is one.")
    elif "tool" in lowered and ("support" in lowered or "endpoint" in lowered):
        lines.append(f"'{model}' cannot do tool calling, which this agent is "
                     "built entirely around.")
        lines.append("  Pick a model whose provider page lists Tools support.")
    elif error.code == 404:
        lines.append(f"'{model}' is not a model this endpoint can route to.")
        lines.append("  Check the exact id (provider/model), or drop --model "
                     "to use the default.")
    else:
        lines.append(f"HTTP {error.code}: {message}")

    lines.append("")
    lines.append(f"  Default model:  {DEFAULT_MODEL}")
    lines.append(f"  Endpoint:       {base_url}")
    lines.append("  The agent needs a model that supports tool calling; "
                 "everything it does is tools.")
    return lines


def _confirm(name, arguments):
    """Ask the human before a tool that costs real time and real bandwidth."""
    models = ", ".join(arguments.get("models") or []) or "(none named)"
    print(f"\n  The model wants to START QUANTIZATION for: {models}")
    print("  This runs for hours, uses the disk, and publishes to the Hub.")
    try:
        answer = input("  Allow? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def main(model=None, base_url=None, prompt=None, max_steps=40):
    key, key_name = _api_key()
    if not key:
        print("No API key found. Set one of OPENROUTER_API_KEY, AQX_API_KEY, "
              "OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY or "
              "TOGETHER_API_KEY.")
        return 1

    model = model or os.getenv("AQX_MODEL") or DEFAULT_MODEL
    base_url = (base_url or os.getenv("AQX_BASE_URL")
                or DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/chat/completions"

    print(f"AgentQuantix  model={model}  via={base_url}  key={key_name}")
    print("-" * 72)

    # With no prompt, ask before doing anything. The model is not consulted
    # until the user has said what they want, so an interactive session costs
    # nothing until it is actually given a job.
    if not prompt:
        print(WELCOME)
        try:
            prompt = input("\n> ").strip()
        except EOFError:
            return 0
        if not prompt or prompt.lower() in ("quit", "exit"):
            return 0

    messages = [
        {"role": "system", "content": prompt_mod.SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for step in range(max_steps):
        try:
            response = _post(url, {"model": model, "messages": messages,
                                   "tools": _openai_tools(),
                                   "tool_choice": "auto"}, key)
        except ApiError as e:
            # A failed call here is nearly always configuration, not a crash.
            # A traceback would bury the one line that says what to change.
            print()
            for line in _diagnose(e, model, base_url):
                print(line)
            return 1

        choices = response.get("choices") or []
        if not choices:
            print(f"No choices in the response: {json.dumps(response)[:400]}")
            return 1

        message = choices[0].get("message") or {}
        messages.append(message)

        if text := (message.get("content") or "").strip():
            print(f"\n{text}\n")

        calls = message.get("tool_calls") or []
        if not calls:
            # No tool call and nothing left to say: the model is waiting on the
            # human. Hand the turn back rather than looping.
            try:
                follow_up = input("> ").strip()
            except EOFError:
                return 0
            if not follow_up or follow_up.lower() in ("quit", "exit"):
                return 0
            messages.append({"role": "user", "content": follow_up})
            continue

        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}

            print(f"  [{step + 1}] {name}({', '.join(f'{k}={v}' for k, v in arguments.items())})")

            if name in CONFIRM_BEFORE and not _confirm(name, arguments):
                result = json.dumps({
                    "error": "The user declined. Do not retry this call; ask "
                             "them what they would like to do instead."})
            else:
                result = tools_mod.call_json(name, arguments)

            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "name": name, "content": result})

    print(f"\nStopped after {max_steps} steps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
