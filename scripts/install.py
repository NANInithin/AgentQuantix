"""Install the harness wiring into the directory where the agent gets used.

AgentQuantix's home is AGI/AgentQuantix, but it is driven from INF — that is
where llama.cpp, the .venv and the scratch tree live, and where the existing
quantization scripts run. So the skill and the MCP config have to exist there
too, or `/quantix` in that directory finds nothing.

Copies rather than links: Claude Code does not follow Windows symlinks
reliably, and a broken link is a silent failure. Re-run after editing SKILL.md
(scripts/sync_adapters.py first, so AGENTS.md keeps up as well).

    python scripts/install.py                 # into INF
    python scripts/install.py <other-dir>
"""

from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / ".claude" / "skills" / "quantix" / "SKILL.md"
AGENTS = ROOT / "adapters" / "AGENTS.md"

def _default_target():
    """Where the harness files go: the work root the agent will actually use.

    Resolved through config so it matches wherever llama.cpp and the scratch
    tree ended up — an existing ~/Documents/INF, or ~/.agentquantix on a fresh
    machine. Hardcoding one user's Documents path made this Windows-only.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from agentquantix import config
    return config.WORK_ROOT


DEFAULT_TARGET = _default_target()


def mcp_servers():
    """Our MCP server entries, with paths resolved for THIS machine.

    Generated rather than read from a checked-in file: the old version hardcoded
    one user's Windows paths, which is fine for one laptop and useless for a
    Linux VM or a pip install. Prefers the installed `aqx` console script and
    falls back to running the module out of a source checkout.
    """
    aqx = shutil.which("aqx")
    if aqx:
        command, args, env = aqx, ["mcp"], {}
    else:
        command = sys.executable
        args = ["-m", "agentquantix.mcp_server"]
        env = {"PYTHONPATH": str(ROOT / "src")}

    env.update({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})
    return {
        "agentquantix": {"command": command, "args": args, "env": env},
        # The Hugging Face server, for ad-hoc Hub lookups alongside the agent's
        # own deterministic trending query.
        "huggingface": {"type": "http", "url": "https://huggingface.co/mcp"},
    }


def merge_mcp(target: Path):
    """Add our servers to the target's .mcp.json without touching the rest.

    The target may already define servers for other work; replacing the file
    wholesale would silently remove them.
    """
    ours = mcp_servers()
    path = target / ".mcp.json"

    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  {path} is not valid JSON - leaving it alone.")
            return False

    servers = existing.setdefault("mcpServers", {})
    changed = False
    for name, spec in ours.items():
        if servers.get(name) != spec:
            servers[name] = spec
            changed = True

    if changed:
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return changed


def main():
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_TARGET
    if not target.exists():
        print(f"target does not exist: {target}")
        return 1

    skill_dest = target / ".claude" / "skills" / "quantix" / "SKILL.md"
    skill_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SKILL, skill_dest)
    print(f"  skill      {skill_dest}")

    if AGENTS.exists():
        shutil.copyfile(AGENTS, target / "AGENTS.md")
        print(f"  agents     {target / 'AGENTS.md'}")

    print(f"  mcp        {target / '.mcp.json'}"
          f"{'  (updated)' if merge_mcp(target) else '  (already current)'}")

    print(f"\nInstalled into {target}.")
    print("Claude Code: restart the session there, approve the MCP servers, "
          "then /quantix.")
    print("The Hugging Face server needs a token from "
          "https://huggingface.co/settings/mcp on first connect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
