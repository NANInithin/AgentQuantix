"""Regenerate the harness instruction files from the one source.

src/agentquantix/agent/prompt.py holds the agent's instructions. Everything
else that carries them is generated from it:

    .claude/skills/quantix/SKILL.md   Claude Code (frontmatter + extra notes)
    adapters/AGENTS.md                Codex, OpenCode, Kimi

This exists because the alternative was tried and failed. When the skill and
the system prompt were maintained as two documents, a correction to the
peak-disk arithmetic landed in one and not the other, and the agent went on
confidently telling the user the old wrong number. Generated files cannot
drift from their source.

    python scripts/sync_adapters.py

Edit prompt.py, run this, commit all three. Do not edit the markdown by hand;
it is overwritten.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentquantix.agent import prompt as prompt_mod   # noqa: E402

SKILL = ROOT / ".claude" / "skills" / prompt_mod.SKILL_NAME / "SKILL.md"
AGENTS = ROOT / "adapters" / "AGENTS.md"

GENERATED = ("<!-- GENERATED from src/agentquantix/agent/prompt.py by "
             "scripts/sync_adapters.py. Edit the prompt, not this file. -->")


def write_skill():
    """Claude Code's skill: YAML frontmatter plus the prompt and its notes."""
    front = "\n".join([
        "---",
        f"name: {prompt_mod.SKILL_NAME}",
        f"description: {prompt_mod.SKILL_DESCRIPTION}",
        "---",
        "",
        GENERATED,
        "",
    ])
    SKILL.parent.mkdir(parents=True, exist_ok=True)
    SKILL.write_text(front + prompt_mod.markdown(include_claude_notes=True),
                     encoding="utf-8")
    return SKILL


def write_agents():
    """AGENTS.md for the harnesses without a skill format of their own."""
    AGENTS.parent.mkdir(parents=True, exist_ok=True)
    AGENTS.write_text(
        GENERATED + "\n\n"
        + prompt_mod.markdown(include_claude_notes=False)
        + "\n## Tools\n\n"
        "These come from the `agentquantix` MCP server. Without it, the CLI "
        "drives identical code: `aqx research`, `aqx show <model>`, "
        "`aqx run <model>`, `aqx verify <repo>`, `aqx card <repo>`.\n",
        encoding="utf-8")
    return AGENTS


def main():
    for path in (write_skill(), write_agents()):
        print(f"  wrote {path.relative_to(ROOT)}")
    print("\nRun scripts/install.py to push these into the working directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
