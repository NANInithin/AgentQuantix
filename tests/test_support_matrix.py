from pathlib import Path

from agentquantix.tasks.registry import TASKS


def test_support_matrix_covers_every_task_spec():
    matrix = Path(__file__).resolve().parents[1] / "docs" / "SUPPORT_MATRIX.md"
    content = matrix.read_text(encoding="utf-8")

    assert all(f"`{task}`" in content for task in TASKS)
