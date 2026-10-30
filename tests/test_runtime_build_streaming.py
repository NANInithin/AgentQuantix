"""Offline tests for streamed native command output."""
import json
import sys
import threading
import time

import pytest

from agentquantix import runtime_builders as builders


def test_live_log_and_record_capture_output_before_process_exit(tmp_path, capsys):
    log_path = tmp_path / "build.json"
    live_path = tmp_path / "build.log"
    result = {}
    script = (
        "import sys,time\n"
        "print('configure-start', flush=True)\n"
        "print('compiler-note', file=sys.stderr, flush=True)\n"
        "time.sleep(0.5)\n"
        "print('build-finished', flush=True)\n")

    def execute():
        result["record"] = builders._execute(
            [sys.executable, "-c", script], log_path, timeout=5)

    worker = threading.Thread(target=execute)
    worker.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if live_path.is_file() and "configure-start" in live_path.read_text(encoding="utf-8"):
            break
        if not worker.is_alive():
            break
        time.sleep(0.01)

    assert live_path.is_file()
    assert "configure-start" in live_path.read_text(encoding="utf-8")
    assert worker.is_alive()
    worker.join(timeout=5)
    assert not worker.is_alive()

    record = result["record"]
    assert set(record) == {
        "argv", "cwd", "returncode", "stdout", "stderr", "duration_seconds"}
    assert record["returncode"] == 0
    assert record["stdout"] == "configure-start\nbuild-finished\n"
    assert record["stderr"] == "compiler-note\n"
    assert set(json.loads(log_path.read_text(encoding="utf-8"))) == set(record)
    assert "compiler-note" in live_path.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_label_echoes_child_output_to_stderr_only(tmp_path, capsys):
    script = (
        "import sys\n"
        "print('native stdout line', flush=True)\n"
        "print('native stderr line', file=sys.stderr, flush=True)\n")

    builders._execute(
        [sys.executable, "-c", script], tmp_path / "labelled.json",
        timeout=5, label="llama.cpp: build")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "native stdout line" in captured.err
    assert "native stderr line" in captured.err


def test_nonzero_exit_reports_tail_and_writes_record(tmp_path):
    log_path = tmp_path / "failed.json"
    script = (
        "import sys\n"
        "print('stdout detail', flush=True)\n"
        "print('compiler failed at end', file=sys.stderr, flush=True)\n"
        "sys.exit(7)\n")

    with pytest.raises(builders.RuntimeBuildError, match="compiler failed at end"):
        builders._execute([sys.executable, "-c", script], log_path, timeout=5)

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["returncode"] == 7
    assert record["stdout"] == "stdout detail\n"
    assert record["stderr"] == "compiler failed at end\n"


def test_timeout_kills_child_and_writes_partial_output(tmp_path):
    log_path = tmp_path / "timeout.json"
    finished = tmp_path / "finished"
    script = (
        "from pathlib import Path\n"
        "import sys,time\n"
        "print('still building', flush=True)\n"
        "time.sleep(5)\n"
        "Path(sys.argv[1]).touch()\n")

    with pytest.raises(builders.RuntimeBuildError, match="timed out after"):
        builders._execute(
            [sys.executable, "-c", script, str(finished)],
            log_path, timeout=0.1)

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["stdout"] == "still building\n"
    assert record["returncode"] is None
    assert "timed out" in record["stderr"]
    assert not finished.exists()
