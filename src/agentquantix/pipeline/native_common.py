"""Shared helpers for revision-bound native release pipelines."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from contextlib import contextmanager
from urllib.request import urlopen


def _slug(repo_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", repo_id).strip("-")[:120]


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
                       encoding="utf-8")
    os.replace(pending, path)


def _command(argv, *, cwd=None, env=None):
    result = subprocess.run([str(value) for value in argv], cwd=cwd, capture_output=True,
                            text=True, errors="replace", stdin=subprocess.DEVNULL, env=env)
    if result.returncode:
        raise RuntimeError(f"native command failed ({result.returncode}): {' '.join(map(str, argv))}\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout + result.stderr


def _record_stage(journal_path, journal, key, path):
    journal[key] = {"path": str(path), "sha256": _hash(path), "bytes": Path(path).stat().st_size}
    _write(journal_path, journal)


def _valid_stage(journal, key):
    item = journal.get(key, {})
    path = Path(item.get("path", ""))
    return path.is_file() and item.get("sha256") == _hash(path)


def _free_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


@contextmanager
def _server(binary, model, root, arguments, *, free_port_fn=None, open_url=None,
            spawn=None, monotonic=None, sleep=None):
    free_port_fn = free_port_fn or _free_port
    open_url = open_url or urlopen
    spawn = spawn or subprocess.Popen
    monotonic = monotonic or time.monotonic
    sleep = sleep or time.sleep
    port = free_port_fn()
    base_url = f"http://127.0.0.1:{port}"
    process = spawn([str(binary), "-m", str(model), *map(str, arguments),
                     "--host", "127.0.0.1", "--port", str(port)],
                    cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = monotonic() + 120
        while monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("llama-server exited before becoming healthy")
            try:
                with open_url(f"{base_url}/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except Exception:
                sleep(0.25)
        else:
            raise RuntimeError("llama-server health check timed out")
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
