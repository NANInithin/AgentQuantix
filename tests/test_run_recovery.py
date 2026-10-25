"""Failure injection at the real text orchestrator boundary; no Hub or models."""
from pathlib import Path
import threading

import pytest

from agentquantix import cli, config
from agentquantix.pipeline import run
from agentquantix.pipeline.job import Job, Status


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(config, "DELETE_AFTER_UPLOAD", True)
    monkeypatch.setattr(run.transfer, "pin", lambda *args: None)
    monkeypatch.setattr(run.time, "sleep", lambda *args: None)
    monkeypatch.setattr(run.sanity, "unloadable_reason", lambda *args: None)
    monkeypatch.setattr(run.imatrix_mod, "build", lambda *args: (True, [], None))
    monkeypatch.setattr(run.source_mod, "ensure_bf16",
                        lambda job, *args: job.bf16_path.write_bytes(b"source"))
    job = Job("org/Demo", "Demo", "owner/Demo-GGUF",
              ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"])

    class Api:
        remote = {job.bf16_path.name}
        fail = False
        error = None
        uploads = []

        def list_repo_files(self, **kwargs):
            if self.error:
                raise self.error
            return sorted(self.remote)

        def create_repo(self, **kwargs):
            pass

        def upload_file(self, **kwargs):
            if self.fail:
                raise OSError("outage")
            self.uploads.append(kwargs["path_in_repo"])
            self.remote.add(kwargs["path_in_repo"])

    api = Api()
    monkeypatch.setattr(run, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(run.build_mod, "run",
                        lambda cmd: Path(cmd[2]).write_bytes(b"quant"))
    return job, api, tmp_path


@pytest.mark.parametrize("sequential,workers", [(True, 1), (False, 1), (False, 3)])
def test_outage_stops_producers_without_exceeding_reserved_disk(harness, sequential, workers):
    job, api, root = harness
    api.fail = True
    options = run.RunOptions(sequential=sequential, quantize_workers=workers)
    result = run.process(job, root, "quantizer", "imatrix", options)
    retained = [p for p in job.models_dir.glob("*.gguf") if p != job.bf16_path]
    assert 1 <= len(retained) <= options.concurrent_quants
    assert result["failures"]
    assert not any(t.name.startswith("aqx-upload") for t in threading.enumerate())
    api.fail = False
    resumed = run.process(job, root, "quantizer", "imatrix", options)
    assert not resumed["failures"]
    assert all(job.quant_path(q).name in api.remote for q in job.quants)


def test_remote_listing_overrides_stale_local_upload_status(harness):
    job, api, root = harness
    Status(job.status_file).mark("uploaded", job.quant_path("Q4_0").name)
    result = run.process(job, root, "quantizer", "imatrix", run.RunOptions(sequential=True))
    assert not result["failures"]
    assert job.quant_path("Q4_0").name in api.uploads


def test_remote_outage_does_not_delete_or_requantize(harness):
    job, api, root = harness
    job.models_dir.mkdir(parents=True)
    artifact = job.quant_path("Q4_0")
    artifact.write_bytes(b"valuable local quant")
    api.error = OSError("Hub unavailable")
    with pytest.raises(OSError, match="unavailable"):
        run.process(job, root, "quantizer", "imatrix")
    assert artifact.read_bytes() == b"valuable local quant"


def test_interrupted_quantizer_never_leaves_a_final_artifact(harness, monkeypatch):
    job, api, root = harness

    def interrupt(cmd):
        Path(cmd[2]).write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(run.build_mod, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run.process(job, root, "quantizer", "imatrix")
    assert not job.quant_path("Q4_0").exists()
    assert not list(job.models_dir.glob("*.partial"))


@pytest.mark.parametrize("failure", ["pipeline", "verify", "missing", "suspect", "card", None])
def test_cli_exit_status_covers_pipeline_verification_and_card_failures(harness, monkeypatch, failure):
    job, api, root = harness
    job.assessment = {"bf16_gb": 1, "peak_disk_gb": 3, "upload_gb": 1,
                      "hours": {"total": 0.1}, "quant_sizes_gb": {"Q4_0": 1},
                      "peak_convert_gb": 2}
    monkeypatch.setattr(cli.research, "load_latest", lambda: None)
    monkeypatch.setattr(cli, "_select_interactively", lambda *args, **kwargs: [job.assessment])
    monkeypatch.setattr(cli.Job, "from_assessment", lambda *args: job)
    monkeypatch.setattr(cli.source_mod, "converter_missing", lambda: [])
    monkeypatch.setattr(cli.run_mod, "run_jobs", lambda *args, **kwargs: {
        "jobs": {job.base_name: {"failures": [("Q4_0", "failed")] if failure == "pipeline" else []}},
        "incomplete": [job.base_name] if failure == "pipeline" else []})
    verification = {"count": 1, "total_gb": 1, "last_modified": "now", "url": "unused", "files": [],
                    "missing": ["Q4_0"] if failure == "missing" else [],
                    "suspect": ["Q4_0"] if failure == "suspect" else []}
    if failure == "verify":
        verification["error"] = "offline"
    monkeypatch.setattr(cli.card, "verify", lambda *args: verification)

    def publish(*args, **kwargs):
        if failure == "card":
            raise OSError("card upload failed")

    monkeypatch.setattr(cli.card, "publish", publish)
    args = cli.build_parser().parse_args(["run", "org/Demo", "--yes"])
    assert cli.cmd_run(args) == (1 if failure else 0)
