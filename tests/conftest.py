"""Shared test setup.

Every test in this suite is OFFLINE and needs no Hugging Face token. That is a
deliberate constraint, not a limitation: CI has no credentials, and a suite
that silently depends on the network fails for reasons that have nothing to do
with the change under test.

Anything that genuinely needs the Hub belongs behind an explicit marker and is
not part of the default run.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Isolate every test from the developer's real environment.

    Without this, results depend on whether the machine running the tests
    happens to have a token, an existing work root, or a populated timing
    history — so a suite that passes locally fails in CI, or worse, the other
    way round.
    """
    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MLE2",
                     "AQX_NAMESPACE", "AQX_XET", "AQX_HOME",
                     "AQX_WORK_ROOT", "INF_ROOT", "HF_HUB_DISABLE_XET",
                     "AQX_SEQUENTIAL", "AQX_UPLOAD_WORKERS",
                     "AQX_QUANTIZE_WORKERS", "AQX_QUEUE_DEPTH",
                     "AQX_QUANTIZE_THREADS", "GITHUB_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # config.TOKEN is resolved at import, and one source is the token file that
    # `hf auth login` writes — so clearing the environment is not enough to
    # keep a developer's real credentials out of the tests. Blanked explicitly;
    # anything that wants a token sets one itself.
    from agentquantix import config
    monkeypatch.setattr(config, "TOKEN", None)
    monkeypatch.setattr(config, "_namespace_cache", None)
    yield


@pytest.fixture
def candidate():
    """A plain, supported, dense 7B-ish candidate to vary from."""
    from agentquantix.hub import Candidate

    model = Candidate(repo_id="someorg/Some-Model-7B", rank=1)
    model.params = 7_000_000_000
    model.architectures = ["LlamaForCausalLM"]
    model.model_type = "llama"
    model.n_layers = 32
    model.hidden_size = 4096
    model.vocab_size = 32000
    model.source_bytes = 14 * 1024 ** 3
    return model


@pytest.fixture
def sysinfo():
    """A machine roughly like the one this was built on, with room to work."""
    return {
        "ram_total_gb": 16.0,
        "ram_available_gb": 10.0,
        "vram_free_gb": 7.5,
        "fast_memory_gb": 17.5,
        "disk_free_gb": 400.0,
        "disk_gbs": 1.2,
        "hf_token": True,
        "llama": {"present": True, "binaries": {"llama-quantize": "x"}},
        "hub_backends": {"hf_xet": True, "hf_transfer": False,
                         "xet_disabled": False, "can_exceed_http_limit": True},
    }
