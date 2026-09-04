"""Reporting failures from the converter in a form that can be acted on.

A conversion is the first expensive thing a run does, and it is the step most
exposed to things outside our control: legacy checkpoints, architectures
upstream cannot map, and — the case here — custom modeling code that imports
whatever its author happened to have installed.

None of those are fixable by AgentQuantix. What IS ours is whether the message
tells you what to do next.
"""

import subprocess

from agentquantix.pipeline import build, source


# =====================================================
# CUSTOM-CODE DEPENDENCIES
# =====================================================
TRANSFORMERS_MESSAGE = (
    "ImportError: This modeling file requires the following packages that "
    "were not found in your environment: PIL, addict, matplotlib, "
    "torchvision. Run `pip install PIL addict matplotlib torchvision`"
)


def test_import_names_are_translated_to_package_names():
    """REGRESSION. transformers reports IMPORT names and tells you to pip
    install them verbatim. `pip install PIL` installs a dead 2005 stub, so
    following that advice literally does not fix the run.
    """
    hint = source.remote_code_hint(TRANSFORMERS_MESSAGE)

    assert hint is not None
    assert "pillow" in hint
    assert "--with PIL" not in hint
    # The ones that need no translation must survive untouched.
    for package in ("addict", "matplotlib", "torchvision"):
        assert f"--with {package}" in hint


def test_the_hint_explains_the_rename():
    hint = source.remote_code_hint(TRANSFORMERS_MESSAGE)
    assert "PIL -> pillow" in hint
    # Only the renamed one is worth calling out.
    assert "addict -> addict" not in hint


def test_the_hint_is_a_runnable_command():
    hint = source.remote_code_hint(TRANSFORMERS_MESSAGE)
    assert "uv tool install --force --reinstall" in hint
    assert "agentquantix[all]" in hint


def test_cv2_and_sklearn_are_translated():
    """Neither exists on PyPI under its import name at all."""
    hint = source.remote_code_hint(
        "ImportError: This modeling file requires the following packages "
        "that were not found in your environment: cv2, sklearn")
    assert "opencv-python-headless" in hint
    assert "scikit-learn" in hint


def test_an_unrelated_failure_is_left_alone():
    """Returning None is what lets the caller re-raise the original error
    rather than replacing a real diagnosis with a guess."""
    assert source.remote_code_hint(
        "ValueError: Can not map tensor 'h.0.attn.bias'") is None
    assert source.remote_code_hint("exit status 1") is None


# =====================================================
# NO SUBPROCESS MAY ASK A QUESTION
# =====================================================
def test_run_verbose_closes_stdin(monkeypatch):
    """REGRESSION. A model with custom code makes transformers ask

        Do you wish to run the custom code? [y/N]

    on stdin. Inherited stdin means a converter running inside a quantize
    worker blocks forever on a prompt nobody can see. Closed stdin turns that
    into an immediate failure that gets reported.
    """
    seen = {}

    class FakeProcess:
        stdout = iter(())

        def wait(self):
            return 0

    def fake_popen(cmd, **kwargs):
        seen.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(build.subprocess, "Popen", fake_popen)
    build.run_verbose(["echo", "hi"], label="test")

    assert seen["stdin"] is subprocess.DEVNULL
