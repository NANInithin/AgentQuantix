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
    assert "agentquantix[full]" in hint


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


# =====================================================
# WHAT THE CONVERTER ACTUALLY NEEDS
# =====================================================
def test_sentencepiece_is_a_converter_requirement():
    """REGRESSION. The check listed only torch and transformers, so a run
    passed the plan-time gate, downloaded 1.27 GB of Qwen3 weights, and then
    died in set_vocab() with ModuleNotFoundError: sentencepiece.

    SentencePiece vocabularies are the default path for Llama, Qwen and Gemma
    -- most of what trends -- so this was the common case, not an edge one.
    """
    assert "sentencepiece" in source.CONVERTER_REQUIREMENTS
    assert "protobuf" in source.CONVERTER_REQUIREMENTS


def test_the_set_matches_llama_cpp_s_own_requirements():
    """These come from requirements-convert_legacy_llama.txt. gguf and numpy
    are already hard dependencies of ours, so they are not repeated here."""
    assert set(source.CONVERTER_REQUIREMENTS) == {
        "torch", "transformers", "sentencepiece", "protobuf"}


def test_protobuf_is_checked_by_its_import_name():
    """REGRESSION-IN-WAITING. protobuf installs, but `import protobuf` fails;
    the module is google.protobuf. Checking the package name would report it
    permanently missing and refuse every conversion on a working machine.
    """
    assert source.CONVERTER_REQUIREMENTS["protobuf"] == "google.protobuf"


def test_missing_deps_are_reported_by_package_name(monkeypatch):
    """The user has to type the PACKAGE name, not the module name."""
    monkeypatch.setattr(source, "CONVERTER_REQUIREMENTS",
                        {"protobuf": "definitely_not_installed_xyz"})
    assert source.converter_missing() == ["protobuf"]


def test_the_converter_hint_points_at_the_full_extra():
    """`[all]` does not contain them -- naming it is what sent a user round
    the loop twice."""
    hint = source.converter_hint(["sentencepiece"])
    assert "agentquantix[full]" in hint
    assert "agentquantix[all]" not in hint


def test_the_package_list_stops_at_the_end_of_the_sentence():
    """REGRESSION. transformers follows the list with "Run `pip install ...`".
    A character class containing "." ran straight past the full stop and
    produced a phantom package named "Run", which would have gone into the
    install command."""
    hint = source.remote_code_hint(TRANSFORMERS_MESSAGE)
    assert "Run" not in hint.split("(transformers")[0].replace(
        "not found", "")
    assert "--with Run" not in hint
    assert "torchvision." not in hint
    assert "--with torchvision" in hint
