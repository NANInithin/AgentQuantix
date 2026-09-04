"""How the Hugging Face token is found.

This project grew up reading a private variable, MLE2, which is meaningless to
anyone else and appeared nowhere in the Hub's own documentation. A public tool
has to accept what the ecosystem already sets, and the most common way for a
working setup to look unconfigured is requiring an environment variable from
someone who has already run `hf auth login`.
"""

from agentquantix import config


def _resolve(monkeypatch, env=None, login_token=None):
    """Run the resolver with a controlled environment and login file."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MLE2"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: login_token)
    return config._resolve_token()


def test_hf_token_is_the_documented_variable(monkeypatch):
    assert _resolve(monkeypatch, {"HF_TOKEN": "hf_standard"}) == "hf_standard"


def test_the_other_official_variable_also_works(monkeypatch):
    assert _resolve(monkeypatch,
                    {"HUGGING_FACE_HUB_TOKEN": "hf_legacy"}) == "hf_legacy"


def test_a_login_is_enough_on_its_own(monkeypatch):
    """`hf auth login` writes ~/.cache/huggingface/token. Someone who has done
    that is configured, and should not be told to export a variable."""
    assert _resolve(monkeypatch, login_token="hf_from_login") == "hf_from_login"


def test_the_environment_wins_over_a_stored_login(monkeypatch):
    """An explicitly exported token is a deliberate override - typically a
    write-scoped one, where the stored login may be read-only."""
    assert _resolve(monkeypatch, {"HF_TOKEN": "hf_explicit"},
                    login_token="hf_stored") == "hf_explicit"


def test_the_legacy_private_variable_still_works(monkeypatch):
    """Undocumented, but an existing machine should not break on upgrade."""
    assert _resolve(monkeypatch, {"MLE2": "hf_old"}) == "hf_old"


def test_hf_token_beats_the_legacy_variable(monkeypatch):
    assert _resolve(monkeypatch,
                    {"MLE2": "hf_old", "HF_TOKEN": "hf_new"}) == "hf_new"


def test_no_token_anywhere_is_none_not_an_exception(monkeypatch):
    """The research step runs fine without one; only uploading needs it."""
    assert _resolve(monkeypatch) is None


def test_a_broken_hub_import_does_not_crash_startup(monkeypatch):
    """The login lookup is a convenience. If it fails, the CLI must still run
    and simply report that no token was found."""
    import huggingface_hub

    def explode():
        raise RuntimeError("hub unavailable")

    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "MLE2"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", explode)
    assert config._resolve_token() is None
