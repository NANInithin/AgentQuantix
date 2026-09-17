from types import SimpleNamespace

from agentquantix import archsupport, config


def _candidate():
    return SimpleNamespace(
        repo_id="org/NewVoice", org="org", name="NewVoice",
        model_type="new_voice", architectures=["NewVoiceForConditionalGeneration"])


def test_voice_fork_hunt_searches_all_runtime_upstreams(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    requested = []

    def github(url, timeout=20):
        requested.append(url)
        if "search/repositories" in url and "audio.cpp" in url:
            return {"items": [{"full_name": "org/audio.cpp",
                                "html_url": "https://github.com/org/audio.cpp"}]}
        if "/repos/org/audio.cpp/branches" in url:
            return [{"name": "model/new-voice"}]
        return {"items": []}

    monkeypatch.setattr(archsupport, "_github_json", github)
    leads = archsupport.find_voice_forks(_candidate(), use_cache=False)
    assert leads[0]["backend"] == "audio.cpp"
    assert leads[0]["kind"] == "publisher-fork"
    assert leads[0]["ref"] == "model/new-voice"
    assert leads[0]["actionable"] is False
    for upstream in archsupport.VOICE_UPSTREAMS.values():
        assert any(upstream in url for url in requested)


def test_voice_fork_hunt_caches_hits(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    calls = {"count": 0}

    def github(url, timeout=20):
        calls["count"] += 1
        return {"items": []}

    monkeypatch.setattr(archsupport, "_github_json", github)
    assert archsupport.find_voice_forks(_candidate()) == []
    first = calls["count"]
    assert archsupport.find_voice_forks(_candidate()) == []
    assert calls["count"] == first
