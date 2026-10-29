"""Offline build-adapter control-flow tests; these are not native acceptance."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from agentquantix import runtime_builders as builders


@pytest.fixture
def native(tmp_path, monkeypatch):
    source, build = tmp_path / "source", tmp_path / "candidate-build"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("project(native)\n", encoding="utf-8")
    (source / "convert_hf_to_gguf.py").write_text("# converter\n", encoding="utf-8")
    commands = []
    control = {"fail_build": False, "diagnostic": "Usage: native --help", "probe_code": 0}

    def run(command, **kwargs):
        command = list(map(str, command))
        commands.append(command)
        if command[0] == "git":
            if "rev-parse" in command:
                output = "a" * 40
            else:
                output = ""
            return SimpleNamespace(returncode=0, stdout=output, stderr="")
        if command[0] == "cmake":
            if "--build" in command:
                if control["fail_build"]:
                    return SimpleNamespace(returncode=2, stdout="", stderr="compiler failed")
                start, end = command.index("--target") + 1, command.index("--parallel")
                for target in command[start:end]:
                    path = Path(command[command.index("--build") + 1]) / "bin" / (target + ".exe")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"native executable")
                (path.parent / "ggml-cpu.dll").write_bytes(b"native backend")
            return SimpleNamespace(returncode=0, stdout="Configured in C:/Documents/INF/source", stderr="")
        return SimpleNamespace(returncode=control["probe_code"], stdout=control["diagnostic"], stderr="")

    monkeypatch.setattr(builders.subprocess, "run", run)
    monkeypatch.setattr(builders.native_build, "build_jobs", lambda: "2")
    options = {"engine": "llama.cpp", "flags": ["-DGGML_CUDA=OFF"],
               "targets": ["llama-quantize", "llama-imatrix"], "compute_backend": "cpu"}
    return SimpleNamespace(source=source, build=build, commands=commands, control=control, options=options)


def test_candidate_build_uses_separate_output_and_retains_native_contract(native):
    result = builders.build_runtime("llama.cpp", native.source, native.build, native.options)
    configure = next(command for command in native.commands if command[:2] == ["cmake", "-S"])
    assert configure[configure.index("-B") + 1] == str(native.build)
    assert not (native.source / "build").exists()
    assert set(result["binaries"]) == {"llama-quantize", "llama-imatrix"}
    assert result["provenance"]["revision"] == "a" * 40
    assert result["resources"]["convert_hf_to_gguf.py"]["sha256"]
    assert result["resources"]["native/bin/ggml-cpu.dll"]["sha256"]
    assert json.loads((native.build / "aqx-build.json").read_text(encoding="utf-8"))["source"] == str(native.source)


def test_failed_candidate_keeps_other_installation_and_diagnostics(native, tmp_path):
    current = tmp_path / "current" / "bin" / "llama-quantize.exe"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"working version")
    native.control["fail_build"] = True
    with pytest.raises(builders.RuntimeBuildError, match="compiler failed"):
        builders.build_runtime("llama.cpp", native.source, native.build, native.options)
    assert current.read_bytes() == b"working version"
    assert not (native.build / "aqx-build.json").exists()
    assert json.loads((native.build / "aqx-logs/build.json").read_text(encoding="utf-8"))["returncode"] == 2


def test_probe_existing_does_not_reconfigure_or_write_to_checkout(native):
    build = native.source / "build/bin"
    build.mkdir(parents=True)
    for name in ("llama-quantize", "llama-imatrix", "llama-tts"):
        (build / (name + ".exe")).write_bytes(b"existing")
    before = sorted(path.relative_to(native.source) for path in native.source.rglob("*"))
    result = builders.probe_existing("llama.cpp", native.source)
    assert result["provenance"]["adopted"]
    assert "llama-tts" in result["binaries"]
    assert before == sorted(path.relative_to(native.source) for path in native.source.rglob("*"))
    assert not any(command[0] == "cmake" for command in native.commands)
    assert not any("update" in command for command in native.commands)


def test_legacy_whisper_quantizer_is_mapped_without_renaming_artifact(native):
    config = native.source / "examples/quantize/CMakeLists.txt"
    config.parent.mkdir(parents=True)
    config.write_text("add_executable(quantize main.cpp)\n", encoding="utf-8")
    build = native.source / "build/bin"
    build.mkdir(parents=True)
    for name in ("whisper-cli", "quantize"):
        (build / (name + ".exe")).write_bytes(b"native whisper ggml runtime")
    result = builders.probe_existing("whisper.cpp", native.source)
    assert Path(result["binaries"]["whisper-quantize"]).name == "quantize.exe"
    assert result["provenance"]["engine"] == "whisper.cpp"


def test_adopted_media_pin_gets_shipped_catalog_without_writing_checkout(native, monkeypatch, tmp_path):
    from agentquantix import media_catalog
    catalog_path = tmp_path / "shipped-capabilities.json"
    catalog_path.write_text(json.dumps({"revision": "a" * 40}), encoding="utf-8")
    monkeypatch.setattr(media_catalog, "ENGINE_REVISION", "a" * 40)
    monkeypatch.setattr(media_catalog, "CATALOG_PATH", catalog_path)
    binary = native.source / "build/bin/sd-cli.exe"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native")
    result = builders.probe_existing("stable-diffusion.cpp", native.source)
    assert result["resources"]["capability_catalog"]["path"] == str(catalog_path)
    assert not (native.source / "build/capabilities.json").exists()
    monkeypatch.setattr(media_catalog, "ENGINE_REVISION", "b" * 40)
    assert builders.probe_existing("stable-diffusion.cpp", native.source) is None


@pytest.mark.parametrize("diagnostic", ["value = NaN", "value = -inf", "assertion failed: weights"])
def test_native_command_rejects_nonfinite_and_assertions(native, diagnostic):
    native.control["diagnostic"] = diagnostic
    with pytest.raises(builders.RuntimeBuildError, match="native command failed"):
        builders._execute(["native", "inference"])


def test_help_probe_accepts_infinity_documentation_but_inference_rejects_nan(tmp_path):
    binary = tmp_path / "fake-native.py"
    binary.write_text(
        "import sys\n"
        "if '--help' in sys.argv:\n"
        "    print('(default: -1, -1 = infinity)')\n"
        "else:\n"
        "    print('nan')\n",
        encoding="utf-8")
    assert builders._execute([sys.executable, binary, "--help"], help_probe=True)["returncode"] == 0
    with pytest.raises(builders.RuntimeBuildError, match="native command failed"):
        builders._execute([sys.executable, binary, "inference"])


def test_quantizer_usage_exit_is_accepted_but_other_failure_is_not(native):
    native.control["probe_code"] = 1
    assert builders._execute(["native", "--help"], help_probe=True)["returncode"] == 1
    native.control["diagnostic"] = "missing native shared library"
    with pytest.raises(builders.RuntimeBuildError):
        builders._execute(["native", "--help"], help_probe=True)


def test_build_fingerprint_records_backend_targets_and_toolchain(monkeypatch):
    monkeypatch.setattr(builders.native_build, "has_cuda_toolkit", lambda: False)
    monkeypatch.setattr(builders.native_build, "find_tool", lambda name: None)
    monkeypatch.setattr(builders.native_build, "msvc_env_bat", lambda: None)
    result = builders.fingerprint("llama.cpp", required_targets=["llama-tts"])
    assert result["compute_backend"] == "cpu"
    assert "-DGGML_CUDA=OFF" in result["flags"]
    assert result["targets"] == ["llama-quantize", "llama-imatrix", "llama-tts"]
    assert result["adapter_schema"] == 1
    with pytest.raises(builders.RuntimeBuildError, match="CUDA toolkit"):
        builders.fingerprint("audio.cpp", "cuda")


def test_retrieval_runtime_targets_and_clip_cpp_cmake4_policy(native, monkeypatch):
    monkeypatch.setattr(builders.native_build, "has_cuda_toolkit", lambda: False)
    monkeypatch.setattr(builders.native_build, "find_tool",
                        lambda name: "cmake" if name == "cmake" else None)
    monkeypatch.setattr(builders.native_build, "msvc_env_bat", lambda: None)
    monkeypatch.setattr(builders.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(
                            stdout="cmake version 4.0.0\n", returncode=0))
    llama = builders.fingerprint(
        "llama.cpp", required_targets=["llama-server", "llama-quantize"])
    clip = builders.fingerprint("clip.cpp")
    assert {"llama-server", "llama-quantize"} <= set(llama["targets"])
    assert clip["compute_backend"] == "cpu"
    assert {"extract", "zsl", "quantize"} <= set(clip["targets"])
    assert "-DGGML_CUDA=OFF" not in clip["flags"]
    assert "-DGGML_CUBLAS=OFF" in clip["flags"]
    assert "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" in clip["flags"]
    assert "-DCLIP_NATIVE=OFF" in clip["flags"]


def test_msvc_configuration_and_build_share_developer_environment(native, monkeypatch):
    monkeypatch.setattr(builders.sys, "platform", "win32")
    run = builders.subprocess.run
    scripts = []

    def windows_run(command, **kwargs):
        if str(command[0]) == "cmd.exe":
            # `.\aqx-build.bat` is a cmd.exe path; the test may run on POSIX.
            script = Path(kwargs["cwd"]) / command[-1].rsplit("\\", 1)[-1]
            scripts.append(script.read_text(encoding="utf-8"))
            for target in native.options["targets"]:
                path = native.build / "bin" / (target + ".exe")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"windows native executable")
        return run(command, **kwargs)

    monkeypatch.setattr(builders.subprocess, "run", windows_run)
    options = {**native.options, "vcvars": "C:/Build Tools/vcvars64.bat", "generator": "Ninja"}
    result = builders.build_runtime("llama.cpp", native.source, native.build, options)
    assert len(scripts) == 1
    assert 'call "C:/Build Tools/vcvars64.bat"' in scripts[0]
    assert '"cmake" "-S"' in scripts[0]
    assert '"cmake" "--build"' in scripts[0]
    assert '"-G" "Ninja"' in scripts[0]
    assert result["binaries"]["llama-quantize"]


@pytest.mark.skipif(builders.sys.platform != "win32", reason="Windows cmd quoting")
def test_windows_build_script_path_with_spaces_and_ampersand(tmp_path):
    directory = tmp_path / "build & candidate"
    directory.mkdir()
    script = directory / "probe.bat"
    script.write_text("@echo off\r\necho native-command-ok\r\n", encoding="utf-8")
    result = builders._execute(["cmd.exe", "/d", "/v:off", "/c", ".\\" + script.name], cwd=directory)
    assert result["stdout"].strip() == "native-command-ok"


def test_media_initializes_only_its_build_submodules_and_records_catalog(native, monkeypatch):
    from agentquantix import media_catalog
    monkeypatch.setattr(media_catalog, "generate_catalog", lambda source, expected_revision:
                        {"revision": expected_revision, "capabilities": []})
    options = {**native.options, "engine": "stable-diffusion.cpp", "targets": ["sd-cli"]}
    result = builders.build_runtime("stable-diffusion.cpp", native.source, native.build, options)
    command = next(command for command in native.commands if "update" in command)
    assert command[-3:] == ["ggml", "thirdparty/libwebp", "thirdparty/libwebm"]
    assert "capability_catalog" in result["resources"]
    assert json.loads((native.build / "capabilities.json").read_text(encoding="utf-8"))["revision"] == "a" * 40


@pytest.mark.parametrize("value", ['bad"argument', "bad\nargument"])
def test_msvc_batch_arguments_reject_shell_injection(value):
    with pytest.raises(builders.RuntimeBuildError):
        builders._batch_quote(value)
    assert builders._batch_quote("C:/a&b/100% native") == '"C:/a&b/100%% native"'
