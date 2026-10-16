"""Offline integration with real fake-native subprocesses and real GGUF files."""
import json
from pathlib import Path
import shutil
import struct
import sys
from types import SimpleNamespace

import pytest

from agentquantix import config, media, media_quality
from agentquantix.pipeline import media_release


def safetensors(path):
    header = json.dumps({"model.weight": {"dtype": "F32", "shape": [256, 256],
                        "data_offsets": [0, 256 * 256 * 4]}}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.125) * (256 * 256))


@pytest.fixture
def native(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.setattr(config, "TEMP_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    fixtures = tmp_path / "fixtures"
    shutil.copytree(media.FIXTURES, fixtures)
    fixture_index = json.loads((fixtures / "fixtures.json").read_text(encoding="utf-8"))
    fixture_index["prompts"] = fixture_index["prompts"][:1]
    (fixtures / "fixtures.json").write_text(json.dumps(fixture_index), encoding="utf-8")
    monkeypatch.setattr(media, "FIXTURES", fixtures)
    script = tmp_path / "fake_sd.py"
    script.write_text('''
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from gguf import GGUFWriter
parser=argparse.ArgumentParser()
parser.add_argument('-M', default='img_gen')
parser.add_argument('-o')
parser.add_argument('-W', type=int, default=64)
parser.add_argument('-H', type=int, default=64)
args, rest=parser.parse_known_args()
if args.M == 'convert':
    writer=GGUFWriter(args.o, 'stable-diffusion')
    writer.add_tensor('model.weight', np.full((256,256), 0.125, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
else:
    image=Image.new('RGB', (args.W,args.H), 'ivory')
    ImageDraw.Draw(image).rectangle((4,4,args.W//2,args.H//2), fill='red')
    image.save(args.o)
print('fake native completed')
''', encoding="utf-8")
    execute = media_release.execute

    def run(command, *args, **kwargs):
        return execute([sys.executable, script, *command[1:]], *args, **kwargs)

    monkeypatch.setattr(media_release, "execute", run)
    monkeypatch.setattr(media_release, "ensure_backend", lambda *args: (Path(sys.executable),
                        {"engine_revision": media_release.ENGINE_REVISION, "compute_backend": "cpu", "binary_sha256": media.file_hash(sys.executable)}))
    from agentquantix.media_catalog import load_catalog
    catalog = load_catalog()
    catalog["source_types"] = ["f16", "q4_0"]
    monkeypatch.setattr(media_release, "load_catalog", lambda: catalog)
    source_dir = tmp_path / "upstream"
    source_dir.mkdir()
    for name in ("model.safetensors", "vae.safetensors"):
        safetensors(source_dir / name)
    downloads = []

    def download(repo_id, filename, *, local_dir, **kwargs):
        downloads.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source_dir / filename, target)
        return str(target)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text('''import json, sys
request=json.load(open(sys.argv[1]))
print(json.dumps({'model_revision':request['model_revision'], 'metrics':dict.fromkeys(
['perceptual_similarity','prompt_alignment','determinism','conditioning_preservation','temporal_consistency'],0.95)}))
''', encoding="utf-8")
    profile = {"family": "flux", "task": "text-to-image", "calibration_evidence": "synthetic test-only profile",
               "thresholds": {"perceptual_similarity": 0.9, "prompt_alignment": 0.9, "determinism": 0.9},
               "evaluator": {"command": [sys.executable, str(evaluator)], "sha256": media.file_hash(sys.executable),
                             "artifacts": {str(evaluator): media.file_hash(evaluator)}, "model_revision": "d" * 40, "name": "test-evaluator"}}
    metadata = {"repo_id": "org/Synthetic", "revision": "a" * 40, "pipeline_tag": "text-to-image", "tags": [],
                "library_name": "diffusers", "configs": {}, "license": "apache-2.0",
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": media.file_hash(p)} for p in source_dir.iterdir()}}
    contract = {"task": "text-to-image", "required_roles": ["diffusion_model", "vae"], "evidence": {"native": "docs/flux.md"},
                "quality_profile": profile, "settings": {"width": 64, "height": 64},
                "components": [{"role": "diffusion_model", "source_path": "model.safetensors", "quantizable": True},
                               {"role": "vae", "source_path": "vae.safetensors", "precision": "f32", "quantizable": False}]}
    plan = media.plan_release("org/Synthetic", family="flux", metadata=metadata, contract=contract,
                              quants=["q4_0"], hunt_forks=False, compute_backend="cpu", sequential=True)
    return SimpleNamespace(plan=plan, downloads=downloads, script=script, evaluator=evaluator)


def test_native_converter_probe_overrides_static_fallback(native):
    result = media_release.converter_types(binary=Path(sys.executable))
    assert result["probed"] and result["types"] == ["f16", "q4_0"]
    assert media_release.converter_types()["probed"]


def test_real_subprocess_conversion_runtime_review_and_resume(native, monkeypatch):
    result = media_release.run_release(native.plan, user_approved=True)
    assert len(result["candidates"]) == 2
    assert all(c["quality"]["passed"] for c in result["candidates"])
    root = media_release.run_dir(result["run_id"])
    before = len(native.downloads)
    journal = json.loads((root / "journal.json").read_text())
    assert journal and all(j["sha256"] and j["command"] for j in journal.values())
    manifest_path = root / "candidates/q4_0/manifest.json"
    bundle = media_release.load_bundle(manifest_path)
    assert not bundle.problems(manifest_path.parent)
    assert bundle.components[1].precision == "f32"
    review = media_release.record_review(result["run_id"], "q4_0", "Test human", True)
    assert review["manifest_sha256"] == media.canonical_hash(bundle.manifest())
    # Conversion resume validates the journal and actual bytes, not existence.
    media_release.run_release(native.plan, user_approved=True)
    assert len(native.downloads) == before
    with pytest.raises(media.MediaValidationError, match="approval"):
        media_release.run_release(native.plan, user_approved=False)


def test_evaluator_script_change_blocks_quality(native, tmp_path):
    native.evaluator.write_text("print('tampered')", encoding="utf-8")
    recipe = {"mode": "img_gen", "settings": {"width": 64, "height": 64}, "components": {}, "prompt": "test"}
    result = media_quality.assess(media.FIXTURES / "reference.png", recipe, None, native.plan["quality_profile"], tmp_path)
    assert not result["passed"] and "not pinned" in result["evaluation_error"]


def test_atomic_publication_requires_review_and_reuploads_hash_mismatches(native):
    result = media_release.run_release(native.plan, user_approved=True)
    root = media_release.run_dir(result["run_id"])
    candidate = root / "candidates/q4_0"
    bundle = media_release.load_bundle(candidate / "manifest.json")

    class Api:
        remote = {}
        commits = []

        def create_repo(self, **kwargs):
            pass

        def model_info(self, repo_id, **kwargs):
            return SimpleNamespace(card_data={"license": "apache-2.0"}, siblings=[
                SimpleNamespace(rfilename=name, size=data["bytes"], lfs={"sha256": data["sha256"]})
                for name, data in self.remote.items()])

        def create_commit(self, *, operations, **kwargs):
            self.commits.append([operation.path_in_repo for operation in operations])
            for operation in operations:
                path = Path(operation.path_or_fileobj)
                self.remote[operation.path_in_repo] = {"bytes": path.stat().st_size, "sha256": media.file_hash(path)}

    api = Api()
    with pytest.raises(media.MediaValidationError, match="human review"):
        media_release.publish_bundle(bundle, candidate, "org/release", {}, api)
    assert not api.commits
    review = media_release.record_review(result["run_id"], "q4_0", "Human", True)
    publication = media_release.publish_bundle(bundle, candidate, "org/release", review, api)
    assert publication["verified"] and len(api.commits) == 1
    assert {"manifest.json", "quality.json", "README.md", "components/diffusion_model.gguf", "components/vae.safetensors"} <= set(api.commits[0])
    # A size match alone must not skip a corrupted/mismatched remote artifact.
    api.remote["components/diffusion_model.gguf"]["sha256"] = "0" * 64
    media_release.publish_bundle(bundle, candidate, "org/release", review, api)
    assert "components/diffusion_model.gguf" in api.commits[-1]


def test_review_is_bound_to_exact_artifacts_and_quality(native):
    result = media_release.run_release(native.plan, user_approved=True)
    root = media_release.run_dir(result["run_id"])
    candidate = root / "candidates/q4_0"
    review = media_release.record_review(result["run_id"], "q4_0", "Human", True)
    bundle = media_release.load_bundle(candidate / "manifest.json")
    bundle.quality_results["runtime_seconds"] += 1
    with pytest.raises(media.MediaValidationError, match="exact artifacts"):
        media_release.publish_bundle(bundle, candidate, "org/release", review)


@pytest.mark.parametrize("issue", ["frozen", "missing-audio", "frames", "malformed"])
def test_video_structural_rejection_without_weakening_decoder_checks(tmp_path, monkeypatch, issue):
    import numpy as np
    monkeypatch.setattr(media_quality.shutil, "which", lambda name: name)
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    frames[:, :, :4] = 200
    if issue != "frozen":
        frames[1, :, 4:] = 100
    facts = {"streams": [{"codec_type": "video", "width": 8, "height": 8, "avg_frame_rate": "24/1",
                           "nb_read_frames": "2" if issue == "frames" else "3"}]}

    def native(command):
        if command[0] == "ffprobe":
            return json.dumps(facts).encode()
        return b"bad" if issue == "malformed" else frames.tobytes()

    monkeypatch.setattr(media_quality, "_native", native)
    clip = tmp_path / "clip.avi"
    clip.write_bytes(b"fake-container")
    with pytest.raises(media.MediaValidationError):
        media_quality.inspect_video(clip, {"width": 8, "height": 8, "frames": 3, "fps": 24}, require_audio=issue == "missing-audio")
