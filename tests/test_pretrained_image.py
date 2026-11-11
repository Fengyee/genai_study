"""Local integrity and reader CLI checks; no weights or optional packages needed."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from genai_study.pretrained_image import (
    MODEL_ID, REVISION, REQUIRED_FILES, load_manifest, make_parser,
    prepare_model, require_empty_output, verify_model,
)


@pytest.fixture
def model_fixture(tmp_path):
    root = tmp_path / "model"
    entries = []
    for relative in sorted(REQUIRED_FILES):
        content = (relative + "\n").encode()
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append({"path": relative, "size_bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"schema_version": 1, "model_id": MODEL_ID,
                "revision": REVISION, "files": entries}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return root, manifest_path, manifest


def test_verified_components_detect_same_size_corruption_and_missing_file(model_fixture):
    root, manifest_path, manifest = model_fixture
    loaded = load_manifest(manifest_path)
    verified = verify_model(loaded, root)
    assert verified == {"verified_files": len(REQUIRED_FILES),
                        "verified_bytes": sum(item["size_bytes"] for item in manifest["files"])}
    weight = root / "unet/diffusion_pytorch_model.safetensors"
    original = weight.read_bytes()
    weight.write_bytes(b"!" + original[1:])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_model(loaded, root)
    weight.write_bytes(original[:-1])
    with pytest.raises(ValueError, match="byte size mismatch"):
        verify_model(loaded, root)
    weight.unlink()
    with pytest.raises(ValueError, match="missing model file"):
        verify_model(loaded, root)


def test_manifest_rejects_unpinned_incomplete_duplicate_or_unsafe_entries(model_fixture):
    _, path, original = model_fixture
    invalid = []
    candidate = copy.deepcopy(original)
    candidate["revision"] = "main"
    invalid.append(candidate)
    candidate = copy.deepcopy(original)
    candidate["files"] = candidate["files"][:-1]
    invalid.append(candidate)
    candidate = copy.deepcopy(original)
    candidate["files"].append(candidate["files"][0])
    invalid.append(candidate)
    for bad_path in ("../weights", "/tmp/weights", "unet/../../weights", "pipeline.py"):
        candidate = copy.deepcopy(original)
        candidate["files"][0]["path"] = bad_path
        invalid.append(candidate)
    for candidate in invalid:
        path.write_text(json.dumps(candidate))
        with pytest.raises(ValueError):
            load_manifest(path)


def test_verification_rejects_symlink_outside_model_directory(model_fixture, tmp_path):
    root, _, manifest = model_fixture
    target = root / "tokenizer/vocab.json"
    external = tmp_path / "external-vocab.json"
    external.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(external)
    with pytest.raises(ValueError, match="escapes model directory"):
        verify_model(manifest, root)


def test_repository_manifest_covers_required_components():
    manifest = load_manifest(Path(__file__).resolve().parents[1]
                             / "examples/pretrained-sd14/manifest.json")
    assert manifest["total_model_bytes"] == sum(item["size_bytes"] for item in manifest["files"])
    assert sum(item["path"].endswith(".safetensors") for item in manifest["files"]) == 4


def test_prepare_reuses_verified_files_and_refetches_corruption_at_fixed_revision(model_fixture, monkeypatch):
    root, _, manifest = model_fixture
    originals = {item["path"]: (root / item["path"]).read_bytes() for item in manifest["files"]}
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        (root / kwargs["filename"]).write_bytes(originals[kwargs["filename"]])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=downloader))
    monkeypatch.setenv("HF_HOME", str(root / "cache"))
    reused = prepare_model(manifest, root)
    assert reused["reused_verified_files"] == len(REQUIRED_FILES)
    assert not calls
    corrupt = "text_encoder/config.json"
    (root / corrupt).write_bytes(b"!" + originals[corrupt][1:])
    repaired = prepare_model(manifest, root)
    assert repaired["download_calls"] == 1
    assert calls[0]["filename"] == corrupt
    assert calls[0]["repo_id"] == MODEL_ID
    assert calls[0]["revision"] == REVISION
    assert calls[0]["token"] is False
    assert calls[0]["force_download"] is True
    assert calls[0]["cache_dir"] == str(root / ".cache/huggingface-hub")
    assert verify_model(manifest, root)["verified_files"] == len(REQUIRED_FILES)


@pytest.mark.parametrize("option,value", [
    ("--guidance-scale", "1"), ("--guidance-scale", "nan"), ("--guidance-scale", "inf"),
    ("--steps", "0"), ("--steps", "51"), ("--height", "129"), ("--width", "1024"),
    ("--threads", "5"), ("--seed", "-1"),
])
def test_reader_arguments_enforce_measured_protocol(option, value):
    with pytest.raises(SystemExit) as error:
        make_parser().parse_args(["run", "--model-dir", "model", "--output", "out", option, value])
    assert error.value.code == 2


def test_output_guard_and_help_work_before_optional_imports(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("reader's previous result")
    with pytest.raises(ValueError, match="nonempty"):
        require_empty_output(output)
    command = [sys.executable, "-m", "genai_study.pretrained_image"]
    help_result = subprocess.run(command + ["run", "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "--guidance-scale" in help_result.stdout
    protected = subprocess.run(command + ["run", "--model-dir", str(tmp_path / "missing-model"),
                                          "--manifest", str(tmp_path / "missing-manifest"),
                                          "--output", str(output)], capture_output=True, text=True)
    assert protected.returncode == 2
    assert "output directory is nonempty" in protected.stderr
    assert marker.read_text() == "reader's previous result"
    assert list(output.iterdir()) == [marker]
