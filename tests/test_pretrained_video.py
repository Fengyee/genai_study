"""SVD reader-path and preprocessing checks using tiny local fixtures."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from genai_study.pretrained_video import (
    MODEL_ID, REVISION, REQUIRED_FILES, frame_diagnostics, load_manifest,
    make_noise_inputs, make_parser, prepare_input_image, prepare_model,
    validate_run_args, verify_model,
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
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return root, path, manifest


def test_manifest_pins_svd_components_license_and_well_formed_hashes(model_fixture):
    root, path, original = model_fixture
    loaded = load_manifest(path)
    assert verify_model(loaded, root)["verified_files"] == 11
    invalid = []
    for key, value in (("schema_version", True), ("model_id", "CompVis/stable-diffusion-v1-4"),
                       ("revision", "main")):
        candidate = copy.deepcopy(original)
        candidate[key] = value
        invalid.append(candidate)
    candidate = copy.deepcopy(original)
    candidate["files"] = [item for item in candidate["files"] if item["path"] != "LICENSE.md"]
    invalid.append(candidate)
    candidate = copy.deepcopy(original)
    candidate["files"].append(candidate["files"][0])
    invalid.append(candidate)
    for key, value in (("path", "../weights"), ("path", "unet/../model.safetensors"),
                       ("path", "pipeline.py"), ("size_bytes", True), ("sha256", "not-a-hash")):
        candidate = copy.deepcopy(original)
        candidate["files"][0][key] = value
        invalid.append(candidate)
    for candidate in invalid:
        path.write_text(json.dumps(candidate))
        with pytest.raises(ValueError):
            load_manifest(path)


def test_prepare_repairs_svd_files_with_correct_identity_and_anonymous_access(model_fixture, monkeypatch):
    root, _, manifest = model_fixture
    originals = {item["path"]: (root / item["path"]).read_bytes() for item in manifest["files"]}
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        (root / kwargs["filename"]).write_bytes(originals[kwargs["filename"]])

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=downloader))
    monkeypatch.setenv("HF_HOME", str(root / "cache"))
    assert prepare_model(manifest, root)["reused_verified_files"] == 11
    assert not calls
    corrupt = "image_encoder/config.json"
    missing = "unet/diffusion_pytorch_model.safetensors"
    (root / corrupt).write_bytes(b"!" + originals[corrupt][1:])
    (root / missing).unlink()
    repaired = prepare_model(manifest, root)
    assert repaired["download_calls"] == 2
    assert repaired["reused_verified_files"] == 9
    assert {call["filename"] for call in calls} == {corrupt, missing}
    for call in calls:
        assert call["repo_id"] == MODEL_ID and call["revision"] == REVISION
        assert call["token"] is False
        assert call["force_download"] is (call["filename"] == corrupt)
    assert verify_model(manifest, root)["verified_files"] == 11


def test_prepare_rejects_file_escape_before_downloader(model_fixture, tmp_path, monkeypatch):
    root, _, manifest = model_fixture
    path = root / "LICENSE.md"
    external = tmp_path / "external-license"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)

    def downloader(**kwargs):
        pytest.fail("outside-model symlink must fail before download")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=downloader))
    monkeypatch.setenv("HF_HOME", str(root / "cache"))
    with pytest.raises(ValueError, match="escapes model directory"):
        prepare_model(manifest, root)


def test_input_crop_preserves_center_and_records_exif_orientation(tmp_path):
    # EXIF rotation changes an 8x12 image to 12x8. The 2:1 crop is y=[1,7).
    pixels = np.zeros((12, 8, 3), dtype=np.uint8)
    pixels[:, :4, 0] = 255
    pixels[:, 4:, 2] = 255
    original = Image.fromarray(pixels)
    exif = original.getexif()
    exif[274] = 6
    path = tmp_path / "oriented.png"
    original.save(path, exif=exif)
    source = path.read_bytes()
    first, info = prepare_input_image(path, 4, 8)
    second, again = prepare_input_image(path, 4, 8)
    assert info["original_size_wh"] == [8, 12]
    assert info["exif_transposed_size_wh"] == [12, 8]
    assert info["crop_box_xyxy"] == [0, 1, 12, 7]
    assert first.size == (8, 4) and first.mode == "RGB"
    assert first.tobytes() == second.tobytes() and info == again
    assert info["prepared_rgb_sha256"] == hashlib.sha256(first.tobytes()).hexdigest()
    # Rotated vertical color halves become horizontal halves, keeping the center.
    observed = np.asarray(first)
    assert observed[0, :, 0].mean() > observed[0, :, 2].mean()
    assert observed[-1, :, 2].mean() > observed[-1, :, 0].mean()
    assert path.read_bytes() == source


def test_landscape_crop_keeps_center_and_excludes_outer_bands(tmp_path):
    image = np.zeros((4, 12, 3), dtype=np.uint8)
    image[:, 4:8, 1] = 255
    path = tmp_path / "wide.png"
    Image.fromarray(image).save(path)
    prepared, info = prepare_input_image(path, 4, 4)
    assert info["crop_box_xyxy"] == [4, 0, 8, 4]
    assert np.all(np.asarray(prepared)[:, :, 1] == 255)
    assert np.all(np.asarray(prepared)[:, :, [0, 2]] == 0)


def test_video_noise_and_augmentation_use_independent_streams():
    torch.manual_seed(1234)
    global_state = torch.random.get_rng_state().clone()
    small, augmentation = make_noise_inputs(7, 8, (1, 4, 4, 2, 2))
    large, same_augmentation = make_noise_inputs(7, 8, (1, 14, 4, 2, 2))
    changed, third_augmentation = make_noise_inputs(19, 8, (1, 4, 4, 2, 2))
    same, other_augmentation = make_noise_inputs(7, 43, (1, 4, 4, 2, 2))
    assert small.shape != large.shape
    assert torch.equal(small, same) and not torch.equal(small, changed)
    draws = [torch.randn((1, 3, 8, 8), generator=g)
             for g in (augmentation, same_augmentation, third_augmentation, other_augmentation)]
    assert torch.equal(draws[0], draws[1]) and torch.equal(draws[0], draws[2])
    assert not torch.equal(draws[0], draws[3])
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_frame_change_diagnostics_have_known_scale_and_stationary_counterexample():
    frames = np.broadcast_to(np.arange(4, dtype=np.float32)[:, None, None, None] / 4,
                             (4, 3, 5, 3)).copy()
    diagnostics = frame_diagnostics(frames)
    assert diagnostics["adjacent_rgb_rms"] == [0.25, 0.25, 0.25]
    assert diagnostics["first_last_rgb_rms"] == 0.75
    still = np.repeat(frames[:1], 4, axis=0)
    assert frame_diagnostics(still)["adjacent_rgb_rms"] == [0, 0, 0]


@pytest.mark.parametrize("option,value", [
    ("--frames", "3"), ("--frames", "15"), ("--steps", "0"), ("--steps", "51"),
    ("--height", "145"), ("--width", "512"), ("--fps", "0"),
    ("--motion-bucket-id", "256"), ("--noise-aug-strength", "nan"),
    ("--noise-aug-strength", "-0.1"), ("--max-guidance-scale", "inf"),
    ("--min-guidance-scale", "0"), ("--threads", "5"), ("--augmentation-seed", "-1"),
])
def test_reader_arguments_bound_runtime_and_condition_parameters(option, value):
    with pytest.raises(SystemExit) as error:
        make_parser().parse_args(["run", "--model-dir", "model", "--input", "image.png",
                                 "--output", "out", option, value])
    assert error.value.code == 2


def test_cfg_cross_parameter_validation_and_seed_wrap(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    argv = ["run", "--model-dir", "model", "--input", str(image), "--output", str(tmp_path / "out")]
    for options in (["--max-guidance-scale", "1"],
                    ["--min-guidance-scale", "3", "--max-guidance-scale", "2"]):
        with pytest.raises(ValueError, match="CFG requires"):
            validate_run_args(make_parser().parse_args(argv + options))
    assert validate_run_args(make_parser().parse_args(argv + ["--seed", str(2**32 - 1)])) == 0
    assert validate_run_args(make_parser().parse_args(argv + ["--augmentation-seed", "123"])) == 123


def test_help_and_output_protection_work_without_site_packages(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    command = [sys.executable, "-S", "-m", "genai_study.pretrained_video"]
    help_result = subprocess.run(command + ["run", "--help"], env=env, capture_output=True, text=True)
    assert help_result.returncode == 0 and "--augmentation-seed" in help_result.stdout
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("reader's previous result")
    result = subprocess.run(command + ["run", "--model-dir", "missing", "--manifest", "missing",
                                       "--input", "missing.png", "--output", str(output)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "output directory is nonempty" in result.stderr
    assert marker.read_text() == "reader's previous result"
    assert list(output.iterdir()) == [marker]
