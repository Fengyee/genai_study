"""Analytic pixels, RNG isolation and reader failures; no pretrained weights."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from genai_study.vae_study import (
    array_identity, make_parser, prepare_input, rgb_diagnostics,
    sample_with_epsilon, validate_args,
)


def test_clipping_projection_has_known_error_and_keeps_raw_diagnostics():
    target = np.array([0.0, 0.5, 1.0]).reshape(1, 3, 1, 1)
    raw = np.array([-1.0, 0.5, 2.0]).reshape(1, 3, 1, 1)
    result = rgb_diagnostics(target, raw)
    assert result["raw_rgb_mse"] == pytest.approx(2 / 3)
    assert result["clipped_rgb_mse"] == 0
    assert result["out_of_range_fraction"] == pytest.approx(2 / 3)
    assert result["clipping_rms"] == pytest.approx((2 / 3) ** 0.5)
    assert result["raw_rgb_range"] == [-1.0, 2.0]
    # Projection onto an interval cannot increase squared distance to any point in it.
    targets = np.linspace(0, 1, 41)
    for prediction in (-2.0, -0.01, 0.3, 1.01, 3.0):
        stats = rgb_diagnostics(targets, np.full_like(targets, prediction))
        assert stats["clipped_rgb_mse"] <= stats["raw_rgb_mse"] + 1e-15


@pytest.mark.parametrize("target,raw,message", [
    ([0], [np.nan], "finite"), ([1.1], [0], "target RGB"), ([0], [0, 1], "matching"),
])
def test_invalid_pixel_statistics_are_rejected(target, raw, message):
    with pytest.raises(ValueError, match=message):
        rgb_diagnostics(target, raw)


def test_exif_orientation_preserves_pixels_without_resampling(tmp_path):
    pixels = np.zeros((128, 136, 3), dtype=np.uint8)
    pixels[:, :68, 0] = 255
    pixels[::2, :, 2] = 170
    image = Image.fromarray(pixels)
    exif = image.getexif()
    exif[274] = 6  # Clockwise 90 degrees; width and height exchange.
    path = tmp_path / "oriented.png"
    image.save(path, exif=exif)
    prepared, rgb, normalized, info = prepare_input(path)
    expected = np.rot90(pixels, k=-1)
    assert np.array_equal(np.asarray(prepared), expected)
    assert rgb.shape == (1, 3, 136, 128)
    assert np.array_equal(rgb[0].transpose(1, 2, 0), expected.astype(np.float32) / 255)
    assert normalized.dtype == np.float32
    assert float(normalized.min()) == -1 and float(normalized.max()) == 1
    assert info["original_size_wh"] == [136, 128]
    assert info["prepared_size_wh"] == [128, 136]
    assert info["exif_orientation"] == 6
    assert info["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("size", [(127, 128), (129, 128), (520, 128)])
def test_invalid_input_size_is_rejected_instead_of_resized(tmp_path, size):
    path = tmp_path / "input.png"
    Image.new("RGB", size).save(path)
    with pytest.raises(ValueError, match="dimensions"):
        prepare_input(path)


class AnalyticPosterior:
    """Small stand-in for the sampling API; no optional dependency is imported."""
    mean = torch.tensor([2.0, -1.0])
    std = torch.tensor([0.25, 2.0])

    def sample(self, generator):
        return torch.normal(self.mean, self.std, generator=generator)


def test_posterior_sampling_uses_isolated_reproducible_cpu_streams():
    posterior = AnalyticPosterior()
    global_state = torch.random.get_rng_state().clone()
    first, noise, residual = sample_with_epsilon(posterior, 7)
    repeat, repeat_noise, _ = sample_with_epsilon(posterior, 7)
    another, _, _ = sample_with_epsilon(posterior, 19)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert torch.equal(first, repeat) and torch.equal(noise, repeat_noise)
    assert not torch.equal(first, another)
    assert residual == 0
    # Invert the two analytic affine distributions to recover standard noise.
    torch.testing.assert_close((first - torch.tensor([2.0, -1.0])) / torch.tensor([0.25, 2.0]), noise)


def test_posterior_sampling_mismatch_is_detected():
    class WrongSampler(AnalyticPosterior):
        def sample(self, generator):
            return self.mean

    with pytest.raises(ValueError, match="independently reconstructed"):
        sample_with_epsilon(WrongSampler(), 7)


def test_hash_metadata_uses_logical_c_order_for_noncontiguous_arrays():
    array = np.arange(12, dtype=np.float32).reshape(3, 4).T
    identity = array_identity(array)
    expected = np.array([[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]], dtype=np.float32)
    assert identity["sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    assert identity["shape"] == [4, 3] and identity["dtype"] == "float32"


@pytest.mark.parametrize("extra", [
    ["--seeds", "1", "2"], ["--seeds", "-1", "2", "3"],
    ["--seeds", "0", "1", str(2**32)], ["--threads", "0"], ["--threads", "5"],
])
def test_cli_rejects_protocol_parameters(extra):
    with pytest.raises(SystemExit) as error:
        make_parser().parse_args(["--model-dir", "model", "--input", "image.png", "--output", "out", *extra])
    assert error.value.code == 2


def test_repeated_seeds_rejected_before_optional_imports(tmp_path):
    path = tmp_path / "input"
    path.touch()
    args = make_parser().parse_args(["--model-dir", "model", "--input", str(path),
                                     "--output", str(tmp_path / "out"), "--seeds", "7", "7", "19"])
    with pytest.raises(ValueError, match="distinct"):
        validate_args(args)


def test_stdlib_help_and_output_protection_preserve_reader_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    command = [sys.executable, "-S", "-m", "genai_study.vae_study"]
    help_result = subprocess.run(command + ["--help"], env=env, text=True, capture_output=True)
    assert help_result.returncode == 0
    assert "exactly three distinct" in help_result.stdout
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.json"
    marker.write_text(json.dumps({"previous": "result"}))
    before = marker.read_bytes()
    failed = subprocess.run(command + ["--model-dir", "missing", "--manifest", "absent.json",
                                       "--input", "absent.png", "--output", str(output)],
                            env=env, capture_output=True, text=True)
    assert failed.returncode == 2
    assert "output directory is nonempty" in failed.stderr
    assert marker.read_bytes() == before
    assert list(output.iterdir()) == [marker]
