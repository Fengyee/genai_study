"""Image checkpoints, paired Euler analysis, and independent solver properties."""
import copy
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from genai_study.checkpoint import save_checkpoint
from genai_study.image_checkpoint import load_image_checkpoint, save_image_checkpoint
from genai_study.image_flow import ConditionalConvFlow, integrate_images, sample_images
from genai_study.toy import TimeMLP


def test_image_checkpoint_roundtrip_and_safe_load(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = ConditionalConvFlow(width=4)
    path = tmp_path / "image.pt"
    rng_before = torch.get_rng_state().clone()
    save_image_checkpoint(path, model, {"seed": 31, "torch": torch.__version__})
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    original_load = torch.load
    calls = []

    def observe_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", observe_load)
    loaded, payload = load_image_checkpoint(path)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert payload["architecture"]["width"] == 4
    assert payload["method"] == "image_flow"
    assert type(payload["metadata"]["torch"]) is str
    assert "optimizer_state_dict" not in payload
    assert loaded.training is False
    initial = torch.randn(4, 1, 16, 16, generator=torch.Generator().manual_seed(19))
    labels = torch.tensor([0, 1, 0, 1])
    times = torch.tensor([0.0, 0.25, 0.75, 1.0])
    torch.testing.assert_close(loaded(initial, times, labels), model(initial, times, labels), rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("method", "flow"),
    ("architecture", {"name": "OtherNet", "width": 4}),
    ("architecture", {"name": "ConditionalConvFlow-v1", "width": True}),
    ("architecture", {"name": "ConditionalConvFlow-v1", "width": 6}),
    ("metadata", {"nonfinite": float("nan")}),
    ("metadata", {"tensor": torch.tensor(1)}),
    ("metadata", {1: "non-string key"}), ("metadata", ["list"]),
])
def test_image_checkpoint_rejects_incompatible_payload(tmp_path, field, value):
    path = tmp_path / "image.pt"
    save_image_checkpoint(path, ConditionalConvFlow(width=4), {})
    payload = torch.load(path, weights_only=True)
    payload[field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_image_checkpoint(path)


def test_image_checkpoint_rejects_shape_change_and_old_torch(tmp_path, monkeypatch):
    path = tmp_path / "image.pt"
    save_image_checkpoint(path, ConditionalConvFlow(width=4), {})
    payload = torch.load(path, weights_only=True)
    payload["architecture"]["image_shape"] = [1, 32, 32]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="architecture"):
        load_image_checkpoint(path)
    monkeypatch.setattr(torch, "__version__", "2.5.1+cpu")
    with pytest.raises(RuntimeError, match="CVE-2025-32434"):
        load_image_checkpoint(tmp_path / "unread.pt")


def test_image_euler_constant_velocity_and_linear_field():
    initial = torch.linspace(-1, 1, 512, dtype=torch.float64).reshape(2, 1, 16, 16)
    labels = torch.tensor([0, 1])
    calls = []

    def constant(x, t, condition):
        calls.append(t.clone())
        assert torch.equal(condition, labels)
        return torch.full_like(x, 2)

    for steps in (1, 4, 16):
        calls.clear()
        actual = integrate_images(constant, initial, labels, steps)
        torch.testing.assert_close(actual, initial + 2, rtol=1e-14, atol=1e-14)
        assert len(calls) == steps
        torch.testing.assert_close(torch.stack(calls)[:, 0], torch.arange(steps, dtype=torch.float64) / steps)

    errors = []
    for steps in (10, 20, 40):
        actual = integrate_images(lambda x, t, condition: 0.5 * x, initial, labels, steps)
        # Euler's exact recurrence and the independent continuous solution dx/dt=.5x.
        torch.testing.assert_close(actual, initial * (1 + 0.5 / steps) ** steps, rtol=1e-13, atol=1e-13)
        errors.append(float((actual - initial * math.exp(0.5)).abs().max()))
    assert errors[0] > errors[1] > errors[2]
    assert 1.9 < errors[0] / errors[1] < 2.1
    assert 1.9 < errors[1] / errors[2] < 2.1


def test_image_sampling_preserves_inputs_parameters_and_original_rng_protocol():
    torch.set_num_threads(1)
    model = ConditionalConvFlow(width=4).eval()
    labels = torch.tensor([0, 1, 0, 1])
    initial = torch.randn(4, 1, 16, 16, generator=torch.Generator().manual_seed(5))
    untouched, saved_labels = initial.clone(), labels.clone()
    original_weights = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        expected = initial.clone()
        # Reader's original implementation, retained here to check unchanged RNG/arithmetic.
        for step in range(3):
            expected = expected + model(expected, torch.full((4,), step / 3), labels) / 3
    actual = integrate_images(model, initial, labels, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sampled = sample_images(model, labels, 3, torch.Generator().manual_seed(5))
    torch.testing.assert_close(sampled, expected, rtol=0, atol=0)
    torch.testing.assert_close(initial, untouched, rtol=0, atol=0)
    torch.testing.assert_close(labels, saved_labels, rtol=0, atol=0)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original_weights[key], rtol=0, atol=0)
    assert not actual.requires_grad
    with pytest.raises(ValueError, match="positive integer"):
        integrate_images(model, initial, labels, 0)


def test_image_solver_cli_frozen_paired_arrays_and_diagnostics(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    model = ConditionalConvFlow(width=4)
    checkpoint = tmp_path / "image.pt"
    save_image_checkpoint(checkpoint, model, {"training_steps": 0})
    sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = tmp_path / "study"
    subprocess.run([sys.executable, "-m", "genai_study.image_solver_study",
                    "--checkpoint", str(checkpoint), "--steps", "4", "2", "4",
                    "--samples", "16", "--seed", "123", "--output", str(output)],
                   check=True, capture_output=True, text=True, timeout=60,
                   env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")})
    result = json.loads((output / "metrics.json").read_text())
    assert result["checkpoint"]["sha256"] == sha256
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == sha256
    assert result["checkpoint"]["metadata"] == {"training_steps": 0}
    assert result["config"]["steps"] == [2, 4]
    assert [run["nfe"] for run in result["runs"]] == [2, 4]
    assert result["runs"][0]["previous_steps"] is None
    assert result["runs"][0]["paired_pixel_rms_difference_from_previous"] is None
    assert result["runs"][1]["previous_steps"] == 2
    with np.load(output / "arrays.npz") as arrays:
        initial = torch.randn(16, 1, 16, 16, generator=torch.Generator().manual_seed(123))
        labels = torch.arange(16) % 2
        np.testing.assert_array_equal(arrays["initial"], initial.numpy())
        np.testing.assert_array_equal(arrays["labels"], labels.numpy())
        for run in result["runs"]:
            expected = integrate_images(model, initial, labels, run["steps"])
            np.testing.assert_array_equal(arrays[f"steps_{run['steps']}"], expected.numpy())
            assert run["elapsed_seconds"] > 0
            assert run["diagnostics"]["circle"]["count"] == 8
            assert run["diagnostics"]["square"]["count"] == 8
        raw = float(np.sqrt(np.mean((arrays["steps_4"] - arrays["steps_2"]) ** 2)))
        clipped = float(np.sqrt(np.mean((arrays["steps_4"].clip(0, 1) - arrays["steps_2"].clip(0, 1)) ** 2)))
        assert result["runs"][1]["paired_pixel_rms_difference_from_previous"] == pytest.approx(raw)
        assert result["runs"][1]["paired_clipped_pixel_rms_difference_from_previous"] == pytest.approx(clipped)
        assert clipped <= raw + 1e-7  # Scalar clipping is 1-Lipschitz in pixel space.
        assert not np.array_equal(arrays["reference"], arrays["independent_target"])
    assert (output / "samples.png").stat().st_size > 1000


def test_image_solver_cli_rejects_2d_checkpoint(tmp_path):
    checkpoint = tmp_path / "flow.pt"
    save_checkpoint(checkpoint, TimeMLP(), "flow", {})
    output = tmp_path / "study"
    process = subprocess.run([sys.executable, "-m", "genai_study.image_solver_study",
                              "--checkpoint", str(checkpoint), "--output", str(output)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 2
    assert "requires an image_flow checkpoint" in process.stderr
    assert not output.exists()
