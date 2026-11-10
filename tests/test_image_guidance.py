"""Classifier-free image guidance: analytic fields and the paired reader path."""
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
from torch import nn

from genai_study.image_checkpoint import load_image_checkpoint, save_image_checkpoint
from genai_study.image_flow import (ConditionalConvFlow, condition_channels,
                                    drop_conditions, integrate_images)
from genai_study.image_guidance_study import guided_velocity, integrate_guided_images


def test_null_condition_and_dropout_preserve_labels_and_rng_streams():
    labels = torch.tensor([-1, 0, 1])
    torch.testing.assert_close(condition_channels(labels, torch.float64),
                               torch.tensor([[0., 0.], [1., 0.], [0., 1.]], dtype=torch.float64))
    with pytest.raises(ValueError):
        condition_channels(torch.tensor([2]), torch.float32)

    labels = torch.arange(20000) % 2
    saved_labels = labels.clone()
    rng = torch.Generator().manual_seed(93)
    rng_before = rng.get_state().clone()
    global_before = torch.get_rng_state().clone()
    unchanged = drop_conditions(labels, 0, rng)
    torch.testing.assert_close(unchanged, saved_labels, rtol=0, atol=0)
    torch.testing.assert_close(rng.get_state(), rng_before, rtol=0, atol=0)
    dropped = drop_conditions(labels, .25, rng)
    repeated = drop_conditions(labels, .25, torch.Generator().manual_seed(93))
    torch.testing.assert_close(dropped, repeated, rtol=0, atol=0)
    torch.testing.assert_close(labels, saved_labels, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_before, rtol=0, atol=0)
    mask = dropped == -1
    assert abs(float(mask.float().mean()) - .25) < .015
    for label in (0, 1):
        assert abs(float(mask[labels == label].float().mean()) - .25) < .02
    torch.testing.assert_close(dropped[~mask], labels[~mask], rtol=0, atol=0)
    for probability in (-.1, 1., float("nan")):
        with pytest.raises(ValueError):
            drop_conditions(labels, probability, rng)


class AffineConditionalField(nn.Module):
    """dx/dt = rate*x + b, with known conditional/unconditional intercepts."""
    def __init__(self, rate):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float64))
        self.calls = []

    def forward(self, x, t, labels):
        self.calls.append((t.clone(), labels.clone()))
        intercept = torch.where(labels == -1, 1, 3 + 2 * labels).to(x.dtype)
        return self.rate * x + intercept[:, None, None, None]


def test_guided_velocity_endpoints_and_network_evaluations():
    x = torch.linspace(-2, 2, 512, dtype=torch.float64).reshape(2, 1, 16, 16)
    labels = torch.tensor([0, 1])
    t = torch.tensor([.2, .7], dtype=torch.float64)
    model = AffineConditionalField(.25)
    for scale in (0., 1., 2.5):
        model.calls.clear()
        actual = guided_velocity(model, x, t, labels, scale)
        intercept = torch.tensor([1 + 2 * scale, 1 + 4 * scale], dtype=torch.float64)
        expected = .25 * x + intercept[:, None, None, None]
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        assert len(model.calls) == 2
        assert sum(bool((condition == -1).all()) for _, condition in model.calls) == 1
        assert sum(torch.equal(condition, labels) for _, condition in model.calls) == 1
        for time, _ in model.calls:
            torch.testing.assert_close(time, t, rtol=0, atol=0)


@pytest.mark.parametrize("rate", [0., .25])
def test_guided_euler_matches_affine_analytic_recurrence(rate):
    initial = torch.linspace(-1, 1, 512, dtype=torch.float64).reshape(2, 1, 16, 16)
    labels = torch.tensor([0, 1])
    original = initial.clone()
    model = AffineConditionalField(rate)
    for scale in (0., 1., 2.):
        intercept = torch.tensor([1 + 2 * scale, 1 + 4 * scale], dtype=torch.float64)[:, None, None, None]
        errors = []
        for steps in (10, 20, 40):
            model.calls.clear()
            actual = integrate_guided_images(model, initial, labels, steps, scale)
            if rate:
                multiplier = (1 + rate / steps) ** steps
                expected = multiplier * initial + intercept * (multiplier - 1) / rate
                exact = math.exp(rate) * initial + intercept * math.expm1(rate) / rate
                errors.append(float((actual - exact).abs().max()))
            else:
                expected = initial + intercept
            torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
            assert len(model.calls) == 2 * steps
            for time, _ in model.calls:
                assert 0 <= time.min() <= time.max() < 1
        if rate:
            assert 1.9 < errors[0] / errors[1] < 2.1
            assert 1.9 < errors[1] / errors[2] < 2.1
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    assert float(model.rate.detach()) == rate


def test_real_network_guidance_preserves_inputs_weights_and_endpoint_branches():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = ConditionalConvFlow(width=4).eval()
    initial = torch.randn(4, 1, 16, 16, generator=torch.Generator().manual_seed(19)).repeat_interleave(2, dim=0)
    labels = torch.arange(8) % 2
    original, original_labels = initial.clone(), labels.clone()
    weights = copy.deepcopy(model.state_dict())
    grad_flags = [parameter.requires_grad for parameter in model.parameters()]
    conditional = integrate_images(model, initial, labels, 3)
    guided = integrate_guided_images(model, initial, labels, 3, 1.)
    torch.testing.assert_close(guided, conditional, rtol=0, atol=0)
    unconditional = integrate_guided_images(model, initial, labels, 3, 0.)
    torch.testing.assert_close(unconditional[::2], unconditional[1::2], rtol=0, atol=0)
    torch.testing.assert_close(initial, original, rtol=0, atol=0)
    torch.testing.assert_close(labels, original_labels, rtol=0, atol=0)
    assert not guided.requires_grad and not unconditional.requires_grad
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
    assert [parameter.requires_grad for parameter in model.parameters()] == grad_flags
    assert model.training is False


def test_cfg_training_to_guidance_cli_and_independent_paired_diagnostics(tmp_path):
    train, study = tmp_path / "train", tmp_path / "study"
    environment = {**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")}
    subprocess.run([sys.executable, "-m", "genai_study.image_flow", "--steps", "20",
                    "--batch-size", "4", "--samples", "16", "--sampling-steps", "2",
                    "--condition-dropout", "0.2", "--seed", "7", "--output", str(train)],
                   check=True, capture_output=True, text=True, timeout=60, env=environment)
    training = json.loads((train / "metrics.json").read_text())
    checkpoint = train / "checkpoint.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model, payload = load_image_checkpoint(checkpoint)
    counts = training["training_condition_counts"]
    assert counts["conditional"] > 0 and counts["unconditional"] > 0
    assert counts["conditional"] + counts["unconditional"] == 80
    assert payload["metadata"]["training_condition_counts"] == counts
    assert payload["metadata"]["config"]["condition_dropout"] == .2
    assert payload["metadata"]["config"]["null_condition_encoding"] == "zero-vector"
    for field in ("initial_unconditional_heldout_loss", "final_unconditional_heldout_loss"):
        assert np.isfinite(training[field]["all"])
        assert payload["metadata"][field] == training[field]
    subprocess.run([sys.executable, "-m", "genai_study.image_guidance_study", "--checkpoint", str(checkpoint),
                    "--scales", "2", "0", "1", "1.0000001", "2", "--sampling-steps", "2", "--samples", "16",
                    "--seed", "123", "--output", str(study)],
                   check=True, capture_output=True, text=True, timeout=60, env=environment)
    result = json.loads((study / "metrics.json").read_text())
    assert result["checkpoint"]["sha256"] == checkpoint_hash
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
    assert [run["scale"] for run in result["runs"]] == [0, 1, 1.0000001, 2]
    array_keys = [run["array_key"] for run in result["runs"]]
    assert len(set(array_keys)) == 4
    assert array_keys[1] != array_keys[2]  # Nearby scales must retain separate saved arrays.
    torch.set_num_threads(1)
    with np.load(study / "arrays.npz") as arrays:
        assert set(array_keys).issubset(arrays.files)
        initial = torch.randn(8, 1, 16, 16, generator=torch.Generator().manual_seed(123)).repeat_interleave(2, dim=0)
        labels = torch.arange(16) % 2
        np.testing.assert_array_equal(arrays["initial"], initial.numpy())
        np.testing.assert_array_equal(arrays["labels"], labels.numpy())
        np.testing.assert_array_equal(arrays["scale_0"][::2], arrays["scale_0"][1::2])
        np.testing.assert_array_equal(arrays["scale_1"], integrate_images(model, initial, labels, 2).numpy())
        reference = arrays["scale_1"].astype(np.float64)
        for run in result["runs"]:
            assert run["steps"] == 2 and run["nfe"] == 4
            assert run["elapsed_seconds"] > 0
            assert run["diagnostics"]["circle"]["count"] == 8
            assert run["diagnostics"]["square"]["count"] == 8
            samples = arrays[run["array_key"]].astype(np.float64)
            raw = float(np.sqrt(np.mean((samples - reference) ** 2)))
            clipped = float(np.sqrt(np.mean((samples.clip(0, 1) - reference.clip(0, 1)) ** 2)))
            between_classes = float(np.sqrt(np.mean((samples[::2] - samples[1::2]) ** 2)))
            clipped_classes = float(np.sqrt(np.mean((samples[::2].clip(0, 1) - samples[1::2].clip(0, 1)) ** 2)))
            assert run["paired_pixel_rms_difference_from_scale1"] == pytest.approx(raw, abs=1e-7)
            assert run["paired_clipped_pixel_rms_difference_from_scale1"] == pytest.approx(clipped, abs=1e-7)
            assert run["paired_class_pixel_rms_difference"] == pytest.approx(between_classes, abs=1e-7)
            assert run["paired_class_clipped_pixel_rms_difference"] == pytest.approx(clipped_classes, abs=1e-7)
            assert clipped <= raw + 1e-12
            assert clipped_classes <= between_classes + 1e-12
        assert not np.array_equal(arrays["reference"], arrays["independent_target"])
    assert (study / "samples.png").stat().st_size > 1000


@pytest.mark.parametrize("metadata", [
    {},
    {"config": {"condition_dropout": .2, "null_condition_encoding": "zero-vector"},
     "training_condition_counts": {"conditional": 8, "unconditional": 0}},
])
def test_guidance_cli_rejects_checkpoint_without_null_training(tmp_path, metadata):
    checkpoint = tmp_path / "image.pt"
    save_image_checkpoint(checkpoint, ConditionalConvFlow(width=4), metadata)
    output = tmp_path / "study"
    process = subprocess.run([sys.executable, "-m", "genai_study.image_guidance_study",
                              "--checkpoint", str(checkpoint), "--output", str(output)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 2
    assert "condition" in process.stderr.lower()
    assert not output.exists()
