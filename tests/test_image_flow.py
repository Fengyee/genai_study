"""Synthetic image mechanics, diagnostic counterexamples, and the reader CLI."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from genai_study.image_flow import (ConditionalConvFlow, image_diagnostics,
                                    image_flow_path, render_shapes, sample_shapes,
                                    template_bank)


def test_shape_layout_range_balance_and_known_boundary():
    images, labels = sample_shapes(64, torch.Generator().manual_seed(7))
    assert images.shape == (64, 1, 16, 16)
    assert images.dtype == torch.float32
    assert 0 <= images.min() <= images.max() <= 1
    assert torch.bincount(labels).tolist() == [32, 32]
    known = render_shapes(torch.tensor([0, 1]), torch.full((2, 2), 8.0), torch.full((2,), 3.0))
    torch.testing.assert_close(known[:, 0, 8, 11], torch.full((2,), 0.5))
    assert (known[:, 0, 8, 8] > 0.99).all()
    # Square corners contain more foreground than a circle with equal size.
    assert known[1, 0, 10, 10] > known[0, 0, 10, 10]
    with pytest.raises(ValueError):
        sample_shapes(3, torch.Generator())


def test_bchw_flow_time_broadcast_and_conditional_network():
    torch.set_num_threads(1)
    data, labels = sample_shapes(4, torch.Generator().manual_seed(9))
    noise = torch.randn(data.shape, generator=torch.Generator().manual_seed(10))
    mixed, target = image_flow_path(noise, data, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(mixed[:2], noise[:2])
    torch.testing.assert_close(mixed[2:], data[2:])
    torch.testing.assert_close(noise + target, data)
    model = ConditionalConvFlow()
    velocity = model(mixed, torch.full((4,), 0.5), labels)
    assert velocity.shape == data.shape
    loss = (velocity - target).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_template_diagnostics_targets_wrong_images_and_condition():
    torch.set_num_threads(1)
    bank = template_bank()
    assert bank.shape == (2, 847, 256)
    for seed in (11, 19):
        data, labels = sample_shapes(256, torch.Generator().manual_seed(seed))
        target = image_diagnostics(data, labels, bank)
        for result in target.values():
            assert result["coarse_recognizable_fraction"] >= 0.99
            assert result["raw_out_of_range_fraction"] == 0
            assert result["center_x_std_px"] > 1
            assert result["foreground_mass_std"] > 5
    wrong_condition = render_shapes(1 - labels, torch.full((256, 2), 7.5), torch.full((256,), 2.75))
    for wrong in (torch.zeros_like(data), torch.randn(data.shape, generator=torch.Generator().manual_seed(3)),
                  wrong_condition):
        for result in image_diagnostics(wrong, labels, bank).values():
            assert result["coarse_recognizable_fraction"] < 0.05


def test_one_prototype_per_condition_passes_but_lacks_diversity():
    labels = torch.arange(128) % 2
    prototype = render_shapes(labels, torch.full((128, 2), 7.5), torch.full((128,), 2.75))
    for result in image_diagnostics(prototype, labels).values():
        assert result["coarse_recognizable_fraction"] == 1
        assert result["center_x_std_px"] < 1e-5
        assert result["center_y_std_px"] < 1e-5
        assert result["foreground_mass_std"] < 1e-4
    shifted = prototype - 0.05
    for result in image_diagnostics(shifted, labels).values():
        assert result["raw_out_of_range_fraction"] > 0
        assert result["mean_absolute_clip_change"] > 0


def test_image_cli_small_run_and_rng_isolation(tmp_path):
    measurements = []
    for samples in (16, 32):
        output = tmp_path / str(samples)
        subprocess.run([sys.executable, "-m", "genai_study.image_flow", "--steps", "2",
                        "--batch-size", "4", "--samples", str(samples), "--sampling-steps", "2",
                        "--output", str(output)], check=True, capture_output=True, text=True,
                       timeout=60, env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")})
        metrics = json.loads((output / "metrics.json").read_text())
        assert np.isfinite(metrics["final_heldout_loss"]["all"])
        assert metrics["config"]["coarse_template_mse_threshold"] == 0.01
        assert metrics["diagnostics"]["circle"]["count"] == samples // 2
        assert metrics["diagnostics"]["square"]["count"] == samples // 2
        for name in ("samples.png", "loss.png"):
            assert (output / name).stat().st_size > 1000
        with np.load(output / "arrays.npz") as data:
            assert data["generated_raw"].shape == (samples, 1, 16, 16)
            assert data["labels"].tolist() == (np.arange(samples) % 2).tolist()
            assert np.isfinite(data["generated_raw"]).all()
            measurements.append((metrics, data["training_loss"].copy()))
    # Changing sampling/reference budget must not perturb training or fixed heldout RNG.
    assert measurements[0][0]["initial_heldout_loss"] == measurements[1][0]["initial_heldout_loss"]
    assert measurements[0][0]["final_heldout_loss"] == measurements[1][0]["final_heldout_loss"]
    np.testing.assert_array_equal(measurements[0][1], measurements[1][1])
