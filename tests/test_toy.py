"""Analytic or statistical checks for the mathematical teaching primitives."""
import math

import pytest
import torch
from torch import nn

from genai_study.toy import (GAUSSIAN_3SIGMA_MASS_2D, centers, diagnostics, flow_path, make_schedule,
                             q_sample, reverse_parameters, sample_data,
                             sample_ddpm, sample_flow)


def rng(seed=12):
    return torch.Generator().manual_seed(seed)


def test_mixture_moments():
    # For a uniform circle of centers, E[x]=0, Cov[x]=(R²/2+sigma²)I.
    x = sample_data(100000, rng())
    assert torch.allclose(x.mean(0), torch.zeros(2), atol=0.02)
    assert torch.allclose(torch.cov(x.T), torch.eye(2) * (2 + 0.08**2), atol=0.03)


def test_forward_conditional_moments():
    schedule = make_schedule()
    x0 = torch.tensor([1.5, -0.5]).expand(100000, 2)
    noise = torch.randn(x0.shape, generator=rng())
    k = torch.full((len(x0),), 35)
    x = q_sample(x0, k, noise, schedule)
    assert torch.allclose(x.mean(0), schedule.alpha_bar[35].sqrt() * x0[0], atol=0.01)
    assert torch.allclose(x.var(0), (1 - schedule.alpha_bar[35]).expand(2), atol=0.01)


def test_posterior_against_gaussian_precision_conditioning():
    # Independent derivation: multiply N(xprev; prior_mean, prior_var) by
    # N(xt; sqrt(alpha)*xprev, beta), then add their precisions.
    s = make_schedule()
    k = 40
    x0 = torch.tensor([[0.3, -0.7]])
    xt = torch.tensor([[1.2, 0.4]])
    prior_mean = s.alpha_bar_prev[k].sqrt() * x0
    prior_variance = 1 - s.alpha_bar_prev[k]
    precision = 1 / prior_variance + s.alpha[k] / s.beta[k]
    expected_variance = 1 / precision
    expected_mean = (prior_mean / prior_variance + s.alpha[k].sqrt() * xt / s.beta[k]) / precision
    epsilon = (xt - s.alpha_bar[k].sqrt() * x0) / (1 - s.alpha_bar[k]).sqrt()
    mean, variance = reverse_parameters(xt, k, epsilon, s)
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(variance, expected_variance)


def test_ddpm_last_step_is_clean_and_deterministic():
    s = make_schedule()
    x0 = torch.tensor([[1.0, -0.8], [0.2, 0.5]])
    epsilon = torch.randn(x0.shape, generator=rng())
    xt = q_sample(x0, torch.zeros(2, dtype=torch.long), epsilon, s)
    mean, variance = reverse_parameters(xt, 0, epsilon, s)
    torch.testing.assert_close(mean, x0, atol=3e-6, rtol=1e-5)
    assert variance.item() == 0


def test_ddpm_oracle_sampler_recovers_point_mass():
    s = make_schedule(steps=10)
    target = torch.tensor([0.5, -0.25])

    class Oracle(nn.Module):
        def forward(self, x, t):
            k = (t * len(s.beta)).round().long() - 1
            abar = s.alpha_bar[k, None]
            return (x - abar.sqrt() * target) / (1 - abar).sqrt()

    samples = sample_ddpm(Oracle(), 64, s, rng())
    torch.testing.assert_close(samples, target.expand_as(samples), atol=1e-5, rtol=1e-5)


def test_flow_endpoints_and_derivative():
    noise = torch.randn(16, 2, generator=rng())
    data = sample_data(16, rng(9))
    start, velocity = flow_path(noise, data, torch.zeros(16))
    end, _ = flow_path(noise, data, torch.ones(16))
    torch.testing.assert_close(start, noise)
    torch.testing.assert_close(end, data)
    dt = 1e-3
    left, _ = flow_path(noise, data, torch.full((16,), 0.4))
    right, _ = flow_path(noise, data, torch.full((16,), 0.4 + dt))
    torch.testing.assert_close((right - left) / dt, velocity, atol=3e-4, rtol=1e-4)


def test_euler_constant_velocity_translation():
    class Constant(nn.Module):
        def forward(self, x, t):
            return torch.tensor([2.0, -1.0]).expand_as(x)

    expected = torch.randn(32, 2, generator=rng()) + torch.tensor([2.0, -1.0])
    for steps in (1, 7, 100):
        torch.testing.assert_close(sample_flow(Constant(), 32, steps, rng()), expected,
                                   atol=2e-5, rtol=1e-5)


def test_energy_diagnostic_and_mode_coverage():
    data = centers().repeat(10, 1)
    same = diagnostics(data, data)
    assert abs(same["energy_distance_v_statistic"]) < 1e-6
    assert same["covered_modes_at_1pct"] == 8
    assert same["fraction_within_3sigma"] == 1
    collapsed = diagnostics(centers()[:1].repeat(80, 1), data)
    assert collapsed["covered_modes_at_1pct"] == 1
    assert collapsed["energy_distance_v_statistic"] > 0.1
    with pytest.raises(ValueError, match="NaN"):
        diagnostics(torch.full((4, 2), math.nan), data)


@pytest.mark.parametrize("steps,start,end", [(1, 0.001, 0.1), (10, 0, 0.1), (10, 0.2, 0.1), (10, 0.1, 1)])
def test_schedule_rejects_invalid_parameters(steps, start, end):
    with pytest.raises(ValueError):
        make_schedule(steps, start, end)


def test_two_dimensional_gaussian_radius_mass():
    # Independently verify the chi-square(2) CDF, not the 1D 99.73% rule.
    noise = torch.randn(100000, 2, generator=rng(24))
    observed = float((noise.square().sum(1) < 9).float().mean())
    assert abs(GAUSSIAN_3SIGMA_MASS_2D - 0.98889100346) < 1e-10
    assert abs(observed - GAUSSIAN_3SIGMA_MASS_2D) < 0.002


def test_full_mode_coverage_can_hide_mostly_wrong_samples():
    # Each of 8 modes gets 2% of total mass, but 84% is far outside the target.
    deceptive = torch.cat((centers().repeat(20, 1), torch.full((840, 2), 50.0)))
    result = diagnostics(deceptive, sample_data(1000, rng()))
    assert result["covered_modes_at_1pct"] == 8
    assert result["fraction_within_3sigma"] == pytest.approx(0.16)
    assert result["mean_distance_to_nearest_center"] > 50
    assert result["energy_distance_v_statistic"] > 10
