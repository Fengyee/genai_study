"""Eight-Gaussian data and DDPM / conditional flow-matching primitives.

DDPM array index k in [0, T-1] represents paper time t=k+1. At k=0,
alpha_bar_prev=1 and posterior variance=0: the final reverse step is deterministic.
Flow time runs in the opposite direction: 0 is noise and 1 is data.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

# For an isotropic 2D Gaussian: R²/sigma² ~ chi-square(df=2).
# This is mass around its own center, not a 1D ±3sigma interval.
GAUSSIAN_3SIGMA_MASS_2D = 1 - math.exp(-9 / 2)


def centers(*, dtype: torch.dtype = torch.float32) -> Tensor:
    angles = torch.arange(8, dtype=dtype) * (2 * math.pi / 8)
    return 2 * torch.stack((angles.cos(), angles.sin()), dim=1)


def sample_data(n: int, generator: torch.Generator) -> Tensor:
    """Balanced-in-expectation 8-component mixture, radius 2, sigma 0.08."""
    labels = torch.randint(8, (n,), generator=generator)
    return centers()[labels] + 0.08 * torch.randn(n, 2, generator=generator)


class TimeMLP(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.register_buffer("frequencies", torch.arange(1, 9) * math.pi)
        self.net = nn.Sequential(
            nn.Linear(18, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        phase = t[:, None] * self.frequencies[None, :]
        return self.net(torch.cat((x, phase.sin(), phase.cos()), dim=1))


@dataclass(frozen=True)
class Schedule:
    beta: Tensor
    alpha: Tensor
    alpha_bar: Tensor
    alpha_bar_prev: Tensor
    posterior_variance: Tensor


def make_schedule(steps: int = 100, beta_start: float = 1e-4,
                  beta_end: float = 0.1) -> Schedule:
    """A deliberately short toy schedule; not a standard image-model recipe."""
    if steps < 2 or not (0 < beta_start <= beta_end < 1):
        raise ValueError("Need steps>=2 and 0 < beta_start <= beta_end < 1")
    beta = torch.linspace(beta_start, beta_end, steps)
    alpha = 1 - beta
    alpha_bar = alpha.cumprod(0)
    previous = torch.cat((torch.ones(1), alpha_bar[:-1]))
    variance = beta * (1 - previous) / (1 - alpha_bar)
    return Schedule(beta, alpha, alpha_bar, previous, variance)


def q_sample(x0: Tensor, k: Tensor, noise: Tensor, schedule: Schedule) -> Tensor:
    abar = schedule.alpha_bar[k, None]
    return abar.sqrt() * x0 + (1 - abar).sqrt() * noise


def reverse_parameters(xt: Tensor, k: int, epsilon: Tensor,
                       schedule: Schedule) -> tuple[Tensor, Tensor]:
    """DDPM epsilon-parameterized mean and fixed posterior variance."""
    mean = (xt - schedule.beta[k] * epsilon /
            (1 - schedule.alpha_bar[k]).sqrt()) / schedule.alpha[k].sqrt()
    return mean, schedule.posterior_variance[k]


@torch.no_grad()
def sample_ddpm(model: nn.Module, n: int, schedule: Schedule,
                generator: torch.Generator) -> Tensor:
    x = torch.randn(n, 2, generator=generator)
    steps = len(schedule.beta)
    for k in range(steps - 1, -1, -1):
        # Map paper t=1..T to normalized time 1/T..1.
        time = torch.full((n,), (k + 1) / steps)
        mean, variance = reverse_parameters(x, k, model(x, time), schedule)
        x = mean if k == 0 else mean + variance.sqrt() * torch.randn(
            x.shape, generator=generator)
    return x


def flow_path(noise: Tensor, data: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Independent endpoint coupling; conditional target is data-noise."""
    return (1 - t[:, None]) * noise + t[:, None] * data, data - noise


@torch.no_grad()
def sample_flow(model: nn.Module, n: int, steps: int,
                generator: torch.Generator) -> Tensor:
    initial = torch.randn(n, 2, generator=generator)
    return integrate_flow(model, initial, steps)


@torch.no_grad()
def integrate_flow(model: nn.Module, initial: Tensor, steps: int) -> Tensor:
    """Euler from supplied noise; keep the initial tensor unchanged for comparisons."""
    if steps < 1:
        raise ValueError("Euler steps must be positive")
    x = initial.clone()
    for k in range(steps):
        x = x + model(x, torch.full((len(x),), k / steps)) / steps
    return x


def diagnostics(generated: Tensor, reference: Tensor) -> dict[str, float | int]:
    """Toy diagnostics only. Energy uses first <=512 samples, a biased V statistic.

    Coverage counts modes having >=1% of all generated points within 3 sigma.
    The circles do not overlap. In 2D a 3-sigma radius is not a 99.7% interval.
    """
    if not torch.isfinite(generated).all():
        raise ValueError("Generated samples contain NaN or infinity")
    x, y = generated[:512], reference[:512]
    energy = (2 * torch.cdist(x, y).mean() - torch.cdist(x, x).mean()
              - torch.cdist(y, y).mean())
    distance, nearest = torch.cdist(generated, centers()).min(dim=1)
    within = distance < 3 * 0.08
    count = torch.bincount(nearest[within], minlength=8)
    return {
        "energy_distance_v_statistic": float(energy),
        "energy_sample_count": len(x),
        "mean_distance_to_nearest_center": float(distance.mean()),
        "fraction_within_3sigma": float(within.float().mean()),
        "covered_modes_at_1pct": int((count >= 0.01 * len(generated)).sum()),
    }
