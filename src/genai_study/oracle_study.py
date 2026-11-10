"""Measure the existing DDPM sampler with exact Gaussian epsilon predictors.

No training or downloads. The standard-normal oracle gives an analytic variance
recursion; the eight-Gaussian oracle evaluates a stable mixture score. Both use
the existing 100-step sampler and its fixed posterior variance unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import Tensor, nn

from . import toy
from .toy import Schedule, centers, diagnostics, make_schedule, sample_data, sample_ddpm


DATA_SIGMA = 0.08
STREAM_OFFSETS = {
    "normal_sampling": 0,
    "mixture_sampling": 1,
    "normal_reference": 2,
    "mixture_reference": 3,
    "mixture_independent_target": 4,
}


def mixture_quantities(x: Tensor, alpha_bar: Tensor | float) -> tuple[Tensor, Tensor, Tensor]:
    """Return log q_t(x), its score, and E[X_0|X_t=x] for the eight-Gaussian data.

    Each noised component has mean sqrt(alpha_bar)*center and scalar variance
    alpha_bar*0.08**2 + 1-alpha_bar. alpha_bar is scalar or has one entry per x.
    Computation follows x's dtype; logsumexp/softmax stabilize the normalization.
    Extremely small relative weights can still underflow at finite precision.
    """
    a = torch.as_tensor(alpha_bar, dtype=x.dtype, device=x.device).broadcast_to((len(x),))
    component_centers = centers(dtype=x.dtype).to(x.device)
    variance = a * DATA_SIGMA**2 + (1 - a)
    difference = x[:, None, :] - a.sqrt()[:, None, None] * component_centers[None]
    logits = -difference.square().sum(2) / (2 * variance[:, None])
    weights = logits.softmax(dim=1)
    log_density = torch.logsumexp(logits, dim=1) - math.log(8) - torch.log(2 * math.pi * variance)
    score = -(weights[:, :, None] * difference).sum(1) / variance[:, None]
    conditional_component_mean = (
        component_centers[None]
        + (a.sqrt() * DATA_SIGMA**2 / variance)[:, None, None] * difference
    )
    posterior_mean = (weights[:, :, None] * conditional_component_mean).sum(1)
    return log_density, score, posterior_mean


class NormalEpsilonOracle(nn.Module):
    """For stationary N(0,I) data, E[epsilon|x_t]=sqrt(1-alpha_bar_t)*x_t."""

    def __init__(self, schedule: Schedule):
        super().__init__()
        self.register_buffer("alpha_bar", schedule.alpha_bar.clone())

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # sample_ddpm passes normalized paper time (k+1)/T; index zero is t=1.
        k = (t * len(self.alpha_bar)).round().long() - 1
        return (1 - self.alpha_bar[k, None]).sqrt() * x


class MixtureEpsilonOracle(NormalEpsilonOracle):
    """Exact marginal epsilon predictor obtained from the Gaussian-mixture score."""

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        k = (t * len(self.alpha_bar)).round().long() - 1
        alpha_bar = self.alpha_bar[k]
        _, score, _ = mixture_quantities(x, alpha_bar)
        return -(1 - alpha_bar[:, None]).sqrt() * score


def normal_variance_path(schedule: Schedule) -> list[float]:
    """Scalar analytic covariance recursion, ordered T,T-1,...,0.

    With the normal oracle the reverse mean is sqrt(alpha_t)*x_t. Independent
    reverse noise gives V_{t-1}=alpha_t*V_t+tilde_beta_t, starting from V_T=1.
    Schedule values are converted to Python floats before the independent scalar
    recursion; model evaluation and random samples are not used here.
    """
    variance = 1.0
    path = [variance]
    for alpha, posterior_variance in zip(
        reversed(schedule.alpha.tolist()), reversed(schedule.posterior_variance.tolist())
    ):
        variance = alpha * variance + posterior_variance
        path.append(variance)
    return path


def normal_statistics(samples: Tensor) -> dict:
    x = samples.double()
    centered = x - x.mean(0)
    covariance = centered.T @ centered / len(x)
    return {
        "mean": x.mean(0).tolist(),
        "covariance_population_denominator": covariance.tolist(),
        "mean_coordinate_variance": float(covariance.diag().mean()),
        "mean_squared_norm_over_dimension": float(x.square().mean()),
    }


def mixture_statistics(samples: Tensor, reference: Tensor) -> dict:
    result = diagnostics(samples, reference)
    distance2 = (samples.double()[:, None] - centers().double()[None]).square().sum(2).min(1).values
    result["nearest_center_second_moment_over_2sigma2"] = float(distance2.mean() / (2 * DATA_SIGMA**2))
    return result


def bounded_integer(low: int, high: int):
    def parse(value: str) -> int:
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return number
    return parse


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=bounded_integer(16, 65536), default=4096)
    parser.add_argument("--seeds", nargs="+", type=bounded_integer(0, 2**32 - 1), default=[7, 19, 43],
                        help="Distinct base seeds; each role uses 8*seed+its recorded offset")
    parser.add_argument("--threads", type=bounded_integer(1, 4), default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    started = time.perf_counter()
    parser = make_parser()
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("base seeds must be distinct")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be an empty directory; preserve existing results and choose a new path")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    schedule = make_schedule()
    variance_path = normal_variance_path(schedule)
    arrays = {}
    runs = []
    for seed in args.seeds:
        streams = {name: 8 * seed + offset for name, offset in STREAM_OFFSETS.items()}
        rng = lambda name: torch.Generator(device="cpu").manual_seed(streams[name])
        prefix = f"seed_{seed}"
        normal_initial = torch.randn(args.samples, 2, generator=rng("normal_sampling"))
        mixture_initial = torch.randn(args.samples, 2, generator=rng("mixture_sampling"))
        begin = time.perf_counter()
        normal = sample_ddpm(NormalEpsilonOracle(schedule), args.samples, schedule, rng("normal_sampling"))
        normal_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        mixture = sample_ddpm(MixtureEpsilonOracle(schedule), args.samples, schedule, rng("mixture_sampling"))
        mixture_seconds = time.perf_counter() - begin
        normal_reference = torch.randn(args.samples, 2, generator=rng("normal_reference"))
        reference = sample_data(args.samples, rng("mixture_reference"))
        independent_target = sample_data(args.samples, rng("mixture_independent_target"))
        values = {"normal_initial": normal_initial, "mixture_initial": mixture_initial,
                  "normal": normal, "mixture": mixture, "normal_reference": normal_reference,
                  "mixture_reference": reference, "mixture_independent_target": independent_target}
        for name, value in values.items():
            arrays[f"{prefix}_{name}"] = value.numpy()
        runs.append({
            "base_seed": seed, "random_stream_seeds": streams,
            "array_keys": {name: f"{prefix}_{name}" for name in values},
            "normal": {"generated": normal_statistics(normal), "reference": normal_statistics(normal_reference),
                       "analytic_final_variance": variance_path[-1], "sampling_seconds": normal_seconds},
            "mixture": {"generated": mixture_statistics(mixture, reference),
                        "independent_target": mixture_statistics(independent_target, reference),
                        "sampling_seconds": mixture_seconds},
        })
    array_path = args.output / "arrays.npz"
    np.savez_compressed(array_path, **arrays)
    source = Path(__file__).resolve()
    root = source.parents[2]
    identity = {"oracle_study.py": sha256(source), "toy.py": sha256(Path(toy.__file__).resolve())}
    lock = root / "requirements-lock.txt"
    if lock.is_file():
        identity["requirements-lock.txt"] = sha256(lock)
    report = {
        "schema_version": 1,
        "config": {"samples": args.samples, "seeds": args.seeds, "threads": args.threads,
                   "device": "cpu", "dtype": "float32", "sampling_steps": len(schedule.beta),
                   "training_steps": 0, "initial_distribution": "N(0,I)",
                   "stream_rule": "8 * base_seed + role_offset", "stream_offsets": STREAM_OFFSETS},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__},
        "source_sha256": identity,
        "schedule": {name: getattr(schedule, name).tolist() for name in
                     ("beta", "alpha", "alpha_bar", "alpha_bar_prev", "posterior_variance")},
        "normal_analytic": {"paper_times": list(range(len(schedule.beta), -1, -1)),
                            "variance": variance_path, "target_variance": 1.0,
                            "recursion": "V[t-1] = alpha[t]*V[t] + posterior_variance[t]; V[T]=1"},
        "mixture_target": {"components": 8, "radius": 2.0, "component_sigma": DATA_SIGMA,
                           "own_component_3sigma_probability": toy.GAUSSIAN_3SIGMA_MASS_2D,
                           "own_component_expected_second_moment_over_2sigma2": 1.0,
                           "qT_mean": [0.0, 0.0],
                           "qT_coordinate_variance": 1 + float(schedule.alpha_bar[-1]) * (2 + DATA_SIGMA**2 - 1)},
        "statistic_scope": {
            "normal_covariance": "Empirical covariance divides by N; analytic prediction is population variance.",
            "nearest_center_second_moment_over_2sigma2": "Mean squared distance to the nearest center divided by 2*0.08^2. The analytic value 1 uses each sample's own component center; wrong-center Gaussian tails make nearest-center expectation slightly smaller.",
            "energy": "Existing diagnostics V-statistic on the first min(N,512) samples in generated random order; finite-sample biased.",
            "sampling": "Each oracle uses the existing DDPM sampler, fixed posterior variance and a fresh RNG for its role. Sampling initial and reverse-step noise consume the same role stream in order.",
        },
        "limitations": [
            "Exact epsilon prediction does not make the fixed-variance finite-step reverse kernel exact.",
            "N(0,I) is the exact forward marginal for the standard-normal task; for the eight-Gaussian task it approximates q_T. q_T is still a mixture, even when its covariance is close to I.",
            "Neighborhood fraction and nearest-center distances must be read together with within-mode spread and the independent target baseline.",
            "Samples and measured diagnostics have finite-sample variation. No training, learned-model performance claim or image/video quality inference.",
        ],
        "runs": runs,
        "outputs": {"arrays_npz_sha256": sha256(array_path),
                    "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                      "raw_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                               for name, value in arrays.items()}},
        "timing_seconds_before_metrics_write": time.perf_counter() - started,
        "timing_scope": "Per-oracle sampling includes predictor construction and the full 100-step call; excludes diagnostics and saving. Program total begins after imports and includes arguments, setup, statistics, arrays and hashes; no warm-up.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "runs": len(runs),
                      "normal_analytic_variance": variance_path[-1]}, allow_nan=False))


if __name__ == "__main__":
    main()
