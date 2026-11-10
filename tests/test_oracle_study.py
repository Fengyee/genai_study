"""Independent score, conditioning, variance and reader-path checks; no training."""
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from genai_study.oracle_study import (
    MixtureEpsilonOracle, NormalEpsilonOracle, main, make_parser,
    mixture_quantities, normal_variance_path,
)
from genai_study.toy import make_schedule, reverse_parameters, sample_ddpm


def test_mixture_score_matches_independent_finite_difference_and_tail_is_finite():
    x = torch.tensor([[.3, -.7], [1.85, .05], [-1.2, 1.5]], dtype=torch.float64)
    for alpha_bar in (.006, .4, .995):
        logp, score, _ = mixture_quantities(x, alpha_bar)
        difference = torch.zeros_like(x)
        for axis in range(2):
            delta = torch.zeros_like(x)
            delta[:, axis] = 1e-5
            plus, _, _ = mixture_quantities(x + delta, alpha_bar)
            minus, _, _ = mixture_quantities(x - delta, alpha_bar)
            difference[:, axis] = (plus - minus) / 2e-5
        torch.testing.assert_close(score, difference, rtol=2e-7, atol=2e-7)
        assert torch.isfinite(logp).all()
    tail = torch.tensor([[1000., -900.]], dtype=torch.float64)
    assert all(torch.isfinite(value).all() for value in mixture_quantities(tail, .995))


def test_component_conditioning_matches_tweedie_and_both_marginal_endpoints():
    x = torch.tensor([[.3, -.7], [1.85, .05], [-1.2, 1.5]], dtype=torch.float64)
    a = torch.tensor([.01, .3, .99], dtype=x.dtype)
    _, score, posterior = mixture_quantities(x, a)
    torch.testing.assert_close(posterior, (x + (1-a[:, None])*score)/a[:, None].sqrt(),
                               rtol=2e-13, atol=2e-13)
    logp0, score0, mean0 = mixture_quantities(x, 0.0)
    expected_logp0 = -.5*x.square().sum(1) - np.log(2*np.pi)
    torch.testing.assert_close(logp0, expected_logp0, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(score0, -x, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(mean0, torch.zeros_like(x), rtol=0, atol=1e-15)
    _, _, mean1 = mixture_quantities(x, 1.0)
    torch.testing.assert_close(mean1, x, rtol=1e-14, atol=1e-14)


def test_oracle_normalized_time_indices_and_reverse_mean():
    schedule = make_schedule()
    x = torch.tensor([[.7, -.2], [-1.1, .6], [.3, .5]])
    indices = torch.tensor([0, 49, 99])
    times = (indices + 1).float() / 100
    a = schedule.alpha_bar[indices, None]
    normal_epsilon = NormalEpsilonOracle(schedule)(x, times)
    torch.testing.assert_close(normal_epsilon, (1-a).sqrt()*x, rtol=0, atol=0)
    mixture_epsilon = MixtureEpsilonOracle(schedule)(x, times)
    _, _, posterior = mixture_quantities(x, a[:, 0])
    reconstructed = (x-(1-a).sqrt()*mixture_epsilon)/a.sqrt()
    torch.testing.assert_close(reconstructed, posterior, atol=2e-6, rtol=2e-6)
    for row, k in enumerate(indices.tolist()):
        mean, _ = reverse_parameters(x[row], k, normal_epsilon[row], schedule)
        torch.testing.assert_close(mean, schedule.alpha[k].sqrt()*x[row], atol=2e-7, rtol=2e-7)


def test_normal_variance_recursion_matches_closed_form_and_sampler_with_same_noise():
    torch.set_num_threads(1)
    schedule = make_schedule()
    path = normal_variance_path(schedule)
    alpha = schedule.alpha.double().numpy()
    posterior_variance = schedule.posterior_variance.double().numpy()
    # Independent unrolled linear-system covariance: product of transition
    # coefficients plus each injected noise variance transported to time zero.
    transport = np.concatenate(([1.0], np.cumprod(alpha[:-1])))
    expected = np.prod(alpha) + np.dot(posterior_variance, transport)
    assert len(path) == 101 and path[0] == 1.0
    assert path[-1] == pytest.approx(expected, abs=3e-15)
    assert .94 < path[-1] < .95  # Detect replacing posterior variance by beta.
    seed, n = 31, 128
    generated = sample_ddpm(NormalEpsilonOracle(schedule), n, schedule,
                            torch.Generator().manual_seed(seed))
    generator = torch.Generator().manual_seed(seed)
    reference = torch.randn(n, 2, generator=generator)
    for k in range(99, -1, -1):
        reference = schedule.alpha[k].sqrt()*reference
        if k:
            reference += schedule.posterior_variance[k].sqrt()*torch.randn(n, 2, generator=generator)
    torch.testing.assert_close(generated, reference, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("option,value", [("--samples", "0"), ("--samples", "65537"),
                                         ("--seeds", "-1"), ("--seeds", str(2**29)),
                                         ("--threads", "0")])
def test_protocol_bounds_reject_invalid_arguments(option, value):
    with pytest.raises(SystemExit) as error:
        make_parser().parse_args(["--output", "unused", option, value])
    assert error.value.code == 2


def test_duplicate_seeds_and_existing_file_are_rejected_before_writing(tmp_path, monkeypatch):
    output = tmp_path / "study"
    monkeypatch.setattr(sys, "argv", ["oracle-study", "--output", str(output), "--seeds", "7", "7"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not output.exists()
    output.write_text("preserve me")
    monkeypatch.setattr(sys, "argv", ["oracle-study", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and output.read_text() == "preserve me"


def test_cli_saves_all_arrays_and_roles_and_protects_results(tmp_path):
    output = tmp_path / "oracle"
    command = [sys.executable, "-m", "genai_study.oracle_study", "--output", str(output),
               "--samples", "32", "--seeds", "7", "19"]
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    metrics_path = output / "metrics.json"
    saved_metrics = metrics_path.read_bytes()
    report = json.loads(saved_metrics)
    array_path = output / "arrays.npz"
    assert report["outputs"]["arrays_npz_sha256"] == hashlib.sha256(array_path.read_bytes()).hexdigest()
    assert report["config"]["training_steps"] == 0
    assert report["config"]["sampling_steps"] == 100
    all_seeds = [seed for run in report["runs"] for seed in run["random_stream_seeds"].values()]
    assert len(all_seeds) == len(set(all_seeds)) == 10
    with np.load(array_path) as arrays:
        assert len(arrays.files) == 14
        for run in report["runs"]:
            for role in ("normal", "mixture"):
                stream = run["random_stream_seeds"][f"{role}_sampling"]
                expected = torch.randn(32, 2, generator=torch.Generator().manual_seed(stream))
                np.testing.assert_array_equal(arrays[run["array_keys"][f"{role}_initial"]], expected.numpy())
            for name in run["array_keys"].values():
                value = arrays[name]
                assert value.shape == (32, 2) and value.dtype == np.float32
                assert report["outputs"]["arrays"][name]["raw_sha256"] == hashlib.sha256(value.tobytes()).hexdigest()
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode == 2 and "empty directory" in repeated.stderr
    assert metrics_path.read_bytes() == saved_metrics
    assert report["outputs"]["arrays_npz_sha256"] == hashlib.sha256(array_path.read_bytes()).hexdigest()
