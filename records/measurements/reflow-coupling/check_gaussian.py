#!/usr/bin/env python3
"""Small float64 checks for two exact Gaussian rectification calculations.

Run with the existing repository environment, for example:
  .venv/bin/python records/measurements/reflow-coupling/check_gaussian.py \
    --output artifacts/reflow-independent/gaussian.json

The output must not already exist. No Monte Carlo estimation, model, or plotting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    identity = torch.eye(2)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                           [1.25, -0.75], [-2.0, 0.5], [0.25, 2.0]])
    times = torch.unique(torch.cat((torch.linspace(0.0, 1.0, 101),
                                   torch.tensor([1e-12, 1.0 - 1e-12]))))
    error_lists: dict[str, list[float]] = {}

    def error(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        value = float((actual - expected).abs().max())
        error_lists.setdefault(name, []).append(value)

    def q(t: torch.Tensor) -> torch.Tensor:
        return (1.0 - t).square() + t.square()

    def a_matrix(t: torch.Tensor) -> torch.Tensor:
        return (1.0 - t) * identity + t * rotation

    def independent_path(t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(q(t)) * points

    def rotation_path(t: torch.Tensor) -> torch.Tensor:
        return points @ a_matrix(t).T

    def independent_field(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return (2.0 * t - 1.0) / q(t) * x

    def rotation_field(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b = ((2.0 * t - 1.0) * identity + rotation) / q(t)
        return x @ b.T

    unit = torch.tensor(1.0)
    speed_expected = points @ (rotation - identity).T
    for t in times:
        qt = q(t)
        at = a_matrix(t)
        inverse = ((1.0 - t) * identity - t * rotation) / qt
        bt = ((2.0 * t - 1.0) * identity + rotation) / qt
        error("rotation_inverse", at @ inverse, identity)
        error("rotation_determinant", torch.linalg.det(at), qt)
        error("same_covariance_path", at @ at.T, qt * identity)
        error("rotation_field_coefficients", (rotation - identity) @ inverse, bt)
        independent_xt, independent_derivative = torch.func.jvp(
            independent_path, (t,), (unit,))
        rotation_xt, rotation_derivative = torch.func.jvp(
            rotation_path, (t,), (unit,))
        error("independent_ode_jvp", independent_derivative,
              independent_field(t, independent_xt))
        error("rotation_ode_jvp", rotation_derivative,
              rotation_field(t, rotation_xt))
        error("rotation_constant_velocity", rotation_field(t, rotation_xt),
              speed_expected)
        error("rotation_endpoint_chord", rotation_xt,
              (1.0 - t) * points + t * (points @ rotation.T))
        _, rotation_acceleration = torch.func.jvp(
            lambda s: rotation_field(s, rotation_path(s)), (t,), (unit,))
        error("rotation_material_acceleration", rotation_acceleration,
              torch.zeros_like(points))
        # For independent endpoints Cov(V|X_t) = 2I - Cov(V,X_t)^2/q.
        conditional_covariance = (2.0 - (2.0 * t - 1.0).square() / qt) * identity
        error("independent_conditional_covariance", conditional_covariance,
              identity / qt)
        cross_product = (points[:, 0] * independent_xt[:, 1]
                         - points[:, 1] * independent_xt[:, 0])
        error("independent_radial_collinearity", cross_product,
              torch.zeros(points.shape[0]))
        # Retraining on the unchanged rotation endpoint map gives same field.
        reflow_rotation_field = rotation_xt @ ((rotation - identity) @ inverse).T
        error("rotation_reflow_field", reflow_rotation_field,
              rotation_field(t, rotation_xt))
        # Difference in velocity has zero divergence when weighted by p_t.
        difference_field = independent_xt @ rotation.T / qt
        divergence_over_density = (torch.trace(rotation) / qt
                                   + ((-independent_xt / qt) * difference_field).sum(-1))
        error("weighted_divergence_difference", divergence_over_density,
              torch.zeros(points.shape[0]))

    epsilon = 1e-6
    for t in torch.linspace(0.01, 0.99, 25):
        for name, path, field in (("independent", independent_path, independent_field),
                                  ("rotation", rotation_path, rotation_field)):
            derivative = (path(t + epsilon) - path(t - epsilon)) / (2.0 * epsilon)
            error(f"{name}_ode_finite_difference", derivative, field(t, path(t)))

    zero, half, one = torch.tensor(0.0), torch.tensor(0.5), torch.tensor(1.0)
    error("independent_start", independent_path(zero), points)
    error("independent_endpoint_identity", independent_path(one), points)
    error("independent_midpoint", independent_path(half), points / torch.sqrt(torch.tensor(2.0)))
    error("rotation_start", rotation_path(zero), points)
    error("rotation_endpoint", rotation_path(one), points @ rotation.T)
    error("rotation_endpoint_field_zero", ((-identity + rotation) / q(zero)), rotation - identity)
    error("rotation_endpoint_field_one", ((identity + rotation) / q(one)), identity + rotation)

    # This four-point rule matches the standard Gaussian first two moments.
    # It validates polynomial costs, not conditional expectations of a Gaussian.
    moment_points = torch.sqrt(torch.tensor(2.0)) * torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    error("moment_rule_mean", moment_points.mean(0), torch.zeros(2))
    error("moment_rule_covariance", moment_points.T @ moment_points / 4.0, identity)
    independent_cost_moment = float(
        (moment_points[:, None, :] - moment_points[None, :, :]).square().sum(-1).mean())
    rotation_cost_moment = float(
        (moment_points @ rotation.T - moment_points).square().sum(-1).mean())
    rotation_cost_trace = float(torch.trace((rotation - identity).T @ (rotation - identity)))
    error("independent_cost_moment", torch.tensor(independent_cost_moment), torch.tensor(4.0))
    error("rotation_cost_moment", torch.tensor(rotation_cost_moment), torch.tensor(4.0))
    error("rotation_cost_trace", torch.tensor(rotation_cost_trace), torch.tensor(4.0))

    errors = {name: max(values) for name, values in sorted(error_lists.items())}
    tolerances = {name: (1e-8 if "finite_difference" in name else 2e-14)
                  for name in errors}
    passed_checks = {name: value <= tolerances[name] for name, value in errors.items()}
    independent_chord_deviation = float((independent_path(half) - points).abs().max())
    grid_q_minimum = float(q(times).min())
    grid_independent_lipschitz_maximum = float(((2.0 * times - 1.0) / q(times)).abs().max())
    grid_rotation_lipschitz_maximum = float(torch.sqrt(2.0 / q(times)).max())
    assertions = {
        "all_numerical_equalities_within_tolerance": all(passed_checks.values()),
        "nonzero_independent_paths_deviate_from_constant_endpoint_chord": independent_chord_deviation > 0.1,
        "rotation_population_cost_strictly_exceeds_identity_cost": rotation_cost_trace > 0.0,
        "q_minimum_at_least_half": grid_q_minimum >= 0.5,
        "independent_lipschitz_at_most_one": grid_independent_lipschitz_maximum <= 1.0 + 1e-14,
        "rotation_lipschitz_at_most_two": grid_rotation_lipschitz_maximum <= 2.0 + 1e-14,
    }
    result = {
        "schema_version": 1,
        "all_checks_passed": all(assertions.values()),
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dtype": "torch.float64",
        "threads": torch.get_num_threads(),
        "method": "Analytic Gaussian conditioning and exact ODE formulas; deterministic small-array JVP and finite-difference checks. No model, training, or Monte Carlo.",
        "coordinates": "Two-dimensional column-vector formulas; arrays store samples as rows. R is counterclockwise 90 degrees.",
        "inputs": {"R": rotation.tolist(), "points": points.tolist(), "times": times.tolist(),
                   "finite_difference_times": torch.linspace(0.01, 0.99, 25).tolist(),
                   "finite_difference_epsilon": epsilon},
        "population_cost_convention": "E[||X1-X0||_2^2], without a factor 1/2",
        "population_costs_2d": {"independent_original": 4.0, "independent_rectified": 0.0,
                                "independent_after_reflow": 0.0, "rotation_original": 4.0,
                                "rotation_rectified": 4.0, "rotation_after_reflow": 4.0,
                                "global_optimum_identical_marginals": 0.0},
        "general_dimension": {"independent_original_cost": "2d", "independent_rectified_cost": 0.0},
        "straightness_definition": "Affine, constant-velocity interpolation between an ODE trajectory's own endpoints.",
        "straightness": {"original_conditional_paths_both_cases": True,
                         "independent_ode_affine_endpoint_chord_for_nonzero_start": False,
                         "independent_ode_lies_on_radial_line": True,
                         "independent_reflow_ode_constant": True,
                         "rotation_ode_affine_endpoint_chord": True},
        "wellposedness": {"time_interval": [0.0, 1.0],
                           "analytic_q_lower_bound": 0.5,
                           "analytic_independent_global_lipschitz_bound": 1.0,
                           "analytic_rotation_global_lipschitz_bound": 2.0,
                           "rotation_interpolation_inverse": "((1-t)I-tR)/q_t",
                           "rotation_interpolation_determinant": "q_t",
                           "endpoint_singularities": False},
        "grid_observations": {"q_minimum": grid_q_minimum,
                              "independent_lipschitz_maximum": grid_independent_lipschitz_maximum,
                              "rotation_lipschitz_maximum": grid_rotation_lipschitz_maximum,
                              "independent_midpoint_max_coordinate_deviation_from_endpoint_chord": independent_chord_deviation},
        "moment_rule": {"purpose": "Exact first/second-moment polynomial-cost check; not an estimate of Gaussian conditional expectations.",
                        "points": moment_points.tolist(), "weights": [0.25] * 4,
                        "independent_cartesian_product_cost": independent_cost_moment,
                        "rotation_cost": rotation_cost_moment},
        "maximum_absolute_errors": errors,
        "absolute_tolerances": tolerances,
        "numerical_checks_passed": passed_checks,
        "assertions": assertions,
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"output": str(args.output), "all_checks_passed": result["all_checks_passed"],
                      "numerical_equalities": len(errors),
                      "largest_jvp_residual": max(v for k, v in errors.items() if "jvp" in k),
                      "largest_finite_difference_residual": max(v for k, v in errors.items() if "finite_difference" in k)},
                     ensure_ascii=False))
    if not result["all_checks_passed"]:
        raise SystemExit("One or more independent mathematical checks failed.")


if __name__ == "__main__":
    main()
