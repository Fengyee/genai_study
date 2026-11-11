"""Run a CPU toy experiment; see --help. No datasets or model downloads."""
import argparse
import json
import os
import tempfile
import platform
from pathlib import Path
import time

import torch

from .checkpoint import save_checkpoint
from .toy import (GAUSSIAN_3SIGMA_MASS_2D, TimeMLP, diagnostics, flow_path, make_schedule, q_sample,
                  sample_data, sample_ddpm, sample_flow)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ddpm", "flow"), required=True)
    parser.add_argument("--steps", type=positive_int, default=800,
                        help="optimizer updates (not sampling steps)")
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--samples", type=positive_int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("output directory is nonempty; choose another or use --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    eval_rng = torch.Generator().manual_seed(args.seed + 2)
    sample_rng = torch.Generator().manual_seed(args.seed + 3)
    reference_rng = torch.Generator().manual_seed(args.seed + 4)
    baseline_rng = torch.Generator().manual_seed(args.seed + 5)
    model = TimeMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    schedule = make_schedule()

    def batch(n: int, rng: torch.Generator):
        data = sample_data(n, rng)
        noise = torch.randn(n, 2, generator=rng)
        if args.method == "ddpm":
            k = torch.randint(len(schedule.beta), (n,), generator=rng)
            return (q_sample(data, k, noise, schedule),
                    (k + 1).float() / len(schedule.beta), noise)
        t = torch.rand(n, generator=rng)
        xt, target = flow_path(noise, data, t)
        return xt, t, target

    eval_x, eval_t, eval_target = batch(4096, eval_rng)

    @torch.no_grad()
    def eval_loss():
        return float((model(eval_x, eval_t) - eval_target).square().mean())

    start = time.perf_counter()
    initial_loss = eval_loss()
    losses = []
    for step in range(args.steps):
        x, t, target = batch(args.batch_size, train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x, t) - target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at update {step + 1}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    final_loss = eval_loss()
    if args.method == "ddpm":
        generated = sample_ddpm(model, args.samples, schedule, sample_rng)
    else:
        generated = sample_flow(model, args.samples, 100, sample_rng)
    reference = sample_data(args.samples, reference_rng)
    measured = diagnostics(generated, reference)
    baseline = diagnostics(sample_data(args.samples, baseline_rng), reference)
    elapsed = time.perf_counter() - start

    # The managed cloud home can be read-only; respect supplied cache paths.
    with tempfile.TemporaryDirectory(prefix="genai-study-plot-") as plot_cache:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(plot_cache) / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", plot_cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        config = vars(args).copy()
        config["output"] = str(args.output)
        config.update({"device": "cpu", "learning_rate": 1e-3, "hidden_width": 128,
                       "sampling_steps": 100, "data_radius": 2.0, "data_sigma": 0.08,
                       "ddpm_beta_start": 1e-4, "ddpm_beta_end": 0.1,
                       "ddpm_alpha_bar_terminal": float(schedule.alpha_bar[-1]),
                       "eval_examples": 4096})
        result = {
            "method": args.method, "config": config,
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__, "matplotlib": matplotlib.__version__},
            "elapsed_seconds": elapsed,
            "timing_scope": "evaluation, training, sampling and metrics; excludes imports and plots",
            "initial_eval_loss": initial_loss, "final_eval_loss": final_loss,
            "diagnostics": measured,
            "baseline_diagnostics": baseline,
            "data_theory": {"own_center_3sigma_mass_2d": GAUSSIAN_3SIGMA_MASS_2D},
            "limitations": "2D toy only; fixed-seed diagnostics are not image/video quality or uncertainty estimates. DDPM and flow losses have different targets and are not directly comparable.",
        }
        save_checkpoint(args.output / "checkpoint.pt", model, args.method,
                        {"config": config, "versions": result["versions"],
                         "purpose": "inference only; optimizer state is not retained"})
        result["checkpoint"] = "checkpoint.pt"
        (args.output / "metrics.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        np.savez_compressed(args.output / "samples.npz", generated=generated.numpy(),
                            reference=reference.numpy(), training_loss=np.array(losses))
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        # Shared full-data limits prevent cropping outliers into apparent success.
        extent = max(3.5, 1.05 * float(torch.cat((reference, generated)).abs().max()))
        for ax, points, title in zip(axes, (reference, generated), ("Target data", args.method)):
            ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.4)
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent), title=title, xlabel="x1", ylabel="x2")
            ax.set_aspect("equal")
        fig.savefig(args.output / "samples.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
        ax.plot(range(1, len(losses) + 1), losses, linewidth=0.7)
        ax.set(xlabel="Optimizer update", ylabel="Batch target MSE", title=args.method)
        fig.savefig(args.output / "loss.png", dpi=140)
        plt.close(fig)
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
