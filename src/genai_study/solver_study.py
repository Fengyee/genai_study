"""Compare Euler step counts for one frozen flow checkpoint and identical noise."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import torch

from .cli import seed_with_offset
from .checkpoint import load_checkpoint
from .experiment import positive_int
from .toy import diagnostics, integrate_flow, sample_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=positive_int, default=[10, 25, 100, 200],
                        help="Euler step counts, sorted and deduplicated; no training")
    parser.add_argument("--samples", type=positive_int, default=2048)
    parser.add_argument("--seed", type=seed_with_offset(max_offset=2), default=123)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    model, payload = load_checkpoint(args.checkpoint)
    if payload["method"] != "flow":
        parser.error("solver_study requires a flow checkpoint; DDPM is not a velocity field")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("output directory is nonempty; choose another or use --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model.requires_grad_(False)
    counts = sorted(set(args.steps))
    initial = torch.randn(args.samples, 2, generator=torch.Generator().manual_seed(args.seed))
    reference = sample_data(args.samples, torch.Generator().manual_seed(args.seed + 1))
    baseline = sample_data(args.samples, torch.Generator().manual_seed(args.seed + 2))
    outputs = []
    runs = []
    for steps in counts:
        start = time.perf_counter()
        generated = integrate_flow(model, initial, steps)
        elapsed = time.perf_counter() - start
        measured = diagnostics(generated, reference)
        difference = None
        if outputs:
            difference = float((generated - outputs[-1]).square().sum(1).mean().sqrt())
        runs.append({"steps": steps, "nfe": steps, "elapsed_seconds": elapsed,
                     "diagnostics": measured,
                     "previous_steps": counts[len(outputs) - 1] if outputs else None,
                     "paired_rms_l2_difference_from_previous": difference})
        outputs.append(generated)
    result = {
        "schema_version": 1,
        "checkpoint": {"path": str(args.checkpoint),
                       "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                       "metadata": payload["metadata"]},
        "config": {"steps": counts, "samples": args.samples, "seed": args.seed,
                   "threads": args.threads, "device": "cpu", "solver": "explicit Euler",
                   "noise_seed": args.seed, "reference_seed": args.seed + 1,
                   "baseline_seed": args.seed + 2},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "baseline_diagnostics": diagnostics(baseline, reference),
        "runs": runs,
        "timing_scope": "one sampling call only; excludes loading, metrics and plotting; no warm-up",
        "limitations": "Same checkpoint and initial noise; no training. Paired endpoint differences compare two discretizations, NOT error against an exact solution. More steps need not improve target-distribution diagnostics. Single-seed timings are not a hardware benchmark.",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="genai-study-solver-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(cache) / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Store paired initial/final samples so the comparison can be independently checked.
        np.savez_compressed(args.output / "samples.npz", initial=initial.numpy(),
                            reference=reference.numpy(), baseline=baseline.numpy(),
                            **{f"steps_{n}": x.numpy() for n, x in zip(counts, outputs)})
        extent = max(3.5, 1.05 * float(torch.cat([reference, *outputs]).abs().max()))
        columns = min(3, len(counts) + 1)
        rows = (len(counts) + columns) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows),
                                 squeeze=False, constrained_layout=True)
        for ax, points, title in zip(axes.flat, [reference, *outputs],
                                    ["Target data", *[f"Euler: {n} steps / {n} NFE" for n in counts]]):
            ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.4)
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent),
                   xlabel="x1", ylabel="x2", title=title)
            ax.set_aspect("equal")
        for ax in list(axes.flat)[len(counts) + 1:]:
            ax.set_visible(False)
        fig.savefig(args.output / "samples.png", dpi=140)
        plt.close(fig)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
