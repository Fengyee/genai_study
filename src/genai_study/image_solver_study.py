"""Compare Euler step counts using one image checkpoint and paired initial noise."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import torch

from .image_checkpoint import load_image_checkpoint
from .image_flow import image_diagnostics, integrate_images, positive, sample_shapes, template_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=positive, default=[10, 20, 40, 80],
                        help="Euler step counts, sorted and deduplicated")
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < 16 or args.samples % 2:
        parser.error("samples must be even and >=16")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("output directory is nonempty; choose another or use --overwrite")
    try:
        model, payload = load_image_checkpoint(args.checkpoint)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model.requires_grad_(False)
    counts = sorted(set(args.steps))
    initial = torch.randn(args.samples, 1, 16, 16, generator=torch.Generator().manual_seed(args.seed))
    reference, labels = sample_shapes(args.samples, torch.Generator().manual_seed(args.seed + 1))
    baseline, _ = sample_shapes(args.samples, torch.Generator().manual_seed(args.seed + 2))
    templates = template_bank()
    outputs, runs = [], []
    for steps in counts:
        start = time.perf_counter()
        generated = integrate_images(model, initial, labels, steps)
        elapsed = time.perf_counter() - start
        raw_difference = clipped_difference = None
        if outputs:
            previous = outputs[-1]
            raw_difference = float((generated - previous).square().mean().sqrt())
            clipped_difference = float((generated.clamp(0, 1) - previous.clamp(0, 1)).square().mean().sqrt())
        runs.append({"steps": steps, "nfe": steps, "elapsed_seconds": elapsed,
                     "diagnostics": image_diagnostics(generated, labels, templates),
                     "previous_steps": counts[len(outputs) - 1] if outputs else None,
                     "paired_pixel_rms_difference_from_previous": raw_difference,
                     "paired_clipped_pixel_rms_difference_from_previous": clipped_difference})
        outputs.append(generated)
    result = {
        "schema_version": 1,
        "checkpoint": {"path": str(args.checkpoint),
                       "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                       "schema_version": payload["schema_version"], "method": payload["method"],
                       "architecture": payload["architecture"], "metadata": payload["metadata"]},
        "config": {"steps": counts, "samples": args.samples, "seed": args.seed,
                   "threads": args.threads, "device": "cpu", "solver": "explicit Euler",
                   "layout": "B,C,H,W", "image_shape": [1, 16, 16],
                   "noise_seed": args.seed, "reference_seed": args.seed + 1,
                   "baseline_seed": args.seed + 2, "labels": {"0": "circle", "1": "square"}},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "reference_diagnostics": image_diagnostics(reference, labels, templates),
        "independent_target_baseline": image_diagnostics(baseline, labels, templates),
        "runs": runs,
        "paired_difference_definition": "sqrt(mean over B,C,H,W of squared paired endpoint differences); clipped variant clamps each endpoint to [0,1] first",
        "timing_scope": "one integration call only; excludes loading, metrics and plotting; no warm-up",
        "limitations": "Paired endpoint differences measure changes between two Euler discretizations of the same frozen model. Exact-solution error and target-distribution quality require separate evidence. More steps can change geometric diagnostics in either direction. Synthetic 16x16 shapes; single-run CPU timings depend on hardware and workload.",
    }
    with tempfile.TemporaryDirectory(prefix="genai-study-image-solver-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(cache) / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        result["versions"].update(numpy=np.__version__, matplotlib=matplotlib.__version__)
        np.savez_compressed(args.output / "arrays.npz", initial=initial.numpy(),
                            labels=labels.numpy(), reference=reference.numpy(),
                            independent_target=baseline.numpy(),
                            **{f"steps_{n}": x.numpy() for n, x in zip(counts, outputs)})
        fig, axes = plt.subplots(len(counts) + 1, 8, figsize=(10, 1.5 * (len(counts) + 1)),
                                 squeeze=False, constrained_layout=True)
        indices = torch.cat([torch.where(labels == label)[0][:4] for label in (0, 1)])
        for row, (images, title) in enumerate(zip([reference, *outputs],
                                                ["Target", *[f"Euler {n}\nNFE={n}" for n in counts]])):
            for column, index in enumerate(indices):
                axes[row, column].imshow(images[index, 0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
                if row == 0:
                    name = "circle" if labels[index] == 0 else "square"
                    axes[row, column].set_title(f"{name} / index {int(index)}", fontsize=8)
                if column == 0:
                    axes[row, column].set_ylabel(title)
        fig.suptitle("Paired generated rows; independent targets; display clipped to [0,1]")
        fig.savefig(args.output / "samples.png", dpi=140)
        plt.close(fig)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
