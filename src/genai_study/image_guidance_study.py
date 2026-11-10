"""Frozen-model classifier-free velocity guidance on paired 16x16 shapes."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time

import torch
from torch import Tensor, nn

from .image_checkpoint import load_image_checkpoint
from .image_flow import image_diagnostics, integrate_images, positive, sample_shapes, template_bank


def guidance_scale(value: str) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale < 0:
        raise argparse.ArgumentTypeError("guidance scales must be finite and nonnegative")
    return scale


@torch.no_grad()
def guided_velocity(model: nn.Module, x: Tensor, t: Tensor, labels: Tensor, scale: float) -> Tensor:
    """Evaluate both branches, then form v_u+s(v_c-v_u); two calls per batch."""
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("guidance scale must be finite and nonnegative")
    unconditional = model(x, t, torch.full_like(labels, -1))
    conditional = model(x, t, labels)
    # Preserve the exact endpoint branches while retaining the declared two calls.
    if scale == 0:
        return unconditional
    if scale == 1:
        return conditional
    return unconditional + scale * (conditional - unconditional)


@torch.no_grad()
def integrate_guided_images(model: nn.Module, initial: Tensor, labels: Tensor,
                            steps: int, scale: float) -> Tensor:
    """Euler integration with 2*steps network evaluations per generated image."""
    return integrate_images(lambda x, t, c: guided_velocity(model, x, t, c, scale),
                            initial, labels, steps)


def require_guidance_training(payload: dict) -> None:
    """Require the training command's null-branch protocol and observed counts."""
    metadata = payload.get("metadata", {})
    config = metadata.get("config", {})
    counts = metadata.get("training_condition_counts", {})
    if not isinstance(config, dict) or not isinstance(counts, dict):
        raise ValueError("Guidance requires metadata from condition-dropout training")
    probability = config.get("condition_dropout")
    if (type(probability) not in (float, int) or not math.isfinite(probability)
            or not 0 < probability < 1 or config.get("null_condition_encoding") != "zero-vector"
            or any(type(counts.get(key)) is not int or counts[key] <= 0
                   for key in ("conditional", "unconditional"))):
        raise ValueError("Guidance requires a checkpoint trained with --condition-dropout in (0,1), "
                         "zero-vector null encoding, and observed conditional/unconditional examples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", type=guidance_scale, default=[0.0, 1.0, 2.0, 4.0],
                        help="Finite nonnegative scales; include 1 as the paired reference")
    parser.add_argument("--sampling-steps", type=positive, default=40)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < 16 or args.samples % 2:
        parser.error("samples must be even and >=16")
    if 1 not in args.scales:
        parser.error("scales must include 1 as the paired conditional reference")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("output directory is nonempty; choose another or use --overwrite")
    try:
        model, payload = load_image_checkpoint(args.checkpoint)
        require_guidance_training(payload)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model.requires_grad_(False)
    scales = sorted(set(args.scales))
    noise = torch.randn(args.samples // 2, 1, 16, 16,
                        generator=torch.Generator().manual_seed(args.seed))
    initial = noise.repeat_interleave(2, dim=0)
    reference, labels = sample_shapes(args.samples, torch.Generator().manual_seed(args.seed + 1))
    baseline, _ = sample_shapes(args.samples, torch.Generator().manual_seed(args.seed + 2))
    templates = template_bank()
    outputs, runs = {}, []
    for scale in scales:
        start = time.perf_counter()
        generated = integrate_guided_images(model, initial, labels, args.sampling_steps, scale)
        elapsed = time.perf_counter() - start
        outputs[scale] = generated
        runs.append({"scale": scale, "array_key": f"scale_{scale:.17g}", "steps": args.sampling_steps,
                     "nfe": 2 * args.sampling_steps, "per_branch_nfe": args.sampling_steps,
                     "elapsed_seconds": elapsed,
                     "diagnostics": image_diagnostics(generated, labels, templates),
                     "paired_class_pixel_rms_difference": float((generated[0::2] - generated[1::2]).square().mean().sqrt()),
                     "paired_class_clipped_pixel_rms_difference": float((generated[0::2].clamp(0, 1)
                                                                         - generated[1::2].clamp(0, 1)).square().mean().sqrt())})
    conditional_reference = outputs[1.0]
    for run in runs:
        generated = outputs[run["scale"]]
        run["paired_pixel_rms_difference_from_scale1"] = float((generated - conditional_reference).square().mean().sqrt())
        run["paired_clipped_pixel_rms_difference_from_scale1"] = float((generated.clamp(0, 1)
                                                                        - conditional_reference.clamp(0, 1)).square().mean().sqrt())
    result = {
        "schema_version": 1,
        "checkpoint": {"path": str(args.checkpoint),
                       "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                       "schema_version": payload["schema_version"], "method": payload["method"],
                       "architecture": payload["architecture"], "metadata": payload["metadata"]},
        "config": {"scales": scales, "sampling_steps": args.sampling_steps,
                   "samples": args.samples, "independent_noise_pairs": args.samples // 2,
                   "seed": args.seed, "threads": args.threads, "device": "cpu",
                   "solver": "explicit Euler", "guidance": "v_unconditional + scale * (v_conditional - v_unconditional)",
                   "null_condition_encoding": "zero-vector", "null_label": -1,
                   "layout": "B,C,H,W", "image_shape": [1, 16, 16],
                   "noise_seed": args.seed, "reference_seed": args.seed + 1,
                   "baseline_seed": args.seed + 2, "labels": {"0": "circle", "1": "square"}},
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "reference_diagnostics": image_diagnostics(reference, labels, templates),
        "independent_target_baseline": image_diagnostics(baseline, labels, templates),
        "runs": runs,
        "paired_difference_definition": "sqrt(mean over B,C,H,W of squared endpoint differences from scale1); class-paired fields compare labels0/1 for the same noise; clipped variants clamp each endpoint first",
        "nfe_definition": "two separate full-batch model calls per Euler step, at every scale including 0 and 1; NFE is per generated image",
        "timing_scope": "one integration call only; excludes loading, metrics and plotting; no warm-up",
        "limitations": "Synthetic 16x16 shape geometry and finite-sample variation. Scales modify the learned velocity; exact tempered-density sampling requires additional theoretical conditions. Generated circle/square pairs share their initial noise; uncertainty analysis must preserve those pairs. Raw outputs and clipping diagnostics are retained.",
    }
    with tempfile.TemporaryDirectory(prefix="genai-study-image-guidance-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(cache) / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        result["versions"].update(numpy=np.__version__, matplotlib=matplotlib.__version__)
        np.savez_compressed(args.output / "arrays.npz", initial=initial.numpy(), labels=labels.numpy(),
                            reference=reference.numpy(), independent_target=baseline.numpy(),
                            **{run["array_key"]: outputs[run["scale"]].numpy() for run in runs})
        fig, axes = plt.subplots(len(scales) + 1, 8, figsize=(10, 1.5 * (len(scales) + 1)),
                                 squeeze=False, constrained_layout=True)
        images_per_row = [reference, *[outputs[scale] for scale in scales]]
        titles = ["Target", *[f"scale={scale:.17g}\nNFE={2 * args.sampling_steps}" for scale in scales]]
        for row, (images, title) in enumerate(zip(images_per_row, titles)):
            for index in range(8):
                axes[row, index].imshow(images[index, 0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
                axes[row, index].set_xticks([])
                axes[row, index].set_yticks([])
                if row == 0:
                    name = "circle" if index % 2 == 0 else "square"
                    axes[row, index].set_title(f"{name} / pair {index // 2}", fontsize=8)
                if index == 0:
                    axes[row, index].set_ylabel(title)
        fig.suptitle("Independent targets; matched-noise generated pairs; display clipped to [0,1]")
        fig.savefig(args.output / "samples.png", dpi=140)
        plt.close(fig)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
