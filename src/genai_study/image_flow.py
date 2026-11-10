"""Conditional 16x16 shape generation with a tiny CPU convolutional flow model.

Data are program-rendered; no images, pretrained weights, or downloads are used.
The fixed template rule below is a coarse synthetic diagnostic, not FID or proof
of distribution coverage. Plotting/geometry clip to [0,1]; ODE outputs do not.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import torch
from torch import Tensor, nn
import torch.nn.functional as F

IMAGE_SIZE = 16
EDGE_WIDTH = 0.35
TEMPLATE_MSE_THRESHOLD = 0.01  # Defined before training, validated on counterexamples.


def render_shapes(labels: Tensor, center: Tensor, size: Tensor) -> Tensor:
    """B1HW images. center is B,2 in (x,y); size is radius/half-width in pixels."""
    axis = torch.arange(IMAGE_SIZE, dtype=center.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    dx = xx[None] - center[:, 0, None, None]
    dy = yy[None] - center[:, 1, None, None]
    circle = (dx.square() + dy.square()).sqrt()
    square = torch.maximum(dx.abs(), dy.abs())
    distance = torch.where(labels[:, None, None] == 0, circle, square)
    return torch.sigmoid((size[:, None, None] - distance) / EDGE_WIDTH)[:, None]


def sample_shapes(n: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    if n < 2 or n % 2:
        raise ValueError("Use an even sample count >=2 for balanced conditions")
    labels = torch.arange(n) % 2
    center = 5 + 5 * torch.rand(n, 2, generator=generator)
    size = 2 + 1.5 * torch.rand(n, generator=generator)
    return render_shapes(labels, center, size), labels


def image_flow_path(noise: Tensor, data: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    time_map = t[:, None, None, None]
    return (1 - time_map) * noise + time_map * data, data - noise


def block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                         nn.GroupNorm(4, out_channels), nn.SiLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1),
                         nn.GroupNorm(4, out_channels), nn.SiLU())


class ConditionalConvFlow(nn.Module):
    """Tiny U-Net: noisy image, time, two condition channels, and x/y coordinates."""
    def __init__(self, width: int = 16):
        super().__init__()
        self.enc1 = block(6, width)
        self.enc2 = block(width, 2 * width)
        self.middle = block(2 * width, 4 * width)
        self.dec2 = block(6 * width, 2 * width)
        self.dec1 = block(3 * width, width)
        self.out = nn.Conv2d(width, 1, 1)
        axis = torch.linspace(-1, 1, IMAGE_SIZE)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coordinates", torch.stack((xx, yy))[None])

    def forward(self, x: Tensor, t: Tensor, labels: Tensor) -> Tensor:
        condition = F.one_hot(labels, num_classes=2).to(x.dtype)[:, :, None, None]
        inputs = torch.cat((x, t[:, None, None, None].expand(-1, 1, 16, 16),
                            condition.expand(-1, -1, 16, 16),
                            self.coordinates.expand(len(x), -1, -1, -1)), dim=1)
        first = self.enc1(inputs)
        second = self.enc2(F.avg_pool2d(first, 2))
        middle = self.middle(F.avg_pool2d(second, 2))
        up = self.dec2(torch.cat((F.interpolate(middle, scale_factor=2, mode="nearest"), second), dim=1))
        up = self.dec1(torch.cat((F.interpolate(up, scale_factor=2, mode="nearest"), first), dim=1))
        return self.out(up)


@torch.no_grad()
def integrate_images(model: nn.Module, initial: Tensor, labels: Tensor, steps: int) -> Tensor:
    """Explicit Euler on t in [0,1], with supplied BCHW noise and class labels.

    One model evaluation per step; retain raw values and leave inputs unchanged.
    The caller controls the model's evaluation mode.
    """
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    if initial.ndim != 4 or initial.shape[1:] != (1, IMAGE_SIZE, IMAGE_SIZE) or not initial.is_floating_point():
        raise ValueError("initial must be floating B,1,16,16 images")
    if labels.shape != (len(initial),) or labels.dtype != torch.long or labels.device != initial.device:
        raise ValueError("labels must be a B-element int64 tensor on the image device")
    if not len(initial) or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Use nonempty images with labels 0 or 1")
    x = initial.clone()
    for step in range(steps):
        t = torch.full((len(x),), step / steps, dtype=x.dtype, device=x.device)
        x = x + model(x, t, labels) / steps
    return x


@torch.no_grad()
def sample_images(model: nn.Module, labels: Tensor, steps: int, generator: torch.Generator) -> Tensor:
    initial = torch.randn(len(labels), 1, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    return integrate_images(model, initial, labels, steps)


def template_bank() -> Tensor:
    """2 x 847 x 256 templates: centers 5..10 by .5, size 2..3.5 by .25."""
    x, y, size = torch.meshgrid(torch.arange(5.0, 10.01, 0.5),
                                torch.arange(5.0, 10.01, 0.5),
                                torch.arange(2.0, 3.51, 0.25), indexing="ij")
    center = torch.stack((x.flatten(), y.flatten()), dim=1)
    return torch.stack([render_shapes(torch.full((len(center),), label), center, size.flatten()).flatten(1)
                        for label in (0, 1)])


@torch.no_grad()
def image_diagnostics(raw: Tensor, labels: Tensor, templates: Tensor | None = None) -> dict:
    """Synthetic recognizability and diversity summaries, grouped by given condition.

    Template matching and mass/centers use clipped intensities, with raw clipping
    distortion reported separately. Std uses denominator n, not sample n-1.
    """
    if not torch.isfinite(raw).all():
        raise ValueError("Nonfinite image output")
    bank = template_bank() if templates is None else templates
    clipped = raw.clamp(0, 1)
    flat = clipped.flatten(1)
    distance = (flat.square().mean(1)[:, None, None]
                + bank.square().mean(2)[None]
                - 2 * torch.einsum("bd,ckd->bck", flat, bank) / flat.shape[1]).clamp_min(0)
    best = distance.min(2).values
    own = best[torch.arange(len(labels)), labels]
    inferred = best.argmin(1)
    mass = flat.sum(1)
    axis = torch.arange(16, dtype=clipped.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    center_x = (clipped[:, 0] * xx).sum((1, 2)) / mass.clamp_min(1e-12)
    center_y = (clipped[:, 0] * yy).sum((1, 2)) / mass.clamp_min(1e-12)
    result = {}
    for label, name in enumerate(("circle", "square")):
        selected = labels == label
        if not selected.any():
            raise ValueError("Diagnostics require both conditions")
        group = raw[selected]
        valid_mass = mass[selected] > 1e-8
        result[name] = {
            "label": label, "count": int(selected.sum()),
            "coarse_recognizable_fraction": float(((own[selected] <= TEMPLATE_MSE_THRESHOLD)
                                                    & (inferred[selected] == label)).float().mean()),
            "condition_match_fraction": float((inferred[selected] == label).float().mean()),
            "nearest_own_template_mse_mean": float(own[selected].mean()),
            "raw_out_of_range_fraction": float(((group < 0) | (group > 1)).float().mean()),
            "mean_absolute_clip_change": float((group - group.clamp(0, 1)).abs().mean()),
            "foreground_mass_mean": float(mass[selected].mean()),
            "foreground_mass_std": float(mass[selected].std(unbiased=False)),
            "zero_mass_fraction": float((~valid_mass).float().mean()),
            "center_x_mean_px": float(center_x[selected][valid_mass].mean()) if valid_mass.any() else None,
            "center_y_mean_px": float(center_y[selected][valid_mass].mean()) if valid_mass.any() else None,
            "center_x_std_px": float(center_x[selected][valid_mass].std(unbiased=False)) if valid_mass.any() else None,
            "center_y_std_px": float(center_y[selected][valid_mass].std(unbiased=False)) if valid_mass.any() else None,
        }
    return result


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=positive, default=1500)
    parser.add_argument("--batch-size", type=positive, default=64)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--sampling-steps", type=positive, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=positive, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size % 2 or args.samples % 2 or args.samples < 16:
        parser.error("batch-size must be even; samples must be even and >=16")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("output directory is nonempty; choose another or use --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = ConditionalConvFlow()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    eval_rng = torch.Generator().manual_seed(args.seed + 2)
    sample_rng = torch.Generator().manual_seed(args.seed + 3)
    reference_rng = torch.Generator().manual_seed(args.seed + 4)
    baseline_rng = torch.Generator().manual_seed(args.seed + 5)

    def batch(n, rng):
        data, labels = sample_shapes(n, rng)
        noise = torch.randn(data.shape, generator=rng)
        t = torch.rand(n, generator=rng)
        xt, velocity = image_flow_path(noise, data, t)
        return xt, t, labels, velocity

    eval_x, eval_t, eval_labels, eval_velocity = batch(256, eval_rng)

    @torch.no_grad()
    def eval_loss():
        errors = (model(eval_x, eval_t, eval_labels) - eval_velocity).square().mean((1, 2, 3))
        return {"all": float(errors.mean()), "circle": float(errors[eval_labels == 0].mean()),
                "square": float(errors[eval_labels == 1].mean())}

    start = time.perf_counter()
    initial_loss = eval_loss()
    losses = []
    for _ in range(args.steps):
        xt, t, labels, target = batch(args.batch_size, train_rng)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(xt, t, labels) - target).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    final_loss = eval_loss()
    training_seconds = time.perf_counter() - start
    reference, labels = sample_shapes(args.samples, reference_rng)
    baseline, _ = sample_shapes(args.samples, baseline_rng)
    start = time.perf_counter()
    generated = sample_images(model, labels, args.sampling_steps, sample_rng)
    sampling_seconds = time.perf_counter() - start
    templates = template_bank()
    observed = image_diagnostics(generated, labels, templates)
    target_summary = image_diagnostics(reference, labels, templates)
    baseline_summary = image_diagnostics(baseline, labels, templates)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config.update({"device": "cpu", "layout": "B,C,H,W", "image_shape": [1, 16, 16],
                   "learning_rate": 1e-3, "base_width": 16, "eval_examples": 256,
                   "data_range": [0, 1], "center_range_px": [5, 10], "size_range_px": [2, 3.5],
                   "edge_width_px": EDGE_WIDTH, "coarse_template_mse_threshold": TEMPLATE_MSE_THRESHOLD,
                   "rng_offsets": {"initialization": 0, "training": 1, "heldout": 2,
                                   "sampling": 3, "reference": 4, "target_baseline": 5}})
    result = {"schema_version": 1, "config": config,
              "versions": {"python": platform.python_version(), "torch": str(torch.__version__)},
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "initial_heldout_loss": initial_loss, "final_heldout_loss": final_loss,
              "training_and_eval_seconds": training_seconds, "sampling_seconds": sampling_seconds,
              "diagnostics": observed, "reference_diagnostics": target_summary,
              "independent_target_baseline": baseline_summary,
              "checkpoint": "checkpoint.pt",
              "limitations": "Synthetic 16x16 shapes. Coarse template recognizability and distribution coverage require separate assessment. Geometry and figures clip to [0,1]; raw outputs and clipping distortion are retained. The checkpoint contains final inference weights and training metadata; optimizer state is omitted."}
    with tempfile.TemporaryDirectory(prefix="genai-study-image-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(cache) / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        result["versions"].update(numpy=np.__version__, matplotlib=matplotlib.__version__)
        # Local import keeps the model module independent of its checkpoint loader.
        from .image_checkpoint import save_image_checkpoint
        save_image_checkpoint(args.output / "checkpoint.pt", model,
                              {"config": config, "versions": result["versions"],
                               "initial_heldout_loss": initial_loss, "final_heldout_loss": final_loss})
        np.savez_compressed(args.output / "arrays.npz", generated_raw=generated.numpy(),
                            reference=reference.numpy(), independent_target=baseline.numpy(),
                            labels=labels.numpy(), training_loss=np.array(losses))
        fig, axes = plt.subplots(4, 8, figsize=(10, 6), constrained_layout=True)
        for label, name in enumerate(("circle", "square")):
            for offset, (images, source) in enumerate(((reference, "target"), (generated, "generated"))):
                row = 2 * label + offset
                indices = torch.where(labels == label)[0][:8]
                for col, index in enumerate(indices):
                    axes[row, col].imshow(images[index, 0].clamp(0, 1), cmap="gray", vmin=0, vmax=1)
                    axes[row, col].set_xticks([])
                    axes[row, col].set_yticks([])
                    axes[row, col].set_title(f"index={int(index)}", fontsize=8)
                    if col == 0:
                        axes[row, col].set_ylabel(f"{name} ({label})\n{source}")
        fig.suptitle("First 8 samples per condition; display clipped to [0,1]")
        fig.savefig(args.output / "samples.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
        ax.plot(range(1, len(losses) + 1), losses, linewidth=0.7)
        ax.set(xlabel="Optimizer update", ylabel="Velocity MSE over B,C,H,W")
        fig.savefig(args.output / "loss.png", dpi=140)
        plt.close(fig)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
