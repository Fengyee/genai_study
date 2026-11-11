"""A synthetic video ordering counterexample, not a generative model or FID/FVD."""
import argparse
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np


def make_videos(frames: int = 16, seed: int = 7) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return T,H,W float64 intensities and the shuffled source-frame indices."""
    if not 2 <= frames <= 32:
        raise ValueError("Use 2..32 frames for this bounded contact-sheet experiment")
    y, x = np.mgrid[:32, :48]
    center_x = np.linspace(8.0, 39.0, frames)
    reference = np.exp(-((x[None] - center_x[:, None, None]) ** 2
                         + (y[None] - 15.5) ** 2) / (2 * 2.0**2))
    permutation = np.random.default_rng(seed).permutation(frames)
    # Ensure even tiny T yields a counterexample instead of an accidental identity.
    if np.array_equal(permutation, np.arange(frames)):
        permutation = np.roll(permutation, 1)
    return {"reference": reference, "shuffled": reference[permutation].copy(),
            "static_first_frame": np.repeat(reference[:1], frames, axis=0)}, permutation


def centroids(video: np.ndarray) -> np.ndarray:
    """Intensity-weighted (x,y) pixel coordinates, for these nonnegative frames."""
    y, x = np.mgrid[:video.shape[1], :video.shape[2]]
    mass = video.sum(axis=(1, 2))
    if np.any(mass <= 0):
        raise ValueError("Centroids require positive mass in every frame")
    return np.stack(((video * x).sum(axis=(1, 2)) / mass,
                     (video * y).sum(axis=(1, 2)) / mass), axis=1)


def video_metrics(video: np.ndarray, reference: np.ndarray, fps: float) -> dict:
    """Metrics with an exact reference, specialized to a right-moving bright blob.

    Symmetric frame-set MSE averages both nearest-frame directions. It ignores
    order and does not completely characterize frame multiplicities; it is not
    a distribution metric, FID, FVD, optical flow, or a perceptual score.
    """
    if video.shape != reference.shape or video.ndim != 3 or len(video) < 2:
        raise ValueError("Expected matching T,H,W videos with T>=2")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(video).all() or not np.isfinite(reference).all():
        raise ValueError("Video intensities must be finite")
    position = centroids(video)
    target_position = centroids(reference)
    steps = np.diff(position, axis=0)
    pair_mse = ((video[:, None] - reference[None, :]) ** 2).mean(axis=(2, 3))
    return {
        "ordered_mse_to_reference": float(((video - reference) ** 2).mean()),
        "time_mean_image_mse": float(((video.mean(0) - reference.mean(0)) ** 2).mean()),
        "symmetric_frame_set_mse": float((pair_mse.min(1).mean() + pair_mse.min(0).mean()) / 2),
        "centroid_rmse_px": float(np.sqrt(((position - target_position) ** 2).sum(1).mean())),
        "mean_centroid_speed_px_per_second": float(np.linalg.norm(steps, axis=1).mean() * fps),
        "wrong_direction_steps": int((steps[:, 0] < -1e-10).sum()),
        "endpoint_displacement_px": float(np.linalg.norm(position[-1] - position[0])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.frames <= 32:
        parser.error("--frames must be between 2 and 32")
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("output directory is nonempty; choose another or use --overwrite")
    duration = args.frames / args.fps
    if not np.isfinite(duration):
        parser.error("--fps must keep playback duration finite")
    videos, permutation = make_videos(args.frames, args.seed)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            timestamps = np.arange(args.frames) / args.fps
            metrics = {name: video_metrics(video, videos["reference"], args.fps)
                       for name, video in videos.items()}
    except FloatingPointError:
        parser.error("--fps must keep timestamps and centroid speeds finite")
    result = {
        "schema_version": 1,
        "config": {"frames": args.frames, "fps": args.fps, "seed": args.seed,
                   "shape": [args.frames, 32, 48], "layout": "T,H,W",
                   "dtype": "float64", "intensity_range": [0.0, 1.0],
                   "gaussian_sigma_px": 2.0, "center_x_start_px": 8.0,
                   "center_x_end_px": 39.0, "center_y_px": 15.5,
                   "first_frame_time_seconds": 0.0,
                   "last_frame_time_seconds": (args.frames - 1) / args.fps,
                   "nominal_playback_duration_seconds": duration,
                   "rng": "NumPy default_rng (PCG64)"},
        "shuffled_source_indices": permutation.tolist(),
        "timestamps_seconds": timestamps.tolist(),
        "metrics": metrics,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "limitations": "Synthetic known-trajectory counterexample; no learned generator or perceptual evaluator. Frame-set metrics ignore order. Static motion has zero speed but fails the moving-target task. These diagnostics are not FID/FVD or general-purpose video quality metrics.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "frames.npz", **videos,
                        shuffled_source_indices=permutation,
                        timestamps_seconds=timestamps)
    with tempfile.TemporaryDirectory(prefix="genai-study-video-") as cache:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(cache) / "matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        result["versions"]["matplotlib"] = matplotlib.__version__
        fig, axes = plt.subplots(3, args.frames, figsize=(1.2 * args.frames, 5),
                                 squeeze=False, constrained_layout=True)
        names = {"reference": "Reference: move right", "shuffled": "Same frames: shuffled",
                 "static_first_frame": "Static: fails motion task"}
        for row, (name, video) in enumerate(videos.items()):
            for index in range(args.frames):
                ax = axes[row, index]
                ax.imshow(video[index], cmap="gray", vmin=0, vmax=1, origin="upper")
                source = (int(permutation[index]) if name == "shuffled" else
                          0 if name == "static_first_frame" else index)
                ax.set_title(f"t={index / args.fps:.3f}s\nsrc={source}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
                if index == 0:
                    ax.set_ylabel(names[name], fontsize=8)
        fig.suptitle("Video ordering counterexample | intensities [0,1] | all frames shown")
        fig.savefig(args.output / "contact_sheet.png", dpi=140)
        plt.close(fig)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
