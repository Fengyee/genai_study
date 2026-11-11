#!/usr/bin/env python3
"""Recompute and plot raw SVD decoder chunk differences without loading a model.

Use the repository's existing base environment. All output paths must be new:
  .venv/bin/python records/measurements/svd-decoder-chunks/plot.py \
    --comparison artifacts/svd-decoder-chunks/comparison.json \
    --run-root artifacts/svd-decoder-chunks \
    --output-dir artifacts/svd-chunk-figure-new
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error("--output-dir must be a new directory")
    comparison = json.loads(args.comparison.read_text())
    expected = {row["chunk_size"]: row for row in comparison["comparisons"]}
    arrays, input_arrays, inputs = {}, {}, {}
    for chunk in (14, 7, 1):
        folder = args.run_root / f"chunk{chunk}"
        metrics_path = folder / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        raw_path = folder / "decoded_raw.npy"
        video_path = folder / "video_latents.npy"
        raw = np.load(raw_path, allow_pickle=False)
        video = np.load(video_path, allow_pickle=False)
        if raw.shape != (1, 3, 14, 144, 256) or raw.dtype != np.float32:
            raise ValueError(f"Unexpected decoder coordinates in {raw_path}")
        if video.shape != (1, 14, 4, 18, 32) or video.dtype != np.float32:
            raise ValueError(f"Unexpected latent coordinates in {video_path}")
        if not np.isfinite(raw).all() or not np.isfinite(video).all():
            raise ValueError("Non-finite source values")
        for name, path, array in (("decoded_raw", raw_path, raw),
                                  ("video_latents", video_path, video)):
            recorded = metrics["outputs"][name]
            if sha256(path) != recorded["sha256"] or raw_hash(array) != recorded["array_bytes_sha256"]:
                raise ValueError(f"Source identity mismatch: {path}")
        if raw_hash(raw) != expected[chunk]["raw_output_array_sha256"]:
            raise ValueError(f"Comparison source identity mismatch for chunk {chunk}")
        arrays[chunk] = raw
        input_arrays[chunk] = video
        inputs[str(chunk)] = {"metrics": identity(metrics_path), "decoded_raw": identity(raw_path),
                              "raw_array_sha256": raw_hash(raw), "video_latents": identity(video_path),
                              "video_latents_array_sha256": raw_hash(video)}

    if not all(np.array_equal(input_arrays[c], input_arrays[14]) for c in (7, 1)):
        raise ValueError("Input latent arrays differ between chunk runs")
    if not np.array_equal(input_arrays[14], np.repeat(input_arrays[14][:, :1], 14, axis=1)):
        raise ValueError("Expected the same latent in all fourteen frames")
    equal_single_frames = np.array_equal(arrays[1], np.repeat(arrays[1][:, :, :1], 14, axis=2))
    equal_seven_segments = np.array_equal(arrays[7][:, :, :7], arrays[7][:, :, 7:])
    if not equal_single_frames or not equal_seven_segments:
        raise ValueError("Repeated-output claims do not hold for the source arrays")

    reference = arrays[14].astype(np.float64)
    per_frame, adjacent, differences = {}, {}, {}
    for chunk, raw in arrays.items():
        values = raw.astype(np.float64)
        # Each reduction independently processes a frame or pair; source NPY stays unchanged.
        per_frame[chunk] = np.array([np.sqrt(np.mean((values[:, :, f] - reference[:, :, f]) ** 2))
                                     for f in range(14)], dtype=np.float64)
        adjacent[chunk] = np.array([np.sqrt(np.mean((values[:, :, f + 1] - values[:, :, f]) ** 2))
                                    for f in range(13)], dtype=np.float64)
        differences[str(chunk)] = {
            "per_frame_max_abs_difference": float(np.max(np.abs(per_frame[chunk] - expected[chunk]["per_frame_raw_rms_vs_chunk14"]))),
            "adjacent_max_abs_difference": float(np.max(np.abs(adjacent[chunk] - expected[chunk]["adjacent_raw_rms"])))
        }
        np.testing.assert_allclose(per_frame[chunk], expected[chunk]["per_frame_raw_rms_vs_chunk14"], rtol=0, atol=5e-15)
        np.testing.assert_allclose(adjacent[chunk], expected[chunk]["adjacent_raw_rms"], rtol=0, atol=5e-15)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Keep Matplotlib's writable cache inside the explicitly requested output.
    os.environ["MPLCONFIGDIR"] = str(args.output_dir / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    frame_index = np.arange(1, 15)
    pair_start = np.arange(1, 14)
    npz_data = {"frame_index_1based": frame_index,
                "pair_start_1based": pair_start, "pair_end_1based": pair_start + 1}
    for chunk in arrays:
        npz_data[f"chunk{chunk}_per_frame_raw_rms_vs_chunk14"] = per_frame[chunk]
        npz_data[f"chunk{chunk}_adjacent_raw_rms"] = adjacent[chunk]
    npz_path = args.output_dir / "data.npz"
    np.savez(npz_path, **npz_data)
    data = {
        "schema_version": 1,
        "scope": "One VAE-only repeated-reference latent experiment; chunk14 is the reference execution for differences.",
        "unit": "raw decoder coordinate",
        "source_coordinate": "float32 B,C,F,H,W; no clipping, normalization or RGB quantization of decoded output",
        "calculation": "Per-frame and adjacent-pair RMS recomputed directly from source NPY with float64 differences and reductions across B,C,H,W.",
        "script": identity(Path(__file__)), "comparison": identity(args.comparison), "inputs": inputs,
        "frame_index_1based": frame_index.tolist(),
        "pair_start_1based": pair_start.tolist(), "pair_end_1based": (pair_start + 1).tolist(),
        "curves": {str(c): {"per_frame_raw_rms_vs_chunk14": per_frame[c].tolist(),
                            "adjacent_raw_rms": adjacent[c].tolist()} for c in arrays},
        "input_and_output_equalities": {
            "all_three_video_latents_bitwise_equal": True,
            "all_fourteen_input_frames_bitwise_equal": True,
            "chunk1_all_output_frames_bitwise_equal": equal_single_frames,
            "chunk7_output_segments_1_to_7_and_8_to_14_bitwise_equal": equal_seven_segments,
        },
        "data_npz": {"file": npz_path.name, "sha256": sha256(npz_path)},
    }
    data_path = args.output_dir / "data.json"
    data_path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.labelsize": 11.5, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": "genai-svd-decoder-chunks-v1",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    styles = {
        14: {"color": "#555555", "linestyle": "--", "marker": "^", "label": "chunk 14 (reference)"},
        7: {"color": "#0072B2", "linestyle": "-", "marker": "o", "label": "chunk 7"},
        1: {"color": "#CC79A7", "linestyle": "-.", "marker": "s", "label": "chunk 1"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.1), dpi=180)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.765, bottom=0.25, wspace=0.25)
    title = fig.suptitle("SVD temporal decoder: the same latent, different chunks", y=0.98,
                        fontsize=16, fontweight="bold")
    subtitle = fig.text(0.5, 0.925, "14 identical latent frames  |  one VAE-only execution per chunk",
                        ha="center", va="center", fontsize=11.5, color="#444444")
    legend = fig.legend(handles=[Line2D([], [], linewidth=2, markersize=5, **styles[c]) for c in (14, 7, 1)],
                        loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=3,
                        frameon=False, columnspacing=2.8, handlelength=3.1, fontsize=11)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDE2E6", linewidth=0.7)
        ax.set_ylabel("RMS [raw decoder coordinate]")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.tick_params(axis="both", length=3.5, color="#777777")

    ax = axes[0]
    ax.set_title("A  Difference from chunk 14", loc="left", pad=13, fontweight="bold")
    ax.axhline(0, color=styles[14]["color"], linestyle="--", linewidth=1.3)
    for chunk in (7, 1):
        ax.plot(frame_index, per_frame[chunk], linewidth=2, markersize=4.6, **styles[chunk])
    ax.axvline(7.5, color="#777777", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(7.5, 0.049, "7 | 8 cut", ha="center", va="center", fontsize=10.5,
            color="#444444", bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    ax.set_xlim(0.6, 14.4)
    ax.set_ylim(-0.0015, 0.052)
    ax.set_xticks(frame_index)
    ax.set_yticks(np.arange(0, 0.051, 0.01))
    ax.set_xlabel("Frame (1-based)", labelpad=9)

    ax = axes[1]
    ax.set_title("B  Difference between adjacent frames", loc="left", pad=13, fontweight="bold")
    for chunk in (14, 7, 1):
        ax.plot(pair_start, adjacent[chunk], linewidth=2, markersize=4.6, **styles[chunk])
    ax.axvline(7, color="#777777", linestyle=":", linewidth=1.2, zorder=1)
    ax.text(7, 0.0258, "7 → 8 cut", ha="center", va="center", fontsize=10.5,
            color="#444444", bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})
    ax.set_xlim(0.6, 13.4)
    ax.set_ylim(-0.0015, 0.0275)
    ax.set_xticks(pair_start, [f"{i}→{i + 1}" for i in pair_start], rotation=50, ha="right")
    ax.set_yticks(np.arange(0, 0.026, 0.005))
    ax.set_xlabel("Adjacent frame pair (1-based)", labelpad=8)
    footer = fig.text(0.5, 0.065,
                      "Repeated inputs: chunk 7 repeats a 7-frame output; chunk 1 repeats one decoded frame.",
                      ha="center", va="center", fontsize=11, color="#444444")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    layout_objects = {"title": title, "subtitle": subtitle, "legend": legend, "footer": footer}
    for index, ax in enumerate(axes):
        layout_objects[f"panel{index + 1}_title"] = ax._left_title
        layout_objects[f"panel{index + 1}_xlabel"] = ax.xaxis.label
        layout_objects[f"panel{index + 1}_ylabel"] = ax.yaxis.label
        for j, tick in enumerate(ax.get_xticklabels() + ax.get_yticklabels()):
            layout_objects[f"panel{index + 1}_tick{j}"] = tick
    extents = {name: list(artist.get_window_extent(renderer).extents)
               for name, artist in layout_objects.items()}
    inside = {name: box[0] >= canvas.x0 - 0.5 and box[1] >= canvas.y0 - 0.5
                    and box[2] <= canvas.x1 + 0.5 and box[3] <= canvas.y1 + 0.5
              for name, box in extents.items()}
    if not all(inside.values()):
        raise ValueError(f"Text falls outside canvas: {[name for name, okay in inside.items() if not okay]}")
    png = args.output_dir / "svd-decoder-chunks.png"
    svg = args.output_dir / "svd-decoder-chunks.svg"
    fig.savefig(png, dpi=180, metadata={"Software": f"Matplotlib {matplotlib.__version__}; original scientific plot"})
    fig.savefig(svg, metadata={"Date": None, "Creator": f"Matplotlib {matplotlib.__version__}; original scientific plot"})
    plt.close(fig)
    verification = {
        "all_numerical_checks_passed": True,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "script": identity(Path(__file__)),
        "recomputed_vs_comparison_absolute_errors": differences,
        "comparison_absolute_tolerance": 5e-15,
        "input_and_output_equalities": data["input_and_output_equalities"],
        "cut_coordinates": {"panel_A_frame_axis": 7.5, "panel_B_pair_start_axis": 7},
        "chunk7_cut_adjacent_raw_rms": float(adjacent[7][6]),
        "chunk1_adjacent_all_exact_zero": bool(np.all(adjacent[1] == 0)),
        "plot_pixels": [int(canvas.width), int(canvas.height)],
        "all_recorded_text_inside_canvas": all(inside.values()),
        "text_extents_pixels": extents,
        "artifacts": {path.name: identity(path) for path in (png, svg, data_path, npz_path)},
        "scope": "Original Matplotlib visualization of already recorded raw arrays; no image generation, trained-model execution, or model loading.",
    }
    (args.output_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "numerical_checks_passed": True,
                      "pixels": verification["plot_pixels"], "png_sha256": sha256(png)}))


if __name__ == "__main__":
    main()
