#!/usr/bin/env python3
"""Recompute the fixed SVD chunk experiment's raw-array comparisons on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be a new JSON path")
    arrays, metrics, runs = {}, {}, {}
    for chunk in (14, 7, 1):
        root = args.input_root / f"chunk{chunk}"
        metrics[chunk] = json.loads((root / "metrics.json").read_text())
        runs[chunk] = json.loads((args.input_root / f"record-chunk{chunk}/run.json").read_text())
        assert runs[chunk]["status"] == "succeeded" and runs[chunk]["exit_code"] == 0
        arrays[chunk] = {}
        for name, meta in metrics[chunk]["outputs"].items():
            path = root / meta["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"]
            a = np.load(path, allow_pickle=False)
            assert list(a.shape) == meta["shape"] and str(a.dtype) == meta["dtype"]
            assert a.dtype == np.float32 and np.isfinite(a).all()
            assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() == meta["array_bytes_sha256"]
            arrays[chunk][name] = a
    identities = {}
    for name in ("normalized_input", "latent_mode", "video_latents"):
        equal = all(arrays[14][name].tobytes(order="C") == arrays[k][name].tobytes(order="C")
                    for k in (7, 1))
        assert equal, f"different inputs: {name}"
        identities[name] = {"all_three_bitwise_equal": equal,
                            "array_bytes_sha256": metrics[14]["outputs"][name]["array_bytes_sha256"]}
    reference = arrays[14]["decoded_raw"].astype(np.float64)
    rows = []
    for chunk in (14, 7, 1):
        raw = arrays[chunk]["decoded_raw"].astype(np.float64)
        delta = raw - reference
        row = {"chunk_size": chunk,
               "global_raw_rms_vs_chunk14": float(np.sqrt(np.mean(delta**2))),
               "per_frame_raw_rms_vs_chunk14": np.sqrt(np.mean(delta**2, axis=(0, 1, 3, 4))).tolist(),
               "max_abs_raw_vs_chunk14": float(np.max(np.abs(delta))),
               "per_frame_max_abs_raw_vs_chunk14": np.max(np.abs(delta), axis=(0, 1, 3, 4)).tolist(),
               "adjacent_raw_rms": np.sqrt(np.mean(np.diff(raw, axis=2)**2, axis=(0, 1, 3, 4))).tolist(),
               "raw_range": [float(raw.min()), float(raw.max())],
               "fraction_outside_minus1_plus1": float(np.mean((raw < -1) | (raw > 1))),
               "decode_seconds": metrics[chunk]["timing_seconds"]["decode_latents"],
               "command_wall_seconds": runs[chunk]["wall_seconds"],
               "peak_rss_gib": metrics[chunk]["memory"]["peak_rss_kib"] / 2**20,
               "observed_call_frames": [x["num_frames"] for x in metrics[chunk]["decoder_calls"]],
               "raw_output_array_sha256": metrics[chunk]["outputs"]["decoded_raw"]["array_bytes_sha256"]}
        assert row["observed_call_frames"] == [chunk] * (14 // chunk)
        # The upstream B,F,C,H,W -> B,C,F,H,W permutation retains strided storage.
        # Reloaded .npy data are C-contiguous; reproduce the original reduction
        # layout when checking its archived float64 diagnostic bit for bit.
        original_view = raw.transpose(0, 2, 1, 3, 4).copy().transpose(0, 2, 1, 3, 4)
        original_rms = np.sqrt(np.mean(np.diff(original_view, axis=2)**2, axis=(0, 1, 3, 4)))
        np.testing.assert_array_equal(original_rms, metrics[chunk]["diagnostics"]["adjacent_raw_rms"])
        row["max_adjacent_rms_reduction_layout_difference"] = float(
            np.max(np.abs(np.asarray(row["adjacent_raw_rms"]) - original_rms)))
        rows.append(row)
    one = arrays[1]["decoded_raw"]
    seven = arrays[7]["decoded_raw"]
    result = {"schema_version": 1, "numpy": np.__version__,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "primary_coordinates": "Raw float32 VAE output B,C,F,H,W; differences and reductions computed in float64. No clipping or RGB quantization.",
              "input_identity": identities, "comparisons": rows,
              "chunk1_all_frames_equal": bool(np.array_equal(one, np.repeat(one[:, :, :1], 14, axis=2))),
              "chunk7_two_segments_equal": bool(np.array_equal(seven[:, :, :7], seven[:, :, 7:])),
              "all_file_and_array_hashes_validated": True,
              "scope": "One repeated-reference latent and one execution per chunk in declared order; chunk14 is a reference execution for output differences. Sampling quality, memory/speed generalization, and prior denoising failure causation require separate evidence."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"rows": [{k: r[k] for k in ("chunk_size", "global_raw_rms_vs_chunk14",
                      "max_abs_raw_vs_chunk14", "command_wall_seconds", "peak_rss_gib")} for r in rows],
                      "chunk1_all_frames_equal": result["chunk1_all_frames_equal"],
                      "chunk7_two_segments_equal": result["chunk7_two_segments_equal"]}, indent=2))


if __name__ == "__main__":
    main()
