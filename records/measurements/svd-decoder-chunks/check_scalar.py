#!/usr/bin/env python3
"""Exact scalar temporal convolution checks; no trained decoder or weights.

From the repository root, use the existing base environment and a new output:
  .venv/bin/python records/measurements/svd-decoder-chunks/check_scalar.py \
    --output artifacts/svd-scalar-check.json
Optionally hash already installed Diffusers sources with --diffusers-root PATH.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


def full_exact(sequence: list[Fraction]) -> list[Fraction]:
    padded = [Fraction(0)] + sequence + [Fraction(0)]
    return [(padded[i] + 2 * padded[i + 1] + padded[i + 2]) / 4
            for i in range(len(sequence))]


def chunks_exact(sequence: list[Fraction], size: int) -> list[Fraction]:
    return [value for start in range(0, len(sequence), size)
            for value in full_exact(sequence[start:start + size])]


def missing_neighbor_residual(sequence: list[Fraction], size: int) -> list[Fraction]:
    result = [Fraction(0) for _ in sequence]
    for cut in range(size, len(sequence), size):
        # A cut is after the 1-based frame `cut`, before frame `cut+1`.
        result[cut - 1] -= sequence[cut] / 4
        result[cut] -= sequence[cut - 1] / 4
    return result


def halo_exact(sequence: list[Fraction], size: int) -> list[Fraction]:
    result = []
    for start in range(0, len(sequence), size):
        stop = min(start + size, len(sequence))
        left, right = max(0, start - 1), min(len(sequence), stop + 1)
        extended = full_exact(sequence[left:right])
        result.extend(extended[start - left:stop - left])
    return result


def array_record(values: list[Fraction]) -> dict:
    return {"rational": [str(value) for value in values],
            "float64": [float(value) for value in values]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diffusers-root", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    inputs = {"ramp7": list(range(1, 8)), "ramp14": list(range(1, 15)),
              "minimal2": [1, 2], "constant14": [1] * 14}
    cases = {}
    for name, values in inputs.items():
        sequence = list(map(Fraction, values))
        exact_full = full_exact(sequence)
        # Explicit pad + valid retains the requested length even for one frame.
        np_full = np.convolve(np.pad(np.array(values, dtype=np.float64), (1, 1)),
                              kernel, mode="valid")
        assert np.array_equal(np_full, np.array(list(map(float, exact_full))))
        records = {}
        for size in (1, 2, 7):
            result = chunks_exact(sequence, size)
            residual = [a - b for a, b in zip(result, exact_full)]
            expected_residual = missing_neighbor_residual(sequence, size)
            assert residual == expected_residual
            assert halo_exact(sequence, size) == exact_full
            np_chunks = np.concatenate([
                np.convolve(np.pad(np.array(values[start:start + size], dtype=np.float64), (1, 1)),
                            kernel, mode="valid")
                for start in range(0, len(values), size)])
            assert np.array_equal(np_chunks, np.array(list(map(float, result))))
            records[str(size)] = {
                "output": array_record(result),
                "residual_chunk_minus_full": array_record(residual),
                "internal_cuts_after_1based_frame": list(range(size, len(sequence), size)),
                "changed_1based_frames": [i + 1 for i, value in enumerate(residual) if value],
                "max_abs_residual": str(max(map(abs, residual))),
                "squared_residual_sum": str(sum(value ** 2 for value in residual)),
                "analytic_missing_neighbor_formula_matches": True,
                "numpy_convolution_matches_exact_rationals": True,
                "one_frame_halo_recovers_full_for_this_single_layer": True,
            }
        cases[name] = {"input": values, "full_output": array_record(exact_full), "chunks": records}

    sources = {}
    if args.diffusers_root:
        for rel in ("models/autoencoders/autoencoder_kl_temporal_decoder.py",
                    "pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py",
                    "models/resnet.py", "models/unets/unet_3d_blocks.py"):
            file = args.diffusers_root / rel
            sources[rel] = {"path": str(file.resolve()), "bytes": file.stat().st_size,
                            "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    report = {
        "schema_version": 1,
        "all_checks_passed": True,
        "scope": "Independent scalar arithmetic counterexample and optional local-source identities. No network, model construction, weights, sampling, or trained-model quality measurement.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "operator": "y_t=(x_(t-1)+2*x_t+x_(t+1))/4; stride 1; a single linear layer; zero outside each supplied sequence",
        "chunk_semantics": "At most chunk_size consecutive frames per call, independent zero padding per chunk; concatenate outputs; keep the final short chunk at its actual length.",
        "indexing": "1-based frame indices in explanatory output; residual means chunk output minus full output.",
        "kernel": {"rational": ["1/4", "1/2", "1/4"], "float64": kernel.tolist()},
        "cases": cases,
        "analytic_residual_rule": "For every internal cut b|b+1, add -x_(b+1)/4 to residual b and -x_b/4 to residual b+1. Add both contributions where cuts meet (chunk_size=1).",
        "source_identities": sources,
        "source_review": {
            "diffusers_version_observed_separately": "0.35.2",
            "pipeline_lines": "290-317: flatten frames, scale latents, slice chunks, pass chunk's actual num_frames to vae.decode, concatenate",
            "decoder_lines": "75-83 and 128-133: Conv3d temporal output with kernel (3,1,1), padding (1,0,0), and B,C,F,H,W layout",
            "temporal_blocks": "unet_3d_blocks.py:899-1009 links decoder blocks to SpatioTemporalResBlock; resnet.py:542-745 has two temporal Conv3d layers and GroupNorm on a B,C,F,H,W tensor",
            "normalization_implication": "Temporal GroupNorm aggregates over each group and all remaining dimensions, including F, so changing chunks can change normalization throughout a chunk.",
        },
        "supported_svd_conclusion": "Separate temporal decoder calls change available neighboring frames and normalization groups even for fixed input latents; chunk size can change decoded frames.",
        "excluded_inferences": [
            "The chosen scalar kernel equals learned SVD weights.",
            "SVD errors are confined to the two frames adjacent to a cut.",
            "A one-frame halo guarantees exact full-SVD decoding.",
            "Smaller chunks universally worsen temporal metrics or visible flicker monotonically.",
            "This arithmetic example proves the cause of existing SVD artifacts or measures real quality/memory/speed.",
            "The full decoder output is a ground-truth video.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "all_checks_passed": True,
                      "case_count": len(cases), "chunk_checks": sum(len(case["chunks"]) for case in cases.values())}))


if __name__ == "__main__":
    main()
