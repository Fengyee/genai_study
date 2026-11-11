"""Independent checks of a saved six-case VAE study; no network forward."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path, help="six-case experiment output directory")
parser.add_argument("--record", type=Path, required=True, help="run recorder's run.json")
parser.add_argument("--output", type=Path, required=True, help="new independent-check JSON path")
args = parser.parse_args()
base = args.results
metrics = json.loads((base / "metrics.json").read_text())
record = json.loads(args.record.read_text())
assert record["status"] == "succeeded" and record["exit_code"] == 0
with np.load(base / "arrays.npz", allow_pickle=False) as archive:
    arrays = {key: archive[key] for key in archive.files}
assert len(arrays) == 26
for name, value in arrays.items():
    identity = metrics["arrays"][name]
    assert list(value.shape) == identity["shape"]
    assert str(value.dtype) == identity["dtype"]
    assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == identity["sha256"]
for name, identity in metrics["outputs"].items():
    contents = (base / name).read_bytes()
    assert len(contents) == identity["size_bytes"]
    assert hashlib.sha256(contents).hexdigest() == identity["sha256"]
original = Path(metrics["input"]["source_path"]).read_bytes()
assert hashlib.sha256(original).hexdigest() == metrics["input"]["source_sha256"]
assert (base / "input-source.bin").read_bytes() == original
target = np.asarray(Image.open(base / "input.png"), dtype=np.float32).transpose(2, 0, 1)[None] / np.float32(255)
np.testing.assert_array_equal(target, arrays["input_rgb"])
np.testing.assert_array_equal(2 * target - 1, arrays["input_normalized"])
assert target.shape == (1, 3, 512, 512)
assert arrays["posterior_mean"].shape == (1, 4, 64, 64)
assert target.size / arrays["posterior_mean"].size == 48
mean, std, logvar = [arrays[f"posterior_{key}"] for key in ("mean", "std", "logvar")]
assert np.min(logvar) >= -30 and np.max(logvar) <= 20
np.testing.assert_allclose(std, np.exp(logvar.astype(np.float64) / 2), rtol=2e-6, atol=0)
np.testing.assert_array_equal(arrays["case_0_latent"], mean)
for index, seed in enumerate((7, 19, 43), 1):
    epsilon = torch.randn(mean.shape, generator=torch.Generator().manual_seed(seed)).numpy()
    np.testing.assert_array_equal(epsilon, arrays[f"epsilon_seed{seed}"])
    np.testing.assert_array_equal(mean + std * epsilon, arrays[f"case_{index}_latent"])
scale = np.float32(0.18215)
np.testing.assert_array_equal(arrays["case_4_latent"], (mean * scale) / scale)
np.testing.assert_array_equal(arrays["case_5_latent"], mean * scale)

checks = []
target64 = target.astype(np.float64)
mode_rgb = arrays["case_0_rgb_raw"].astype(np.float64)
for index, case in enumerate(metrics["cases"]):
    raw_decoder = arrays[f"case_{index}_decoder_raw"]
    raw_rgb = arrays[f"case_{index}_rgb_raw"]
    np.testing.assert_array_equal(raw_decoder * np.float32(0.5) + np.float32(0.5), raw_rgb)
    raw64 = raw_rgb.astype(np.float64)
    clipped = raw64.clip(0, 1)
    mse = float(np.square(raw64 - target64).mean())
    clipped_mse = float(np.square(clipped - target64).mean())
    expected = {"raw_rgb_mse": mse, "raw_rgb_rms": mse ** 0.5,
                "clipped_rgb_mse": clipped_mse, "clipped_rgb_rms": clipped_mse ** 0.5,
                "out_of_range_fraction": float(((raw64 < 0) | (raw64 > 1)).mean()),
                "clipping_rms": float(np.sqrt(np.square(clipped - raw64).mean()))}
    for key, value in expected.items():
        np.testing.assert_allclose(case["diagnostics"][key], value, rtol=1e-12, atol=1e-15)
    assert clipped_mse <= mse + 1e-15
    pixels = (np.clip(raw_rgb[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
    np.testing.assert_array_equal(np.asarray(Image.open(base / case["image"])), pixels)
    expected.update(id=case["id"], raw_rgb_rms_from_mode=float(np.sqrt(np.square(raw64 - mode_rgb).mean())),
                    raw_rgb_max_abs_from_mode=float(np.abs(raw64 - mode_rgb).max()))
    checks.append(expected)
result = {"scope": "Saved arrays and direct analytic/RNG/pixel checks; no model forward",
          "arrays_verified": len(arrays), "artifact_files_verified": len(metrics["outputs"]),
          "source_commit": record["git"]["head"], "cases": checks,
          "posterior_std_vs_numpy_max_abs": float(np.max(np.abs(std - np.exp(logvar.astype(np.float64) / 2)))),
          "latent_scale_roundtrip_max_abs": float(np.max(np.abs(arrays["case_4_latent"] - mean))),
          "primary_wall_seconds": record["wall_seconds"],
          "process_peak_rss_gib": metrics["memory"]["process_peak_rss_kib"] / 1024**2}
output = args.output
with output.open("x") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")
print(json.dumps(result, indent=2))
