"""Measure six fixed SD1.4 VAE encode-decode cases on one local image.

Only the verified VAE component is loaded. Help and argument validation use
stdlib; inference requires the separate pretrained environment. Raw decoder
values, display transformations and posterior randomness remain distinguishable.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import time

from .pretrained_image import (
    DEFAULT_MANIFEST, MODEL_ID, REVISION, bounded_int, load_manifest,
    require_empty_output, sha256_file, verify_model,
)


VAE_FILES = frozenset({"vae/config.json", "vae/diffusion_pytorch_model.safetensors"})


def validate_args(args):
    require_empty_output(args.output)
    if len(args.seeds) != 3 or len(set(args.seeds)) != 3:
        raise ValueError("the six-case protocol requires three distinct posterior seeds")
    if not args.input.is_file():
        raise ValueError("input must be an existing image file")


def prepare_input(path):
    """Apply EXIF orientation and RGB conversion, preserving the oriented size."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        original_size = list(source.size)
        original_mode = source.mode
        orientation = source.getexif().get(274)
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    if any(size < 128 or size > 512 or size % 8 for size in (height, width)):
        raise ValueError("oriented image dimensions must be 128..512 and multiples of 8")
    rgb = np.ascontiguousarray(np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255)
    normalized = np.float32(2) * rgb - np.float32(1)
    return image, rgb, normalized, {
        "source_path": str(path), "source_sha256": sha256_file(path),
        "original_size_wh": original_size, "original_mode": original_mode,
        "exif_orientation": orientation, "prepared_size_wh": [width, height],
        "transform": "EXIF transpose -> RGB -> float32/255 -> BCHW -> 2*x-1; no crop or resize",
    }


def rgb_diagnostics(target, raw):
    """Pixel statistics over all BCHW entries, accumulated in float64."""
    import numpy as np

    target = np.asarray(target, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64)
    if target.shape != raw.shape or target.size == 0:
        raise ValueError("target and reconstruction must have matching nonempty shapes")
    if not np.isfinite(target).all() or not np.isfinite(raw).all():
        raise ValueError("pixel statistics require finite arrays")
    if np.any((target < 0) | (target > 1)):
        raise ValueError("target RGB must lie in [0,1]")
    clipped = np.clip(raw, 0, 1)
    raw_mse = float(np.mean((raw - target) ** 2))
    clipped_mse = float(np.mean((clipped - target) ** 2))
    return {
        "raw_rgb_range": [float(raw.min()), float(raw.max())],
        "raw_rgb_mse": raw_mse, "raw_rgb_rms": raw_mse ** 0.5,
        "clipped_rgb_mse": clipped_mse, "clipped_rgb_rms": clipped_mse ** 0.5,
        "out_of_range_fraction": float(np.mean((raw < 0) | (raw > 1))),
        "clipping_rms": float(np.sqrt(np.mean((clipped - raw) ** 2))),
    }


def sample_with_epsilon(posterior, seed):
    """Call the actual posterior API and independently reconstruct its random draw."""
    import torch

    sample = posterior.sample(generator=torch.Generator(device="cpu").manual_seed(seed))
    epsilon = torch.randn(posterior.mean.shape, dtype=posterior.mean.dtype, device="cpu",
                          generator=torch.Generator(device="cpu").manual_seed(seed))
    expected = posterior.mean + posterior.std * epsilon
    residual = float((sample - expected).abs().max())
    if not torch.equal(sample, expected):
        raise ValueError("posterior sample differs from the independently reconstructed CPU draw")
    return sample, epsilon, residual


def array_identity(array):
    import numpy as np

    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "hash_format": "C-order raw bytes"}


def run_study(args, manifest):
    started = time.perf_counter()
    validate_args(args)
    component_files = [item for item in manifest["files"] if item["path"] in VAE_FILES]
    verification = verify_model({**manifest, "files": component_files}, args.model_dir)
    configuration = json.loads((args.model_dir / "vae/config.json").read_text())
    if configuration.get("scaling_factor") != 0.18215:
        raise ValueError("the fixed VAE config must explicitly set scaling_factor=0.18215")
    verification_seconds = time.perf_counter() - started

    preprocess_start = time.perf_counter()
    input_image, input_rgb, normalized, input_info = prepare_input(args.input)
    preprocess_seconds = time.perf_counter() - preprocess_start
    load_start = time.perf_counter()
    import numpy as np
    import torch
    from diffusers import AutoencoderKL
    from PIL import Image, ImageDraw

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    vae = AutoencoderKL.from_pretrained(
        str(args.model_dir), subfolder="vae", local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).to("cpu").eval().requires_grad_(False)
    vae.disable_tiling()
    vae.disable_slicing()
    attention = {name: f"{type(value).__module__}.{type(value).__name__}"
                 for name, value in vae.attn_processors.items()}
    if not attention or any(not value.endswith(".AttnProcessor2_0") for value in attention.values()):
        raise ValueError("this fixed CPU protocol expects the installed default AttnProcessor2_0")
    scale = float(vae.config.scaling_factor)
    if scale != configuration["scaling_factor"]:
        raise ValueError("loaded VAE scaling differs from the verified configuration")
    load_seconds = time.perf_counter() - load_start

    arrays = {"input_rgb": input_rgb, "input_normalized": normalized}
    with torch.inference_mode():
        print("VAE encode started", flush=True)
        encode_start = time.perf_counter()
        posterior = vae.encode(torch.from_numpy(normalized)).latent_dist
        encode_seconds = time.perf_counter() - encode_start
        mode = posterior.mode()
        if not torch.equal(mode, posterior.mean):
            raise ValueError("diagonal Gaussian mode must equal its mean")
        if tuple(mode.shape) != (1, 4, normalized.shape[2] // 8, normalized.shape[3] // 8):
            raise ValueError("encoded shape differs from the fixed 4-channel, factor-8 VAE")
        for name in ("mean", "logvar", "std"):
            arrays[f"posterior_{name}"] = getattr(posterior, name).numpy().copy()

        cases = [("mode", mode, None, None)]
        for seed in args.seeds:
            sample, epsilon, residual = sample_with_epsilon(posterior, seed)
            arrays[f"epsilon_seed{seed}"] = epsilon.numpy().copy()
            cases.append((f"posterior-seed{seed}", sample, seed, residual))
        roundtrip = (scale * mode) / scale
        scale_residual = float((roundtrip - mode).abs().max())
        cases.extend([("scale-roundtrip", roundtrip, None, None),
                      ("scale-mismatch", scale * mode, None, None)])

        results = []
        for index, (name, latent, seed, residual) in enumerate(cases):
            print(f"VAE decode {index + 1}/6: {name}", flush=True)
            decode_start = time.perf_counter()
            decoded = vae.decode(latent).sample
            decode_seconds = time.perf_counter() - decode_start
            if tuple(decoded.shape) != tuple(normalized.shape) or decoded.dtype != torch.float32:
                raise ValueError("decoded image must have the original BCHW shape and float32 dtype")
            prefix = f"case_{index}"
            arrays[f"{prefix}_latent"] = latent.numpy().copy()
            arrays[f"{prefix}_decoder_raw"] = decoded.numpy().copy()
            arrays[f"{prefix}_rgb_raw"] = (decoded * 0.5 + 0.5).numpy().copy()
            results.append({
                "id": name, "seed": seed, "array_prefix": prefix,
                "reparameterization_max_abs_error": residual,
                "decode_seconds": decode_seconds,
                "diagnostics": rgb_diagnostics(input_rgb, arrays[f"{prefix}_rgb_raw"]),
            })

    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("all stored arrays must be finite")
    save_start = time.perf_counter()
    require_empty_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    source_copy = args.output / "input-source.bin"
    shutil.copyfile(args.input, source_copy)
    if sha256_file(source_copy) != input_info["source_sha256"]:
        raise ValueError("input file changed during the experiment")
    input_image.save(args.output / "input.png")
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    panels = [("input", input_image)]
    for result in results:
        rgb = arrays[f"{result['array_prefix']}_rgb_raw"][0].transpose(1, 2, 0)
        pixels = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
        image = Image.fromarray(pixels)
        filename = result["id"] + ".png"
        image.save(args.output / filename)
        result["image"] = filename
        panels.append((result["id"], image))
    width, height = input_image.size
    label_height = 24
    contact = Image.new("RGB", (4 * width, 2 * (height + label_height)), "white")
    draw = ImageDraw.Draw(contact)
    for index, (label, image) in enumerate(panels):
        left, top = (index % 4) * width, (index // 4) * (height + label_height)
        draw.text((left + 6, top + 4), label, fill="black")
        contact.paste(image, (left, top + label_height))
    contact.save(args.output / "contact_sheet.png")

    repo = Path(__file__).resolve().parents[2]
    versions = {name: metadata.version(name) for name in (
        "torch", "numpy", "diffusers", "transformers", "accelerate", "safetensors", "Pillow",
    )}
    versions["python"] = platform.python_version()
    result = {
        "schema_version": 1,
        "model": {"model_id": MODEL_ID, "revision": REVISION,
                  "manifest_path": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
                  "verification_scope": "VAE config and safetensors only; other SD components are unused",
                  "files": component_files, **verification},
        "input": input_info,
        "config": {"seeds": args.seeds, "threads": args.threads, "device": "cpu", "dtype": "float32",
                   "deterministic_algorithms": True, "tiling": vae.use_tiling, "slicing": vae.use_slicing,
                   "scaling_factor": scale, "vae_config": dict(vae.config),
                   "attention_processors": attention},
        "versions": versions,
        "source": {"vae_study_sha256": sha256_file(Path(__file__)),
                   "pretrained_image_sha256": sha256_file(Path(__file__).with_name("pretrained_image.py")),
                   "pretrained_lock_sha256": sha256_file(repo / "requirements-pretrained-lock.txt")},
        "posterior": {
            "family": "encoder approximate posterior: diagonal Gaussian",
            "logvar_clamp": [-30.0, 20.0], "mode_equals_mean": True,
            "mean_coordinate_mean": float(np.mean(arrays["posterior_mean"], dtype=np.float64)),
            "mean_coordinate_std": float(np.std(arrays["posterior_mean"], dtype=np.float64)),
            "std_range": [float(arrays["posterior_std"].min()), float(arrays["posterior_std"].max())],
            "std_coordinate_mean": float(np.mean(arrays["posterior_std"], dtype=np.float64)),
        },
        "representation": {"image_scalar_count": int(input_rgb.size),
                           "latent_scalar_count": int(mode.numel()),
                           "scalar_count_ratio": input_rgb.size / mode.numel(),
                           "scale_roundtrip_max_abs_error": scale_residual},
        "network_evaluations": {"vae_encode_calls": 1, "vae_decode_calls": len(results)},
        "cases": results,
        "arrays": {name: array_identity(array) for name, array in arrays.items()},
        "diagnostic_scope": "RGB coordinate [0,1]; statistics reduce all BCHW entries in float64. Raw RGB is (decoder+1)/2 before clipping. Display applies clip, round(255*x), uint8.",
        "timing_seconds": {"verification": verification_seconds, "preprocess": preprocess_seconds,
                           "vae_load": load_seconds, "encode": encode_seconds,
                           "decode_total": sum(item["decode_seconds"] for item in results)},
        "timing_scope": "No warm-up. VAE load includes optional imports; encode/decode fields time API calls. Per-case diagnostics and artifact saving are included in total_before_metrics_write. Full command wall is recorded externally.",
        "memory": {"process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                   "scope": "Linux process lifetime high-water RSS, including imports, weights and saving"},
        "outputs": {path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for path in sorted(args.output.iterdir()) if path.is_file()},
    }
    result["timing_seconds"]["save_and_metadata"] = time.perf_counter() - save_start
    result["timing_seconds"]["total_before_metrics_write"] = time.perf_counter() - started
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs=3, type=bounded_int(0, 2**32 - 1), default=[7, 19, 43],
                        help="exactly three distinct posterior seeds for the fixed six-case protocol")
    parser.add_argument("--threads", type=bounded_int(1, 4), default=1)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        manifest = load_manifest(args.manifest)
        result = run_study(args, manifest)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
