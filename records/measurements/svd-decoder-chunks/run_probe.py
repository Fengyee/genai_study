#!/usr/bin/env python3
"""Decode one fixed 14-frame SVD latent with a declared temporal chunk size.

Uses the optional pretrained environment and the cached fixed SVD VAE. Output
is raw decoder coordinates B,C,F,H,W. Powered by Stability AI; model terms:
examples/pretrained-svd/LICENSE-model.txt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import platform
import resource
import time
from types import SimpleNamespace

from genai_study.pretrained_image import checked_model_path, sha256_file
from genai_study.pretrained_video import load_manifest


INPUT_SHA256 = "e76d50ef91a5ece55252b9c867d58927ca8b21a303ee112014491608f79ed962"
FRAMES = 14


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, choices=(14, 7, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be a new directory")
    if not args.input.is_file() or sha256_file(args.input) != INPUT_SHA256:
        parser.error("input must match the archived SVD reference PNG SHA")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    manifest_path = Path("examples/pretrained-svd/manifest.json")
    manifest = load_manifest(manifest_path)
    verified = []
    for entry in manifest["files"]:
        if entry["path"] in {"vae/config.json", "vae/diffusion_pytorch_model.safetensors",
                              "LICENSE.md"}:
            path = checked_model_path(args.model_dir, entry["path"])
            if (not path.is_file() or path.stat().st_size != entry["size_bytes"]
                    or sha256_file(path) != entry["sha256"]):
                parser.error(f"fixed asset does not match manifest: {entry['path']}")
            verified.append(entry)
    if len(verified) != 3:
        parser.error("expected VAE config, weights and model license")
    verify_seconds = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"Verified fixed VAE assets; starting chunk={args.chunk_size}", flush=True)

    load_started = time.perf_counter()
    import numpy as np
    import torch
    from diffusers import AutoencoderKLTemporalDecoder, StableVideoDiffusionPipeline
    from diffusers.models import resnet
    from PIL import Image

    versions = {name: metadata.version(name)
                for name in ("torch", "numpy", "diffusers", "transformers", "Pillow")}
    if versions["diffusers"] != "0.35.2" or versions["torch"] != "2.6.0+cpu":
        parser.error("this fixed protocol requires Diffusers 0.35.2 and torch 2.6.0+cpu")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    vae, loading_info = AutoencoderKLTemporalDecoder.from_pretrained(
        str(args.model_dir / "vae"), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True, output_loading_info=True,
    )
    if any(loading_info.values()):
        raise ValueError(f"unexpected model loading diagnostics: {loading_info}")
    vae.to("cpu").eval().requires_grad_(False)
    scaling_factor = float(vae.config.scaling_factor)
    if scaling_factor != 0.18215 or vae.config.latent_channels != 4:
        raise ValueError("unexpected fixed VAE configuration")
    load_seconds = time.perf_counter() - load_started

    with Image.open(args.input) as im:
        if im.mode != "RGB" or im.size != (256, 144):
            raise ValueError("reference must be RGB at 256 by 144 pixels")
        pixels = np.asarray(im).copy()
    image = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    print("Encoding reference posterior mode", flush=True)
    encode_started = time.perf_counter()
    with torch.inference_mode():
        latent_mode = vae.encode(image).latent_dist.mode()
        latents = latent_mode[:, None].repeat(1, FRAMES, 1, 1, 1) * scaling_factor
    encode_seconds = time.perf_counter() - encode_started
    if tuple(latents.shape) != (1, FRAMES, 4, 18, 32):
        raise ValueError("unexpected encoded latent shape")
    if not torch.equal(latents, latents[:, :1].expand_as(latents)):
        raise ValueError("all frame latents must be identical")
    calls = []

    def decoder_input(module, positional, keyword):
        sample = positional[0] if positional else keyword["sample"]
        calls.append({"shape": list(sample.shape), "num_frames": keyword["num_frames"]})

    hook = vae.decoder.register_forward_pre_hook(decoder_input, with_kwargs=True)
    print(f"Decoding all {FRAMES} frames with chunk={args.chunk_size}", flush=True)
    decode_started = time.perf_counter()
    try:
        with torch.inference_mode():
            # Invoke the installed upstream method unchanged, with only its VAE dependency.
            decoded = StableVideoDiffusionPipeline.decode_latents(
                SimpleNamespace(vae=vae), latents, num_frames=FRAMES,
                decode_chunk_size=args.chunk_size,
            )
    finally:
        hook.remove()
    decode_seconds = time.perf_counter() - decode_started
    if tuple(decoded.shape) != (1, 3, FRAMES, 144, 256) or not torch.isfinite(decoded).all():
        raise ValueError("decoder output shape or finite-value contract failed")
    if [c["num_frames"] for c in calls] != [args.chunk_size] * (FRAMES // args.chunk_size):
        raise ValueError("observed decoder calls differ from the planned partition")

    save_started = time.perf_counter()
    arrays = {"normalized_input": image.numpy(), "latent_mode": latent_mode.numpy(),
              "video_latents": latents.numpy(), "decoded_raw": decoded.numpy()}
    outputs = {}
    for name, array in arrays.items():
        array = np.ascontiguousarray(array)
        path = args.output / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        outputs[name] = {"file": path.name, "shape": list(array.shape), "dtype": str(array.dtype),
                         "sha256": sha256_file(path),
                         "array_bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    raw64 = arrays["decoded_raw"].astype(np.float64)
    diagnostics = {
        "raw_min": float(raw64.min()), "raw_max": float(raw64.max()),
        "fraction_outside_minus1_plus1": float(np.mean((raw64 < -1) | (raw64 > 1))),
        "adjacent_raw_rms": np.sqrt(np.mean(np.diff(raw64, axis=2)**2, axis=(0, 1, 3, 4))).tolist(),
        "coordinate": "Raw VAE decoder output; normalization, clipping, RGB quantization and playback are not applied.",
    }
    source_files = {"pipeline": Path(inspect.getfile(StableVideoDiffusionPipeline)),
                    "temporal_vae": Path(inspect.getfile(AutoencoderKLTemporalDecoder)),
                    "resnet": Path(inspect.getfile(resnet))}
    source_identity = {key: {"path": str(path), "sha256": sha256_file(path)}
                       for key, path in source_files.items()}
    serialization_seconds = time.perf_counter() - save_started
    metrics = {
        "schema_version": 1, "started_utc": started_utc,
        "model_id": manifest["model_id"], "revision": manifest["revision"],
        "manifest_sha256": sha256_file(manifest_path), "verified_assets": verified,
        "input_png_sha256": INPUT_SHA256, "script_sha256": sha256_file(Path(__file__)),
        "pretrained_lock_sha256": sha256_file(Path("requirements-pretrained-lock.txt")),
        "source_identity": source_identity, "versions": versions,
        "model_loading_info": loading_info,
        "environment": {"python": platform.python_version(), "machine": platform.machine(),
                        "platform": platform.platform()},
        "config": {"frames": FRAMES, "chunk_size": args.chunk_size, "threads": 2,
                   "dtype": "float32", "device": "cpu", "deterministic_algorithms": True,
                   "scaling_factor": scaling_factor, "latent_construction": "Posterior mode of the fixed normalized reference, repeated identically 14 times, then multiplied by scaling_factor. No augmentation or random draws."},
        "decoder_calls": calls, "outputs": outputs, "diagnostics": diagnostics,
        "timing_seconds": {"asset_verification": verify_seconds,
                           "imports_and_component_load": load_seconds,
                           "encode_and_latent_construction": encode_seconds,
                           "decode_latents": decode_seconds,
                           "serialization_diagnostics_source_hashes": serialization_seconds,
                           "post_input_validation_to_metrics": time.perf_counter() - started},
        "memory": {"peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "scope": "Linux process lifetime through output serialization, diagnostics and metadata construction; queried before metrics JSON writing/printing/exit."},
        "scope": "One fixed-latent temporal decoder run. The protocol compares three independent processes using the same reference and encoded mode. The UNet, denoising, CLIP, random video generation and generated-video quality are outside this experiment.",
        "attribution": "Powered by Stability AI; examples/pretrained-svd/LICENSE-model.txt",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"chunk_size": args.chunk_size, "decode_seconds": decode_seconds,
                      "peak_rss_kib": metrics["memory"]["peak_rss_kib"],
                      "output_sha256": outputs["decoded_raw"]["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
