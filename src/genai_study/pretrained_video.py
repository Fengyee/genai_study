"""Optional, fixed-revision SVD image-to-video inference on CPU.

Use the separate pretrained dependency lock. --help and manifest inspection
only need the standard library. Powered by Stability AI; model terms are in
examples/pretrained-svd/LICENSE-model.txt.
"""
import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import resource
import sys
import time

from genai_study.pretrained_image import (
    bounded_int, checked_model_path, require_empty_output, sha256_file, verify_model,
)


MODEL_ID = "stabilityai/stable-video-diffusion-img2vid"
REVISION = "9cf024d5bfa8f56622af86c884f26a52f6676f2e"
DEFAULT_MANIFEST = Path("examples/pretrained-svd/manifest.json")
REQUIRED_FILES = frozenset({
    "model_index.json", "feature_extractor/preprocessor_config.json",
    "image_encoder/config.json", "image_encoder/model.safetensors",
    "scheduler/scheduler_config.json", "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors", "README.md", "LICENSE.md",
})


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict)
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION):
        raise ValueError("manifest must identify the fixed SVD model and revision (schema 1)")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files must be a nonempty list")
    paths = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("invalid manifest file entry")
        name = item.get("path")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", name)
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                or str(PurePosixPath(name)) != name or name in paths
                or name not in REQUIRED_FILES):
            raise ValueError(f"unexpected, duplicate or unsafe manifest path: {name!r}")
        if (type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError(f"invalid byte size or SHA256 for {name}")
        paths.add(name)
    if paths != REQUIRED_FILES:
        raise ValueError(f"manifest missing required files: {sorted(REQUIRED_FILES - paths)}")
    return manifest


def prepare_model(manifest: dict, model_dir: Path) -> dict:
    os.environ.setdefault("HF_HOME", str((model_dir / ".cache/huggingface").resolve()))
    from huggingface_hub import hf_hub_download

    started = time.perf_counter()
    model_dir.mkdir(parents=True, exist_ok=True)
    reused = 0
    for item in manifest["files"]:
        path = checked_model_path(model_dir, item["path"])
        if (path.is_file() and path.stat().st_size == item["size_bytes"]
                and sha256_file(path) == item["sha256"]):
            reused += 1
            continue
        hf_hub_download(repo_id=MODEL_ID, revision=REVISION, filename=item["path"],
                        local_dir=str(model_dir),
                        cache_dir=str(model_dir / ".cache/huggingface-hub"), token=False,
                        force_download=path.exists())
    verification_start = time.perf_counter()
    result = verify_model(manifest, model_dir)
    result.update(model_id=MODEL_ID, revision=REVISION, model_dir=str(model_dir),
                  reused_verified_files=reused, download_calls=len(manifest["files"]) - reused,
                  verification_seconds=time.perf_counter() - verification_start,
                  prepare_seconds=time.perf_counter() - started)
    return result


def video_dimension(value: str) -> int:
    number = bounded_int(128, 256)(value)
    if number % 8:
        raise argparse.ArgumentTypeError("video dimensions must be multiples of 8")
    return number


def finite_float(low: float, high: float | None = None):
    def parse(value: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number < low or (high is not None and number > high):
            raise argparse.ArgumentTypeError(f"value must be finite, >= {low}" +
                                             (f" and <= {high}" if high is not None else ""))
        return number
    return parse


def validate_run_args(args: argparse.Namespace) -> int:
    require_empty_output(args.output)
    if args.min_guidance_scale > args.max_guidance_scale or args.max_guidance_scale <= 1:
        raise ValueError("CFG requires 1 <= min-guidance-scale <= max-guidance-scale and max > 1")
    if not args.input.is_file():
        raise ValueError("input must be an existing image file")
    return (args.seed + 1) % (2**32) if args.augmentation_seed is None else args.augmentation_seed


def prepare_input_image(path: Path, height: int, width: int):
    """Apply a recorded EXIF orientation, integer center crop and Lanczos resize."""
    from PIL import Image, ImageOps

    source_hash = sha256_file(path)
    with Image.open(path) as original:
        original_size = list(original.size)
        original_mode = original.mode
        oriented = ImageOps.exif_transpose(original).convert("RGB")
    source_width, source_height = oriented.size
    if source_width * height > source_height * width:
        crop_width = max(1, source_height * width // height)
        left = (source_width - crop_width) // 2
        box = (left, 0, left + crop_width, source_height)
    else:
        crop_height = max(1, source_width * height // width)
        top = (source_height - crop_height) // 2
        box = (0, top, source_width, top + crop_height)
    prepared = oriented.crop(box).resize((width, height), Image.Resampling.LANCZOS)
    return prepared, {
        "source_path": str(path), "source_sha256": source_hash,
        "original_size_wh": original_size, "original_mode": original_mode,
        "exif_transposed_size_wh": [source_width, source_height],
        "crop_box_xyxy": list(box), "prepared_size_wh": [width, height],
        "transform": "PIL EXIF transpose -> RGB -> integer center crop -> Lanczos resize",
        "prepared_rgb_sha256": hashlib.sha256(prepared.tobytes()).hexdigest(),
        "prepared_rgb_hash_format": "HWC RGB uint8 raw bytes",
    }


def make_noise_inputs(seed: int, augmentation_seed: int, shape: tuple[int, ...]):
    """Keep video initialization and reference augmentation on separate streams."""
    import torch

    initial = torch.randn(shape, dtype=torch.float32, device="cpu",
                          generator=torch.Generator(device="cpu").manual_seed(seed))
    augmentation = torch.Generator(device="cpu").manual_seed(augmentation_seed)
    return initial, augmentation


def frame_diagnostics(frames) -> dict:
    """Descriptive changes in the postprocessed RGB output, with explicit units."""
    import numpy as np

    differences = np.diff(frames.astype(np.float64), axis=0)
    return {
        "pixel_range": [float(frames.min()), float(frames.max())],
        "adjacent_rgb_rms": np.sqrt(np.mean(differences**2, axis=(1, 2, 3))).tolist(),
        "first_last_rgb_rms": float(np.sqrt(np.mean((frames[-1].astype(np.float64)
                                                    - frames[0].astype(np.float64))**2))),
        "definition": "RMS over H,W,RGB of postprocessed [0,1] frame differences. These describe image change; stationary repetition has zero difference and motion correctness requires a separate task criterion.",
    }


def run_inference(args: argparse.Namespace, manifest: dict) -> dict:
    started = time.perf_counter()
    augmentation_seed = validate_run_args(args)
    input_started = time.perf_counter()
    prepared_image, input_info = prepare_input_image(args.input, args.height, args.width)
    input_seconds = time.perf_counter() - input_started
    verification_started = time.perf_counter()
    verification = verify_model(manifest, args.model_dir)
    verification_seconds = time.perf_counter() - verification_started
    load_started = time.perf_counter()
    os.environ.setdefault("HF_HOME", str((args.model_dir / ".cache/huggingface").resolve()))
    import numpy as np
    import torch
    from diffusers import StableVideoDiffusionPipeline
    from PIL import Image

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        str(args.model_dir), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).to("cpu")
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    load_seconds = time.perf_counter() - load_started
    if pipe.unet.config.in_channels != 8 or pipe.vae_scale_factor != 8:
        raise ValueError("the fixed SVD protocol expects 8 UNet channels and VAE spatial factor 8")

    initial, generator = make_noise_inputs(
        args.seed, augmentation_seed,
        (1, args.frames, 4, args.height // 8, args.width // 8),
    )
    augmentation_state_hash = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    call_shapes = []
    extra_conditions = {}
    augmented_inputs = []
    timestamps = {}

    def count_unet(module, positional, keyword):
        sample = positional[0] if positional else keyword["sample"]
        call_shapes.append(list(sample.shape))
        if len(call_shapes) == 1:
            timestamps["denoising_start"] = time.perf_counter()
            extra_conditions["added_time_ids"] = keyword["added_time_ids"].detach().cpu().tolist()
            extra_conditions["image_embedding_shape"] = list(keyword["encoder_hidden_states"].shape)
        step = len(call_shapes)
        if step == 1 or step % max(1, args.steps // 5) == 0 or step == args.steps:
            print(f"SVD Euler step {step}/{args.steps} started", file=sys.stderr, flush=True)

    def capture_augmented_input(module, positional):
        augmented_inputs.append(positional[0].detach().cpu().clone())

    def step_finished(pipeline, index, timestep, callback_kwargs):
        if index == args.steps - 1:
            timestamps["denoising_end"] = time.perf_counter()
        return callback_kwargs

    unet_hook = pipe.unet.register_forward_pre_hook(count_unet, with_kwargs=True)
    encoder_hook = pipe.vae.encoder.register_forward_pre_hook(capture_augmented_input)
    inference_started = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = pipe(
                image=prepared_image, height=args.height, width=args.width,
                num_frames=args.frames, num_inference_steps=args.steps,
                min_guidance_scale=args.min_guidance_scale, max_guidance_scale=args.max_guidance_scale,
                fps=args.fps, motion_bucket_id=args.motion_bucket_id,
                noise_aug_strength=args.noise_aug_strength, decode_chunk_size=args.frames,
                num_videos_per_prompt=1, generator=generator, latents=initial.clone(),
                output_type="np", return_dict=True, callback_on_step_end=step_finished,
            )
    finally:
        unet_hook.remove()
        encoder_hook.remove()
    inference_ended = time.perf_counter()
    frames = np.asarray(generated.frames[0], dtype=np.float32)
    expected_shape = (args.frames, args.height, args.width, 3)
    if frames.shape != expected_shape or not np.isfinite(frames).all():
        raise ValueError(f"unexpected or nonfinite video output: {frames.shape}, expected {expected_shape}")
    if frames.min() < 0 or frames.max() > 1:
        raise ValueError("pipeline postprocessed RGB frames must lie in [0,1]")
    expected_call = [2, args.frames, 8, args.height // 8, args.width // 8]
    if call_shapes != [expected_call] * args.steps or len(augmented_inputs) != 1:
        raise ValueError("actual SVD calls differ from the fixed joint-video/augmentation protocol")
    augmented = augmented_inputs[0].numpy()
    guidance = pipe.guidance_scale.detach().cpu().reshape(-1).tolist()

    require_empty_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    input_path = args.output / "input.png"
    prepared_image.save(input_path)
    np.save(args.output / "initial_noise.npy", initial.numpy(), allow_pickle=False)
    np.save(args.output / "augmentation_input.npy", augmented, allow_pickle=False)
    np.save(args.output / "frames.npy", frames, allow_pickle=False)
    frame_files = []
    for index, frame in enumerate(frames):
        filename = f"frame_{index:03d}.png"
        path = args.output / filename
        Image.fromarray((frame * 255).round().astype(np.uint8)).save(path)
        frame_files.append({"path": filename, "sha256": sha256_file(path)})
    versions = {name: metadata.version(name) for name in (
        "torch", "numpy", "diffusers", "transformers", "accelerate",
        "safetensors", "huggingface-hub", "Pillow",
    )}
    versions["python"] = platform.python_version()
    result = {
        "schema_version": 1,
        "model": {"model_id": MODEL_ID, "revision": REVISION,
                  "manifest_path": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
                  **verification},
        "config": {"seed": args.seed, "augmentation_seed": augmentation_seed,
                   "frames": args.frames, "steps": args.steps, "height": args.height, "width": args.width,
                   "min_guidance_scale": args.min_guidance_scale,
                   "max_guidance_scale": args.max_guidance_scale, "fps_input": args.fps,
                   "fps_condition": args.fps - 1, "motion_bucket_id": args.motion_bucket_id,
                   "noise_aug_strength": args.noise_aug_strength, "decode_chunk_size": args.frames,
                   "threads": args.threads, "device": "cpu", "dtype": "float32",
                   "num_videos_per_prompt": 1, "deterministic_algorithms": True},
        "input": {**input_info, "prepared_png_sha256": sha256_file(input_path)},
        "versions": versions,
        "scheduler": {"class": type(pipe.scheduler).__name__, "config": dict(pipe.scheduler.config),
                      "timesteps": pipe.scheduler.timesteps.tolist(), "sigmas": pipe.scheduler.sigmas.tolist(),
                      "init_noise_sigma": float(pipe.scheduler.init_noise_sigma)},
        "representation": {"initial_noise_shape": list(initial.shape),
                           "initial_noise_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                           "initial_noise_hash_format": "C-order float32 raw bytes before pipeline init_noise_sigma multiplication",
                           "augmentation_generator_initial_state_sha256": augmentation_state_hash,
                           "augmentation_input_shape": list(augmented.shape),
                           "augmentation_input_sha256": hashlib.sha256(augmented.tobytes()).hexdigest(),
                           "augmentation_input_hash_format": "C-order float32 raw bytes at the VAE encoder input",
                           "vae_spatial_factor": pipe.vae_scale_factor,
                           "vae_scaling_factor": float(pipe.vae.config.scaling_factor),
                           "reference_vae_latent_scaling": "latent_dist.mode(), without multiplying scaling_factor",
                           "generated_latent_decoding": "divide by scaling_factor before temporal VAE decode",
                           "guidance_per_frame": guidance, **extra_conditions},
        "network_evaluations": {"unet_forward_calls": len(call_shapes), "unet_input_shapes": call_shapes,
                                "unet_video_branch_evaluations": sum(shape[0] for shape in call_shapes),
                                "definition": "One joint-video UNet call per Euler interval; batch2 contains image-conditioned and zero-image branches. Each branch processes all F frames jointly. Excludes image encoder and VAE."},
        "timing_seconds": {"input_preparation": input_seconds, "verification": verification_seconds,
                           "pipeline_load": load_seconds, "inference": inference_ended - inference_started,
                           "conditioning_before_first_unet": timestamps["denoising_start"] - inference_started,
                           "denoising_first_unet_through_last_euler": timestamps["denoising_end"] - timestamps["denoising_start"],
                           "decoding_and_postprocessing": inference_ended - timestamps["denoising_end"],
                           "total_before_metrics_write": time.perf_counter() - started},
        "timing_scope": "Fresh process, no warm-up. Inference includes image conditioning, all Euler steps, temporal VAE decode and RGB postprocessing. Denoising begins at the first UNet prehook and ends in the callback after the last Euler update. Total also includes verification, loading, input preparation and saving, stopping before metrics write. Download is separate.",
        "memory": {"process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                   "scope": "Linux process lifetime high-water RSS in KiB, including imports/model loading."},
        "diagnostics": frame_diagnostics(frames),
        "outputs": {"input": "input.png", "initial_noise": "initial_noise.npy",
                    "augmentation_input": "augmentation_input.npy", "frames": "frames.npy",
                    "frames_shape": list(frames.shape), "frames_sha256": sha256_file(args.output / "frames.npy"),
                    "frames_raw_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
                    "frames_format": "F,H,W,RGB float32 after pipeline (decoded/2+0.5).clamp(0,1)",
                    "png_conversion": "round(frames*255) to uint8, matching Diffusers numpy_to_pil",
                    "frame_files": frame_files},
        "attribution": "Powered by Stability AI; fixed model uses Stability AI Community License (2024-07-05).",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        child = commands.add_parser(command)
        child.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        child.add_argument("--model-dir", type=Path, required=True)
        if command == "run":
            child.add_argument("--input", type=Path, required=True)
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--seed", type=bounded_int(0, 2**32 - 1), default=7)
            child.add_argument("--augmentation-seed", type=bounded_int(0, 2**32 - 1))
            child.add_argument("--frames", type=bounded_int(4, 14), default=14)
            child.add_argument("--steps", type=bounded_int(1, 50), default=25)
            child.add_argument("--height", type=video_dimension, default=144)
            child.add_argument("--width", type=video_dimension, default=256)
            child.add_argument("--min-guidance-scale", type=finite_float(1), default=1.0)
            child.add_argument("--max-guidance-scale", type=finite_float(1), default=2.5)
            child.add_argument("--fps", type=bounded_int(1, 30), default=7)
            child.add_argument("--motion-bucket-id", type=bounded_int(0, 255), default=127)
            child.add_argument("--noise-aug-strength", type=finite_float(0, 1), default=0.02)
            child.add_argument("--threads", type=bounded_int(1, 4), default=1)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    try:
        if args.command == "run":
            validate_run_args(args)
        manifest = load_manifest(args.manifest)
        result = (prepare_model(manifest, args.model_dir) if args.command == "prepare"
                  else run_inference(args, manifest))
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
