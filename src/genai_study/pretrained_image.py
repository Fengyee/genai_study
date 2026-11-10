"""Optional, versioned Stable Diffusion 1.4 inference on CPU.

Install the separate pretrained dependency lock before prepare/run. Importing
this module and using --help only require Python's standard library.
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


MODEL_ID = "CompVis/stable-diffusion-v1-4"
REVISION = "133a221b8aa7292a167afc5127cb63fb5005638b"
DEFAULT_MANIFEST = Path("examples/pretrained-sd14/manifest.json")
DEFAULT_PROMPT = "a watercolor painting of a small cabin beside a lake, pine trees, daytime"
REQUIRED_FILES = frozenset({
    "model_index.json", "feature_extractor/preprocessor_config.json",
    "safety_checker/config.json", "safety_checker/model.safetensors",
    "scheduler/scheduler_config.json", "text_encoder/config.json",
    "text_encoder/model.safetensors", "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json", "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json", "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors", "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
})


def sha256_file(path: Path) -> str:
    """Hash large weights without making a second in-memory copy."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("model_id") != MODEL_ID or manifest.get("revision") != REVISION):
        raise ValueError("manifest must identify the fixed SD 1.4 model and revision (schema 1)")
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
                or name not in REQUIRED_FILES | {"README.md"}):
            raise ValueError(f"unexpected, duplicate or unsafe manifest path: {name!r}")
        if (type(item.get("size_bytes")) is not int or item["size_bytes"] <= 0
                or not isinstance(item.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError(f"invalid byte size or SHA256 for {name}")
        paths.add(name)
    if not REQUIRED_FILES <= paths:
        raise ValueError(f"manifest missing required files: {sorted(REQUIRED_FILES - paths)}")
    return manifest


def checked_model_path(model_dir: Path, relative: str) -> Path:
    root = model_dir.resolve()
    candidate = model_dir / relative
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"model file escapes model directory: {relative}")
    return candidate


def verify_model(manifest: dict, model_dir: Path) -> dict:
    """Verify every selected component, including tokenizer and scheduler configs."""
    total = 0
    for item in manifest["files"]:
        path = checked_model_path(model_dir, item["path"])
        if not path.is_file():
            raise ValueError(f"missing model file: {item['path']}; run prepare first")
        if path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"byte size mismatch: {item['path']}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {item['path']}")
        total += item["size_bytes"]
    return {"verified_files": len(manifest["files"]), "verified_bytes": total}


def prepare_model(manifest: dict, model_dir: Path) -> dict:
    # local_dir keeps downloads and Hugging Face download metadata together.
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


def require_empty_output(output: Path) -> None:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory is nonempty or is a file; choose a new directory")


def bounded_int(low: int, high: int):
    def parse(value: str) -> int:
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"value must be between {low} and {high}")
        return number
    return parse


def image_dimension(value: str) -> int:
    number = bounded_int(128, 512)(value)
    if number % 64:
        raise argparse.ArgumentTypeError("image dimensions must be multiples of 64")
    return number


def cfg_scale(value: str) -> float:
    scale = float(value)
    if not math.isfinite(scale) or scale <= 1:
        raise argparse.ArgumentTypeError("this CFG experiment requires a finite scale > 1")
    return scale


def run_inference(args: argparse.Namespace, manifest: dict) -> dict:
    started = time.perf_counter()
    require_empty_output(args.output)
    verification = verify_model(manifest, args.model_dir)
    verification_seconds = time.perf_counter() - started
    load_start = time.perf_counter()
    os.environ.setdefault("HF_HOME", str((args.model_dir / ".cache/huggingface").resolve()))
    import numpy as np
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    pipe = StableDiffusionPipeline.from_pretrained(
        str(args.model_dir), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True, requires_safety_checker=True,
    ).to("cpu")
    if pipe.safety_checker is None or pipe.feature_extractor is None:
        raise ValueError("the fixed pipeline requires its safety checker and feature extractor")
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component.eval().requires_grad_(False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    load_seconds = time.perf_counter() - load_start

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    initial = torch.randn(1, pipe.unet.config.in_channels,
                          args.height // pipe.vae_scale_factor,
                          args.width // pipe.vae_scale_factor,
                          generator=generator, dtype=torch.float32)
    token_count = len(pipe.tokenizer(args.prompt, truncation=False)["input_ids"])
    maximum_tokens = int(pipe.tokenizer.model_max_length)
    batch_sizes = []

    def count_unet(module, positional, keyword):
        sample = positional[0] if positional else keyword["sample"]
        batch_sizes.append(int(sample.shape[0]))
        step = len(batch_sizes)
        if step == 1 or step % max(1, args.steps // 5) == 0 or step == args.steps:
            print(f"DDIM step {step}/{args.steps} started", file=sys.stderr, flush=True)

    hook = pipe.unet.register_forward_pre_hook(count_unet, with_kwargs=True)
    inference_start = time.perf_counter()
    try:
        with torch.inference_mode():
            generated = pipe(
                prompt=args.prompt, negative_prompt="", height=args.height, width=args.width,
                num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                eta=0.0, generator=generator, latents=initial.clone(),
                num_images_per_prompt=1, output_type="pil", return_dict=True,
            )
    finally:
        hook.remove()
    inference_seconds = time.perf_counter() - inference_start
    if len(batch_sizes) != args.steps or any(batch != 2 for batch in batch_sizes):
        raise RuntimeError("observed UNet calls differ from the declared one-image CFG protocol")
    if generated.nsfw_content_detected is None or len(generated.nsfw_content_detected) != 1:
        raise RuntimeError("the pipeline did not return the required image safety result")

    # Model loading and inference may take minutes; recheck before saving.
    require_empty_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    # Only the pipeline's checked, postprocessed output is saved.
    image_path = args.output / "image.png"
    generated.images[0].save(image_path)
    np.save(args.output / "initial_noise.npy", initial.numpy(), allow_pickle=False)
    versions = {name: metadata.version(name) for name in (
        "torch", "numpy", "diffusers", "transformers", "accelerate", "safetensors",
        "huggingface-hub", "tokenizers", "Pillow",
    )}
    versions["python"] = platform.python_version()
    result = {
        "schema_version": 1,
        "model": {"model_id": MODEL_ID, "revision": REVISION,
                  "manifest_path": str(args.manifest),
                  "manifest_sha256": sha256_file(args.manifest), **verification},
        "config": {"prompt": args.prompt, "negative_prompt": "", "seed": args.seed,
                   "steps": args.steps, "guidance_scale": args.guidance_scale,
                   "height": args.height, "width": args.width, "threads": args.threads,
                   "device": "cpu", "dtype": "float32", "eta": 0.0,
                   "batch_size": 1, "deterministic_algorithms": True},
        "versions": versions,
        "scheduler": {"class": type(pipe.scheduler).__name__,
                      "config": dict(pipe.scheduler.config),
                      "timesteps": pipe.scheduler.timesteps.tolist(),
                      "init_noise_sigma": float(pipe.scheduler.init_noise_sigma)},
        "representation": {"initial_noise_shape": list(initial.shape),
                           "initial_noise_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                           "initial_noise_hash_format": "C-order float32 raw bytes, before init_noise_sigma multiplication",
                           "vae_spatial_factor": pipe.vae_scale_factor,
                           "vae_scaling_factor": float(pipe.vae.config.scaling_factor),
                           "prompt_tokens_with_special_tokens": token_count,
                           "tokenizer_max_length": maximum_tokens,
                           "prompt_truncated": token_count > maximum_tokens},
        "network_evaluations": {
            "unet_forward_calls": len(batch_sizes), "unet_batch_sizes": batch_sizes,
            "unet_sample_branch_evaluations": sum(batch_sizes),
            "definition": "One UNet forward call per DDIM timestep, batch 2 concatenates empty-text and prompt branches for one output image; excludes text encoder, VAE and safety checker.",
        },
        "timing_seconds": {"verification": verification_seconds, "pipeline_load": load_seconds,
                           "inference": inference_seconds, "total_before_metrics_write": time.perf_counter() - started},
        "timing_scope": "Verification hashes all selected model files; pipeline_load includes optional imports and local model loading; inference includes text encoding, denoising, VAE decoding, safety checking and PIL postprocessing. No warm-up. Download is a separate prepare command.",
        "memory": {"process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                   "scope": "Linux process lifetime high-water RSS, including imports/model loading; measured in KiB, not incremental inference allocation."},
        "safety": {"checker_enabled": True,
                   "nsfw_content_detected": bool(generated.nsfw_content_detected[0]),
                   "saved_output": "pipeline safety-checked PIL output"},
        "outputs": {"image": "image.png", "image_sha256": sha256_file(image_path),
                    "initial_noise": "initial_noise.npy"},
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
            child.add_argument("--output", type=Path, required=True)
            child.add_argument("--prompt", default=DEFAULT_PROMPT)
            child.add_argument("--seed", type=bounded_int(0, 2**32 - 1), default=7)
            child.add_argument("--steps", type=bounded_int(1, 50), default=20)
            child.add_argument("--guidance-scale", type=cfg_scale, default=7.5)
            child.add_argument("--height", type=image_dimension, default=512)
            child.add_argument("--width", type=image_dimension, default=512)
            child.add_argument("--threads", type=bounded_int(1, 4), default=1)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    try:
        if args.command == "run":
            require_empty_output(args.output)
            if not args.prompt.strip():
                raise ValueError("prompt must contain text")
        manifest = load_manifest(args.manifest)
        result = (prepare_model(manifest, args.model_dir) if args.command == "prepare"
                  else run_inference(args, manifest))
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
