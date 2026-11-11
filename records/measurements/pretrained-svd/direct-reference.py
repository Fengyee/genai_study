from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/workspace/genai_study')
OUT = ROOT / 'artifacts/svd-current/direct-reference'
SMOKE = ROOT / 'artifacts/svd-current/smoke'
MODEL = ROOT / 'artifacts/models/svd'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def worker():
    import platform
    import resource
    from importlib import metadata
    started = time.perf_counter()
    manifest_path = ROOT / 'examples/pretrained-svd/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['model_id'] == 'stabilityai/stable-video-diffusion-img2vid'
    assert manifest['revision'] == '9cf024d5bfa8f56622af86c884f26a52f6676f2e'
    for item in manifest['files']:
        path = MODEL / item['path']
        assert path.stat().st_size == item['size_bytes']
        assert digest(path) == item['sha256']
    verification_seconds = time.perf_counter() - started
    load_start = time.perf_counter()
    import numpy as np
    import torch
    import diffusers
    from diffusers import StableVideoDiffusionPipeline
    from PIL import Image
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        str(MODEL), local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
    ).to('cpu')
    for component in pipe.components.values():
        if isinstance(component, torch.nn.Module):
            component.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    load_seconds = time.perf_counter() - load_start
    with Image.open(SMOKE / 'input.png') as opened:
        image = opened.convert('RGB')
    assert image.size == (256, 144)
    initial = torch.randn((1, 4, 4, 18, 32), dtype=torch.float32,
                          generator=torch.Generator(device='cpu').manual_seed(7))
    augmentation_generator = torch.Generator(device='cpu').manual_seed(8)
    inference_start = time.perf_counter()
    print('DIRECT_REFERENCE_INFERENCE_STARTED', flush=True)
    with torch.no_grad():
        result = pipe(
            image=image, height=144, width=256, num_frames=4,
            num_inference_steps=2, min_guidance_scale=1.0, max_guidance_scale=2.5,
            fps=7, motion_bucket_id=127, noise_aug_strength=0.02,
            decode_chunk_size=4, num_videos_per_prompt=1,
            generator=augmentation_generator, latents=initial.clone(),
            output_type='np', return_dict=True,
        )
    inference_seconds = time.perf_counter() - inference_start
    frames = np.asarray(result.frames[0])
    assert frames.dtype == np.float32 and frames.shape == (4,144,256,3)
    assert np.isfinite(frames).all()
    np.save(OUT / 'frames.npy', frames, allow_pickle=False)
    np.save(OUT / 'initial_noise.npy', initial.numpy(), allow_pickle=False)
    reference = np.load(SMOKE / 'frames.npy', allow_pickle=False)
    reference_noise = np.load(SMOKE / 'initial_noise.npy', allow_pickle=False)
    pixels = pipe.video_processor.numpy_to_pil(frames)
    png_comparisons = []
    for index, frame in enumerate(pixels):
        filename = f'frame_{index:03d}.png'
        path = OUT / filename
        frame.save(path)
        original = SMOKE / filename
        with Image.open(original) as observed:
            same_pixels = np.array_equal(np.asarray(frame), np.asarray(observed))
        png_comparisons.append({'frame': index, 'png_sha256': digest(path),
                                'smoke_png_sha256': digest(original),
                                'byte_identical': path.read_bytes() == original.read_bytes(),
                                'quantized_rgb_equal': bool(same_pixels),
                                'float_frame_equal': bool(np.array_equal(frames[index], reference[index])),
                                'float_max_abs_difference': float(np.max(np.abs(frames[index].astype(np.float64)-reference[index])))})
    package_root = Path(diffusers.__file__).parent
    sources = ['pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py',
               'schedulers/scheduling_euler_discrete.py',
               'models/unets/unet_spatio_temporal_condition.py',
               'models/autoencoders/autoencoder_kl_temporal_decoder.py',
               'video_processor.py', 'image_processor.py']
    report = {
        'scope': 'Independent direct call to the installed official StableVideoDiffusionPipeline; no repository inference/noise/preprocessing helpers and no hooks. This compares the four-frame/two-step smoke wrapper with its direct API equivalent; it does not establish a cause for full-run visual failures.',
        'model': {'model_id':manifest['model_id'],'revision':manifest['revision'],
                  'manifest_sha256':digest(manifest_path),'verified_files':len(manifest['files']),
                  'verified_bytes':sum(x['size_bytes'] for x in manifest['files'])},
        'identity': {'script_sha256':digest(__file__),
                     'requirements_lock_sha256':digest(ROOT/'requirements-pretrained-lock.txt'),
                     'prepared_input_png_sha256':digest(SMOKE/'input.png'),
                     'prepared_input_rgb_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                     'diffusers_source_sha256':{name:digest(package_root/name) for name in sources}},
        'versions': {name:metadata.version(name) for name in ['torch','numpy','diffusers','transformers','accelerate','safetensors','huggingface-hub','Pillow']},
        'python':platform.python_version(),
        'config': {'height':144,'width':256,'frames':4,'steps':2,'min_guidance_scale':1.0,
                   'max_guidance_scale':2.5,'fps_input':7,'motion_bucket_id':127,'noise_aug_strength':0.02,
                   'decode_chunk_size':4,'initial_seed':7,'augmentation_seed':8,'threads':1,
                   'device':'cpu','dtype':'float32','grad_context':'torch.no_grad','module_mode':'eval, requires_grad False'},
        'scheduler': {'class':type(pipe.scheduler).__name__,'config':dict(pipe.scheduler.config),
                      'sigmas':pipe.scheduler.sigmas.tolist(),'timesteps':pipe.scheduler.timesteps.tolist(),
                      'init_noise_sigma':float(pipe.scheduler.init_noise_sigma)},
        'comparisons': {'all_float_frames_equal':bool(np.array_equal(frames, reference)),
                        'different_float_elements':int(np.count_nonzero(frames != reference)),
                        'max_abs_float_difference':float(np.max(np.abs(frames.astype(np.float64)-reference))),
                        'initial_noise_equal':bool(np.array_equal(initial.numpy(), reference_noise)),
                        'all_png_bytes_equal':all(x['byte_identical'] for x in png_comparisons),
                        'all_quantized_pixels_equal':all(x['quantized_rgb_equal'] for x in png_comparisons),
                        'per_frame':png_comparisons},
        'outputs': {'frames_npy_sha256':digest(OUT/'frames.npy'),
                    'frames_raw_sha256':hashlib.sha256(frames.tobytes()).hexdigest(),
                    'initial_noise_raw_sha256':hashlib.sha256(initial.numpy().tobytes()).hexdigest()},
        'timing_seconds': {'verification':verification_seconds,'pipeline_load':load_seconds,
                           'inference':inference_seconds,'total_before_report_write':time.perf_counter()-started},
        'timing_scope': 'Independent fresh process; inference includes official image conditioning, denoising, decoding and output conversion. Total includes full weight hashing, loading, saving and comparison. Shared filesystem cache/host load uncontrolled.',
        'memory': {'process_peak_rss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                   'scope':'Linux process-lifetime peak RSS, including model loading; KiB.'},
    }
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'float_equal':report['comparisons']['all_float_frames_equal'],
                      'png_equal':report['comparisons']['all_png_bytes_equal'],
                      'inference_seconds':inference_seconds,
                      'peak_rss_kib':report['memory']['process_peak_rss_kib']}),flush=True)
    if not all((report['comparisons']['all_float_frames_equal'],
                report['comparisons']['all_png_bytes_equal'],
                report['comparisons']['initial_noise_equal'])):
        raise SystemExit(1)


def driver():
    repeat_path = ROOT/'artifacts/svd-current/smoke-repeat-execution.json'
    assert repeat_path.exists(), 'Wait for the prior repeat process before starting another model.'
    repeat = json.loads(repeat_path.read_text())
    assert repeat['exit_code'] == 0
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Direct reference directory already contains work; preserve it.')
    OUT.mkdir(parents=True,exist_ok=True)
    command=[str(ROOT/'.venv-pretrained/bin/python'),str(Path(__file__).resolve()),'--worker']
    start=time.perf_counter()
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (OUT/'run.log').open('w') as log:
        try:
            result=subprocess.run(command,cwd=ROOT,env={**os.environ,'HF_HOME':str(ROOT/'.cache/huggingface')},
                                  stdout=log,stderr=subprocess.STDOUT,timeout=600)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
    execution={'started_utc':utc,'command':command,'timeout_seconds':600,'exit_code':code,
               'wall_seconds':time.perf_counter()-start,'log':'artifacts/svd-current/direct-reference/run.log',
               'prior_repeat_exit_code':repeat['exit_code']}
    (OUT/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    print(json.dumps(execution),flush=True)
    raise SystemExit(code)

if __name__=='__main__':
    worker() if sys.argv[-1:] == ['--worker'] else driver()
