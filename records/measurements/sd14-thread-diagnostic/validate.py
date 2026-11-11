from pathlib import Path
import hashlib, json, math
import numpy as np
from PIL import Image

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def validate(req):
    p=json.loads(Path('records/measurements/sd14-thread-diagnostic/plan.json').read_text())
    base=json.loads(Path('records/measurements/sd14-attributes-pilot/plan.json').read_text())
    for name,v in p['source_identities'].items():assert sha(name)==v['sha256']
    d=Path(req['output']);rec=Path(req['record_output'])
    r=json.loads((rec/'run.json').read_text());m=json.loads((d/'metrics.json').read_text())
    assert r['status']=='succeeded' and r['exit_code']==r['child_returncode']==0 and r['timed_out'] is False
    assert r['argv']==req['argv'][req['argv'].index('--')+1:]
    assert sha(rec/'command.log')==r['log_sha256'] and (rec/'command.log').stat().st_size==r['log_size_bytes']
    assert r['runner']['source_sha256']==p['source_identities']['src/genai_study/run_record.py']['sha256']
    conf={k:p['fixed'][k] for k in ['steps','guidance_scale','height','width','device','dtype','eta','negative_prompt','batch_size','seed','prompt']}
    conf.update(threads=req['threads'],deterministic_algorithms=True); assert m['config']==conf
    assert m['versions']==dict(base['dependency_preflight']['versions'],python=base['dependency_preflight']['python'])
    model=m['model'];assert model['model_id']==p['model_id'] and model['revision']==p['revision']
    assert model['manifest_sha256']==p['source_identities']['examples/pretrained-sd14/manifest.json']['sha256']
    assert model['verified_files']==16 and model['verified_bytes']==5482669095
    sch=m['scheduler']; expected=json.loads(json.dumps(base['expected_scheduler']))
    for conf in [sch['config'],expected['config']]:conf['_use_default_values']=sorted(conf['_use_default_values'])
    assert sch['config']==expected['config'] and sch['timesteps']==[751,501,251,1] and sch['class']=='DDIMScheduler' and sch['init_noise_sigma']==1
    net=m['network_evaluations'];assert net['unet_forward_calls']==4 and net['unet_batch_sizes']==[2]*4 and net['unet_sample_branch_evaluations']==8
    rep=m['representation'];assert rep['prompt_tokens_with_special_tokens']==13 and rep['tokenizer_max_length']==77 and rep['prompt_truncated'] is False
    a=np.load(d/'initial_noise.npy',allow_pickle=False);assert a.shape==(1,4,64,64) and a.dtype==np.float32 and np.isfinite(a).all()
    assert hashlib.sha256(a.tobytes()).hexdigest()==p['expected_noise']['sha256']==rep['initial_noise_sha256']
    with Image.open(d/'image.png') as im:assert im.size==(512,512) and im.mode=='RGB'; im.load()
    assert sha(d/'image.png')==m['outputs']['image_sha256']
    assert m['safety']['checker_enabled'] is True and type(m['safety']['nsfw_content_detected']) is bool
    wall=r['wall_seconds']; inf=m['timing_seconds']['inference'];rss=m['memory']['process_peak_rss_kib']
    assert all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in [wall,inf,rss])
    return {'threads':req['threads'],'validation':'passed','wall_seconds':wall,'inference_seconds':inf,'process_peak_rss_kib':rss,'conservative_20_step_estimate_seconds':wall-inf+5*inf,'image_sha256':sha(d/'image.png'),'initial_noise_sha256':rep['initial_noise_sha256'],'filtered':m['safety']['nsfw_content_detected']}
