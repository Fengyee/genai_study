"""Audit archived SVD chunk arrays/identity/costs without constructing models."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root, output = args.root.resolve(), args.output.absolute()
if output.exists() or output.is_symlink():
    parser.error('output must be new')
if not output.parent.is_dir():
    parser.error('output parent must exist')
commit = '1d97b06613872096c6ab86fc9030be8059a1233a'
plan_path = Path('records/measurements/svd-decoder-chunks/plan.json')

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8*1024*1024):
            h.update(b)
    return h.hexdigest()

def rawsha(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()

def git_bytes(path):
    return subprocess.run(['git','show',f'{commit}:{path}'], cwd=root,
                          check=True, capture_output=True, timeout=10).stdout

def utc(s):
    return datetime.fromisoformat(s.replace('Z','+00:00'))

def comparisons(a, b):
    # Explicit per-frame squared norms; independent of producer diagnostics.
    delta = np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)
    frames = np.moveaxis(delta, 2, 0).reshape(delta.shape[2], -1)
    sums = [float(np.dot(row, row)) for row in frames]
    return {'global_rms':math.sqrt(math.fsum(sums)/delta.size),
            'global_max_abs':float(np.max(np.abs(delta))),
            'per_frame_rms':[math.sqrt(x/frames.shape[1]) for x in sums],
            'per_frame_max_abs':[float(np.max(np.abs(row))) for row in frames],
            'different_float_elements':int(np.count_nonzero(delta)),
            'exact_equal':bool(np.array_equal(a,b))}

plan = read(root/plan_path)
assert (root/plan_path).read_bytes() == git_bytes(plan_path)
source_hashes = {plan['script']['path']:plan['script']['sha256'], **plan['frozen_sources']}
for path, expected in source_hashes.items():
    assert sha(root/path) == expected, path
    assert hashlib.sha256(git_bytes(path)).hexdigest() == expected, path
assert plan['script']['sha256'] == '8ae054b3883d1cb5092f08215589ff378faa2c0d7703942a60532e53ff041e0d'
execution = read(root/'artifacts/svd-decoder-chunks/execution.json')
assert execution['preregistration_head'] == commit and execution['plan_sha256'] == sha(root/plan_path)
assert execution['status']=='succeeded' and all(s['status']=='succeeded' for s in execution['slots'])
assert [s['id'] for s in execution['slots']]==[s['id'] for s in plan['commands']]
source_identity = read(root/'records/measurements/svd-decoder-chunks/source-identity.json')
expected_library_hashes = {e['sha256'] for e in source_identity['files']}
manifest = read(root/'examples/pretrained-svd/manifest.json')
assets = [e for e in manifest['files'] if e['path'] in {'vae/config.json','vae/diffusion_pytorch_model.safetensors','LICENSE.md'}]
for e in assets:
    path = root/'artifacts/models/svd'/e['path']
    assert path.stat().st_size == e['size_bytes'] and sha(path)==e['sha256'], e['path']
with Image.open(root/plan['latent_protocol']['reference']) as im:
    assert im.mode=='RGB' and im.size==(256,144)
    normalized = np.asarray(im).transpose(2,0,1)[None].astype(np.float32)/np.float32(127.5)-np.float32(1)
expected_shapes = {'normalized_input':[1,3,144,256], 'latent_mode':[1,4,18,32],
                   'video_latents':[1,14,4,18,32], 'decoded_raw':[1,3,14,144,256]}
rows, arrays, metadata = [], {}, {}
last_end = None
for slot in plan['commands']:
    name, chunk = slot['id'], slot['chunk_size']
    directory, record_dir = root/slot['output'], root/slot['record_output']
    m, r = read(directory/'metrics.json'), read(record_dir/'run.json')
    assert r['status']=='succeeded' and r['exit_code']==r['child_returncode']==0 and not r['timed_out']
    assert r['argv']==slot['argv'] and Path(r['cwd']).resolve()==root
    assert r['git']['head']==commit and r['runner']['source_sha256']==source_hashes['src/genai_study/run_record.py']
    assert r['timeout_seconds']==plan['budget']['per_command_timeout_seconds']
    log = record_dir/r['log_file']
    assert log.stat().st_size==r['log_size_bytes'] and sha(log)==r['log_sha256']
    begin, end = utc(r['started_utc']), utc(r['finished_utc'])
    assert utc(plan['created_utc']) < begin and begin <= utc(m['started_utc']) <= end
    assert last_end is None or last_end <= begin
    last_end = end
    assert m['model_id']==manifest['model_id'] and m['revision']==manifest['revision']
    assert m['manifest_sha256']==sha(root/'examples/pretrained-svd/manifest.json')
    assert m['script_sha256']==plan['script']['sha256'] and m['verified_assets']==assets
    assert m['input_png_sha256']==source_hashes[plan['latent_protocol']['reference']]
    assert m['pretrained_lock_sha256']==source_hashes['requirements-pretrained-lock.txt']
    assert all(not value for value in m['model_loading_info'].values())
    for e in m['source_identity'].values():
        assert e['sha256'] in expected_library_hashes and sha(e['path'])==e['sha256']
    config = m['config']
    assert (config['frames'],config['chunk_size'],config['threads'],config['dtype'],config['device'],config['deterministic_algorithms'],config['scaling_factor'])==(14,chunk,2,'float32','cpu',True,0.18215)
    assert m['decoder_calls']==[{'shape':[chunk,4,18,32],'num_frames':chunk}]*(14//chunk)
    a = {}
    for key, info in m['outputs'].items():
        p = directory/info['file']; value = np.load(p,allow_pickle=False)
        assert value.dtype==np.dtype('float32') and list(value.shape)==expected_shapes[key]==info['shape']
        assert np.isfinite(value).all() and str(value.dtype)==info['dtype']
        assert sha(p)==info['sha256'] and rawsha(value)==info['array_bytes_sha256']
        a[key]=value
    assert np.array_equal(a['normalized_input'],normalized)
    expected_latents = np.repeat(a['latent_mode'][:,None],14,axis=1)*np.float32(0.18215)
    assert np.array_equal(a['video_latents'],expected_latents)
    assert np.array_equal(a['video_latents'],np.repeat(a['video_latents'][:,:1],14,axis=1))
    raw = a['decoded_raw']
    adj = comparisons(raw[:,:,1:],raw[:,:,:-1])
    diagnostic = m['diagnostics']
    assert float(raw.min())==diagnostic['raw_min'] and float(raw.max())==diagnostic['raw_max']
    assert float(np.mean((raw < -1)|(raw > 1)))==diagnostic['fraction_outside_minus1_plus1']
    np.testing.assert_allclose(adj['per_frame_rms'],diagnostic['adjacent_raw_rms'],rtol=1e-12,atol=1e-15)
    timing=m['timing_seconds']; total=timing['post_input_validation_to_metrics']
    parts=math.fsum(v for k,v in timing.items() if k!='post_input_validation_to_metrics')
    assert all(v>=0 for v in timing.values()) and parts<=total<=r['wall_seconds']<=300
    rows.append({'id':name,'chunk_size':chunk,'wall_seconds':r['wall_seconds'],
                 'decode_seconds':timing['decode_latents'],'internal_total_seconds':total,
                 'unitemized_internal_seconds':total-parts,'peak_rss_kib':m['memory']['peak_rss_kib'],
                 'peak_rss_gib':m['memory']['peak_rss_kib']/1048576,'rss_scope':m['memory']['scope'],
                 'raw_min':float(raw.min()),'raw_max':float(raw.max()),
                 'outside_minus1_plus1_fraction':diagnostic['fraction_outside_minus1_plus1'],
                 'adjacent_raw_rms':adj['per_frame_rms'],'decoder_calls':m['decoder_calls'],
                 'metrics_sha256':sha(directory/'metrics.json'),'run_sha256':sha(record_dir/'run.json'),
                 'raw_npy_sha256':sha(directory/'decoded_raw.npy'),'changed_paths':r['git']['changed_paths']})
    arrays[name],metadata[name]=a,m
identity=[]
pairs={}
for left,right in itertools.combinations(arrays,2):
    for key in ('normalized_input','latent_mode','video_latents'):
        assert np.array_equal(arrays[left][key],arrays[right][key])
        assert arrays[left][key].tobytes()==arrays[right][key].tobytes()
    assert metadata[left]['versions']==metadata[right]['versions']
    identity.append({'pair':[left,right],'all_three_input_arrays_byte_identical':True})
    pairs[f'{left}_vs_{right}']=comparisons(arrays[left]['decoded_raw'],arrays[right]['decoded_raw'])
seven = arrays['chunk7']['decoded_raw']; one=arrays['chunk1']['decoded_raw']
invariance={'chunk7_first_vs_second_segment':comparisons(seven[:,:,:7],seven[:,:,7:]),
            'chunk1_all_vs_first_frame':comparisons(one,np.repeat(one[:,:,:1],14,axis=2))}
assert rows[0]['wall_seconds']<=240 and rows[0]['peak_rss_gib']<=12
assert execution['first_gate']['passed'] and execution['first_gate']['wall_seconds']==rows[0]['wall_seconds']
assert utc(execution['first_gate']['checked_utc'])<=utc(read(root/plan['commands'][1]['record_output']/'run.json')['started_utc'])
wall_sum = math.fsum(r['wall_seconds'] for r in rows)
sequence_seconds = (utc(execution['last_command_finished_utc'])-utc(execution['started_utc'])).total_seconds()
assert abs(sequence_seconds-execution['observed_sequence_seconds'])<1e-9
assert wall_sum==execution['command_wall_sum'] and wall_sum<=sequence_seconds<=1000
assert utc(execution['recorded_complete_utc'])>=utc(execution['last_command_finished_utc'])
for row, slot in zip(rows, execution['slots']):
    assert row['id']==slot['id'] and row['wall_seconds']==slot['wall_seconds'] and row['peak_rss_gib']==slot['peak_rss_gib']
assert execution['first_gate']['peak_rss_gib']==rows[0]['peak_rss_gib']
result={'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
        'preregistration_commit':commit,'plan_sha256':sha(root/plan_path),'source_hashes':source_hashes,
        'assets_full_sha_verified':assets,'input_identity_comparisons':identity,
        'cost_rows':rows,'sum_command_wall_seconds':wall_sum,
        'observed_request_window_seconds':sequence_seconds,
        'inter_request_and_coordinator_seconds':sequence_seconds-wall_sum,
        'pairwise_raw_float64_differences':pairs,'within_chunk_invariance':invariance,
        'versions':metadata['chunk14']['versions'],'execution':execution,
        'method':'Raw float32 NPY validated by file and C-order array SHA; float64 per-frame dot-product squared norms combined with math.fsum. No clipping, RGB conversion, image inspection or model execution.',
        'scope':'One fixed repeated encoded reference; exact input identity checked across processes. Encoder itself was not re-executed. Query-point RSS and full external command wall are distinct measures.'}
with output.open('x',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':result['status'],'output':str(output),'cost_rows':rows,
                  'pairs':{k:{x:v[x] for x in ('global_rms','global_max_abs','exact_equal')} for k,v in pairs.items()},
                  'invariance':invariance},ensure_ascii=False,indent=2))
