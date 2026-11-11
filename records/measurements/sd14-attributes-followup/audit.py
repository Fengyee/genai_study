"""Read-only independent audit of the fixed two-thread SD1.4 campaign; no image viewing."""
import argparse, hashlib, json, math, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

COMMIT = '0bc702c562f9dad44ad446d448ffcb1a30d0d995'
PLAN = Path('records/measurements/sd14-attributes-followup/plan.json')
STATE = Path('artifacts/sd14-attributes-followup-execution/execution.json')
def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text())
def normalize(config):
    result = dict(config); result['_use_default_values'] = sorted(result['_use_default_values']); return result
def positive(value): return type(value) in (int, float) and math.isfinite(value) and value > 0
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--verify-assets', action='store_true', help='Additionally hash all 16 current cached model assets')
args = parser.parse_args()
if args.output.exists(): parser.error('output exists; choose a new snapshot path')
raw = PLAN.read_bytes(); assert subprocess.check_output(['git', 'show', f'{COMMIT}:{PLAN}']) == raw
plan = json.loads(raw); state_raw = STATE.read_bytes(); state = json.loads(state_raw)
assert len(plan['requests']) == len(state['requests']) == 8 and state['plan_commit'] == COMMIT
assert state['plan_sha256'] == hashlib.sha256(raw).hexdigest()
assert state['driver_sha256'] == digest('records/measurements/sd14-attributes-pilot/execute_plan.py')
for name, identity in plan['source_identities'].items():
    assert digest(name) == identity['sha256'] and Path(name).stat().st_size == identity['bytes'], name
asset_check = {'performed': args.verify_assets}
if args.verify_assets:
    manifest = read('examples/pretrained-sd14/manifest.json')
    for item in manifest['files']:
        path = Path(plan['model_directory']) / item['path']
        assert path.stat().st_size == item['size_bytes'] and digest(path) == item['sha256'], str(path)
    asset_check.update(files=len(manifest['files']), bytes=sum(x['size_bytes'] for x in manifest['files']), passed=True)
noise_by_request = {}; checked = []; previous_finished = None; fixed = plan['fixed']; expected_scheduler = plan['expected_scheduler']
for request, live in zip(plan['requests'], state['requests']):
    for key in ('request_id','order','prompt_id','seed','output','record_output','argv'): assert live[key] == request[key], key
    entry = {'request_id': request['request_id'], 'seed': request['seed'], 'status': live['status']}
    if live['status'] not in ('succeeded','filtered'):
        entry['scope'] = 'Retained snapshot status; completed-output checks pending'; checked.append(entry); continue
    output, recdir = Path(request['output']), Path(request['record_output'])
    record, metrics = read(recdir/'run.json'), read(output/'metrics.json')
    begin, end = datetime.fromisoformat(record['started_utc']), datetime.fromisoformat(record['finished_utc'])
    assert begin < end and (previous_finished is None or begin >= previous_finished); previous_finished = end
    assert record['status'] == 'succeeded' and record['exit_code'] == record['child_returncode'] == 0 and record['timed_out'] is False
    assert record['argv'] == request['argv'][request['argv'].index('--')+1:] and record['timeout_seconds'] == 420
    assert record['runner']['source_sha256'] == plan['source_identities']['src/genai_study/run_record.py']['sha256']
    assert record['log_file'] == 'command.log' and digest(recdir/'command.log') == record['log_sha256']
    assert (recdir/'command.log').stat().st_size == record['log_size_bytes']
    assert not set(plan['source_identities']).intersection(x['path'] if isinstance(x,dict) else x for x in record['git']['changed_paths'])
    expected = {k: fixed[k] for k in ('steps','guidance_scale','height','width','threads','device','dtype','eta','negative_prompt','batch_size')}
    expected.update(seed=request['seed'],prompt=plan['prompts'][request['prompt_id']]['text'],deterministic_algorithms=True)
    assert metrics['config'] == expected
    assert metrics['versions'] == dict(plan['dependency_preflight']['versions'],python=plan['dependency_preflight']['python'])
    model = metrics['model']; assert model['model_id'] == plan['model_id'] and model['revision'] == plan['revision']
    assert model['manifest_path'] == 'examples/pretrained-sd14/manifest.json' and model['manifest_sha256'] == digest(model['manifest_path'])
    assert (model['verified_files'],model['verified_bytes']) == (16,5482669095)
    actual = metrics['scheduler']; assert normalize(actual['config']) == normalize(expected_scheduler['config'])
    for k in ('class','timesteps','init_noise_sigma'): assert actual[k] == expected_scheduler[k], k
    network = metrics['network_evaluations']; assert network['unet_forward_calls'] == 20 and network['unet_batch_sizes'] == [2]*20 and network['unet_sample_branch_evaluations'] == 40
    rep = metrics['representation']; assert rep['prompt_tokens_with_special_tokens'] == plan['prompts'][request['prompt_id']]['tokens_with_special_tokens']
    assert rep['tokenizer_max_length'] == 77 and rep['prompt_truncated'] is False
    assert rep['vae_spatial_factor'] == 8 and rep['vae_scaling_factor'] == 0.18215
    assert metrics['outputs']['image'] == 'image.png' and metrics['outputs']['initial_noise'] == 'initial_noise.npy'
    with Image.open(output/'image.png') as im: assert im.format == 'PNG' and im.size == (512,512) and im.mode == 'RGB'; im.load()
    assert digest(output/'image.png') == metrics['outputs']['image_sha256'] == live['image_sha256']
    noise = np.load(output/'initial_noise.npy',allow_pickle=False); assert noise.shape == (1,4,64,64) and noise.dtype == np.float32 and noise.flags.c_contiguous and np.isfinite(noise).all()
    noise_sha = hashlib.sha256(noise.tobytes(order='C')).hexdigest()
    assert noise_sha == plan['expected_initial_noise'][str(request['seed'])]['sha256'] == rep['initial_noise_sha256'] == live['initial_noise_sha256']
    assert rep['initial_noise_shape'] == [1,4,64,64]
    noise_by_request[request['request_id']] = (request['seed'],noise)
    safety = metrics['safety']; assert safety['checker_enabled'] is True and type(safety['nsfw_content_detected']) is bool
    assert (live['status'] == 'filtered') == safety['nsfw_content_detected']
    wall, rss = record['wall_seconds'], metrics['memory']['process_peak_rss_kib']; assert positive(wall) and positive(rss)
    assert wall == live['wall_seconds'] and rss == live['process_peak_rss_kib']
    if request['order'] == 1: assert state['first_request_gate']['passed'] is True and wall <= 360 and rss <= 8388608
    entry.update(validation='passed',wall_seconds=wall,inference_seconds=metrics['timing_seconds']['inference'],rss_kib=rss,png_sha256=metrics['outputs']['image_sha256'],noise_raw_sha256=noise_sha,noise_file_sha256=digest(output/'initial_noise.npy'),run_record_sha256=digest(recdir/'run.json'),metrics_sha256=digest(output/'metrics.json'),launch_git_head=record['git']['head'])
    checked.append(entry)
if state['status'] == 'completed':
    assert len(noise_by_request) == 8 and state['active_request'] is None and state['stop_reason'] is None
    assert positive(state['campaign_wall_seconds']) and state['campaign_wall_seconds'] <= plan['execution_budget']['campaign_wall_budget_seconds']
pairs = []
for i,(name,(seed,a)) in enumerate(noise_by_request.items()):
    for other,(other_seed,b) in list(noise_by_request.items())[i+1:]:
        equal = np.array_equal(a,b); assert equal == (seed == other_seed), (name,other)
        pairs.append({'left':name,'right':other,'same_seed':seed==other_seed,'arrays_equal':equal})
result = {'checked_utc':datetime.now(timezone.utc).isoformat(),'scope':'Independent read-only audit; only terminal succeeded/filtered slots receive completed-output validation. No pixel-content interpretation. Earlier timeout batch and four-step diagnostics excluded.','plan_commit':COMMIT,'plan_sha256':hashlib.sha256(raw).hexdigest(),'execution_snapshot_sha256':hashlib.sha256(state_raw).hexdigest(),'campaign_status_at_snapshot':state['status'],'completed_slots_checked':len(noise_by_request),'all_eight_completed_and_checked':len(noise_by_request)==8 and state['status']=='completed','source_identity_count':len(plan['source_identities']),'cached_asset_verification':asset_check,'requests':checked,'successful_command_wall_sum_seconds':sum(x.get('wall_seconds',0) for x in checked),'campaign_wall_seconds':state.get('campaign_wall_seconds'),'max_query_point_rss_kib':max((x.get('rss_kib',0) for x in checked),default=0),'pairwise_noise_checks':pairs,'memory_scope':'Linux RUSAGE_SELF high-water RSS from process start through source line254 query, after PNG/NPY saving and before PNG hash, metrics serialization/write, printing and exit. Full command wall comes from run_record.','script_sha256':digest(__file__)}
args.output.parent.mkdir(parents=True,exist_ok=True)
with args.output.open('x') as stream: stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('campaign_status_at_snapshot','completed_slots_checked','all_eight_completed_and_checked','cached_asset_verification','script_sha256')},indent=2))
