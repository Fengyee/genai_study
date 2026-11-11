from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
import torch

torch.set_num_threads(1)
ROOT=Path('/workspace/genai_study');OLD=ROOT/'artifacts/first-run-5000-audit';NEW=ROOT/'artifacts/recovered-cpu-reader-path'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def compare(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all() and np.isfinite(b).all()
    d=b.astype(np.float64)-a.astype(np.float64)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'exact_equal':bool(np.array_equal(a,b)),'different_scalars':int(np.count_nonzero(a!=b)),'scalars':int(a.size),'max_abs':float(np.max(np.abs(d))),'rms':float(np.sqrt(np.mean(d*d))),'relative_l2_to_old':float(np.linalg.norm(d.ravel())/np.linalg.norm(a.ravel())) if np.linalg.norm(a.ravel()) else None}
def arr_hash(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
state=json.loads((NEW/'execution.json').read_text()); assert state['status']=='completed'
plan=json.loads((NEW/'plan.json').read_text());assert all(sha(ROOT/p)==h for p,h in plan['historical_file_sha256'].items())
result={'checked_utc':datetime.now(timezone.utc).isoformat(),'scope':'One recovered Intel CPU instance; same saved software versions and numerical training/sampling sources as historical reader run, with checkpoint validation changes. Exact comparisons and measured differences recorded without a chosen tolerance or causal attribution.','training':{},'solver':{},'historical_files_preserved':True,'sources_preserved_during_execution':state['source_files_preserved'],'execution':{'campaign_wall_seconds':state['campaign_wall_seconds'],'runs':[{k:r.get(k) for k in ['label','status','wall_seconds','exit_code']} for r in state['requests']]}}
for label in ['flow','ddpm','solver']:
    oldm=json.loads((OLD/label/'metrics.json').read_text());newm=json.loads((NEW/label/'metrics.json').read_text())
    r=json.loads((NEW/'run-records'/label/'run.json').read_text());assert r['status']=='succeeded' and r['exit_code']==r['child_returncode']==0 and not r['timed_out'];assert sha(NEW/'run-records'/label/'command.log')==r['log_sha256']
    olda=np.load(OLD/label/'samples.npz',allow_pickle=False);newa=np.load(NEW/label/'samples.npz',allow_pickle=False);assert olda.files==newa.files
    arrays={k:dict(compare(olda[k],newa[k]),old_raw_sha256=arr_hash(olda[k]),new_raw_sha256=arr_hash(newa[k])) for k in olda.files}
    config1={k:v for k,v in oldm['config'].items() if k!='output'};config2={k:v for k,v in newm['config'].items() if k!='output'};assert config1==config2 and newm['config']['threads']==1 and oldm['versions']==newm['versions']
    data={'config_equal_except_output':True,'versions_equal':True,'arrays':arrays,'wall_seconds':r['wall_seconds'],'all_expected_files_present':all((NEW/label/p).exists() for p in (['metrics.json','samples.npz','samples.png','loss.png','checkpoint.pt'] if label!='solver' else ['metrics.json','samples.npz','samples.png']))}
    if label!='solver':
        data['metric_equality']={k:oldm[k]==newm[k] for k in ['initial_eval_loss','final_eval_loss','diagnostics','baseline_diagnostics','data_theory']}
        data['old_diagnostics']=oldm['diagnostics'];data['new_diagnostics']=newm['diagnostics']
        data['old_final_eval_loss']=oldm['final_eval_loss'];data['new_final_eval_loss']=newm['final_eval_loss'];data['new_internal_elapsed_seconds']=newm['elapsed_seconds']
        oldp=torch.load(OLD/label/'checkpoint.pt',weights_only=True,map_location='cpu');newp=torch.load(NEW/label/'checkpoint.pt',weights_only=True,map_location='cpu')
        assert oldp.keys()==newp.keys() and oldp['schema_version']==newp['schema_version'] and oldp['method']==newp['method'] and oldp['architecture']==newp['architecture']
        oldstate=oldp['model_state_dict'];newstate=newp['model_state_dict'];assert oldstate.keys()==newstate.keys()
        tensor_comparisons={k:compare(oldstate[k].numpy(),newstate[k].numpy()) for k in oldstate}
        diffs=np.concatenate([(newstate[k].numpy().astype(np.float64)-oldstate[k].numpy().astype(np.float64)).ravel() for k in oldstate])
        data['checkpoint']={'old_file_sha256':sha(OLD/label/'checkpoint.pt'),'new_file_sha256':sha(NEW/label/'checkpoint.pt'),'file_bytes_equal':(OLD/label/'checkpoint.pt').read_bytes()==(NEW/label/'checkpoint.pt').read_bytes(),'state_all_exact_equal':all(v['exact_equal'] for v in tensor_comparisons.values()),'state_different_scalars':int(np.count_nonzero(diffs)),'state_total_scalars':int(diffs.size),'state_max_abs':float(np.max(np.abs(diffs))),'state_rms':float(np.sqrt(np.mean(diffs*diffs))),'tensors':tensor_comparisons,'metadata_config_equal_except_output':{k:v for k,v in oldp['metadata']['config'].items() if k!='output'}=={k:v for k,v in newp['metadata']['config'].items() if k!='output'},'metadata_versions_equal':oldp['metadata']['versions']==newp['metadata']['versions'],'scope':'Raw checkpoint file includes run-specific metadata; tensor equality is checked separately.'}
        result['training'][label]=data
    else:
        assert newm['checkpoint']['sha256']==sha(NEW/'flow/checkpoint.pt')
        data['checkpoint_training_steps']=newm['checkpoint']['metadata']['config']['steps'];data['checkpoint_sha256_matches_actual_file']=True
        data['old_runs']=oldm['runs'];data['new_runs']=newm['runs']
        steps=newm['config']['steps'];data['independent_paired_rms_l2']=[]
        for prev,cur,run in zip(steps,steps[1:],newm['runs'][1:]):
            diff=newa['steps_'+str(cur)].astype(np.float64)-newa['steps_'+str(prev)].astype(np.float64)
            rms=float(np.sqrt(np.mean(np.sum(diff*diff,axis=1))));saved=run['paired_rms_l2_difference_from_previous'];assert abs(rms-saved)<1e-7
            data['independent_paired_rms_l2'].append({'previous':prev,'steps':cur,'recomputed_float64':rms,'recorded':saved,'abs_difference':abs(rms-saved)})
        data['comparison_scope']='The new reader pipeline solver uses the newly trained flow checkpoint; differences can combine retraining and inference-environment effects.'
        result['solver']=data
(NEW/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
Path('/tmp/genai-recovered-reader-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'training':{k:{'generated':v['arrays']['generated'],'state_all_exact':v['checkpoint']['state_all_exact_equal'],'diagnostics':v['new_diagnostics'],'wall':v['wall_seconds']} for k,v in result['training'].items()},'solver_arrays':{k:{'exact_equal':v['exact_equal'],'max_abs':v['max_abs']} for k,v in result['solver']['arrays'].items()}},indent=2))
