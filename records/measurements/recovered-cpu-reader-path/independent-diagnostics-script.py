from pathlib import Path
import json
import numpy as np

base=Path('/workspace/genai_study/artifacts/recovered-cpu-reader-path')
centers=2*np.column_stack((np.cos(np.arange(8)*np.pi/4),np.sin(np.arange(8)*np.pi/4)))
def diagnostic(x,y):
    x,y=x.astype(np.float64),y.astype(np.float64)
    xy=np.sqrt(np.sum((x[:512,None,:]-y[None,:512,:])**2,axis=-1))
    xx=np.sqrt(np.sum((x[:512,None,:]-x[None,:512,:])**2,axis=-1))
    yy=np.sqrt(np.sum((y[:512,None,:]-y[None,:512,:])**2,axis=-1))
    d=np.sqrt(np.sum((x[:,None,:]-centers[None,:,:])**2,axis=-1));nearest=np.argmin(d,axis=1);distance=np.min(d,axis=1);inside=distance<.24
    counts=np.bincount(nearest[inside],minlength=8)
    return {'energy_distance_v_statistic':float(2*xy.mean()-xx.mean()-yy.mean()),'energy_sample_count':min(len(x),512),'mean_distance_to_nearest_center':float(distance.mean()),'fraction_within_3sigma':float(np.mean(inside)),'covered_modes_at_1pct':int(np.sum(counts>=.01*len(x)))}
checks=[]
for label in ['flow','ddpm','solver','fixed-old-checkpoint-solver']:
    m=json.loads((base/label/'metrics.json').read_text());z=np.load(base/label/'samples.npz',allow_pickle=False)
    targets=[('generated',m['diagnostics'])] if label in ['flow','ddpm'] else [('steps_'+str(r['steps']),r['diagnostics']) for r in m['runs']]
    for key,reported in targets:
        independent=diagnostic(z[key],z['reference']);counts=['energy_sample_count','fraction_within_3sigma','covered_modes_at_1pct'];assert all(independent[k]==reported[k] for k in counts)
        checks.append({'label':label,'array':key,'independent':independent,'reported':reported,'integer_and_fraction_fields_exact_equal':True,'energy_abs_difference':abs(independent['energy_distance_v_statistic']-reported['energy_distance_v_statistic']),'nearest_distance_abs_difference':abs(independent['mean_distance_to_nearest_center']-reported['mean_distance_to_nearest_center'])})
result={'method':'NumPy float64 direct pairwise difference/sqrt matrices on first512 samples, plus analytic radius2 eight centers in float64; no calls to repository diagnostics or torch.cdist. Counter/threshold fields checked exactly, float discrepancies reported numerically. Float64 centers differ slightly from the implementation float32 trigonometric coordinates.','checks':checks,'max_energy_abs_difference':max(x['energy_abs_difference'] for x in checks),'max_nearest_distance_abs_difference':max(x['nearest_distance_abs_difference'] for x in checks),'all_count_fields_match':True}
(base/'independent-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
