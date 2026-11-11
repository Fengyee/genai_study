from fractions import Fraction
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import subprocess
import sys
import numpy as np
import torch

started = datetime.now(timezone.utc).isoformat()
root = Path('/workspace/genai_study')
candidate = Path('/tmp/genai-temporal-norm-candidate')
paths = [candidate/'section5-addition.md',candidate/'teaching-example.py',candidate/'closing-and-exercise.md',root/'docs/svd-decoder-chunk-report.md']
files = {str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in paths}
eps = 1e-5

def exact_moments(values):
    values = list(map(Fraction,values))
    mean = sum(values)/len(values)
    variance = sum((v-mean)**2 for v in values)/len(values)
    return mean, variance

# Start from explicit group elements and exact rational reductions.
cases = [
 ('all8',list(range(16)),Fraction(15,2),Fraction(85,4)),
 ('first4',list(range(4))+list(range(8,12)),Fraction(11,2),Fraction(69,4)),
 ('first5',list(range(5))+list(range(8,13)),Fraction(6),Fraction(18)),
 ('remote_plus100',list(range(15))+[115],Fraction(55,4),Fraction(11215,16)),
]
exact = {}
for name,values,want_mean,want_variance in cases:
    mean, variance = exact_moments(values)
    assert (mean,variance)==(want_mean,want_variance)
    exact[name]={'mean_exact':str(mean),'variance_exact':str(variance),'mean':float(mean),'variance':float(variance),'M':len(values)}
    shifted = exact_moments([v+123 for v in values])
    assert shifted==(mean+123,variance)
    assert all(Fraction(v)-mean == Fraction(v+123)-shifted[0] for v in values)

# Independent NumPy reduction keeps batch and group axes separate.
def manual_groupnorm(array, groups):
    grouped=array.reshape(array.shape[0],groups,array.shape[1]//groups,*array.shape[2:])
    axes=tuple(range(2,grouped.ndim))
    mean=grouped.mean(axis=axes,keepdims=True)
    var=((grouped-mean)**2).mean(axis=axes,keepdims=True)
    return ((grouped-mean)/np.sqrt(var+eps)).reshape(array.shape)

x=np.arange(16,dtype=np.float64).reshape(1,2,8,1,1)
full=manual_groupnorm(x,1)
chunk=manual_groupnorm(x[:,:,:4],1)
halo=manual_groupnorm(x[:,:,:5],1)[:,:,:4]
observed=[float(a[0,0,1,0,0]) for a in [full,chunk,halo]]
expected=[(1-exact[k]['mean'])/math.sqrt(exact[k]['variance']+eps) for k in ['all8','first4','first5']]
np.testing.assert_array_equal(observed,expected)
norm=torch.nn.GroupNorm(1,2,eps=eps,affine=False).eval()
np.testing.assert_allclose(norm(torch.from_numpy(x)).numpy(),full,rtol=1e-14,atol=1e-14)

remote=x.copy()
remote[0,1,7,0,0]+=100
remote_full=manual_groupnorm(remote,1)
np.testing.assert_array_equal(manual_groupnorm(remote[:,:,:5],1),manual_groupnorm(x[:,:,:5],1))
assert np.all(remote_full[:,:,:5]!=full[:,:,:5])
remote_changed=int(np.count_nonzero(remote_full[:,:,:5]!=full[:,:,:5]))

# Two groups within each of two samples; six spatial locations per frame.
multi=np.arange(384,dtype=np.float64).reshape(2,4,8,2,3)
layer=torch.nn.GroupNorm(2,4,eps=eps,affine=False).eval()
torch_base=layer(torch.from_numpy(multi)).numpy()
manual=manual_groupnorm(multi,2)
np.testing.assert_allclose(torch_base,manual,rtol=1e-14,atol=1e-14)
changed=multi.copy()
changed[0,0,7,1,2]+=100
perturbed=layer(torch.from_numpy(changed)).numpy()
assert not np.array_equal(perturbed[0,:2],torch_base[0,:2])
np.testing.assert_array_equal(perturbed[0,2:],torch_base[0,2:])
np.testing.assert_array_equal(perturbed[1],torch_base[1])
np.testing.assert_array_equal(layer.train()(torch.from_numpy(multi)).numpy(),torch_base)

# Confirm the epsilon evidence refers to callers, not just class defaults.
identity=json.loads((candidate/'source-identity.json').read_text())
source_matches={}
for key in ['torch_groupnorm','diffusers_resnet','diffusers_blocks','diffusers_vae']:
    entry=identity['files'][key]
    raw=Path(entry['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==entry['sha256']
    source_matches[key]=entry['sha256']
blocks=Path(identity['files']['diffusers_blocks']['path']).read_text()
for cls in ['MidBlockTemporalDecoder','UpBlockTemporalDecoder']:
    body=blocks.split('class '+cls+'(',1)[1].split('\nclass ',1)[0]
    assert 'temporal_eps=1e-5' in body
resnet=Path(identity['files']['diffusers_resnet']['path']).read_text()
assert 'eps=temporal_eps if temporal_eps is not None else eps' in resnet
assert 'torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=eps, affine=True)' in resnet

run=subprocess.run([sys.executable,str(candidate/'teaching-example.py')],capture_output=True,text=True,timeout=30)
assert run.returncode==0,run.stderr
report={
 'status':'passed_no_confirmed_issue',
 'review_started_utc':started,'review_completed_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Read-only candidate, existing-report and pinned-installed-source review; independent exact-rational/NumPy/PyTorch small-tensor checks; no learned-model inference, training, repository edits, Git mutations or downloads.',
 'files':files,
 'independent_derivation':{
  'normalization_domain':'For each fixed sample n and group g, reduce C/G channels and F,H,W only; M=(C/G)FHW. Batch n and group g are excluded from the reduction axes.',
  'exact_moments':exact,
  'derivation':'For m first frames of both channels: mixture of uniform integers0..m-1 and the same integers shifted8 gives mean=(m-1)/2+4 and variance=(m*m-1)/12+16. Full8 gives7.5/21.25; first4 gives5.5/17.25; first5 gives6/18. Replacing15by115 increases sum by100 and sum of squares by13000, giving mean13.75 and variance890-13.75^2=700.9375.',
  'channel0_frame2_outputs':observed,
  'remote_changed_outputs_in_first5_frames':remote_changed,
  'whole_group_translation':'Exact rational calculation preserves centered values and variance after adding123to every group element. Fixed epsilon therefore preserves normalized values in exact arithmetic.',
  'group_isolation':'Two-group test: perturbing sample0/group0 leaves sample0/group1 exactly unchanged.',
  'batch_isolation':'Two-sample test: perturbing sample0 leaves all of sample1 exactly unchanged.',
  'train_eval_equality':'PyTorch CPU float64 output is exactly equal in train/eval for the same tensor.',
  'manual_vs_torch_max_abs':float(np.max(np.abs(torch_base-manual)))
 },
 'candidate_execution':{'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr},
 'source_review':{
  'local_hashes_match_supplied_pinned_identity':source_matches,
  'epsilon':'Both fixed VAE MidBlockTemporalDecoder and UpBlockTemporalDecoder explicitly pass temporal_eps=1e-5; SpatioTemporalResBlock forwards this to TemporalResnetBlock; spatial eps remains separately configured.',
  'real_parameters':'Real temporal norms have32groups and affine=True. Candidate explicitly fixes one group and affine=False to isolate statistics.',
  'network_scope':'Per-group statistics acrossF are established; actual output influence also depends on convolution/nonlinearity/residual/affine parameters. Candidate preserves this distinction and calls for intervention to attribute mechanisms.'
 },
 'reader_review':[
  'Frame indices are consistent with the preceding report section: frame2 is tensor index1, frame5halo is:5, remote frame8 is index7. Channel numbers use standard0-based convention.',
  'One right halo at the beginning of the sequence is an explicit finite-subset counterexample. Remote frame8changes full statistics while the local five-frame tensor is identical. Thus locally recomputed GroupNorm cannot universally reproduce full-group normalization.',
  'The new numerical example adds a global-statistics mechanism to the existing local-convolution example. Translation and remote-perturbation exercise adds a distinct completion criterion rather than repeating the previous halo question.',
  'The repeated-latent caveat accurately permits identical early statistics; subsequent temporal padding can alter features. The experiment does not assign a measured share of decoder error to GroupNorm.'
 ],
 'findings':[],
 'integration_note':'The candidate ends with an editorial suggestion to connect exercise3and add a question; turn that suggestion into the actual final exercise text during integration. Keep the scoped mechanism paragraph with the example.',
 'validation_limits':['No new upstream fetch; fixed installed files were compared with the supplied identity manifest and relevant code read.','No SVD model execution or attribution experiment.'],
 'rerun_script':'/tmp/audit_temporal_norm_reader.py'
}
p=Path('/tmp/genai-temporal-norm-reader-review.json')
p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':report['status'],'outputs':observed,'remote_changed':remote_changed,'manual_vs_torch_max_abs':report['independent_derivation']['manual_vs_torch_max_abs']},ensure_ascii=False))
