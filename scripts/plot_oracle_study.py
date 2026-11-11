"""Plot saved oracle-study arrays and the analytic variance path; preserve outputs."""
import argparse
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from genai_study.toy import centers
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True,help='New PNG filename; JSON provenance uses the same stem')
args=parser.parse_args()
folder=args.input;path=args.output
provenance=path.with_suffix('.json')
if path.suffix.lower()!='.png': parser.error('output must end in .png')
if path.exists() or provenance.exists(): parser.error('output or provenance already exists; choose a new filename')
path.parent.mkdir(parents=True,exist_ok=True)
r=json.loads((folder/'metrics.json').read_text());arrays=np.load(folder/'arrays.npz',allow_pickle=False)
if hashlib.sha256((folder/'arrays.npz').read_bytes()).hexdigest()!=r['outputs']['arrays_npz_sha256']: parser.error('array archive hash differs from metrics')
fig,ax=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
ax[0].plot(np.arange(101),r['normal_analytic']['variance'],color='#da6a18',lw=2,label='Population variance recurrence')
ax[0].axhline(1,color='#222222',ls='--',label='Target variance = 1')
for index,run in enumerate(r['runs']):
 marker=('o','s','^')[index%3]
 value=run['normal']['generated']['mean_coordinate_variance']
 ax[0].scatter([100],[value],marker=marker,s=46,zorder=3,label=f"Seed {run['base_seed']}: {value:.4f}")
variance_values=r['normal_analytic']['variance']+[1.0]+[run['normal']['generated']['mean_coordinate_variance'] for run in r['runs']]
padding=max(.005,(max(variance_values)-min(variance_values))*.15)
ax[0].set(xlabel='Reverse updates completed',ylabel='Variance per coordinate',title='Standard normal: exact predictor, fixed variance',ylim=(min(variance_values)-padding,max(variance_values)+padding))
ax[0].legend(fontsize=8,loc='lower left');ax[0].grid(alpha=.2)
mu=centers().double().numpy();roles=[('mixture','Oracle DDPM (all seeds)','#da6a18'),('mixture_independent_target','Independent target (all seeds)','#2468a0')]
last=0
for role,label,color in roles:
 samples=np.concatenate([arrays[run['array_keys'][role]].astype(np.float64) for run in r['runs']])
 radii=np.sqrt(((samples[:,None]-mu[None])**2).sum(-1).min(1))/.08
 radii.sort();last=max(last,float(radii[-1]));ax[1].step(np.r_[0,radii],np.arange(len(radii)+1)/len(radii),where='post',lw=1.7,color=color,label=label)
x=np.linspace(0,last*1.02,800);ax[1].plot(x,1-np.exp(-x*x/2),'--',color='#222222',lw=1.5,label='Own-component Rayleigh CDF')
ax[1].axvline(3,color='#888888',lw=1,alpha=.7)
ax[1].set(xlabel='Nearest-center radius / component sigma',ylabel='Fraction within radius',title='Eight Gaussians: radial spread after sampling',xlim=(0,last*1.02),ylim=(0,1.02));ax[1].legend(fontsize=8,loc='lower right');ax[1].grid(alpha=.2)
fig.savefig(path,dpi=160);plt.close(fig)
identity={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_metrics_sha256':hashlib.sha256((folder/'metrics.json').read_bytes()).hexdigest(),'input_arrays_sha256':hashlib.sha256((folder/'arrays.npz').read_bytes()).hexdigest(),'scope':'Left uses analytic 101-point variance path and every seed observed mean coordinate variance; right pools every generated/independent-target sample from every recorded seed, without excluding tails. Rayleigh reference uses own component center.'}
provenance.write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity,indent=2))
