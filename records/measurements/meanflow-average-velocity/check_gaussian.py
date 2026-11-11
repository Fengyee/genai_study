"""Independent analytic checks, no paper implementation or trained model."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

parser = argparse.ArgumentParser(description="Independent analytic MeanFlow checks.")
parser.add_argument("--output", type=Path, required=True, help="New JSON output file; parent directory must exist.")
args = parser.parse_args()
if args.output.exists():
    parser.error("output already exists; choose a new path")

torch.set_num_threads(1)
DTYPE = torch.float64
m, s = 1.5, 0.7

def a(t): return (1-t)*m
def q(t): return (1-t)**2*s*s+t*t
def w(x,t): return -m+((1+s*s)*t-s*s)/q(t)*(x-a(t))
def transport(x,r,t): return a(r)+torch.sqrt(q(r)/q(t))*(x-a(t))
def u(x,r,t):
    # Rationalized divided difference, continuously defined also at r=t.
    c=((1+s*s)*(r+t)-2*s*s)/(torch.sqrt(q(t))*(torch.sqrt(q(t))+torch.sqrt(q(r))))
    return -m+c*(x-a(t))

def tensor(v): return torch.as_tensor(v,dtype=DTYPE)
z=tensor(np.linspace(-2,2,9))
rows=[]
for rf,tf in [(0.,1.),(.2,.8),(.05,.4),(.7,.9),(.4,.400001)]:
    r,t=tensor(rf),tensor(tf); x=a(t)+torch.sqrt(q(t))*z
    value,directional=torch.func.jvp(u,(x,r,t),(w(x,t),torch.zeros_like(r),torch.ones_like(t)))
    identity=float((value-w(x,t)+(t-r)*directional).abs().max())
    endpoint=float((x-(t-r)*value-transport(x,r,t)).abs().max())
    h=1e-5
    # Finite difference follows the analytic true trajectory, independent of JVP.
    xp=a(t+h)+torch.sqrt(q(t+h))*z; xm=a(t-h)+torch.sqrt(q(t-h))*z
    fd=(u(xp,r,t+h)-u(xm,r,t-h))/(2*h)
    derivative_error=float((fd-directional).abs().max())
    rows.append(dict(r=rf,t=tf,identity_max_abs=identity,endpoint_max_abs=endpoint,
                     jvp_finite_difference_max_abs=derivative_error))
    assert identity<2e-14 and endpoint<2e-14 and derivative_error<2e-8
# Diagonal including both endpoints; the rationalized extension avoids 0/0.
diagonal=[]
for tf in [0.,.2,.5,.9,1.]:
    t=tensor(tf); x=a(t)+torch.sqrt(q(t))*z
    error=float((u(x,t,t)-w(x,t)).abs().max()); diagonal.append(error); assert error<2e-14
# Reverse one Euler step at t=1 lands at the mean for all nine input states.
r,t=tensor(0.),tensor(1.); x=z.clone(); expected=m+s*z
meanflow=x-u(x,r,t); euler=x-w(x,t)
assert torch.allclose(meanflow,expected,atol=2e-14,rtol=0)
assert torch.allclose(euler,torch.full_like(x,m),atol=2e-14,rtol=0)
# A two-point conditional law matches the exact first two Gaussian moments.
r,t=tensor(.2),tensor(.8); x=tensor(.3); delta=t-r
conditional_mean=w(x,t); conditional_variance=s*s/q(t)
v=conditional_mean+torch.sqrt(conditional_variance)*tensor([-1.,1.])
values=[]
for vi in v:
    value,du=torch.func.jvp(u,(x,r,t),(vi,torch.zeros_like(r),torch.ones_like(t)))
    values.append(vi-delta*du)
targets=torch.stack(values); exact=u(x,r,t)
residual_var=float(((targets-exact)**2).mean())
expected_var=float(q(r)*s*s/q(t)**2)
assert torch.allclose(targets.mean(),exact,atol=2e-14,rtol=0)
assert abs(residual_var-expected_var)<2e-14
# The target is parameter-dependent; stopping it changes the update.
theta=tensor(.4).requires_grad_(); pred=theta*x
moving_targets=v-delta*theta*v
loss_stop=((pred-moving_targets.detach())**2).mean()
loss_full=((pred-moving_targets)**2).mean()
grad_stop=float(torch.autograd.grad(loss_stop,theta,retain_graph=True)[0])
grad_full=float(torch.autograd.grad(loss_full,theta)[0])
assert abs(grad_stop-grad_full)>.1
result={'scope':'Independent Gaussian algebra and numerical checks; no model or paper code executed',
        'torch':torch.__version__,'dtype':'float64','threads':1,'parameters':{'m':m,'s':s},
        'off_diagonal':rows,'diagonal_max_abs':max(diagonal),
        'one_step':{'r':0,'t':1,'z':z.tolist(),'exact_endpoint':expected.tolist(),
                    'meanflow':meanflow.tolist(),'euler':euler.tolist(),
                    'meanflow_max_abs':float((meanflow-expected).abs().max()),
                    'euler_max_abs':float((euler-expected).abs().max()),
                    'population_variance_exact':s*s,'population_variance_euler':0.},
        'conditional_target':{'r':.2,'t':.8,'x':.3,'mean_target':float(targets.mean()),
                    'exact_u':float(exact),'target_residual_variance':residual_var,
                    'analytic_residual_variance':expected_var,
                    'method':'Two conditional velocities with equal weight reproduce Gaussian first and second moments; no Monte Carlo claim'},
        'stopgrad_example':{'theta':.4,'semigradient':grad_stop,'full_target_gradient':grad_full},
        'passed':True}
with args.output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
