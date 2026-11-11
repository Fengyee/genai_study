import argparse
import json
import torch
from torch.func import jvp
from pathlib import Path

parser = argparse.ArgumentParser(description="Independent analytic MeanFlow checks.")
parser.add_argument("--output", type=Path, required=True, help="New JSON output file; parent directory must exist.")
args = parser.parse_args()
if args.output.exists():
    parser.error("output already exists; choose a new path")

torch.set_num_threads(1)
dtype=torch.float64
# Independent flow: dz/dtau=A z, A diagonal, with exact backward endpoint exp(-A h) z.
a=torch.tensor([0.7,-0.4],dtype=dtype)
z=torch.tensor([0.3,-1.2],dtype=dtype)
tau=torch.tensor(0.85,dtype=dtype)
rho=torch.tensor(0.2,dtype=dtype)
h=tau-rho

def average_h(z,tau,h):
    return -torch.expm1(-a*h)/h*z

def average_r(z,tau,rho):
    return average_h(z,tau,tau-rho)

vel=a*z
u,correct_r=jvp(average_r,(z,tau,rho),(vel,torch.ones_like(tau),torch.zeros_like(rho)))
_,correct_h=jvp(average_h,(z,tau,h),(vel,torch.ones_like(tau),torch.ones_like(h)))
_,wrong_h=jvp(average_h,(z,tau,h),(vel,torch.ones_like(tau),torch.zeros_like(h)))
identity=(u+h*correct_r-vel).abs().max().item()
endpoint=(z-h*u-torch.exp(-a*h)*z).abs().max().item()
wrong=(u+h*wrong_h-vel).abs().max().item()
assert identity<1e-14 and endpoint<1e-14
assert torch.equal(correct_r,correct_h) and wrong>1e-2
# Two consecutive exact average-velocity displacements use the intermediate state.
s=torch.tensor(0.5,dtype=dtype)
z_s=torch.exp(-a*(tau-s))*z
split=(tau-s)*average_r(z,tau,s)+(s-rho)*average_r(z_s,s,rho)
additivity=(h*u-split).abs().max().item(); assert additivity<1e-14
# Conditional JVP expectation is linear in the tangent, at fixed z,tau,rho and fixed field.
v1=vel+torch.tensor([0.6,-0.2],dtype=dtype); v2=vel-torch.tensor([0.6,-0.2],dtype=dtype)
conditional_targets=[]
for v in [v1,v2]:
    _,d=jvp(average_r,(z,tau,rho),(v,torch.ones_like(tau),torch.zeros_like(rho)))
    conditional_targets.append(v-h*d)
conditional_mean=(torch.stack(conditional_targets).mean(0)-u).abs().max().item(); assert conditional_mean<1e-14
# Stop-gradient changes the derivative when the target depends on theta.
theta=torch.tensor(0.8,dtype=dtype,requires_grad=True)
raw_target=1.2-0.3*theta
loss=(theta-raw_target.detach())**2
semi_grad=torch.autograd.grad(loss,theta)[0].item()
full_grad=torch.autograd.grad((theta-raw_target)**2,theta)[0].item()
assert abs(semi_grad-(-0.32))<1e-14 and abs(full_grad-(-0.416))<1e-14
out={'scope':'Independent diagonal linear vector field; no model training or author framework imported','identity_max_abs':identity,'endpoint_max_abs':endpoint,'rt_and_th_jvp_exact_equal':True,'wrong_constant_h_identity_max_abs':wrong,'two_interval_additivity_max_abs':additivity,'equal_weight_conditional_tangent_mean_error':conditional_mean,'stopgrad_gradient':semi_grad,'full_target_gradient':full_grad,'constant_target_assumption':'For conditional tangent comparison z,tau,rho and the current field are fixed; this is not a convergence proof.'}
with args.output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
