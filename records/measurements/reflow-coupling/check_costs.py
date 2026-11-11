"""Independent float64 checks for two Gaussian couplings; no training or randomness."""
from pathlib import Path
import argparse
import json
import math
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True, type=Path,
                    help='New JSON output; parent directory must exist')
args = parser.parse_args()
OUT = args.output.expanduser().absolute()
if OUT.exists() or OUT.is_symlink():
    parser.error(f'output already exists: {OUT}')
if not OUT.parent.is_dir():
    parser.error(f'output parent directory does not exist: {OUT.parent}')
I = np.eye(2, dtype=np.float64)
J = np.array([[0., -1.], [1., 0.]], dtype=np.float64)
A = J - I
# Deterministic points check maps; population moments below use covariance formulas.
points = np.array([[1., 0.], [0., 1.], [-1., .5], [.3, -1.2], [2., 3.]], dtype=np.float64)

def q(t): return (1-t)**2 + t*t

def independent_map(t, z0): return math.sqrt(q(t)) * z0

def independent_v(t, x): return ((2*t-1)/q(t)) * x

def M(t): return (1-t) * I + t * J

def rotation_v(t, x): return x @ (((2*t-1)*I+J)/q(t)).T

sample_times = [0., .1, .25, .5, .7, .9, 1.]
checks = []
for t in sample_times:
    matrix = M(t)
    assert np.linalg.det(matrix) >= .5 - 1e-15
    # Direct conditional Gaussian regression from block covariance for independent endpoints.
    B = np.concatenate(((1-t)*I, t*I), axis=1)
    D = np.concatenate((-I, I), axis=1)
    cov_x = B @ B.T
    cov_delta_x = D @ B.T
    regression = np.linalg.solve(cov_x, cov_delta_x.T).T
    np.testing.assert_allclose(regression, ((2*t-1)/q(t))*I, atol=1e-15, rtol=0)
    conditional_cov = D@D.T - regression @ cov_delta_x.T
    np.testing.assert_allclose(conditional_cov, I/q(t), atol=1e-15, rtol=0)
    np.testing.assert_allclose(matrix @ matrix.T, q(t)*I, atol=1e-15, rtol=0)
    # Rotation coupling is invertible at every time; solve directly for X0 from Xt.
    zrot = points @ matrix.T
    direct_v = np.array([A @ np.linalg.solve(matrix, x) for x in zrot])
    closed_v = rotation_v(t, zrot)
    expected_v = points @ A.T
    np.testing.assert_allclose(direct_v, closed_v, atol=1e-14, rtol=0)
    np.testing.assert_allclose(closed_v, expected_v, atol=1e-14, rtol=0)
    zind = independent_map(t, points)
    analytic_derivative = ((2*t-1)/math.sqrt(q(t))) * points
    np.testing.assert_allclose(independent_v(t, zind), analytic_derivative, atol=1e-14, rtol=0)
    checks.append({'time':t,'q':q(t),'rotation_determinant':float(np.linalg.det(matrix)),
                   'rotation_constant_velocity_max_error':float(np.max(np.abs(closed_v-expected_v)))})

# Centered differences of exact maps, separate from the algebraic velocity formulas.
h = 1e-5
finite_difference = []
for t in [.1, .3, .5, .7, .9]:
    d_ind = (independent_map(t+h, points)-independent_map(t-h, points))/(2*h)
    d_rot = (points @ M(t+h).T-points @ M(t-h).T)/(2*h)
    ei = float(np.max(np.abs(d_ind-independent_v(t,independent_map(t,points)))))
    er = float(np.max(np.abs(d_rot-rotation_v(t,points@M(t).T))))
    assert ei < 2e-9 and er < 2e-9
    finite_difference.append({'time':t,'independent_error':ei,'rotation_error':er})

exact_independent_end = independent_map(1.,points)
first_euler_end = points + independent_v(0.,points)
reflow_euler_end = points + np.zeros_like(points)
rotation_end = points @ J.T
rotation_euler_end = points + rotation_v(0.,points)
np.testing.assert_array_equal(exact_independent_end,points)
np.testing.assert_array_equal(first_euler_end,np.zeros_like(points))
np.testing.assert_array_equal(reflow_euler_end,points)
np.testing.assert_allclose(rotation_euler_end,rotation_end,atol=1e-15,rtol=0)
# Population displacement costs with E[X0 X0^T] = I; these are not sample averages.
independent_original_cost = float(np.trace(2*I))
rotation_cost = float(np.trace(A.T@A))
assert independent_original_cost == rotation_cost == 4.0
# Straightness defined here as integral E||dot Z_t-(Z1-Z0)||^2 dt.
# Simpson integration independently checks closed forms S=4-pi and regression residual R=pi.
n = 20000
f = lambda t: 2*(2*t-1)**2/q(t)
g = lambda t: 2/q(t)
def simpson(fn):
    result = fn(0.) + fn(1.)
    for k in range(1,n): result += (4 if k%2 else 2)*fn(k/n)
    return result/(3*n)
S, R = simpson(f), simpson(g)
assert abs(S-(4-math.pi))<1e-12 and abs(R-math.pi)<1e-12 and abs(S+R-4)<1e-12
# For radial N(0,qI), div(p Jx/q)=0: trace J=0 and x^T J x=0.
assert np.trace(J) == 0
assert np.max(np.abs(np.einsum('ni,ij,nj->n',points,J,points))) < 1e-15
result = {'scope':'Independent analytic Gaussian/covariance and deterministic float64 map checks. No trained model, Monte Carlo estimate, paper code, network access or tracked repository files.',
          'numpy':np.__version__,'dtype':'float64','times':checks,'finite_difference_step':h,'finite_difference':finite_difference,
          'population_costs':{'independent':{'input':independent_original_cost,'exact_rectified_endpoint':0.},'rotation':{'input':rotation_cost,'exact_rectified_endpoint':rotation_cost},'identity_global_squared_cost_optimum':0.},
          'one_step':{'independent_field_euler_output':'identically zero','independent_exact_ode_endpoint':'identity','reflow_of_independent_exact_endpoint_coupling':'zero velocity, identity endpoint','rotation_field_euler_endpoint_max_error':float(np.max(np.abs(rotation_euler_end-rotation_end)))},
          'straightness_definition':'S=integral_0^1 E[||dot Z_t-(Z1-Z0)||^2] dt',
          'independent_field_straightness':{'analytic':4-math.pi,'simpson':S},'independent_conditional_regression_residual':{'analytic':math.pi,'simpson':R},
          'rotation_straightness':0.,'same_intermediate_covariance':'q(t) I for both couplings','weighted_divergence_free_velocity_difference':'J x / q(t)',
          'conclusion':'Exact rectification is a fixed point for the rotation coupling although its squared displacement cost4 exceeds attainable cost0. Parameter-time straightness and a one-Euler exact map do not enforce global OT.'}
with OUT.open('x', encoding='utf-8') as handle:
    handle.write(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'independent_straightness':S,'regression_residual':R,'rotation_cost':rotation_cost,'identity_cost':0.,'max_independent_finite_difference_error':max(x['independent_error'] for x in finite_difference)},indent=2))
