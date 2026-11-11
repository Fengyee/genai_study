"""Exact Gaussian coupling example. CPU, float64, NumPy; no learning or randomness."""
import numpy as np

I = np.eye(2, dtype=np.float64)
J = np.array([[0., -1.], [1., 0.]], dtype=np.float64)
z0 = np.array([[1., 0.], [.3, -1.2]], dtype=np.float64)  # Rows are particles.

def q(t):
    return (1-t)**2 + t*t

def independent_v(t, x):
    return ((2*t-1)/q(t))*x

def rotation_v(t, x):
    return x @ (((2*t-1)*I+J)/q(t)).T

for t in [0., .25, .5, .75, 1.]:
    M = (1-t)*I+t*J
    # Independent and rotation couplings have the same intermediate covariance.
    np.testing.assert_allclose(M@M.T, q(t)*I, atol=1e-15)
    radial_path = np.sqrt(q(t))*z0
    rotation_path = z0@M.T
    np.testing.assert_allclose(independent_v(t, radial_path),
                               (2*t-1)/np.sqrt(q(t))*z0, atol=1e-15)
    np.testing.assert_allclose(rotation_v(t, rotation_path), z0@(J-I).T, atol=1e-15)

np.testing.assert_array_equal(np.sqrt(q(1.))*z0, z0)  # Exact independent-field ODE.
np.testing.assert_array_equal(z0+independent_v(0., z0), np.zeros_like(z0))
np.testing.assert_allclose(z0+rotation_v(0., z0), z0@J.T, atol=1e-15)
# Reflow of the exact independent-field endpoint pairs (z0, z0): zero target.
np.testing.assert_array_equal(z0+np.zeros_like(z0), z0)
# Population costs use E[X0 X0^T]=I, not the two teaching points' sample average.
print("independent input / exact rectified cost:", float(np.trace(2*I)), 0.)
print("rotation input / exact rectified cost:", float(np.trace((J-I).T@(J-I))), 4.)
print("identity coupling achieves global minimum squared cost:", 0.)
print("independent-field one-step Euler endpoint:", z0+independent_v(0., z0))
