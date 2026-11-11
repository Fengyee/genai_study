"""Check three frozen author sampler methods using an analytic NumPy substitute.

The source SHA is checked before AST parsing or execution. Only the reviewed
sample_one_step, solver_step and _default_schedule bodies are executed; no
upstream import, JAX runtime, Flax module or trained model is loaded.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_SHA = "b8dcee1033e6ba01de0019d14fad846a399d356d67e549ee311a5e19d2a7c3a4"
EXPECTED_BYTES = 8029
COMMIT = "58d099c0fa197bd31b9a7b51727a76019ae5fdbe"
METHODS = {"sample_one_step", "solver_step", "_default_schedule"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True, help="Downloaded frozen meanflow.py")
parser.add_argument("--output", type=Path, required=True, help="New JSON output; parent must exist")
args = parser.parse_args()
if args.output.exists():
    parser.error("output already exists; choose a new path")
raw = args.source.read_bytes()
if len(raw) != EXPECTED_BYTES or hashlib.sha256(raw).hexdigest() != EXPECTED_SHA:
    parser.error("source size/SHA differs from the reviewed frozen implementation")
tree = ast.parse(raw.decode("utf-8"))
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MeanFlow"]
assert len(classes) == 1
methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
assert len(methods) == 3 and {n.name for n in methods} == METHODS
namespace = {"jnp": np}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(args.source), "exec"), namespace)


class AnalyticSubstitute:
    sampling_timesteps = None

    def __init__(self):
        self.calls = []

    def u_fn(self, z_t, t, h, y, train=False):
        self.calls.append({"t": t.tolist(), "h": h.tolist(), "train": train,
                           "shape": list(z_t.shape)})
        r = t - h
        m, s = 1.5, 0.7
        qt, qr = (1-t)**2*s*s+t*t, (1-r)**2*s*s+r*r
        c = ((1+s*s)*(r+t)-2*s*s)/(np.sqrt(qt)*(np.sqrt(qt)+np.sqrt(qr)))
        return -m + c[:, None, None, None] * (z_t-(1-t)[:, None, None, None]*m)


for name in METHODS:
    setattr(AnalyticSubstitute, name, namespace[name])
x = np.array([-1.25, 0.25, 1.75], dtype=np.float64).reshape(3, 1, 1, 1)
model = AnalyticSubstitute()
grid = model._default_schedule()
output = model.sample_one_step(x, np.array([0, 1, 2]), 0, grid)
assert np.array_equal(grid, np.array([1., 0.])) and len(model.calls) == 1
np.testing.assert_allclose(output, 1.5+0.7*x, atol=1e-14, rtol=0)
record = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_sha256": EXPECTED_SHA, "source_bytes": len(raw), "source_commit": COMMIT,
    "method": "AST-extracted three reviewed source functions; jnp mapped to NumPy; model u_fn replaced by analytic Gaussian average velocity; no JAX/Flax import or weight inference",
    "function_lines": {n.name: [n.lineno, n.end_lineno] for n in methods},
    "grid": grid.tolist(), "model_calls": model.calls,
    "input": x.ravel().tolist(), "output": output.ravel().tolist(),
    "max_error": float(np.max(np.abs(output-(1.5+0.7*x)))),
    "scope": "Actual source control flow and one-call time/interval contract; excludes network computation, compiled JAX, training and image quality",
}
with args.output.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
