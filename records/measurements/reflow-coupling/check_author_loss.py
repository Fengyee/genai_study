"""Execute one pinned upstream loss factory with tiny tensors and a model stub."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', required=True, type=Path, help='Pinned author losses.py')
parser.add_argument('--output', required=True, type=Path, help='New JSON output; parent must exist')
args = parser.parse_args()
source = args.source.expanduser().absolute()
output = args.output.expanduser().absolute()
if output.exists() or output.is_symlink():
    parser.error(f'output already exists: {output}')
if not output.parent.is_dir():
    parser.error(f'output parent directory does not exist: {output.parent}')
if not source.is_file():
    parser.error(f'source is not a readable file: {source}')
source_bytes = source.read_bytes()
expected_sha = 'f59d42f90c982063498c5c6328d184bebb3258a79c8233a712cf06743b4ca5e8'
actual_sha = hashlib.sha256(source_bytes).hexdigest()
if actual_sha != expected_sha:
    parser.error(f'source SHA-256 mismatch: expected {expected_sha}, got {actual_sha}')
module = ast.parse(source_bytes.decode('utf-8'))
fn = next(node for node in module.body if isinstance(node, ast.FunctionDef)
          and node.name == 'get_rectified_flow_loss_fn')
namespace = {'torch': torch,
             'mutils': SimpleNamespace(get_model_fn=lambda model, train: model)}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), namespace)
torch.set_num_threads(1)
eps, seed = 1e-3, 1729
z0 = torch.arange(24, dtype=torch.float32).reshape(4, 1, 2, 3) / 10
z1 = z0.flip(0) * 0.4 + 2
target = z1 - z0
observed = {}
def model(x, label):
    observed.update(x=x.clone(), label=label.clone())
    return target.clone()
sde = SimpleNamespace(reflow_flag=True, reflow_t_schedule='uniform',
                      reflow_loss='l2', T=1.0)
loss_fn = namespace['get_rectified_flow_loss_fn'](sde, train=True)
generator = torch.Generator().manual_seed(seed)
times = torch.rand(4, generator=generator) * (1 - eps) + eps
torch.manual_seed(seed)
loss = loss_fn(model, [z0, z1])
expected_x = times[:, None, None, None] * z1 + (1-times[:, None, None, None]) * z0
torch.testing.assert_close(observed['x'], expected_x, rtol=0, atol=0)
torch.testing.assert_close(observed['label'], times * 999, rtol=0, atol=0)
assert float(loss) == 0.0
assert bool(((times >= eps) & (times < 1)).all())
assert observed['x'].shape == z0.shape
# Independent target mismatch checks the actual L2 and dimension/batch reduction.
shift = torch.tensor([1., 2., 3., 4.])[:, None, None, None]
changed_loss = loss_fn(lambda x, t: target + shift, [z0, z1])
expected_changed_loss = shift.square().mean()
torch.testing.assert_close(changed_loss, expected_changed_loss, rtol=0, atol=0)
# Permuting only the endpoint changes the supplied coupling and its target.
permuted = z1.roll(1, 0)
permuted_loss = loss_fn(lambda x, t: target.clone(), [z0, permuted])
expected_permuted_loss = (z1-permuted).square().mean()
torch.testing.assert_close(permuted_loss, expected_permuted_loss, rtol=1e-6, atol=1e-6)
assert float(permuted_loss) > 0
result = {'status':'passed', 'torch':torch.__version__, 'device':'cpu',
          'threads':torch.get_num_threads(), 'dtype':'float32', 'seed':seed,
          'source_sha256':expected_sha, 'source_path':str(source),
          'output_path':str(output), 'extracted_function':fn.name,
          'source_lines':[fn.lineno, fn.end_lineno], 'eps':eps,
          'times':times.tolist(), 'perfect_target_loss':float(loss),
          'offset_target_loss':float(changed_loss),
          'endpoint_permutation_loss':float(permuted_loss),
          'scope':'Real upstream factory AST executed; stub model/mutils, four tiny tensors. No training, architecture, data loader, solver, or full upstream dependency execution.'}
with output.open('x', encoding='utf-8') as handle:
    handle.write(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
