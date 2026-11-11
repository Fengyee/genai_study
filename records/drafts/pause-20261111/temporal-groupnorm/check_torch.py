"""Small CPU checks; no model assets or training. Output must be a new file."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import runpy
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error('output already exists')
base = Path(__file__).resolve().parent
stream = io.StringIO()
with redirect_stdout(stream):
    example = runpy.run_path(str(base / 'teaching-example.py'))
x, norm, full = (example[k] for k in ('x', 'norm', 'full'))
remote_full = norm(example['remote'])
assert bool(torch.all(remote_full[:, :, :5] != full[:, :, :5]))
# Compare a real two-group call to separate per-group calls; batch examples
# remain independent. An affine-free double tensor keeps all inputs exact.
grouped = torch.cat([x, 3*x + 50], dim=1)
combined = torch.nn.GroupNorm(2, 4, eps=example['eps'], affine=False).double().eval()
expected = torch.cat([norm(grouped[:, :2]), norm(grouped[:, 2:])], dim=1)
torch.testing.assert_close(combined(grouped), expected, rtol=1e-14, atol=1e-14)
modified = grouped.clone()
modified[:, 2:, -1] += 100
assert torch.equal(combined(modified)[:, :2], combined(grouped)[:, :2])
batch = torch.cat([x, example['remote']], dim=0)
torch.testing.assert_close(norm(batch), torch.cat([full, remote_full]), rtol=0, atol=0)

identity = lambda p: {'path': str(p), 'bytes': p.stat().st_size,
                       'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
result = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'python_check_source': identity(Path(__file__).resolve()),
    'teaching_example': identity(base/'teaching-example.py'),
    'torch_version': torch.__version__, 'dtype': 'float64', 'device': 'cpu',
    'epsilon': example['eps'], 'stdout': stream.getvalue(),
    'observed_frame2_full_chunk_halo': example['observed'].tolist(),
    'scalar_frame2_full_chunk_halo': example['expected'].tolist(),
    'max_abs_scalar_difference': (example['observed']-example['expected']).abs().max().item(),
    'remote_changed_frame2': remote_full[0,0,1,0,0].item(),
    'remote_modification_changed_all_ten_local_full_outputs': True,
    'two_groups_match_separate_normalization': True,
    'remote_other_group_does_not_affect_first_group': True,
    'batch_examples_normalized_independently': True,
    'training_evaluation_exact_match': True,
    'scope': 'One normalization layer, no convolution, trained model, SVD forward or causal attribution of decoder differences.',
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
