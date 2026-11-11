import math
import torch

eps = 1e-5
x = torch.arange(16, dtype=torch.float64).reshape(1, 2, 8, 1, 1)
norm = torch.nn.GroupNorm(1, 2, eps=eps, affine=False).eval()
full = norm(x)
chunk4 = norm(x[:, :, :4])
halo = norm(x[:, :, :5])[:, :, :4]  # Frame 5 supplies a one-frame right halo.

# Channel 0, frame 2 lies inside the first four-frame chunk.
observed = torch.stack([full[0, 0, 1, 0, 0], chunk4[0, 0, 1, 0, 0],
                        halo[0, 0, 1, 0, 0]])
expected = torch.tensor([(1 - 7.5) / math.sqrt(21.25 + eps),
                         (1 - 5.5) / math.sqrt(17.25 + eps),
                         (1 - 6.0) / math.sqrt(18.00 + eps)], dtype=torch.float64)
torch.testing.assert_close(observed, expected, rtol=1e-14, atol=1e-14)
assert len(torch.unique(observed)) == 3

remote = x.clone()
remote[0, 1, 7, 0, 0] += 100  # Change frame 8 outside the first chunk and halo.
torch.testing.assert_close(norm(remote[:, :, :5]), norm(x[:, :, :5]), rtol=0, atol=0)
assert not torch.equal(norm(remote)[:, :, :4], full[:, :, :4])
torch.testing.assert_close(norm.train()(x), full, rtol=0, atol=0)
print("frame 2: full, chunk4, one-frame halo", observed.tolist())
