import numpy as np

def decode(x):
    p = np.pad(np.asarray(x, dtype=np.float64), (1, 1))
    return (p[:-2] + 2*p[1:-1] + p[2:]) / 4

def chunk_decode(x, size):
    return np.concatenate([decode(x[i:i+size]) for i in range(0, len(x), size)])

x = np.arange(1, 15, dtype=np.float64)
full, split = decode(x), chunk_decode(x, 7)
expected = np.zeros(14)
expected[6:8] = [-2., -1.75]  # 1-based frames 7 and 8.
np.testing.assert_array_equal(split - full, expected)
np.testing.assert_array_equal(decode([1., 2.]), [1., 1.25])
np.testing.assert_array_equal(chunk_decode([1., 2.], 1), [.5, 1.])

constant = np.ones(14)
single = chunk_decode(constant, 1)
np.testing.assert_array_equal(np.diff(single), np.zeros(13))
assert not np.array_equal(single, decode(constant))
print("frames 7/8: full", full[6:8], "chunk7", split[6:8])
print("constant input: full", decode(constant), "chunk1", single)
