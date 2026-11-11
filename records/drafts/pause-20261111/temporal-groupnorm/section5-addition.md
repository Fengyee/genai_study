### 统计范围怎样影响块内帧

固定一个样本和一个通道组。若共有 $`G`$ 个组，每组在 $`M=(C/G)FHW`$ 个数上计算

```math
\mu=\frac1M\sum_{i=1}^M x_i,\qquad
s^2=\frac1M\sum_{i=1}^M(x_i-\mu)^2,\qquad
y_i=\frac{x_i-\mu}{\sqrt{s^2+\epsilon}}.
```

这里取`affine=False`以单独观察归一化；实际SVD还包含每通道的可学习缩放和平移。方差分母为 $`M`$，对应PyTorch的带偏估计。调用`.eval()`继续使用这次输入的统计量。固定VAE的中间块与上采样块向时间残差层显式传入`temporal_eps=1e-5`，下面沿用该数值。

构造`[1,2,8,1,1]`张量：两个通道分别为`[0,…,7]`和`[8,…,15]`，共同组成一组。比较整段、前4帧，以及前4帧加上第5帧作为右侧上下文（halo）。三者统计如下：

| 统计范围 | 均值 | 总体方差 |
| --- | ---: | ---: |
| 全部8帧 | 7.5 | 21.25 |
| 前4帧 | 5.5 | 17.25 |
| 前5帧，随后裁回4帧 | 6 | 18 |

第0通道第2帧的值为1，处在四帧块的内部。代入公式，三种输出依次约为`-1.410048`、`-1.083472`、`-1.178511`；多取一帧上下文仍未恢复整段的统计量。

<!-- cpu-check: temporal-groupnorm-context -->
```python
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
```

代码还将第1通道第8帧从15改为115。前5帧输入保持相同，因此局部归一化保持；全段统计改为均值13.75、方差700.9375，前5帧的全部10个输出都变化。这说明当上下文只覆盖序列的一部分时，仅在块内重算GroupNorm无法普遍恢复整段结果。覆盖整组或提供相应全局统计是不同的计算条件。

这个小例单独检验归一化的统计范围。真实SVD包含多个时间卷积、非线性、残差与可学习仿射参数，本实验的原始差值仍按完整解码器解释；两种机制各自的贡献需要额外干预实验。原14帧latent严格重复，因此某些早期层的全段/分块统计可能相同，后续时间卷积边界可改变进入归一化层的特征。

建议把练习3的答案接入本例，另加短题：**将一整组所有帧加上同一个常数，归一化结果会怎样？只改变远端一帧呢？** 答案：精确算术下，全组平移同时移动每个值与均值，中心化项和方差保持，包含固定eps的输出也保持；只改变一帧通常同时改变均值和方差，本例给出了明确反例。浮点实现可能有舍入差异。
