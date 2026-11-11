# 从空间网格到Transformer：patch、位置与预测输出

**先修：** [07卷积图像生成](../chapters/07-image-flow-lab.md)、[04潜空间与DiT](../chapters/04-image-video-roadmap.md)、[条件注意力](conditioning-attention.md)的Q/K/V与矩阵形状。**学习目标：** 写清空间patch的排列与投影；解释位置编码如何把内容绑定到网格；将预测token还原为带通道的空间场。

07章通过卷积和坐标通道处理图像位置，DiT则把潜网格转成token序列。本页用一个基础CPU块贯通这条结构链。范围是非重叠空间patch、二维位置的教学原型和固定DiT接口；视频token计数沿用[Wan](wan-video-2025.md)与[CogVideoX](video-training-svd-cogvideox.md)，本文无需训练或下载模型权重。这里的token是连续patch向量或特征；[ResTok](restok-image-ar-2026.md)生成器预测的token则定义为离散词表索引。

## 1. Patch先重排，再投影

设带噪潜变量为$`z\in\mathbb R^{B\times C\times H\times W}`$，patch边长为$`p`$，高宽均可被$`p`$整除。使用无padding、步长等于$`p`$的非重叠划分，得到

```math
H_p=H/p,\qquad W_p=W/p,\qquad L=H_pW_p,\qquad m=p^2C.
```

把过程分成两步：

| 步骤 | 输出形状 | 保留或学习什么 |
| --- | --- | --- |
| 按指定顺序收集patch坐标 | $`R\in\mathbb R^{B\times L\times m}`$ | 每个原始标量恰好出现一次，原网格尺寸与排列规则确定逆重排 |
| 逐patch共享线性投影 | $`U=RW_{\mathrm{emb}}+b\in\mathbb R^{B\times L\times d}`$ | $`W_{\mathrm{emb}}\in\mathbb R^{m\times d}`$学习内容表示，$`d`$为Transformer宽度 |

本页选择**先patch行、后patch列**；每个patch内部选择**先局部行、后局部列、最后通道**。索引均从0开始，明确写为

```math
n=iW_p+j,\qquad k=(rp+s)C+c,\qquad
R_{b,n,k}=z_{b,c,ip+r,jp+s}.
```

其中$`i,j`$为patch网格坐标，$`r,s\in\{0,\ldots,p-1\}`$为patch内部坐标。这个顺序需要一次轴交换，直接对BCHW做`reshape(B,L,m)`通常会把不同位置的标量错误地分组。

**无损重排与表示容量。** 第一步保存所有坐标；第二步能否保留任意patch向量的信息取决于矩阵秩。当$`d<m`$时，映射在整个$`\mathbb R^m`$上存在非零零空间；当$`d\ge m`$时，满行秩是从线性输出恢复任意输入的条件。训练目标、数据分布与后续预测头共同决定实际表示的用途。DiT论文将线性嵌入也包含在“patchify”这个名称中，读实现时可以按这两步检查。

相同的投影可由`Conv2d`完成。若卷积核布局为$`K\in\mathbb R^{d\times C\times p\times p}`$，按本页向量顺序需设置

```math
W_{\mathrm{emb}}[(rp+s)C+c,o]=K[o,c,r,s].
```

卷积使用`stride=p`、无padding，并将输出从`B,d,H_p,W_p`按行列展成`B,L,d`，即可与$`RW_{\mathrm{emb}}+b`$逐元素比较。更换patch内部顺序时，权重的对应轴也要一起变换。

## 2. 自注意力怎样看到空间位置？

条件注意力专题的交叉注意力从当前状态构造查询、从条件构造键和值。空间**自注意力（self-attention）**则从同一内容序列构造三者。省略batch，对$`U\in\mathbb R^{L\times d}`$，一个头写成

```math
Q=UW_Q,\quad K=UW_K,\quad V=UW_V,\qquad
\mathcal A(U)=\mathop{\mathrm{softmax}}\nolimits_{\mathrm{key}}
\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
```

这里每个位置共享投影参数，允许读取全部位置，关闭dropout，并且没有位置嵌入、位置偏置或mask。令$`P`$重排序列的行，则$`Q,K,V`$同时按$`P`$重排，logits变为$`P(QK^\top)P^\top`$。按行softmax也同步重排行列，因此

```math
\mathcal A(PU)=P\mathcal A(U).
```

这称为**置换等变（permutation equivariance）**：输入内容重排后，输出以相同方式重排。共享的逐token前馈层、逐token LayerNorm与残差连接保持这一性质。序列的存储下标本身没有进入上述公式，因而这一模块还没有被告知哪个内容位于网格左上角。

添加位置表$`E_{\mathrm{pos}}\in\mathbb R^{L\times d}`$后，模块接收$`U+E_{\mathrm{pos}}`$。每行位置向量与一个网格坐标绑定，可以表示二维的行列信息。要区分两个具体操作：

| 操作 | 输入怎样变化 | 应检查的关系 |
| --- | --- | --- |
| 仅改变序列的存储顺序，内容与位置一起搬动 | $`PU+PE_{\mathrm{pos}}=P(U+E_{\mathrm{pos}})`$ | $`\mathcal A(PU+PE_{\mathrm{pos}})=P\mathcal A(U+E_{\mathrm{pos}})`$ |
| 保留网格位置，将内容移到其它格子 | $`PU+E_{\mathrm{pos}}`$ | 内容与坐标的绑定改变，一般不满足上面的输出关系 |

下面的数值例把2×2网格每行的内容左右互换，保持位置表不动，可以直接检出后一种变化。位置表只提供坐标信息，模型怎样使用它仍由结构与训练决定。

固定DiT在输入投影后加入**冻结的二维正弦/余弦位置嵌入**，源码分别编码两个坐标轴后拼接。本页CPU例用两个坐标值组成的小向量来隔离“位置绑定”这一机制。它是教学位置原型；DiT的实际频率、维数与嵌入函数按末尾固定源码读取。CogVideoX等系统的具体位置配置仍按各自阅读卡核对。

## 3. 输出token如何回到空间场？

Transformer保持$`L`$个位置，预测头把每个位置映射到$`p^2C_{\mathrm{out}}`$个数，再执行逆重排：

```math
B,L,d\ \longrightarrow\ B,L,p^2C_{\mathrm{out}}
\ \longrightarrow\ B,C_{\mathrm{out}},H,W.
```

预测头学习任务要求的输出，逆重排把这些输出放回对应空间坐标。重排不需要求输入嵌入矩阵的逆；它只使用网格尺寸、patch大小和已约定的通道顺序。

DiT论文第5页的输出包含噪声预测及对角方差相关预测，固定源码默认`learn_sigma=True`，于是$`C_{\mathrm{out}}=2C`$。输入$`C=4,p=2`$时，每个预测token有$`2\cdot2\cdot8=32`$个数，最终张量是`B,8,H,W`；噪声与方差分支各占4个通道。方差分支按对应扩散实现解释为方差参数，具体数值变换需继续读取该实现。

固定`unpatchify`按`B,H_p,W_p,p,p,C_out`解释预测向量，再排列为BCHW；它使用正方形token网格。下面的教学函数显式接收$`H,W`$，把同一索引规则用于长方形网格。作者函数的docstring写为NHWC，实际`einsum`与返回值为NCHW；检查接口时以返回张量的运算为依据。

## 4. CPU检查：坐标、卷积等价与置换

按README安装基础环境后，在仓库根目录把整个代码块保存为`/tmp/spatial_token_check.py`，运行`.venv/bin/python /tmp/spatial_token_check.py`。代码只使用NumPy和PyTorch的小型float64数组，不含随机抽样、模型训练、文件读写或权重下载。前半段检查2通道4×6网格，后半段独立使用2×2网格上的二维内容向量。

<!-- cpu-check: spatial-tokens-and-position -->
```python
import numpy as np
import torch
from torch.nn import functional as F

def patchify(x, p):
    batch, channels, height, width = x.shape
    assert height % p == width % p == 0
    grid = x.reshape(batch, channels, height // p, p, width // p, p)
    return grid.transpose(0, 2, 4, 3, 5, 1).reshape(
        batch, (height // p) * (width // p), p * p * channels)

def unpatchify(tokens, p, channels, height, width):
    batch = tokens.shape[0]
    assert height % p == width % p == 0
    assert tokens.shape[1:] == ((height // p) * (width // p), p*p*channels)
    grid = tokens.reshape(batch, height // p, width // p, p, p, channels)
    return grid.transpose(0, 5, 1, 3, 2, 4).reshape(batch, channels, height, width)

x = np.arange(48, dtype=np.float64).reshape(1, 2, 4, 6)
p, channels, width_hidden = 2, 2, 3
tokens = patchify(x, p)
assert tokens.shape == (1, 6, 8)
np.testing.assert_array_equal(unpatchify(tokens, p, channels, 4, 6), x)
np.testing.assert_array_equal(tokens[0, 0], [0, 24, 1, 25, 6, 30, 7, 31])
np.testing.assert_array_equal(tokens[0, -1], [16, 40, 17, 41, 22, 46, 23, 47])
assert not np.array_equal(tokens, x.reshape(1, 6, 8))

# Convert [out, channel, patch_row, patch_col] to [patch_row, patch_col, channel, out].
kernel = np.arange(24, dtype=np.float64).reshape(width_hidden, channels, p, p) / 24
embedding = kernel.transpose(2, 3, 1, 0).reshape(p*p*channels, width_hidden)
projected = tokens @ embedding
conv = F.conv2d(torch.from_numpy(x), torch.from_numpy(kernel), stride=p)
conv_tokens = conv.numpy().transpose(0, 2, 3, 1).reshape(1, 6, width_hidden)
np.testing.assert_allclose(projected, conv_tokens, rtol=0, atol=1e-12)

# Identity Q/K/V projections; shared parameters, no mask or dropout.
def attention(u):
    logits = u @ u.T / np.sqrt(u.shape[-1])
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ u

u = np.array([[.1, .7], [.3, -.5], [.8, .2], [-.4, .6]])
position = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])  # (row, col)
perm = np.array([1, 0, 3, 2])  # Swap left/right contents within each grid row.
base = attention(u)
with_position = attention(u + position)
np.testing.assert_allclose(attention(u[perm]), base[perm], rtol=0, atol=1e-14)
np.testing.assert_allclose(attention(u[perm] + position[perm]),
                           with_position[perm], rtol=0, atol=1e-14)
fixed_position_error = np.max(np.abs(
    attention(u[perm] + position) - with_position[perm]))
assert fixed_position_error > 1.0
print("Patch/embedding shapes:", tokens.shape, projected.shape)
print("First patch:", tokens[0, 0].astype(int).tolist())
print("Conv2d equivalence max error:", np.max(np.abs(projected - conv_tokens)))
print("Fixed-position content swap max difference:", round(fixed_position_error, 6))
```

解析形状为`(1,6,8)`与`(1,6,3)`；重排往返逐元素相等。共享线性投影与卷积的差异仅来自浮点求和顺序。两个等变性断言检验理论恒等式；保留位置的内容交换是具体反例，其最大差预期约为1.774983。

## 5. 练习与参考答案

1. **输入`B,4,8,12`，patch为2×2。求$`L,m`$，并确定token $`n=7`$内坐标$`k=11`$来自哪个原网格位置。** 提示：逐项反解整数索引。答案：$`H_p=4,W_p=6,L=24,m=16`$；$`n=7`$对应patch $`(i,j)=(1,1)`$，$`k=11`$对应$`(r,s,c)=(1,0,3)`$，所以来自$`z_{b,3,3,2}`$。
2. **把8维patch投影到3维，能否根据“patch重排可逆”保证恢复任意输入？输出头随后给每token预测8个数又说明什么？** 答案：8到3的线性映射秩至多为3，存在至少5维零空间；后续输出头学习目标坐标，无法仅凭线性投影对任意输入建立一一对应。重排的逆只还原这些输出的空间布局。
3. **给定含位置的token，把内容与位置共同重排，同时仍把输出当作原行列顺序unpatchify，会发生什么？** 答案：注意力输出随序列等变，空间恢复也必须使用一致的排列；先用逆置换恢复token顺序，或同步更新位置到网格的映射。若保持原网格规则直接还原，就会把预测放入错误的patch位置。

## 6. 固定来源与核验范围

| 原始来源 | 固定版本与定位 | 本页依据 |
| --- | --- | --- |
| Peebles & Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748v2) | arXiv:2212.09748v2；第4页§3.2 “Patchify”，第5页“Transformer decoder” | 线性patch嵌入、二维正弦位置、预测输出的空间还原及噪声/方差分支 |
| [DiT作者实现：models.py](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py) | commit `ed81ce2229091fd4ecc9a223645f95cf379d582b`；第129–141、149–174、191–198、218–247、274–320行 | 输出通道与预测头、patch投影入口、固定位置初始化、实际NCHW还原及二维正弦函数 |

本文的矩阵索引、卷积等价与置换关系从给定假设独立推导，CPU块检查教学数组。作者`unpatchify`另外以抽取函数和小张量核验输出形状及顺序，无需构造模型。DiT论文身份沿用[图像与视频来源审计](../image-video-source-audit.md)；源码身份、正文运行与独立检查见[核验记录](../../records/measurements/spatial-tokens/README.md)。本页的实际执行范围为所列解析数组与作者重排函数。
