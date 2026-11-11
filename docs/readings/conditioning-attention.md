# 条件怎样进入生成网络：从文本token到单向量图像条件

**先修：** [04研究路线](../chapters/04-image-video-roadmap.md)的潜空间与条件控制、矩阵乘法和softmax；结合[09图像推理](../chapters/09-pretrained-image-lab.md)与[10视频推理](../chapters/10-pretrained-video-lab.md)阅读。**学习目标：** 标出交叉注意力中查询和条件的轴；追踪两种CLIP输出；用单token和置换反例检验条件接口。

本页核对固定论文、模型配置与推理源码，使用小型CPU张量验证代数。既有图像/视频的运行与质量证据沿用对应实验报告，本页没有新增模型推理或训练。

## 1. 查询来自空间状态，键和值来自条件

设某层图像特征展平为$`U\in\mathbb R^{B\times L_q\times d_{\mathrm{img}}}`$，条件编码为$`C\in\mathbb R^{B\times L_c\times d_{\mathrm{ctx}}}`$。$`L_q`$是查询位置数：图像空间层通常为该层的高乘宽，视频时间层则可沿帧轴构造查询；$`L_c`$是送入该层的条件token数。一个头的投影与注意力为

```math
Q=UW_Q,\quad K=CW_K,\quad V=CW_V,\qquad
A=\operatorname{softmax}_{\text{key}}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right),\quad O=AV.
```

$`Q,K`$的每头宽度为$`d_h`$；$`V`$的宽度记为$`d_v`$。这里使用行向量约定，softmax沿条件键轴归一化。$`M`$是加性mask：允许位置为0，屏蔽位置为$`-\infty`$；每个查询至少保留一个有效键。

| 张量 | 单头形状 | 操作含义 |
| --- | --- | --- |
| $`Q`$ | $`B,L_q,d_h`$ | 当前图像/视频特征给出查询 |
| $`K`$ | $`B,L_c,d_h`$ | 条件编码映射成匹配所用的键 |
| $`V`$ | $`B,L_c,d_v`$ | 条件编码映射成被汇聚的内容 |
| $`A`$ | $`B,L_q,L_c`$ | 每个查询对条件位置的归一化权重 |
| $`O`$ | $`B,L_q,d_v`$ | 各位置得到加权条件内容，保留查询位置数 |

多头实现先独立计算，再拼接各头并投影回网络宽度。LDM作者实现将这一步放进带残差的Transformer块：空间自注意力、条件交叉注意力和前馈层依次更新特征。去噪网络还同时接收扩散时间，条件接口与时间接口承担不同作用。

例如SD1.4在64×64空间尺度、77个条件位置时，每头权重形状为$`4096\times77`$，共315,392个元素。密集空间自注意力则为$`4096\times4096`$。这些是形状计数；实际SDPA算子可以融合计算，运行内存和延迟须按完整实现测量。

## 2. 同样使用CLIP，实际取出的表示不同

LDM原论文以$`\tau_\theta`$描述通用条件编码框架；下面分别核对后续发布的SD1.4冻结CLIP实例与SVD接口。CLIP编码器可以提供序列隐藏态，也可以经池化与投影得到全局向量，复现时沿管线查找实际读取的字段及后处理。

| 固定系统 | 实际编码器输出 | 去噪器收到的条件 | 另一条条件路径 |
| --- | --- | --- | --- |
| 09的SD1.4 | `CLIPTextModel`的`last_hidden_state`，经最终LayerNorm的每位置隐藏态 | 单提示为$`B,77,768`$；CFG把空文本与提示合成batch | 本章文本生成使用随机潜变量起点 |
| 10的SVD | `CLIPVisionModelWithProjection.image_embeds`，池化视觉表示经线性投影 | $`B,1,1024`$；进入U-Net后按帧扩展batch，条件长度保持1 | 参考图经VAE编码，沿帧复制，再与带噪视频latent沿通道拼接 |

SD1.4的77个位置包含BPE片段、起止特殊token与padding。文本编码器加入位置嵌入，并使用因果自注意力形成上下文化表示：一个位置的向量已经包含允许读取的前文信息。当前固定管线使用默认`clip_skip=None`，取最终序列隐藏态；这些位置在去噪网络中共同构成条件序列。该配置没有向U-Net传入文本padding mask，任意删除尾部位置会改变它的输入与注意力归一化。

原作者CLIP的图文相似度接口另有清楚的路径：选择文本结束位置的隐藏态，乘`text_projection`得到全局文本向量；图像也得到一个全局向量，随后在相似度计算前分别做L2归一化。09的去噪条件使用上表的序列隐藏态，因此复现时要同时检查字段、轴和投影步骤。

SVD的视觉编码器输入为224×224、patch为14×14，编码器内部有$`16\times16+1=257`$个位置。实际`image_embeds`取池化CLS表示并投影：隐藏宽度1280变为1024，再由管线增加长度为1的条件轴。这条SVD路径没有额外执行L2归一化。编码器内部的257个位置与去噪器收到的1个条件token分别记录。

空条件也按具体模型核对：09把空字符串经过文本编码器，仍产生77个位置；10的CFG把全局图像向量与参考latent分别置零。输入约定与训练方式共同定义空条件含义。

## 3. 一个条件token时，注意力会做什么？

对一个有效条件token，任意查询$`i`$只有一个有限logit $`\ell_i`$，于是

```math
A_{i1}=\frac{e^{\ell_i}}{e^{\ell_i}}=1,\qquad O_i=V_1.
```

在固定$`V`$、关闭dropout的attention core中，所有查询获得相同的值向量；对独立变量$`Q,K`$的偏导均为0。改变这个条件的$`V`$仍会改变输出。SVD管线传入的单向量条件满足长度1这一接口；固定`AttnProcessor2_0`在SDPA调用中设置`dropout_p=0.0`和`is_causal=False`。

这帮助定位两条图像条件路径：全局CLIP向量通过值投影提供全局信息；VAE参考latent保留显式空间网格，通过通道拼接进入生成网络。全局向量本身也可能编码布局信息，单token的形状不说明它只包含某一种语义。

上面的等式限定在注意力汇聚这一步。外层残差保留各位置原有特征，前后的卷积、自注意力、归一化和非线性继续处理它们；整网的条件响应仍可随空间位置与去噪状态变化。单token的权重图始终为1，所以仅查看这张图无法比较两张参考图所携带的值向量及其后续影响。

## 4. 联合置换的条件与提示词顺序

对**已经编码好并固定的**$`Q,K,V`$，设$`P`$是条件位置的置换矩阵。若同时使用

```math
K'=PK,\qquad V'=PV,\qquad M'_{:,j}=M_{:,\pi(j)},
```

其中$`\pi`$对应$`P`$的行置换，mask直接按列索引重排。logits和权重只在键轴重排：$`A'=AP^\top`$，因此

```math
O'=A'V'=AP^\top PV=AV=O.
```

这可以检查代码中的轴和mask是否同步。若只重排$`V`$，或沿原顺序保留mask，通常会改变汇聚结果。

原始文本的词序变化发生在编码器之前。位置嵌入与因果上下文会随新序列重新计算，得到的$`K,V`$通常不只是旧张量的行置换。例如“红色方块在蓝色圆盘左侧”与调换两个对象的提示，需要先重新编码，再研究生成结果。类似地，在注意力层内使用相对位置偏置或其他位置相关项时，也必须把这些项纳入上述变换条件。

## 5. CPU检查：形状、置换与单token梯度

下面只创建小张量，基础环境中的PyTorch即可执行。手写矩阵公式与PyTorch SDPA交叉核对；双key的一维特例另外使用$`\operatorname{sigmoid}(2x)`$及其导数$`2p(1-p)`$作解析参照。

<!-- cpu-check: conditioning-attention -->
```python
import math
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
dtype = torch.float64

def attention(q, k, v, mask=None):
    logits = q @ k.T / math.sqrt(q.shape[-1])
    if mask is not None:
        logits = logits + mask
    weights = logits.softmax(dim=-1)
    return weights @ v, weights

q = torch.tensor([[1., 0.], [0., 1.]], dtype=dtype)
k = torch.tensor([[1., 0.], [0., 1.], [-1., .5]], dtype=dtype)
v = torch.tensor([[2., 0.], [0., 4.], [-1., 2.]], dtype=dtype)
mask = torch.tensor([[0., 0., -math.inf], [0., -math.inf, 0.]], dtype=dtype)
out, weights = attention(q, k, v, mask)
torch.testing.assert_close(weights.sum(-1), torch.ones(2, dtype=dtype))
actual = F.scaled_dot_product_attention(q[None,None], k[None,None], v[None,None],
                                      attn_mask=mask, dropout_p=0., is_causal=False)[0,0]
torch.testing.assert_close(out, actual, atol=1e-14, rtol=1e-14)
perm = torch.tensor([2, 0, 1])
permuted, _ = attention(q, k[perm], v[perm], mask[:,perm])
torch.testing.assert_close(out, permuted, atol=1e-14, rtol=1e-14)
wrong_mask, _ = attention(q, k[perm], v[perm], mask)
assert not torch.allclose(out, wrong_mask)

single_q = q.clone().requires_grad_()
single_k = torch.tensor([[.3, -.2]], dtype=dtype, requires_grad=True)
single_v = torch.tensor([[3., 4.]], dtype=dtype)
single, single_weights = attention(single_q, single_k, single_v)
torch.testing.assert_close(single, single_v.expand_as(single), atol=0, rtol=0)
torch.testing.assert_close(single_weights, torch.ones_like(single_weights), atol=0, rtol=0)
grad_q, grad_k = torch.autograd.grad(single.sum(), (single_q, single_k))
assert torch.count_nonzero(grad_q) == torch.count_nonzero(grad_k) == 0
shifted, _ = attention(single_q.detach() + 1., single_k.detach() - 2., single_v)
torch.testing.assert_close(shifted, single, atol=0, rtol=0)

# With two keys (-1, +1) and values (0, 1), output is sigmoid(2*x).
x = torch.tensor([[.4]], dtype=dtype, requires_grad=True)
k2 = torch.tensor([[-1.], [1.]], dtype=dtype)
v2 = torch.tensor([[0.], [1.]], dtype=dtype)
y, _ = attention(x, k2, v2)
p = torch.sigmoid(2*x)
torch.testing.assert_close(y, p, atol=1e-14, rtol=1e-14)
grad = torch.autograd.grad(y.sum(), x)[0]
expected_grad = 2*p*(1-p)
h = 1e-6
finite_diff = (attention(x.detach()+h, k2, v2)[0] -
               attention(x.detach()-h, k2, v2)[0])/(2*h)
torch.testing.assert_close(grad, expected_grad, atol=1e-14, rtol=1e-14)
torch.testing.assert_close(finite_diff, expected_grad, atol=1e-10, rtol=1e-10)
print({'weight_shape': tuple(weights.shape), 'output_shape': tuple(out.shape),
       'SDPA_max_error': float((out-actual).abs().max()),
       'joint_KV_mask_permutation_error': float((out-permuted).abs().max()),
       'unaligned_mask_difference': float((out-wrong_mask).abs().max()),
       'single_token_Q_K_gradients': [float(grad_q.abs().max()), float(grad_k.abs().max())],
       'two_key_analytic_output': float(p.detach()),
       'two_key_finite_difference_error': float((finite_diff-expected_grad.detach()).abs().max())})
```

本次float64实际运行：权重形状为`2×3`、输出为`2×2`，与SDPA的最大差`2.78e-17`；KV与mask联合置换差为0，漏改mask时差为2。单token的Q/K梯度均为0。双key解析输出为`0.68997448`，中心有限差分与解析导数的差约`3.79e-11`。这些检查验证给定条件下的代数与张量操作。

## 6. 练习与参考答案

1. 图像特征有1024个位置，条件有77个位置，每头值宽度64。注意力权重与汇聚输出各是什么形状？把条件改成1个全局向量会改变哪一轴？

   <details><summary>答案</summary>

   省略batch后，权重为`1024×77`，输出为`1024×64`。改成一个条件token后，权重为`1024×1`，输出仍为`1024×64`；在本文假设下，各查询得到同一值向量，查询位置轴保留。

   </details>

2. SVD单token权重全为1，能否据此判定模型忽略参考图？

   <details><summary>答案</summary>

   先检查值向量：它来自参考图，可以改变每层的条件输出；再检查参考VAE latent的空间拼接路径。若要测量某条条件路径的作用，需要固定其余输入后做对应干预，并跟踪后续网络输出与完整视频。单token权重本身只记录唯一条件位置的归一化选择。

   </details>

3. 在已编码条件上同步重排KV得到相同输出，原提示词是否可任意重排？

   <details><summary>答案</summary>

   编码后联合置换保留每个向量已有的上下文信息，只改变存放顺序。原词序变化会触发位置嵌入和因果编码的重新计算，通常产生另一组向量；二者对应不同操作。mask或位置偏置也应随对应键同步处理。

   </details>

## 7. 固定来源与实际核验

核验日期2026-11-10。[来源身份与独立小数组记录](../../records/measurements/conditioning-attention/README.md)随仓库保存。LDM PDF重新计算SHA与既有审计一致；两份作者源码实际读取，五份固定库源码下载后与本地安装逐字节一致。仅核对所列接口、解析性质与小张量，未载入完整模型权重。

| 来源 | 固定身份、定位与范围 |
| --- | --- |
| LDM论文 | [2112.10752v2](https://arxiv.org/pdf/2112.10752v2)，第4页§3.3的交叉注意力、Q/K/V与条件编码，第5页式(3)的条件目标；45页、40,842,586字节，SHA-256 `46ede043a8dc07ca1f0f445620523fe1ad8b2436bd83856a3835612a47e9f79e`。 |
| LDM作者实现 | [`CompVis/latent-diffusion` a506df5756472e2ebaf9078affdde2c4f1502cd4 的 attention.py](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/attention.py)：`CrossAttention`、`BasicTransformerBlock`、`SpatialTransformer`；8,531字节，SHA `012198f441427ef892da0b6efc5ea9fae8e414aefcc65504577b41c2afd3526e`。 |
| CLIP作者实现 | [`openai/CLIP` cfcffb90e69f37bf2ff1e988237a0fbe41f33c04 的 model.py](https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py)：`encode_text`取结束位置并投影，`forward`归一化两分支后计算相似度；17,242字节，SHA `a8dd35d7ba5877e61fd4de3379ee0cd8a8c3962c50403c2949ad0961d5a02793`。 |
| 固定SD1.4/SVD资产 | SD1.4 revision `133a221b8aa7292a167afc5127cb63fb5005638b`、SVD revision `9cf024d5bfa8f56622af86c884f26a52f6676f2e`；编码器/U-Net配置的逐文件SHA见[SD1.4清单](../../examples/pretrained-sd14/manifest.json)与[SVD清单](../../examples/pretrained-svd/manifest.json)。 |
| Diffusers0.35.2 | commit `b71269675ec1b85193107a691dd35c308e46f0a5`：[SD管线](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py)的`encode_prompt`与U-Net调用；[SVD管线](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py)的`_encode_image`、`_encode_vae_image`与通道拼接；[AttnProcessor2_0](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/models/attention_processor.py)及[时空条件广播](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/models/transformers/transformer_temporal.py)。 |
| Transformers4.57.6 | commit `753d61104116eefc8ffc977327b441ee0c8d599f`：[modeling_clip.py](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/clip/modeling_clip.py)中的文本位置/因果mask、`last_hidden_state`、池化与`CLIPVisionModelWithProjection`；SHA `3e779a99827618d6e6d57583ec4f5c0d765321dfa5054a4ffca6a56bb004d499`。 |

本轮与远端固定commit逐字节相同的Diffusers源码SHA-256：

```text
pipeline_stable_diffusion.py        7466201901e68667164344a9fb58596d9d253e524c9addad38494ee6bc1f7e08
pipeline_stable_video_diffusion.py  c7a8c4afc7944401a1e79aba21ecb5a48221c5b584cf10f43509b3bd8fd577e9
attention_processor.py              d56adb29122171c897568ae0235dba6c5409b9a4176de6196d563c6f56c9df90
transformer_temporal.py             18c9f95946d6597ed9bf2a9584ed7683a6c80fb767acc9dd62b0cdf946442344
```
