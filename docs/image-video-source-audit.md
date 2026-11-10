# 潜扩散、CFG 与 DiT：原始来源核验

核验日期：2026-11-10。对应 [04 研究路线](chapters/04-image-video-roadmap.md) 的表示学习、条件引导与空间 token 三组主张。本轮对照固定论文 PDF，并重新读取 SVD/CogVideoX 的固定官方 README；后两者的论文训练细节继续列入后续核验。实验成本与质量数值在下文标明为原作者报告。

## 1. 固定版本与文件身份

版本日期取自 arXiv submission history，文件大小、页数和 SHA-256 来自本次实际下载；页码采用 PDF 页序，以下定位与印刷页码一致。

| 来源 | 固定版本与提交日期（UTC） | PDF 页数 | 下载字节数 |
| --- | --- | ---: | ---: |
| Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* | [2112.10752v2](https://arxiv.org/abs/2112.10752v2)，2022-04-13；CVPR 2022 | 45 | 40842586 |
| Ho & Salimans, *Classifier-Free Diffusion Guidance* | [2207.12598v1](https://arxiv.org/abs/2207.12598v1)，2022-07-26 | 14 | 3773509 |
| Peebles & Xie, *Scalable Diffusion Models with Transformers* | [2212.09748v2](https://arxiv.org/abs/2212.09748v2)，2023-03-02；ICCV 2023 | 25 | 43824062 |

```text
2112.10752v2.pdf  46ede043a8dc07ca1f0f445620523fe1ad8b2436bd83856a3835612a47e9f79e
2207.12598v1.pdf  ccbdf2da65903ed579d35240165aba2d0e9ad6f015ade81af6e30823eb0a7f24
2212.09748v2.pdf  5fbb0ec35e3ae76826240171ca63ed4c27de6cb43180367453019e9d28a8b076
```

原始文件入口：[LDM PDF](https://arxiv.org/pdf/2112.10752v2)、[CFG PDF](https://arxiv.org/pdf/2207.12598v1)、[DiT PDF](https://arxiv.org/pdf/2212.09748v2)。可用 `sha256sum`、`pdfinfo` 与 `pdftotext -layout` 复核。本轮临时文件在 `/tmp/genai-image-video-audit/`；仓库保存来源索引和原创解释。

## 2. 主张到证据的对应

| 课程主张 | 原文位置 | 核对结果与适用条件 |
| --- | --- | --- |
| LDM 将压缩学习与生成学习分为两个阶段 | LDM，第3–4页 §3、§3.1–3.2；DiT，第4页 “Latent diffusion models” | 先学习编码器/解码器，再在表示上训练扩散；DiT 明确在第二阶段冻结编码器，生成表示后用解码器还原图像。 |
| LDM 自编码器含感知与对抗重建目标 | LDM，第3页 §3.1；第4页续文 | 原文为 perceptual loss 与 patch-based adversarial objective，并比较弱 KL 正则和 VQ 正则。具体潜变量尺度、编码输出与正则选择仍按对应模型配置确定。 |
| 潜扩散的噪声回归作用在潜表示上 | LDM，第4页式(2)、第5页式(3) | 无条件/有条件目标均在带噪潜变量上预测噪声；式(3)加入条件编码器与条件输入。 |
| 压缩率同时影响后续成本和重建信息 | LDM，第5页 §4.1、Figures 6–7 | 原作者扫描 $f\in\{1,2,4,8,16,32\}$；高压缩可能丢失细节，低压缩增加生成计算。这一实验支持设计权衡，具体速度需对目标实现测量。 |
| CFG 的空条件分支需要训练 | CFG，第4页 Algorithm 1、§3.2；第5页续文 | 以概率 $p_{\mathrm{uncond}}$ 将条件替换为 $\varnothing$，共享同一网络联合训练有/无条件预测；Algorithm 2 再组合两次预测。 |
| CFG 两种系数约定相差1 | CFG，第5页式(6)、Algorithm 2 | 原文为 $(1+w)\epsilon_c-w\epsilon_u$；本仓库为 $\epsilon_u+s(\epsilon_c-\epsilon_u)$，故 $s=1+w$。两分支应使用相同带噪输入、噪声级别与预测量。 |
| 更强引导改变多样性与评价表现 | CFG，第6页 Figure 3；第7页 Table 1、Figure 4 | 作者图示报告饱和颜色、正文报告样本多样性下降；表1显示 FID 的最佳强度与 IS 的最佳强度不同。原文中一处趋势描述与表格不一致，见下节。 |
| DiT 在二维潜表示上划分 patch | DiT，第3页 Figure 3；第4页 Figure 4、§3.2 “Patchify” | $256\times256\times3$ 图像对应 $32\times32\times4$ 潜变量；patch 尺寸 $p$ 对应 token 数 $(32/p)^2$。 |
| Patch 数与整个模型成本需分别计算 | DiT，第4页 “Patchify”；第5页 Table 1 | 论文说明减半 $p$ 会使 token 数变为4倍，Transformer Gflops 至少变为4倍；总 Gflops 也依赖隐藏维度和层数。密集注意力的 $L^2$ 元素计数见独立推导。 |

04 章本轮补齐 CFG 条件丢弃的训练前提、明确 latent patch 坐标，并更新原文证据入口。

## 3. 读原始表格时发现的趋势记载差异

CFG v1 第7页 §4.1 中有一句将 FID 随引导增强的变化描述为 “monotonically decreasing”。同页 Table 1 的 $p_{\mathrm{uncond}}=0.1$ 列则给出：$w=0$ 时 FID=1.80，$w=0.1$ 时1.55，$w=0.2$ 时2.04，$w=4$ 时26.22；Figure 4 与表格方向一致。该列 IS 从53.71提高到260.20。

本课程据表格与图说明：这一 ImageNet 64×64 配置在较小正引导处取得较低 FID，随后随引导增强而变差，同时 IS 提高。原作者用每组50,000张样本评价；这些数值属于作者报告。本轮保存正文与表格的差异，便于读者复核原始结论。

## 4. 独立尺寸与代数检查

**系数展开。** 对任意同形张量 $\epsilon_c,\epsilon_u$，

$$
\epsilon_u+(1+w)(\epsilon_c-\epsilon_u)
=(1+w)\epsilon_c-w\epsilon_u.
$$

$w=0$ 对应 $s=1$，$w=4$ 对应 $s=5$。本轮以 Python 浮点向量对 $w\in\{0,0.1,4\}$ 逐项核对，两式最大绝对差低于 $10^{-14}$。

**Token 与矩阵大小。** 对固定 $32\times32$ 潜空间，$p=4$ 得64个 token，$p=2$ 得256个 token。每头密集注意力矩阵分别有4096和65536个元素，比例为16。若保持 $p=2$ 而把图像宽高各加倍，潜空间变为 $64\times64$，token 数增至1024，矩阵元素数再次增为16倍。这些是尺寸恒等式；总运算量和峰值显存还取决于网络与算子实现。

**压缩数量。** 对RGB图像、高宽各压缩8倍、4通道潜变量，标量数比例为 $3HW/[4(H/8)(W/8)]=48$。它描述表示大小；延迟需包含去噪网络、解码器和硬件利用率。

## 5. 视频公开接口：固定官方 README 复核

| 来源 | 固定位置 | 本轮核对内容 |
| --- | --- | --- |
| Stability AI `generative-models` | [e8cd657… README，第107–116行](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/README.md#L107) | 2023-11-21 发布说明将 SVD 标为 image-to-video；SVD 接受同尺寸参考帧，输出14帧，SVD-XT 对25帧微调，并给出对应推理脚本。 |
| THUDM `CogVideo` | [7a1af71… README，第41–57行](https://github.com/THUDM/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/README.md#L41) | 2024-08-06 发布3D Causal VAE与CogVideoX-2B；2024-09-19单独发布CogVideoX-5B-I2V，结合图像与文本提示生成视频。 |

本轮下载的 README SHA-256：

```text
SVD README       06c52638c03fb2a966cdbb1de6946b3dc2f1fcef47faed6e583d906f6a10d557
CogVideo README  0aaad019b64e906c54fff8359d655da4b3f6d67a0262d2d82fe34dcbe419e6f8
```

这里核验公开接口、命名与发布顺序。SVD/CogVideoX 的训练论文细节、许可逐权重核查和本机推理测量继续作为独立任务；Wan 的视频压缩与 token 推导另见 [2025 阅读卡](readings/wan-video-2025.md)。
