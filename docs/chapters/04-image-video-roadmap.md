# 04 · 从二维实验走向图像与视频生成

**先修与依赖：** [02 扩散](02-diffusion.md)、[03 Flow matching](03-flow-matching.md)；沿完整路线完成[05评价](05-evaluation.md)、[07图像生成](07-image-flow-lab.md)、[08条件引导](08-image-guidance-lab.md)和[06视频时序反例](06-video-temporal-lab.md)，再设计研究实验。**目标：** 解释从 toy 分布到图像/视频增加了哪些问题；辨别模型、训练方法和采样器；为一个可验证的研究问题设计实验。

**当前范围：** 这是一份以 **2015—2024 年代表工作**为锚点的研究入门路线，连接2025–2026专题阅读卡。可执行路径包括 [07/08的小图像生成与引导](08-image-guidance-lab.md)、[09的预训练SD1.4图像推理](09-pretrained-image-lab.md)，[06的合成视频时序诊断](06-video-temporal-lab.md)，以及[10的SVD视频运行与失败分析](10-pretrained-video-lab.md)。发布尺寸的视频质量基线继续按资源和评价协议推进。

## 1. 高维生成多了什么

一张 $`H\times W`$、三通道图像有 $`3HW`$ 个标量。数据具有局部相关性、长程结构、对象语义和多尺度变化。视频还需要建模跨帧联合分布。对同一文本独立生成每帧，只是在采样 $`\prod_f p(x_f\mid c)`$；它通常无法表达目标的 $`p(x_1,\ldots,x_F\mid c)`$ 中的身份、运动与遮挡依赖。

两个重要后果：首先，网络架构与表示的归纳偏置很重要；其次，训练损失和人类评价之间的差距更明显。“去噪误差下降”不能自动推断手部解剖、物体持久性或物理运动正确。

## 2. 潜空间：先压缩，再生成

Latent Diffusion Models（LDM）先学习图像编码器 $`E`$ 与解码器 $`D`$，再在 $`z=E(x)`$ 的表示中训练生成模型。推理时先生成 $`z`$，再解码为图像。

```mermaid
flowchart LR
  accTitle: 潜空间训练与生成
  accDescr: 训练图像经编码器成为潜变量，用于训练扩散或流模型；生成时从噪声得到潜变量，再经解码器得到图像。
  A[图像 x] --> B[编码器 E]
  B --> C[潜变量 z]
  C --> D[在潜空间训练扩散或流]
  N[噪声] --> G[潜空间生成过程]
  G --> H[生成潜变量]
  H --> I[解码器 D]
  I --> J[生成图像]
```

训练和生成是两条流程；生成阶段没有一张待编码的目标图像。压缩可以降低后续生成模型的空间成本，却可能损失细节。即使潜空间生成完美，也无法自动修复表示阶段已经丢掉的信息。

**要核对的三个假设：** 潜变量是否缩放、使用何种编码输出（均值或采样）、训练与推理是否用相同的自编码器。模型文件名相近不保证潜空间兼容；随意替换 VAE 可能改变尺度或解码质量。

LDM 的自编码器同时使用感知损失和基于局部图块的对抗目标；论文分别研究弱 KL 正则与 VQ 正则。训练潜扩散时复用已经训练好的表示，按具体配置核对编码器、解码器与损失。固定原文的定位见 [图像与视频来源核验](../image-video-source-audit.md)。

完成09章的环境准备后，可用[固定VAE往返实验](../vae-reconstruction-report.md)单独观察表示阶段：同一图像编码后比较后验均值、三个采样种子和缩放接口，保留全部原始数组与显示图。练习将重建误差、解码器的最优latent问题和生成分布评价分别定义。

## 3. 条件控制与引导

文本到图像系统通常包含文本表示模型和带条件的生成网络。交叉注意力用当前空间特征查询已编码条件；条件编码器、tokenization、训练数据和损失共同影响文本遵循。配套[条件接口专题](../readings/conditioning-attention.md)从Q/K/V形状推导汇聚过程，对照SD1.4与SVD的CLIP输出，再用单token与mask置换反例检查接口。

Classifier-free guidance（CFG）常把无条件与有条件噪声预测组合为

```math
\hat\epsilon=\epsilon_{\mathrm{uncond}}
+s(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}).
```

这里两次预测使用**同一个带噪状态、噪声级别和预测参数化**。原始 CFG 训练以概率 $`p_{\mathrm{uncond}}`$ 随机丢弃条件，用空条件标识 $`\varnothing`$ 联合训练无条件分支；推理时再组合两种输出。这个训练步骤使空条件输入具有明确含义。

$`s=1`$ 得到有条件预测，$`s=0`$ 得到无条件预测。CFG 原论文式(6)使用 $`(1+w)\epsilon_{\mathrm{cond}}-w\epsilon_{\mathrm{uncond}}`$，与本式对应 $`s=1+w`$；报告 guidance 值必须说明约定。实际系统若把基准分支换成负向提示，应将其记录为第二个条件，连同空文本约定一并固定。

引导增强条件方向，效果需要沿强度扫描。CFG 原论文的 ImageNet 实验中，更强引导减少样本多样性，图3展示过饱和颜色；表1的 FID 在低强度处较好，继续增加引导后上升。两分支通常各需一次网络预测（可以组成 batch），成本要实测。迁移到其它预测参数化或引导方法时，需重新核对系数与实现。

可运行的配套任务见 [08 条件丢弃与小图像引导](08-image-guidance-lab.md)：在CPU训练空条件分支，并固定模型/噪声比较速度引导强度。该实验沿用flow的预测量与时间方向，同时保存两分支计算成本、几何变化和原始像素诊断。

## 4. 从 U-Net 到 Diffusion Transformer

U-Net 利用多尺度特征与跳连，Transformer 可以把图像或潜变量划分为 token 后建模交互。DiT 的关键对象是在潜空间工作的扩散 Transformer；“Transformer”描述网络，“扩散”描述生成建模框架，采样仍沿扩散反向过程进行。

DiT 论文将 $`256\times256\times3`$ 图像编码为 $`32\times32\times4`$ 潜变量，再按 $`p\times p`$ 的潜空间 patch 划分，得到 $`L=(32/p)^2`$ 个 token。$`p=4`$ 时 $`L=64`$，$`p=2`$ 时 $`L=256`$；patch 尺寸按潜空间坐标计算。

每个注意力头的密集注意力矩阵有 $`L^2`$ 个元素，所以这里将 $`p`$ 从4减到2会使该矩阵元素数增加16倍。实际总成本还包括线性投影、MLP、网络宽度、层数与算子实现；算子融合和重计算也会改变峰值显存。比较延迟和内存时应固定这些条件并实际测量。

## 5. 视频：表示、时间与条件一起改变

视频模型需要处理 $`F`$ 帧、空间尺寸、帧率和持续时长。相同帧数在不同 fps 下表示不同时间跨度；慢动作与正常速度也不应混为一谈。报告输出至少写明：帧数、fps、宽高、采样步数、种子、条件类型和权重版本。

常见设计包括：时空注意力、空间/时间分解、视频潜空间压缩、图像模型向视频扩展，以及对时间条件的显式编码。实际系统常组合其中多种设计。

- **Video Diffusion Models（2022）**：建立联合图像/视频扩散建模的代表性出发点，适合理解时间维如何进入去噪网络。
- **Stable Video Diffusion（2023 预印本）**：阅读数据筛选与分阶段训练。固定官方 README 的 SVD/SVD-XT 发布面向**图像到视频**，输入契约应从对应权重与推理入口核对。
- **CogVideoX（2024 首发）**：阅读 3D 视频压缩与 Transformer 设计，核对具体模型版本支持的条件、分辨率和时长。后续仓库版本可能扩展能力，不应回写成 2024 首发论文的结论。

配套 [SVD与CogVideoX训练阅读卡](../readings/video-training-svd-cogvideox.md) 已核验固定全文与发布配置：SVD的预条件去噪、分阶段训练和帧轴引导，CogVideoX的时间压缩、联合注意力和VP目标，均附可执行CPU检查。论文报告与具体权重的帧数、精度、结构及许可差异集中记录。

**研究反例。** 每帧都很清晰，但角色服装不断变化；单帧图像质量指标可能很好，视频整体仍失败。平均帧级指标不能代替时间一致性检查。先用 [06 视频时序实验](06-video-temporal-lab.md) 检验更简单、可精确控制的情形：保持一组光点帧完全相同，只打乱顺序，即可改变运动。静止对照进一步说明“相邻帧变化越小越好”可能奖励不动的输出。

## 6. 评价协议先于排行榜

| 要回答的问题 | 可采用的证据 | 不能据此自动推出的结论 |
| --- | --- | --- |
| 图像特征分布是否接近参考集 | 固定实现、样本数与预处理的 FID 等指标 | 文本遵循、训练数据未泄漏、所有细粒度模式均被覆盖 |
| 条件是否被表达 | 明确提示集合、人工规则或可信评价器，逐项统计属性/关系 | 评价器无偏或复杂组合全部正确 |
| 时间是否一致 | 跨帧身份、运动、遮挡测试；视频特征指标与人工评价 | 物理规律普遍正确 |
| 生成是否更快 | 同硬件、同精度、同分辨率/帧数、明确 warm-up 的延迟与内存测量 | 质量相同，或更换硬件后仍相同排序 |
| 是否改善多样性 | 多种子结果、覆盖率与近邻检查 | 没有记忆或未授权内容问题 |

FID 的常用形式是在固定特征空间中，用均值和协方差拟合两个高斯，再计算它们之间的**平方二阶 Wasserstein 距离**（通常称 FID）：

```math
\|\mu_r-\mu_g\|_2^2+
\operatorname{Tr}\!\left(\Sigma_r+\Sigma_g
-2(\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2})^{1/2}\right).
```

这是固定特征空间中的高斯近似比较。这里使用平方距离 $`W_2^2`$：若所有特征一起缩放为原来的 $`a`$ 倍，FID 会乘以 $`a^2`$，因此特征定义和尺度必须固定。有限样本估计存在偏差；实现的 resize、特征权重和数值处理也会影响结果。视频特征指标同样依赖编码器、采样与裁剪策略。比较论文数值时须匹配样本量与完整协议，详见 [05 章](05-evaluation.md) 及其来源核验。

## 7. 从阅读到一个可复现实验

### 已提供：无需模型的视频评价反例

完成README的安装后，在仓库根目录首次运行以下命令。若已按学习路线或06章生成`artifacts/video-counterexample`，直接复用结果；重做时将`--output`改为新目录。

```bash
python -m genai_study.video_diagnostics --output artifacts/video-counterexample
```

查看 `contact_sheet.png` 中移动、乱序、静止三组光点序列，并按 [06 章](06-video-temporal-lab.md) 解释 `metrics.json` 的指标定义与单位。原始帧同时保存在 `frames.npz`。预先预测哪些统计对帧顺序不敏感、哪些诊断会变化，再与本次输出核对。这个确定构造的反例可帮助选择真实视频任务的时序指标。

### 已提供起点：预训练图像模型的受控实验

[09章](09-pretrained-image-lab.md)已提供固定SD1.4权重的CPU推理与配对CFG操作。沿下面协议扩展提示和种子集合：

1. 固定代码提交、权重 revision、依赖、硬件、精度、提示集合、种子集合和基础采样器。
2. 先完成一个最低配置的样本，记录实际时间和峰值内存；据此设置总成本上限。
3. 只改变 guidance 系数，保持其它配置一致。保存完整样本集合，不挑选最佳图。
4. 为属性遵循、伪影、多样性定义人工评分规则，明确样本量与评估者局限。
5. 报告各组分布、失败案例和成本；若结果不足以判定，就写“证据不足”。

再把实验迁移到图像到视频时，增加参考图像版本、帧率、时长和时间一致性规则，并重新测量成本。不要仅因为方法名称相近就假定推理参数可互换。

### 可深入的五条研究线

| 方向 | 可起步的问题 | 第一项必须控制的变量 |
| --- | --- | --- |
| 表示学习 | 压缩率如何影响生成细节与成本？ | 固定编解码器、重建协议与潜变量尺度 |
| 采样与求解器 | 更少网络评估次数能否保持质量？ | 同模型、相同条件与质量口径 |
| 条件与可控生成 | 组合属性和空间关系为何失败？ | 提示分布、条件编码及评价器偏差 |
| 视频时空建模 | 哪些机制改善身份与运动一致性？ | 帧率、时长、参考帧与数据筛选 |
| 评价与数据 | 指标改善能否被人工偏好复核？ | 样本量、特征实现与训练数据重叠 |

## 8. 练习与提示

**A · CFG 约定。** 一篇论文使用 $`w=4`$ 的 $`(1+w)\epsilon_c-w\epsilon_u`$，代码使用本章公式。应该填多少 $`s`$？

<details><summary>答案</summary>

$`s=5`$。展开两式后，有条件项系数均为5，无条件项系数均为−4。

</details>

**B · 压缩成本。** 图像的高和宽各压缩 8 倍，潜变量通道数为 4，而原图通道为 3。标量数减少多少倍？是否等于端到端加速？

<details><summary>答案</summary>

原图 $`3HW`$，潜变量 $`4HW/64`$，标量数减少 48 倍。端到端耗时还取决于网络宽度、注意力、解码器、数据类型与硬件效率，需要独立测量。

</details>

**C · 独立帧反例。** 构造两个视频分布，使它们每帧的边缘分布相同，但跨帧联合分布不同。

<details><summary>答案</summary>

两帧都只取黑或白。分布 A 以相同概率生成“黑黑”或“白白”；分布 B 以相同概率生成“黑白”或“白黑”。每帧均为黑白各半，时间关系却完全相反。只看单帧边缘分布无法区分。

</details>

**D · 设计反证。** 模型 A 的平均帧级图像指标优于 B，但你怀疑其运动更不一致。给出一个小规模验证方案。

<details><summary>提示</summary>

固定参考图像、条件、时长、fps 与种子；预先规定身份变化、物体消失和运动跳变的判定标准，保存整段视频并盲评。报告计数与不确定性，不能只挑一个失败例子就断言总体排序。

</details>

## 原始论文、实现与版本边界

| 主题 | 原始来源 | 阅读定位 / 版本 |
| --- | --- | --- |
| U-Net | Ronneberger et al., [U-Net](https://arxiv.org/abs/1505.04597) | MICCAI 2015；原任务是医学分割，不能直接等同于现代扩散 U-Net |
| 潜扩散 | Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)；[作者实现](https://github.com/CompVis/latent-diffusion) | CVPR 2022；§3 的自编码器、潜扩散及条件化 |
| CFG | Ho & Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 2022 预印本；核对引导系数定义 |
| DiT | Peebles & Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)；[作者实现](https://github.com/facebookresearch/DiT) | ICCV 2023；网络结构、规模实验与潜空间设置 |
| 视频扩散 | Ho et al., [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | NeurIPS 2022；联合图像/视频训练与时空架构 |
| SVD | Blattmann et al., [Stable Video Diffusion](https://arxiv.org/abs/2311.15127)；[官方实现](https://github.com/Stability-AI/generative-models) | 2023 首发预印本；区分论文训练研究与公开图像到视频权重 |
| CogVideoX | Yang et al., [CogVideoX](https://arxiv.org/abs/2408.06072)；[官方实现](https://github.com/THUDM/CogVideo) | 2024 首发预印本；后续仓库包含更新版本，运行时需固定 revision |
| FID | Heusel et al., [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) | NeurIPS 2017；FID 定义与评价实验 |
| FID 实现敏感性 | Parmar et al., [On Aliased Resizing and Surprising Subtleties in GAN Evaluation](https://arxiv.org/abs/2104.11222)；[作者实现](https://github.com/GaParmar/clean-fid) | CVPR 2022；resize、压缩与实现差异 |
| 视频特征评价 | Unterthiner et al., [Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/abs/1812.01717) | 2018 首发预印本；FVD，不代表覆盖所有视频质量维度 |

本章为原创教学解释，外部论文、代码和权重归原作者所有。[图像与视频来源核验](../image-video-source-audit.md) 记录 LDM、CFG、DiT 的固定论文版本、页码、公式、PDF 哈希与独立尺寸推导。

### 本轮可访问原始实现核对（2026-11-10）

以下固定提交的 README 已实际读取，用于核对方法定位与公开接口范围：

- [LDM · `a506df5`](https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/README.md)：作者项目定位与原论文链接。
- [DiT · `ed81ce2`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/README.md)：Transformer 作用于 latent patches；README 的默认结果也附采样配置，不能脱离配置比较。
- [SVD · `e8cd657`](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/README.md)：公开 SVD 标为 image-to-video，区分它与仓库后续其它模型。
- [CogVideoX · `7a1af71`](https://github.com/THUDM/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/README.md)：3D Causal VAE，以及图像到视频功能是后续单独发布的模型变体。

发布环境后已恢复 arXiv 访问，LDM/CFG/DiT 的上述主张已对照固定 PDF 核验；SVD/CogVideoX的训练论文、公开配置和许可进一步整理于配套阅读卡。[10章](10-pretrained-video-lab.md)已完成固定SVD的CPU运行与低分辨率失败分析，发布尺寸视频质量基线继续按backlog推进。

## 进一步阅读：把结构映射到真实系统

完成 [07 CPU小图像实验](07-image-flow-lab.md) 后，可用 [Wan 2025阅读卡](../readings/wan-video-2025.md) 对照图像/视频的潜表示、patch token、时间采样与I2V接口。阅读卡提供固定版本的原文定位与独立token数推导，作为后续权重实践的准备。

[ResTok 2026阅读卡](../readings/restok-image-ar-2026.md)补充类别条件的离散图像生成路径：分层tokenizer、分组自回归、分类logits的CFG与重建/生成评价，连接01的因果性和本章的表示学习研究线。

[LTX-2 2026阅读卡](../readings/ltx2-video-audio-2026.md)把视频扩展到音画联合：链式条件模型可表达联合分布，独立边缘模型会丢失对应关系；双流去噪、跨模态条件、时间坐标与具体采样循环都需要逐项核对。CPU反例与固定源码计数连接06的时序评价和08的引导约定。

研究采样加速时，读[少步生成专题](../readings/consistency-endpoint-map.md)：从概率流轨迹推导一致性端点映射，比较Euler网格、引导蒸馏和CD/CT训练对象，并核对正端点、初始分布与实际调用数。
