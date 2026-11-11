# IS、近邻 precision/recall、KID 与 FVD：定义和估计条件核验

核验日期：2026-11-10。本页补充 [05 评价章](chapters/05-evaluation.md) 中四项指标的定义、统计条件和输入协议；FID 定义、有限样本偏差与 resize 的前一轮证据保留在 [FID 来源核验](evaluation-source-audit.md)。本轮读取指定原文和作者源码，运行解析反例；图像/视频特征权重与论文生成实验尚未运行。

## 1. 固定 PDF

版本日期取自 arXiv submission history，页数、字节数和 SHA-256 来自实际下载。下文页码为 PDF 页序，所引用页面与印刷页码一致。

| 来源 | 固定版本与提交日期（UTC） | PDF 页数 | 字节数 |
| --- | --- | ---: | ---: |
| Salimans et al., *Improved Techniques for Training GANs* | [1606.03498v1](https://arxiv.org/abs/1606.03498v1)，2016-06-10；NeurIPS 2016 | 10 | 2346786 |
| Kynkäänniemi et al., *Improved Precision and Recall Metric for Assessing Generative Models* | [1904.06991v3](https://arxiv.org/abs/1904.06991v3)，2019-10-30；NeurIPS 2019 | 16 | 10490223 |
| Bińkowski et al., *Demystifying MMD GANs* | [1801.01401v5](https://arxiv.org/abs/1801.01401v5)，2021-01-14；ICLR 2018 | 36 | 10781114 |
| Unterthiner et al., *Towards Accurate Generative Models of Video: A New Metric & Challenges* | [1812.01717v2](https://arxiv.org/abs/1812.01717v2)，2019-03-27；ICLR 2019 workshop | 17 | 7299504 |

```text
1606.03498v1.pdf  933d18b2f2767ea157e77c73b60d46f96135ac1984633a765c5e7d5b49da3301
1904.06991v3.pdf  ee4efa06f67bfeec5bd7486e37ebebe47e2bded3a6821fe46ee8c8a7d2b795ee
1801.01401v5.pdf  a52724fbb6643f440fd9a85148c077a79d1b3ccb7bc4948b72c454bd608e0b21
1812.01717v2.pdf  6802a958cb841e1de093969b2814fc3e0c79978152d279cd78af5ff0379edadd
```

固定 PDF：[IS](https://arxiv.org/pdf/1606.03498v1)、[precision/recall](https://arxiv.org/pdf/1904.06991v3)、[KID](https://arxiv.org/pdf/1801.01401v5)、[FVD](https://arxiv.org/pdf/1812.01717v2)。本轮临时文件位于 `/tmp/genai-metrics-audit/`；可用 `sha256sum`、`pdfinfo` 与 `pdftotext -layout` 核对文件和定位段落。

## 2. 逐项证据

| 主张 | 原文位置 | 核验结论 |
| --- | --- | --- |
| IS 是标签条件分布与边缘分布之间的期望 KL 的指数 | Salimans，第 4 页 §4，无编号的 IS 公式；第 5 页首段 | 原文用低条件熵、高边缘熵说明动机，原实验采用 50k 张图像。课程中的互信息形式和原型反例由该定义独立展开得到。 |
| 近邻球由各集合内部的距离构造 | Kynkäänniemi，第 3 页 §2，Eq. (1)–(2)；第 11 页 Algorithm 1 | 分别估计真实/生成特征的球并集，然后交叉查询覆盖。算法第 11 行选含自身零距离列表的第 $`k+1`$ 小值，明确排除自身。 |
| 特征与 $`k`$ 是 precision/recall 协议 | Kynkäänniemi，第 3–4 页 §2、Figure 3 | 原实验默认 VGG-16 第二全连接层后的特征、$`k=3`$、两侧各 50k 样本；比较过其他 $`k`$ 及 Inception-v3 特征。 |
| KID 使用平方 MMD 与三次多项式核 | Bińkowski，第 7–8 页 FID/KID 段落 | 核为 $`(u^\top v/d+1)^3`$；$`d`$ 是 Inception 表示维度，论文讨论的 pool3 为 2048 维。三次核涉及均值、方差与三阶矩信息。 |
| 两样本平方 MMD 有无偏估计 | Bińkowski，第 3 页 §2.1/脚注 4；第 4 页 Eq. (4) | 核固定、两侧独立 IID 抽样、相关核期望存在；同组配对排除自身。第 8 页脚注 12 建议小子集多次估计后平均，需同时记录抽样方案。 |
| 有限样本 U 估计可以为负 | Bińkowski，第 4 页 Eq. (4)；本页第 3 节解析反例 | 从原式直接计算得到负数；完整枚举仍恢复非负总体平方 MMD 的期望。将负数裁剪为 0 会改变该期望。 |
| FVD 使用平方高斯特征距离 | Unterthiner，第 2 页 §2，Eq. (1)–(2) | Eq. (1) 采用平方欧氏运输代价，Eq. (2) 给出与 FID 同型的高斯表达，均无外层平方根。 |
| 原论文最终选择 Kinetics-400 I3D logits | Unterthiner，第 3 页 §2；第 6 页 §4.1 | 比较 Kinetics-400/600 RGB I3D 的 logits 与最后池化层；后续实验选用 Kinetics-400 logits。具体层是协议的重要部分。 |
| 视频长度、上下文与样本数依任务设定 | Unterthiner，第 6 页 §4.3；第 8 页 §4.4 | BAIR 人评为 2 帧上下文加 14 帧生成结果、256 条参考视频；KTH 为 10+10 帧，基准使用 1024 条验证样本。论文提供多种任务长度。 |

FVD 第 6 页 §4.2 同样观察到有限样本的同分布距离通常大于 0。控制样本量应结合 [模型依赖偏差的证据](evaluation-source-audit.md) 理解：统一样本量是协议要求，剩余估计偏差仍需分析。

## 3. 独立推导与可运行反例

### IS 的互信息与上界

以 $`p_g(x)p_\phi(y\mid x)`$ 定义联合分布，直接展开：

```math
\mathbb E_X D_{\rm KL}(p_\phi(y\mid X)\Vert p_\phi(y))
=\sum_{x,y}p_g(x)p_\phi(y\mid x)\log p_\phi(y\mid x)
-\sum_y p_\phi(y)\log p_\phi(y)
=H(Y)-H(Y\mid X).
```

连续 $`X`$ 时把相应求和换成积分。有限 $`C`$ 类满足 $`0\le I(X;Y)\le\log C`$；当每类等概率且条件标签为 one-hot 时取上界。每类只有一个反复出现的原型也能满足该条件，所以最高 IS 可以与零类内变化同时出现。

### KID 的负估计与裁剪偏差

对一维特征、$`k(u,v)=(uv+1)^3`$，两组恰好都观测到 $`[0,1]`$ 时，同组的非对角核均为 1，跨组核矩阵为

```math
\begin{pmatrix}1&1\\1&8\end{pmatrix}.
```

故 Eq. (4) 为 $`1+1-2\times11/4=-3.5`$。如果两组各由两个独立 Bernoulli$`(1/2)`$ 特征组成，共有 16 种等概率观测：平方 MMD 的总体值为 0，U 估计在所有观测上的均值也为 0；把每次估计裁剪到非负后，均值变成 0.875。

[05 章第 4 节](chapters/05-evaluation.md) 给出完整 NumPy 代码，同时检验 IS 原型反例与上述 16 种观测。验证时直接从 Markdown 提取该代码块并执行；无需训练、模型权重或额外依赖。其数值来自人工构造的概率与特征，用于核对公式与估计性质。

### 近邻球的单调性

固定两组特征时，第 $`k+1`$ 近邻距离不小于第 $`k`$ 近邻距离，所以增大 $`k`$ 使球并集只会扩大；precision 与 recall 均不减。该性质提醒读者：比较不同方法前应固定 $`k`$ 和特征协议，单独提高 $`k`$ 就可能同时提高两个数值。

## 4. FVD 作者实现的输入协议

论文描述了特征与任务设置；空间缩放、数值范围等细节由作者公开实现补齐。本轮只读核验 [google-research 固定源码](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/frechet_video_distance/frechet_video_distance.py)，该文件最近提交为 `5b09c22d73a9d35eb6c5d2a99b95677a45053466`（2026-02-12），5142 字节，SHA-256 为 `20622e6ef6acc002858cd42ed4e975331b33163678855f94adf4e6f1732825a9`。

| 环节 | 固定源码位置 | 协议 |
| --- | --- | --- |
| 预处理 | `preprocess`，第 38–52 行 | 输入布局 `[B,T,H,W,C]`、范围 `[0,255]`；展平帧后用 `tf.image.resize_bilinear` 缩放，再按 $`2x/255-1`$ 变换。 |
| 特征输入 | `create_id3_embedding`，第 72–73 行 | float32、空间尺寸 $`224\times224`$、3 通道、范围 $`[-1,1]`$。 |
| 模型身份 | 第 83–84 行 | 固定模块标识 `https://tfhub.dev/deepmind/i3d-kinetics-400/1`。 |
| 批大小 | 第 99–103 行 | 该实现检查第一维 $`B=16`$；这是视频批大小。 |
| 输出层 | 第 125–127 行 | `RGB/inception_i3d/Mean:0`，源码注释明确为 Kinetics-400 logits。 |

论文在 BAIR 使用 16 帧、在 KTH 使用 20 帧；作者同一固定提交的 [example.py](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/frechet_video_distance/example.py#L28-L30) 又给出 15 帧示例。以上源码没有显式给出最小帧数或统一 fps。使用其他实现时应核验其时间维要求、窗口、采样率、缩放函数和权重；采用同一个名称不足以确定这些输入选择。

## 5. 验证范围

已核验四份固定 PDF 的字节/SHA、对应页节/公式，以及作者 FVD 代码的输入与输出约定。新增 NumPy 示例验证熵恒等式的端点情况、负 U 估计、完整有限样本期望和裁剪偏差；另用直接 KL 求和和近邻球检查交叉核对公式。文档链接与相关 CPU 诊断测试的具体结果写入 [执行记录](../records/PROGRESS.md)。
