# 05 · 评价生成模型：从可运行结果到可信结论

**先修**：[01 生成模型](01-generative-models.md)，完成 [02 DDPM](02-diffusion.md) 或 [03 Flow matching](03-flow-matching.md) 的 CPU 实验；均值、协方差与基本统计推断。

**完成后应能**：辨别运行成功与分布拟合，设计使指标暴露盲点的反例，解释 FID/FVD、KID、IS 与特征空间 precision/recall 的假设，并写出有对照、版本、成本和不确定性范围的研究计划。

**范围**：本章 CPU 诊断可直接运行；图像与视频评价是方法说明和实验设计，尚未在本仓库执行大模型评测。与 [04 研究路线](04-image-video-roadmap.md) 配合使用。

## 1. “更好”至少包含六种问题

| 维度 | 问题与可用证据 | 常见混淆 |
| --- | --- | --- |
| 密度建模 density estimation | 保留数据的似然或明确界限，使用相同数据处理和测度 | 较高似然不自动等于更符合人类偏好的样本 |
| 保真度 fidelity | 生成样本是否像目标分布中的有效样本；人工评分、特征空间 precision 等 | 只生成一张逼真图可以保真却缺少多样性 |
| 覆盖与多样性 coverage/diversity | 目标分布的哪些区域被生成；模态统计、特征空间 recall 等 | 看起来不同的噪声也“多样”，却可能全都无效 |
| 条件遵循 conditional alignment | 对象、数量、关系、文字或参考图像是否满足条件 | 无条件特征分布接近不保证逐提示正确 |
| 时间一致性 temporal coherence | 身份、动作、遮挡与状态是否跨帧合理延续 | 静止画面可以很一致，却不完成“向前走”的动作 |
| 资源成本 efficiency | 指定硬件、精度、尺寸和质量标准下的延迟、峰值内存、训练成本 | 少网络调用不一定端到端更快；不同质量不构成等质量加速 |

应先写出主张属于哪一行，再选择指标。将所有维度压成一个排行榜数字，会隐藏任务定义与取舍。

## 2. 本仓库的真实反例：覆盖 8 个模式仍然欠拟合

以下是已保存的**实际测量**，CPU、Python 3.12.14、PyTorch 2.6.0+cpu、seed 7、800 次训练更新、100 次采样网络调用、2048 个生成点；完整配置见 [DDPM JSON](../../records/measurements/ddpm-800-seed7.json) 与 [flow JSON](../../records/measurements/flow-800-seed7.json)。这些记录用于同一固定 seed 下的教学诊断；方法总体性能的比较需要另行设计评价。

| 样本来源 | 达到 1% 阈值的模式数 | 模式中心半径 $`3\sigma`$ 内比例 | 到最近模式中心的平均距离 |
| --- | ---: | ---: | ---: |
| DDPM，800 更新 | 8 | 23.14% | 0.4276 |
| Flow，800 更新 | 8 | 27.05% | 0.4110 |
| 独立真实目标分布基线 | 8 | 98.68% | 0.09845 |

这里 $`\sigma=0.08`$，阈值半径是欧氏距离 $`0.24`$。“覆盖 8 个模式”的定义仅要求每个模式圈内至少有全部生成样本的 1%，不要求剩下的大多数样本正确。因此覆盖数达到上限，不等于拟合完成。把表格第一列单独作为成功标准，会掩盖大部分点落在圈外的事实。

**理论基线**：二维各向同性高斯样本到其自身生成中心的距离 $`R`$ 满足 $`R^2/\sigma^2\sim\chi^2_2`$，因此

```math
\Pr(R\lt 3\sigma)=1-\exp(-9/2)\approx98.8891\%.
```

这不是一维正态“正负三倍标准差约 99.7%”的结论。代码测的是到最近中心的距离：落入其他中心圆盘的尾部也被计入，故严格说与“自身中心”的解析概率不同；当前中心间隔很大，这项差异可忽略。有限样本基线 98.68% 与解析概率不同，不是代码必然出错。

### 直接运行一个指标反例

在已安装依赖的仓库根目录运行；无需训练或下载数据：

<!-- cpu-check: coverage-counterexample -->
```bash
python - <<'PY'
import json
import torch
from genai_study.toy import centers, diagnostics, sample_data

torch.set_num_threads(1)
n = 2048
reference = sample_data(n, torch.Generator().manual_seed(101))
target = sample_data(n, torch.Generator().manual_seed(102))
noise = torch.randn(n, 2, generator=torch.Generator().manual_seed(103))
# 12.5% 的点均匀放在 8 个中心，其余点全部远离目标。
misleading = torch.full((n, 2), 8.0)
misleading[:256] = centers().repeat_interleave(32, dim=0)
# 打乱，避免 energy distance 只读取前 512 点时产生次序偏差。
misleading = misleading[torch.randperm(n, generator=torch.Generator().manual_seed(104))]
collapsed = centers()[0].expand(n, -1)
for name, points in [('target', target), ('noise', noise),
                     ('coverage_trap', misleading), ('collapsed', collapsed)]:
    print(name, json.dumps(diagnostics(points, reference)))
PY
```

**应验证的解析性质**：`coverage_trap` 的覆盖模式数为 8，但只有 12.5% 点在圈内；`collapsed` 到最近中心的距离为 0，但覆盖模式数只有 1。其他随机指标以输出为准。

本仓库的 energy distance 诊断使用至多前 512 个点的经验 V-statistic：

```math
\hat D_E=\frac{2}{nm}\sum_{i,j}\|x_i-y_j\|
-\frac{1}{n^2}\sum_{i,i'}\|x_i-x_{i'}\|
-\frac{1}{m^2}\sum_{j,j'}\|y_j-y_{j'}\|.
```

总体形式在欧氏空间、有限一阶矩条件下可区分不同分布；有限样本 V-statistic 包含对角线项，存在有限样本偏差。两组同分布有限样本一般也不等于零。它用原始二维坐标计算，既不是 FID，也不能把同样的数值尺度搬到图像特征空间。前 512 个点若按模式排序，会造成额外的代表性问题。

## 3. FID 究竟比较了什么

Fréchet Inception Distance（FID）先用固定 Inception 特征提取器 $`\phi`$（原论文采用 Inception-v3 最后池化层），对真实与生成特征分别估计均值和协方差 $`(\hat\mu_r,\hat\Sigma_r)`$、$`(\hat\mu_g,\hat\Sigma_g)`$，再计算

```math
\mathop{\mathrm{FID}}\nolimits=
\|\hat\mu_r-\hat\mu_g\|_2^2+
\mathop{\mathrm{Tr}}\nolimits\left(\hat\Sigma_r+\hat\Sigma_g-
2(\hat\Sigma_r^{1/2}\hat\Sigma_g\hat\Sigma_r^{1/2})^{1/2}\right).
```

该公式是两个高斯的**平方 2-Wasserstein 距离**。在 FID 中，它比较的是特征的一、二阶统计所定义的高斯近似；真实特征并不保证高斯。它不是像素分布之间的精确最优传输，也没有直接度量文本条件或人类偏好。特征缩放会改变数值，因此 FID 的量纲与特征坐标平方有关，不是概率或“正确率”。

**解析反例**：在一维恒等特征下，标准正态与等概率取 $`-1,+1`$ 的分布具有相同均值 0、方差 1，对应总体高斯距离为 0，但分布显然不同。这个一维例子说明了只比较前两阶矩的盲点；具体图像的 FID 应依据固定特征协议实际计算。

### 有限样本与实现协议

1. 均值和协方差是估计量；均值差平方与矩阵平方根等非线性运算会带来有限样本偏差。固定真实参考集时，生成侧的偏差仍依赖模型；统一生成样本数控制了样本量，模型间偏差差异仍可能改变排序。具体推导与原论文案例见 [评价来源核验](../evaluation-source-audit.md)。
2. 样本量影响估计稳定性；当样本数小于特征维度时，经验协方差至多有 $`n-1`$ 秩。正则化可能帮助数值计算，但会改变具体估计过程，必须记录。
3. Resize、插值核、抗锯齿、图像范围、裁剪、压缩、特征层和权重都属于协议的一部分。分别记录数据集准备与两侧特征提取前的缩放，以及库版本、函数和显式参数；复用参考统计时核对生成侧管线。下采样中的预滤波影响混叠，论文的库比较结论应对应其测试版本。
4. 真实参考集的 split、内容和样本数也必须相同。对另一参考数据集得到的更低数值不能被解释成在同一任务上的提升。
5. 比较差值应配合采样变化范围及失败案例。不能用单次计算的小数点后几位作为显著性证据，也不能把 bootstrap 区间当作自动消除 FID 偏差的方法。

### KID：无偏估计与负数结果

Kernel Inception Distance（KID）使用 Inception 特征的**平方最大均值差异**（squared maximum mean discrepancy，$`\mathrm{MMD}^2`$），原论文采用三次多项式核

```math
k(u,v)=\left(\frac{u^\top v}{d}+1\right)^3,
```

其中 $`d`$ 为特征维度。对于互相独立的两组 IID 特征 $`u_1,\ldots,u_m`$ 与 $`v_1,\ldots,v_n`$，固定特征提取器和核、$`m,n\ge2`$、相关核期望有限时，无偏 U 统计量为

```math
\widehat{\mathrm{MMD}}_u^2=
\frac{\sum_{i\ne j}k(u_i,u_j)}{m(m-1)}+
\frac{\sum_{i\ne j}k(v_i,v_j)}{n(n-1)}-
\frac{2\sum_{i,j}k(u_i,v_j)}{mn}.
```

同组求和排除对角项，跨组求和保留所有配对。总体 $`\mathrm{MMD}^2\ge0`$，但有限样本的无偏估计可为负；将负值裁剪为 0 会改变估计量并引入偏差。三次核比较特征中至三阶的矩信息；解读时仍需核对特征与核能分辨哪些分布。

报告 KID 时写明特征层/权重、预处理、核、子集大小、抽样方法、重复次数和是否缩放数值（如乘 $`10^3`$）。与 FID 对比应使用一致的图像样本与特征协议，并报告各自的估计不确定性。第 4 节给出不依赖图像网络的解析反例；真实图像评价仍需加载对应特征权重。

## 4. IS、precision/recall 与单分数陷阱

### IS 的熵与互信息形式

Inception Score（IS）常写作

```math
\mathop{\mathrm{IS}}\nolimits=\exp\left(\mathbb E_{x\sim p_g}
D_{\mathrm{KL}}(p_\phi(y\mid x)\Vert p_\phi(y))\right),
\qquad p_\phi(y)=\mathbb E_{x\sim p_g}[p_\phi(y\mid x)].
```

这里 $`y`$ 是分类器标签。在联合分布 $`p_g(x)p_\phi(y\mid x)`$ 下，展开 KL 可得

```math
\log\mathop{\mathrm{IS}}\nolimits=H(Y)-\mathbb E_X[H(Y\mid X=x)]=I(X;Y).
```

IS 因而偏好单张图的分类分布尖锐、总体分类分布多样。若分类器有 $`C`$ 类，$`1\le\mathop{\mathrm{IS}}\nolimits\le C`$。等概率反复输出每类的一张原型，且分类器对这些原型给出对应 one-hot 概率时，$`H(Y)=\log C`$、条件熵为 0，IS 达到上界 $`C`$，类内变化仍为零。这个互信息只涉及分类器标签，语义取决于分类器与目标数据域的匹配；真实性、记忆及条件遵循需要其他证据。

### 近邻球怎样估计 precision/recall

Kynkäänniemi et al. 的 improved precision/recall 在固定特征空间中构造两组球。对每个真实特征 $`r_i`$，令 $`\rho_i`$ 为它到**真实集合中第 $`k`$ 个近邻**的距离，排除自身；令

```math
\mathcal R=\bigcup_i B(r_i,\rho_i).
```

同理，用生成特征内部的近邻距离构造 $`\mathcal G`$。用闭球和欧氏距离时，估计为

```math
\widehat{\mathrm{precision}}=\frac1n\sum_{j=1}^n\mathbf1\{g_j\in\mathcal R\},
\qquad
\widehat{\mathrm{recall}}=\frac1m\sum_{i=1}^m\mathbf1\{r_i\in\mathcal G\}.
```

进入任意一个球即计为覆盖。原论文使用 VGG-16 第二全连接层后的特征，通常两侧各 50,000 个样本、$`k=3`$。这些是作者实验配置；样本数、$`k`$、特征和离群点都会影响覆盖估计。固定样本集时，增大 $`k`$ 会扩大球，使两项估计均不减，因此报告时需固定这些选择。

结合 precision 与 recall 展示保真和覆盖的取舍；条件生成还应按条件分组检查。无条件总体覆盖充分时，逐提示内容仍可能错配。

### 直接运行：原型高分与负的 U 统计量

下面直接构造分类器概率和一维特征，检验两个指标的统计性质。它不加载图像或特征网络，完成 README 安装后，从仓库根目录执行：

<!-- cpu-check: is-kid-counterexamples -->
```bash
python - <<'PY'
from itertools import product
import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float)
    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)  # 约定 0 log 0 = 0
    return -(p * logp).sum(axis=-1)

def inception_score(probabilities):
    return np.exp(entropy(probabilities.mean(0)) - entropy(probabilities).mean())

# 三类各只有一张原型；重复次数增加，类内变化仍为零。
prototypes = np.eye(3)
repeated = np.repeat(prototypes, 100, axis=0)
assert np.isclose(inception_score(prototypes), 3)
assert np.isclose(inception_score(repeated), 3)
assert np.isclose(inception_score(np.full((300, 3), 1 / 3)), 1)

def mmd2_u(x, y):
    m, d = x.shape
    n = len(y)
    kxx = (x @ x.T / d + 1) ** 3
    kyy = (y @ y.T / d + 1) ** 3
    kxy = (x @ y.T / d + 1) ** 3
    return ((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
            + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean())

observed = np.array([[0.], [1.]])
assert np.isclose(mmd2_u(observed, observed), -3.5)
# 两组各2个IID Bernoulli(1/2)特征，共16种等概率的联合观测。
batches = [np.array(bits, dtype=float)[:, None] for bits in product([0, 1], repeat=2)]
values = np.array([mmd2_u(x, y) for x in batches for y in batches])
assert np.isclose(values.mean(), 0)  # 同分布总体平方MMD为0
assert np.isclose(np.maximum(values, 0).mean(), 0.875)
print(f"IS: {inception_score(repeated):.1f}; signed MMD² estimate: {mmd2_u(observed, observed):.1f}")
print(f"Exact expectation: {values.mean():.3f}; clipped expectation: {np.maximum(values, 0).mean():.3f}")
PY
```

解析输出应为 `IS: 3.0; signed MMD² estimate: -3.5`，枚举期望为 `0.000`，裁剪后的期望为 `0.875`。两组独立抽样仍可能恰好观测到相同的 `[0, 1]`；无偏性约束的是所有可能观测的期望。代码既展示负估计，也用完整枚举核对了其期望。

上述定义、原论文公式页码与独立推导见 [IS / precision-recall / KID / FVD 来源核验](../evaluation-metrics-audit.md)。

## 5. 视频：FVD 的表示与任务协议

Fréchet Video Distance（FVD）将平方高斯特征距离用于视频表示。原论文比较 Kinetics-400/600 预训练的 RGB I3D 视频网络和不同特征层，后续实验选用 Kinetics-400 的 logits；网络共同处理一段视频的空间与时间信息。

使用时记录特征权重/层、输入布局与范围、空间预处理、帧数、帧采样策略、fps、时长和参考视频来源。原论文 BAIR 人评使用 2 帧上下文加 14 帧生成结果、256 条参考视频；KTH 实验采用 10 帧上下文加 10 帧生成结果。这些是具体任务协议，FVD 的定义允许其他视频长度。改变 fps 会改变动作速度的含义；短片段只覆盖对应时间范围内的一致性。跨实现比较还需核对特征和完整输入管线，详见 [FVD 来源与协议](../evaluation-metrics-audit.md)。

FVD 将每段视频压缩为特征，再比较一、二阶统计。物体持久性、碰撞守恒、事件先后和条件遵循需要针对性测试，例如：物体遮挡后再出现时是否保留身份，容器倒水时液体是否保持合理状态变化。结合这些事件检查，才能说明分数变化对应了哪些任务能力。

一个可用的多维视频协议应同时报告：逐帧空间伪影、条件动作完成度、跨帧身份、运动连续性、事件逻辑，以及每条完整视频的盲评结果。静止视频可能在某些时间一致性指标上优秀，必须用动作完成度把这一退化解排除。

## 6. 控制变量与统计范围

### 先决定结论的单位

| 随机来源 | 合适的实验单位 | 一个不足以覆盖它的做法 |
| --- | --- | --- |
| 同一模型的采样噪声 | 固定提示与配置下的多个采样 seed | 只挑一个最好 seed |
| 提示/条件集合变化 | 来自声明分布的多个独立条件 | 将同一提示的 100 张图当作 100 个独立提示 |
| 训练随机性 | 多个独立训练 seed 或训练运行 | 对单个已训练模型多采样，称为训练稳定性 |
| 人类评估差异 | 明确评估者和题目抽样；记录一致性与分歧 | 把同一人的重复评分当作独立评估者 |

对同一批提示比较方法 A/B，可以固定采样 seed 与噪声生成方案以构造配对设计，并盲化模型名称与展示顺序。不同算法的随机数消费不一定一致；“seed 数字相同”不自动意味着每一步噪声相同，需保存或说明实际耦合方式。

置信区间必须说明重采样的层次。例如对固定模型、固定参考集，以提示为簇 bootstrap，先抽提示簇并保留簇内样本，再重新计算 A/B 差值；这反映该层次的样本不确定性，不能覆盖新训练 seed、参考集选择偏差或评估器系统偏差。少数训练种子的均值和标准差只是描述性摘要；三个种子不能支撑宽泛的“已证明稳健”。

视频的连续帧强相关，不能将同一视频的 100 帧当作 100 个独立视频来缩窄区间。报告平均分时还要看分组与最差案例：常见提示的大量成功可能掩盖少数重要条件完全失败。

### 可运行反例：复制结果不会增加独立提示

固定模型A/B与评分规则，先对同一提示下配对的输出计算分数差A−B。用一个教学构造：设三个独立提示的平均差为$`d=(-1,0,1)`$，每个提示的结果各复制4次，组成12行记录。提示等权，观察均值为0；四个副本保留同一提示的信息。

如果目标是描述这三个固定提示，直接报告均值0即可。若要估计声明提示分布上的平均配对差，重采样单位应对应独立提示：有放回抽3个提示，保留各自4行，并重新计算平均差。下面的星号$`*`$表示**条件于已观察数据的经验bootstrap分布**。

每次抽中的提示差以相同概率取$`-1,0,1`$，其经验方差为$`2/3`$，因此

```math
\mathrm{Var}^{*}(\bar d^{*})=\frac{1}{3}\left[\frac{1}{3}\sum_{i=1}^{3}(d_i-\bar d)^2\right]=\frac{2}{9}.
```

错误地把12行当作独立观测，会按行有放回抽12次。单次取值的经验分布相同，均值方差却变成$`(2/3)/12=1/18`$，标准差只有按提示重采样的一半。

在基础环境中运行以下代码；无需训练、文件或随机种子。它完整枚举$`3^3=27`$个等概率提示索引序列；错误行抽样的方差使用上面的独立抽样均值公式。

<!-- cpu-check: prompt-cluster-bootstrap -->
```python
from itertools import product
import numpy as np

# Rows are prompts; columns are four exact copies of a paired A-minus-B result.
paired_diff = np.repeat(np.array([-1., 0., 1.])[:, None], 4, axis=1)
indices = np.array(list(product(range(3), repeat=3)))
# Keep all four paired differences whenever a prompt is selected.
cluster_means = paired_diff[indices].mean(axis=(1, 2))
cluster_variance = cluster_means.var(ddof=0)

rows = paired_diff.ravel()
iid_row_variance = rows.var(ddof=0) / rows.size
sd_ratio = np.sqrt(cluster_variance / iid_row_variance)
np.testing.assert_allclose(
    [cluster_means.mean(), cluster_variance, iid_row_variance, sd_ratio],
    [0., 2/9, 1/18, 2.], rtol=0, atol=1e-15)
print(f"Prompt resamples: {len(cluster_means)}")
print(f"Prompt variance: {cluster_variance:.6f}; row variance: {iid_row_variance:.6f}")
print(f"Prompt/row standard-deviation ratio: {sd_ratio:.1f}")
```

应输出27种结果、方差约0.222222与0.055556、标准差比2.0。`ddof=0`把27个结果视为完整的等概率条件分布，方差分母为27；随机抽取$`B`$次复制时，常见的$`B-1`$样本方差则用于估计这个条件分布的方差。

这段计算精确展示了重采样单位的影响。向新提示推广仍依赖原提示的独立性和代表性，本例没有验证任何置信区间的覆盖率。实际每提示样本数不等时，先明确采用提示等权还是按输出条数加权；四次完全复制是本反例的构造。

**练习：** 每个提示改为复制9次，上述两种方差与标准差比怎样变化？

<details><summary>答案</summary>

提示重采样方差保持$`2/9`$；错误的27行独立重采样方差为$`(2/3)/27=2/81`$，其标准差成为提示重采样的$`1/3`$。新增副本增加了行数，独立提示仍为3个。

</details>

**来源：** Efron与Narasimhan，[The automatic construction of bootstrap confidence intervals](https://efron.ckirby.su.domains/papers/2018Automatic-Construction-BCIs.pdf)，作者稿PDF第6–7页§3、式(3.1)–(3.7)：独立同分布观测及从原观测中有放回抽取同样数量样本。这里将独立提示作为观测单位；复制反例与精确方差由上述构造独立推导。

### 防止选择和泄漏影响结论

- 在运行前固定主要假设、主指标、条件集合和预算；把调参集合与最后报告集合分开。
- 保存全部生成结果及失败，不从多次尝试中只报告最好一次；记录丢弃规则和丢弃数量。
- 图像按来源或身份、视频按原始视频/主体拆分，防止近重复跨训练与评价集；检查模型训练数据是否已知包含评测内容。无法知道时应写明未知。
- 最近邻检查可以发现部分重复，没找到近邻不能证明没有记忆或训练数据泄漏。
- 盲评使用预先定义的评分规则、随机顺序、允许平局/不确定，并记录评估者数量与背景限制。评估时隐藏模型身份，按预先规定的规则选取和呈现样本。

## 7. 练习：先设计能推翻自己的实验

### A. 解释 coverage trap

对上述 CPU 代码，为什么覆盖数为 8？怎样同时识别它与单模式坍塌？

**参考答案**：每个中心有 32 个点，超过 $`0.01\times2048=20.48`$ 的阈值；其余 1792 个点完全无效也不减覆盖数。近邻比例识别无效点，覆盖数识别坍塌，energy distance 与完整散点图进一步检查权重、形状和离群点。任何一个指标都不能取代其它证据。

### B. 一次公平的求解器比较

在同一个 flow 模型上比较 25 与 100 步 Euler，应固定哪些量？训练 seed 相同但重新训练一次足够吗？

**参考答案**：固定同一模型权重、同一初始噪声张量、数据/条件、精度和评估参考集；记录 NFE 与实测延迟。重新训练可能引入训练差异，即使 seed 一样也应验证权重是否完全相同。按 03 章的 `solver_study` 命令读取同一个训练 checkpoint，以同一初始噪声比较 10/25/100/200 步；记录权重 SHA、NFE 和配对终点差。配对差不是精确解误差，checkpoint 也不包含可严格续训的优化器状态。

### C. 一个图像生成的受控研究计划

在 [04 章](04-image-video-roadmap.md) 的预训练模型上研究 CFG 系数是否提高属性遵循。填入 [论文阅读与实验模板](../reading-paper-template.md)，在开始前声明 30 个条件、每条件 4 个采样 seed、3 个引导系数的试运行设计。

**参考答案要点**：固定权重/代码 revision、采样器、分辨率、NFE 与条件编码方式；三个系数采用相同条件与可复用初始噪声；先测一条样本成本，再决定是否执行完整 360 张。主指标采用预定义属性判定，辅指标包括伪影、类内多样性与成本。采用按提示配对的比较和提示簇不确定性分析；结论适用于本次固定模型与指定条件集合。30×4×3 为本题提出的设计规模，实际评测尚待执行。

### D. 从图像计划迁移到视频

如果模型只是把输入图像复制为所有帧，身份一致性极好。怎样设计对照识别失败？

**参考答案**：加入这项静态复制基线，同时评价明确的动作完成度；固定参考图像、目标动作、fps 与时长，逐条保存完整视频。盲评判断动作是否发生、身份是否保持、是否存在跳变与状态错误。将评价单位设为视频而非帧。不能只增加 FVD 一个数字就宣称动作和物理问题已解决。

## 8. 原始来源与核验边界

| 主题 | 来源 | 对应内容与版本 |
| --- | --- | --- |
| FID | Heusel et al., [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500v6) | NeurIPS 2017；v6，2018-01-12；第 6 页 Eq. (6)、第 7 页特征配置 |
| Inception Score | Salimans et al., [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498v1) | NeurIPS 2016；v1，2016-06-10；第 4 页 §4，IS 定义 |
| 生成分布 precision/recall | Kynkäänniemi et al., [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991v3) | NeurIPS 2019；v3，2019-10-30；§2 Eq. (1)–(2)、Algorithm 1 |
| FVD | Unterthiner et al., [Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/abs/1812.01717v2) | 2018 预印本；v2，2019-03-27；§2、§4.1、§4.3–4.4；ICLR 2019 workshop |
| FID 有限样本偏差 | Chong & Forsyth, [Effectively Unbiased FID and Inception Score and Where to Find Them](https://arxiv.org/abs/1911.07023v3) | CVPR 2020；v3，2020-06-15；§3.1–3.2、§4.1/Figure 3 |
| 预处理敏感性 | Parmar, Zhang, Zhu, [On Aliased Resizing and Surprising Subtleties in GAN Evaluation](https://arxiv.org/abs/2104.11222v3) | CVPR 2022；v3，2022-01-21；§3.1–3.3、Tables 1–2、§6.3 |
| KID | Bińkowski et al., [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401v5) | ICLR 2018；v5，2021-01-14；第 4 页 Eq. (4)、第 8 页 KID |

已对照 Heusel、Chong/Forsyth、Parmar 三篇固定版本原文，核验 FID 定义、有限样本偏差和预处理协议；PDF SHA-256、页码、假设及独立数学检查见 [评价来源核验](../evaluation-source-audit.md)。IS、precision/recall、KID 与 FVD 的定义、条件及输入协议见 [第二组评价核验](../evaluation-metrics-audit.md)。toy 数值引用仓库内实测 JSON；本章 NumPy 示例验证解析性质，论文中的生成模型实验尚待复现。

进一步实践：[DDPM解析预测器对照](../ddpm-oracle-report.md)固定已知最优噪声函数，检查有限步采样的方差及邻域率。用最近中心径向二阶矩与目标样本参照，补充coverage和邻域率对分量内部多样性的检查。
