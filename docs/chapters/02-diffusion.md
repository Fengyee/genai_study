# 02 · 扩散模型：从高斯后验到可运行的 DDPM

**先修**：[生成模型与训练目标](01-generative-models.md)、高斯条件分布、PyTorch 张量广播。

**完成后应能**：推导正向边缘与反向后验，解释噪声预测目标，独立检查实现的时间索引，并在 CPU 上训练二维扩散模型。

**实践产物**：配置与诊断 JSON、损失曲线、真实与生成样本散点图。这里的实践是二维教学模型，尚不是图像生成器。

## 1. 直觉：把困难的生成分解成许多小步

从数据开始，逐步添加高斯噪声，最终得到接近标准正态的分布。生成时从标准正态开始，学习逐步撤销这一过程。困难在于“当前带噪点来自哪个干净样本”通常没有唯一答案，因此训练的不是一个能逐样本完全还原的逆函数。

本章以无条件 DDPM 为例。真实数据记为 $x_0\in\mathbb R^d$；时间 $t=1,\ldots,T$ 越大，噪声越多。注意下一章 flow matching 的生成方向是从时间 0 的噪声到时间 1 的数据，两个章节的时间约定不同。

| 对象 | 含义 |
| --- | --- |
| $\beta_t\in(0,1)$ | 单步新增噪声方差，无量纲 |
| $\alpha_t=1-\beta_t$ | 单步信号保留系数 |
| $\bar\alpha_t=\prod_{s=1}^t\alpha_s$，$\bar\alpha_0=1$ | 累积信号保留系数 |
| $q$ | 已知的加噪分布 |
| $p_\theta$ | 学习的生成分布 |
| $\epsilon_\theta(x_t,t)$ | 与 $x_t$ 同形状的噪声预测，不是方差 |

图示：`数据 x₀ ──q──> x₁ ──q──> … ──q──> xT ≈ 噪声`；采样沿相反方向运行 $p_\theta$。

## 2. 正向过程为什么可以跳到任意时间

定义 Markov 链

$$
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I).
$$

代入上一时刻的表达式，独立高斯噪声仍是高斯。用归纳法得到

$$
q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I),\qquad
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\quad\epsilon\sim\mathcal N(0,I).
$$

归纳的关键是方差：$\alpha_t(1-\bar\alpha_{t-1})+\beta_t=1-\bar\alpha_t$。因此训练时可以直接随机抽取 $t$，无需先运行 $t$ 次加噪。

对一般数据，减小 $\bar\alpha_T$ 会削弱终点残留的数据信号；终点与 $\mathcal N(0,I)$ 的接近程度还取决于数据分布和尺度。数据具有有限二阶矩时，终点均值为 $\sqrt{\bar\alpha_T}\mathbb E[x_0]$，协方差为 $\bar\alpha_T\operatorname{Cov}(x_0)+(1-\bar\alpha_T)I$。一个特殊情形是标准正态数据：由于独立高斯的线性组合仍为高斯，任意 $\bar\alpha_T\in[0,1]$ 都严格得到标准正态；[练习D](#d-标准正态数据的解析噪声预测器)进一步推导它的最优预测器。

**噪声表与数据尺度**：对100步使用 $\beta_t$ 从 $10^{-4}$ 到 $0.02$ 的线性表，$\bar\alpha_T\approx0.36356$。本仓库半径2、分量标准差0.08的等权八高斯具有协方差 $2.0064I$，对应终点协方差约 $1.36589I$，每坐标方差比标准正态高约36.59%。这给出了该数据与噪声表组合的具体终点偏差；缩短扩散步数后，应重新核对噪声表和采样起点。

## 3. 可计算的后验与不可直接计算的反向条件

当 $x_0$ 已知时，Bayes 公式给出

$$
q(x_{t-1}\mid x_t,x_0)\propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0).
$$

对 $x_{t-1}$ 配方。对于 $t>1$，精度为

$$
\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}
=\frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}.
$$

于是

$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,
\qquad
\tilde\mu_t(x_t,x_0)=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
$$

$t=1$ 时该后验退化为已知的 $x_0$：均值 $x_0$、方差 0，应单独理解，不能直接用上面含零分母的精度表达式。

不要混淆 $q(x_{t-1}\mid x_t,x_0)$ 与 $q(x_{t-1}\mid x_t)$。后者对未知 $x_0$ 积分，通常不是单个高斯。DDPM 用高斯 $p_\theta(x_{t-1}\mid x_t)$ 建模反向一步，这是模型假设；后验公式正确不代表有限步生成已经精确。

由噪声预测还原干净样本估计

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}.
$$

将其代入后验均值，得到等价表达式

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).
$$

本仓库使用固定 $\tilde\beta_t$ 采样：$x_{t-1}=\mu_\theta+\sqrt{\tilde\beta_t}z$，其中每一步的 $z$ 独立；最后一步不加随机噪声。真实图像实现可能使用其他固定方差、学习方差或裁剪 $\hat x_0$。这些选择有各自的假设，不能把某一选择当成 DDPM 唯一形式。

## 4. 为什么是噪声均方误差

负对数似然的变分上界可分解为终点 KL、中间时间的高斯 KL 和最后一步的重建项。若中间反向方差固定为 $\sigma_t^2I$，则均值相关项为

$$
\frac{1}{2\sigma_t^2}\|\tilde\mu_t-\mu_\theta\|^2
=\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}
\|\epsilon-\epsilon_\theta(x_t,t)\|^2,\qquad t>1.
$$

DDPM 常用的简化目标忽略上述时间权重：

$$
L_{\text{simple}}=\mathbb E_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right],\qquad
t\sim\operatorname{Uniform}\{1,\ldots,T\}.
$$

因此，无权重 MSE **不等于**完整负 ELBO，也不能直接作为可比较的对数似然。代码通常对 batch 和坐标都求平均，论文中的向量平方范数则对坐标求和；固定维度下两者相差常数，但报告损失时必须说明 reduction。 原论文 §3.3、公式 (13) 还为整数图像的最后一步定义了离散 decoder；本仓库使用连续二维数据，没有实现该图像似然项。因此这里的噪声 MSE 不能报告成论文的图像 bits/dim。

对给定 $(x_t,t)$，MSE 最优预测为条件均值 $\mathbb E[\epsilon\mid x_t,t]$。由于一份 $x_t$ 可以由不同的 $x_0,\epsilon$ 产生，最优 MSE 通常不是零。在 $t>0$ 且密度光滑时，边缘 score 满足

$$
\nabla_{x_t}\log q_t(x_t)
=-\frac{\mathbb E[\epsilon\mid x_t,t]}{\sqrt{1-\bar\alpha_t}}.
$$

这解释噪声预测与 score 的联系，但有限训练、模型容量与离散采样误差仍然存在。

## 5. 动手：二维八高斯

先完成 [README](../../README.md) 的依赖安装。若已有`artifacts/first-ddpm`或`artifacts/ddpm`，先查看`metrics.json`中的`config.steps`：5000步结果可直接复用；800步结果用于短训练对照，需要5000步时选择新目录，例如`artifacts/ddpm-5000`。首次运行时在仓库根目录执行：

```bash
python -m genai_study.experiment --method ddpm --steps 5000 --seed 7 --output artifacts/ddpm
```

默认CPU、单线程、batch size 256、生成2048个样本。`--steps`控制训练更新次数；推荐显式使用5000次，省略时CLI默认800次。已有三seed实测中，5000次更新的目标邻域比例为91.89%–93.51%，具体成本与800步对照见[实测报告](../experiment-report.md)。默认扩散/采样预算$T=100$，$\beta_t$线性从$10^{-4}$到$0.1$；终点$\bar\alpha_T$约$0.0056$。这张较强的噪声表用于缩短二维CPU教学实验；图像任务需另选配置，且这里仍存在终点分布差异。

实现中的 Python 索引 $k=0,\ldots,T-1$ 对应本文 $t=k+1$。`alpha_bar_prev[0] = 1`，所以索引 0 的后验方差应严格为 0。注意数组索引 0 已是第一次加噪，不能误当作干净数据的 $t=0$。

读图和指标时依次检查：

1. 损失是否有限；固定评估样本上的损失是否相对训练前下降。该评估用于教学诊断，并非最终保留测试集。
2. 生成样本是否落在目标模态附近；是否只覆盖少数模态；是否存在模态之间的伪样本。
3. JSON 的最近中心距离、三倍标准差范围内比例、模态覆盖和经验 energy distance 是否与散点图一致。
4. 在相同配置下改变随机种子，判断观察是否稳定。

**理论预期**：训练足够充分时，模型应学会与八个模式相关的去噪方向。**实际测量**：以本次输出 JSON 中的耗时、损失和诊断为准，不预先承诺数值。**仍待验证**：对图像、高维数据和更强网络的可迁移程度。

二维数据具有无界高斯尾部；不要沿用图像归一化后常用的 `clamp(-1, 1)`，否则目标分布会被修改。二维诊断也不能解释为 FID、视觉质量或语义对齐度。

## 6. 独立验证与练习

### A. 正向边缘是否正确？

固定 $x_0$，重复抽样；样本均值应接近 $\sqrt{\bar\alpha_t}x_0$，每坐标方差接近 $1-\bar\alpha_t$。有限样本误差不能要求严格为零。

**练习**：若误把噪声系数写成 $1-\bar\alpha_t$，方差是什么？

**答案**：$(1-\bar\alpha_t)^2$；只有接近端点时才可能让图看起来“差不多”。这正是需要解析性质测试的原因。

### B. 检查后验的两个表达式

随机生成 $x_0,\epsilon$，用正确的前向公式构造 $x_t$。把构造时的真实 $\epsilon$ 代入噪声参数化均值，与直接用 $x_0$ 的后验均值比较，应在浮点容差内一致。

**提示**：此处是知道真实噪声的代数 oracle；训练网络并不能普遍恢复这份真实噪声。这个测试验证公式和索引，不验证生成模型质量。

### C. 最后一步为什么不能再加噪声？

**答案**：由于 $\bar\alpha_0=1$，$\tilde\beta_1=0$。错误地使用 $\beta_1$ 给最后一步加噪声，便不再是这里声明的后验方差采样器。其他论文有不同 decoder 定义，应按其定义另行实现。

### D. 标准正态数据的解析噪声预测器

设 $x_0\sim\mathcal N(0,I)$。证明 $x_t\sim\mathcal N(0,I)$ 且

$$
\epsilon^*(x_t,t)=\sqrt{1-\bar\alpha_t}\,x_t.
$$

**提示/答案**：联合高斯的 $\operatorname{Cov}(\epsilon,x_t)=\sqrt{1-\bar\alpha_t}I$、$\operatorname{Cov}(x_t)=I$。每坐标最小 MSE 是 $\bar\alpha_t$，所以“网络没把训练噪声 MSE 降到零”本身不证明训练失败。将此 oracle 代入反向均值得到 $\sqrt{\alpha_t}x_t$；但这里使用的 $\tilde\beta_t$ 小于真实平稳反向过程的 $\beta_t$，因此 oracle 均值也不保证有限步采样完全精确。

## 7. 来源、版本与阅读顺序

- Ho, Jain, Abbeel, **Denoising Diffusion Probabilistic Models**, NeurIPS 2020，特别是公式 4、6–8、11–14、Algorithm 1/2：[论文](https://arxiv.org/abs/2006.11239)。本章推导依据该论文的离散 DDPM 约定。
- 原作者实现 `hojonathanho/diffusion`：[固定版本 `diffusion_utils.py`](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils.py)。2026-11-10 核对了前向边缘、posterior variance、后验均值系数及噪声到 $x_0$ 的转换。该实现用零基数组索引，且可选择裁剪重建；本仓库二维实验不裁剪。
- Song et al., **Score-Based Generative Modeling through Stochastic Differential Equations**, ICLR 2021：[论文](https://arxiv.org/abs/2011.13456)。作为后续阅读理解 score、反向 SDE 与 probability-flow ODE；本章未实现这些连续时间采样器。

核验边界：2026-11-10 恢复网络后，已下载并逐项核对 `arXiv:2006.11239v2` 中与本章直接相关的正向过程、后验、噪声参数化、训练目标、末步采样和图像 decoder 定义。版本、PDF 哈希、公式号、页码及核对范围见 [来源审计](../source-audit.md)。没有审计整篇论文的实验结果或复现其图像性能；本节其它论文的全文仍未核验。

下一步：[03 · Flow matching](03-flow-matching.md)，比较同一目标分布上的另一种训练路径。
