# 03 · Flow matching：学习搬运分布的速度场

**先修**：[生成模型](01-generative-models.md)、常微分方程、条件期望；建议先读 [DDPM](02-diffusion.md) 便于比较。

**完成后应能**：写出条件插值与速度目标，解释条件回归为何学习边缘速度，区分路径、耦合、训练目标和 ODE 求解器，并训练二维模型。

**实践产物**：同一八高斯任务的 flow 模型结果。进阶练习是用独立 solver_study 命令完成一次 Euler 步数敏感性分析。

## 1. 换一个生成问题

设 $z\sim p_{\mathrm{base}}=\mathcal N(0,I)$，$y\sim p_{\mathrm{data}}$，本章令二者独立。选取简单路径

$$
x_t=(1-t)z+ty,\qquad t\in[0,1].
$$

时间 0 是噪声，时间 1 是数据；这与上一章 DDPM 的加噪时间方向相反。为避免符号误读，本章用 $z,y$ 表示路径两端，不沿用 DDPM 中“$x_0$ 总是干净数据”的约定。

单条路径的条件速度是

$$
u_t(z,y)=\frac{\mathrm d x_t}{\mathrm dt}=y-z.
$$

训练时同时知道 $z,y,t$，所以不需要先求解 ODE 就能构造监督目标。网络只看到 $(x_t,t)$，用

$$
L_{\mathrm{CFM}}=\mathbb E_{z,y,t}\left[\|v_\theta(x_t,t)-(y-z)\|^2\right],
\qquad t\sim\operatorname{Uniform}[0,1].
$$

这里的 CFM 指 conditional flow matching。时间无物理单位；速度是“数据单位 / 归一化时间”。网络输出与 $x_t$ 同维，代表速度，不能直接代入 DDPM 的噪声预测公式。

**与原论文的参数差别**：Lipman et al. 的 arXiv v2 公式 (20)–(23) 使用正的终点噪声标准差 $\sigma_{\min}$，对应

$$
x_t=[1-(1-\sigma_{\min})t]z+ty,\qquad
\dot x_t=y-(1-\sigma_{\min})z.
$$

其终点为 $y+\sigma_{\min}z$，边缘分布是目标数据经过小高斯平滑的结果。本仓库使用 $\sigma_{\min}=0$ 的端点极限，得到上面的 $y-z$ 目标；条件分布在 $t=1$ 退化到数据点，不能把论文针对正条件方差、可逆条件映射的推导不加说明地套到该端点。训练从 $[0,1)$ 抽样，Euler 在最后一次更新前的时间评价网络；这不会自动证明任意离散目标的端点 ODE 正则性。核对位置与范围见 [来源审计](../source-audit.md)。


## 2. 条件路径很多，为什么只学一个速度场？

同一个 $(x,t)$ 可以由许多不同的 $(z,y)$ 产生，直线之间也可以交叉。平方损失的最优解是

$$
v^*(x,t)=\mathbb E[y-z\mid x_t=x].
$$

通过条件期望展开平方误差，得到正交分解

$$
\mathbb E\|v_\theta(x_t,t)-(y-z)\|^2
=\mathbb E\|v_\theta(x_t,t)-v^*(x_t,t)\|^2
+\mathbb E\|v^*(x_t,t)-(y-z)\|^2.
$$

第二项与参数无关。条件目标和边缘速度回归因此有相同的最优解；随机条件速度并不是每个输入的唯一正确标签。

更直接地，取光滑紧支撑测试函数 $f$，在允许交换求导与期望的正则性条件下，

$$
\frac{\mathrm d}{\mathrm dt}\mathbb E[f(x_t)]
=\mathbb E[\nabla f(x_t)^\top(y-z)]
=\mathbb E[\nabla f(x_t)^\top v^*(x_t,t)].
$$

分部积分得到连续性方程的弱形式

$$
\partial_t p_t(x)+\nabla\cdot\big(p_t(x)v^*(x,t)\big)=0.
$$

若速度场与密度满足所需正则性，ODE $\dot x=v^*(x,t)$ 从 $p_0$ 出发便具有这些边缘分布。实际网络和离散求解器近似这一理想过程；遇到离散数据、端点奇异性或不光滑场时，不能直接忽略正则性条件。

**反例与误区**：条件路径画成直线，不代表学到的 ODE 每条轨迹都是直线。直线交叉时，回归场对条件速度取平均；满足唯一性条件的确定性 ODE 轨迹不能在同一时刻、同一位置拥有两个不同速度。训练路径与生成轨迹的边缘分布可以一致，逐样本配对却不相同。

## 3. “直线”不自动意味着全局最优传输

需要区分四个选择：

| 选择 | 本章实现 | 改变它的影响 |
| --- | --- | --- |
| 端点耦合 $\pi(z,y)$ | 独立抽样 $p_{\mathrm{base}}p_{\mathrm{data}}$ | 改变训练路径、速度方差与几何 |
| 条件路径 | $(1-t)z+ty$ | 改变边缘分布与回归目标 |
| 时间采样与权重 | 均匀时间、无权重 MSE | 改变训练资源分配 |
| ODE 求解器 | 固定步长显式 Euler | 改变离散误差与计算成本 |

固定一对端点的直线路径具有简单的传输解释，但独立端点耦合通常不是源分布与目标分布之间的最优传输耦合。不能因使用线性插值就声称实现了全局 OT、最少步采样或更高质量的图像生成。

“Flow matching”“rectified flow”“diffusion velocity parameterization”之间有联系，也有不同的路径、耦合和训练约定。只看到一个变量名 `v`，不足以认定方法相同；先写出 $x_t$ 和目标的定义。

## 4. 采样为什么需要 ODE 求解器

从新噪声 $x^{(0)}\sim\mathcal N(0,I)$ 出发，在 $t_k=k/K$ 上做显式 Euler：

$$
x^{(k+1)}=x^{(k)}+\Delta t\,v_\theta(x^{(k)},t_k),\qquad
\Delta t=1/K,
\quad k=0,\ldots,K-1.
$$

最后一次网络输入时间是 $(K-1)/K$，更新后到达时间 1。忘记乘 $\Delta t$ 会让每一步位移过大；写成减号则把生成方向反转。Euler 在足够光滑且稳定的条件下具有一阶全局误差 $O(1/K)$，这不是任意训练网络都满足的质量保证。

增加 $K$ 主要减小数值积分误差，不能自动修复错误的速度场；增加训练步数主要改善模型拟合，也不能代替求解器精度检查。不同采样器的比较应报告函数评估次数 NFE；此处 Euler 每步恰好一次网络调用，NFE 为 $K$。

## 5. 动手：与 DDPM 共用数据分布

完成 [README](../../README.md) 的安装后，从仓库根目录首次运行以下命令。已有flow结果时先查看`metrics.json`中的`config.steps`，5000步结果可直接复用；例如使用`artifacts/first-flow`时，下文的`--checkpoint`改为`artifacts/first-flow/checkpoint.pt`。已有800步结果可保留作短训练对照，需要5000步时选择新目录，例如`artifacts/flow-5000`，并同步checkpoint路径。

```bash
python -m genai_study.experiment --method flow --steps 5000 --seed 7 --output artifacts/flow
```

默认CPU、单线程、batch size 256、2048个生成样本，Euler采样100步。`--steps`控制训练更新次数；推荐显式使用5000次，省略时CLI默认800次。已有三seed实测中，5000次更新的目标邻域比例为92.38%–92.48%，具体成本与800步对照见[实测报告](../experiment-report.md)。监督数据是二维八高斯；每次训练独立采样源噪声和目标点。输出的损失采用坐标与batch的均值。

观察损失曲线、样本散点图和 JSON 诊断，特别注意：

- DDPM 的标签是噪声，flow 的标签是端点差，尺度与难度不同。两者最终 MSE 不能直接排名。
- 最近中心距离可能奖励模式坍塌；模态覆盖只反映模式是否出现，不能保证权重或形状正确。结合覆盖、模式附近比例、energy distance 和图形判断。
- 较低训练损失不代表时间积分准确。冻结同一模型、固定同一组初始噪声，分别使用 10、25、100、200 步积分，比较结果，这才是在隔离求解器步数的影响。训练命令会保存 `checkpoint.pt`；使用下面的独立求解器实验冻结同一个模型及初始噪声，不要重新训练四次后声称控制了模型。
- 一次固定 seed 的运行只能证明该配置下的观测；方法比较应使用多个训练 seed，报告变化范围及运行成本。

**理论预期**：充分拟合的速度场与足够准确的积分应把源分布搬运到八高斯附近。**实际测量**：以运行输出为准。**未验证范围**：高维图像或视频、语义条件控制、最优传输耦合、似然评价及大规模训练。

### 可运行的求解器对照

完成上面的训练后运行以下独立采样命令。若`artifacts/flow-solver`已有结果，先核对其`metrics.json`的`checkpoint.metadata.config.steps`与训练配置，再读取JSON和图；改用新的5000步checkpoint时，将`--output`也改为新目录，例如`artifacts/flow-solver-5000`。结果按所用训练预算解释；[历史800步求解器实测](../experiment-report.md#固定模型的求解器对照)另提供独立复现命令。

```bash
python -m genai_study.solver_study --checkpoint artifacts/flow/checkpoint.pt \
  --steps 10 25 100 200 --samples 2048 --seed 123 --output artifacts/flow-solver
```

该命令不训练，记录权重SHA、相同初始噪声、每种步数的NFE、终点诊断与相邻离散方案之间的配对RMS距离。配对距离是两个数值解的差，不是真解误差；更小的差也不保证模型更接近数据。推理checkpoint不含优化器状态，不提供严格续训。只使用自己训练的可信checkpoint，详见 [实验说明](../../examples/README.md)。

## 6. 解析验证与练习

### A. 路径与端点

验证 $x_{t=0}=z$、$x_{t=1}=y$，并用中心有限差分检查内部时间点的导数为 $y-z$。

**练习**：若路径写为 $(1-t)y+tz$ 而标签仍写 $y-z$，哪里错了？

**答案**：路径真正导数是 $z-y$，生成起终点也颠倒。这个错误可能仍能产生下降的损失，因此必须检查路径定义和端点。

### B. 常速度场是求解器的最小解析基准

设 $v(x,t)=c$，真实解 $x(1)=x(0)+c$。任意正整数 $K$ 的 Euler 在浮点容差内都应得到这一结果。它能检查时间步长、循环次数和方向；不能验证一般非线性速度场的精度。

再用 $v(x,t)=x$，真实解 $x(1)=e x(0)$，Euler 解为 $(1+1/K)^Kx(0)$，可检验增大 $K$ 是否带来正确的收敛趋势。

### C. 一个能完全推导的边缘速度

设 $z\sim\mathcal N(0,I)$、$y\sim\mathcal N(m,I)$，二者独立。令

$$
s_t^2=(1-t)^2+t^2.
$$

则 $x_t\sim\mathcal N(tm,s_t^2I)$，且

$$
v^*(x,t)=m+\frac{2t-1}{s_t^2}(x-tm).
$$

**推导提示**：$\mathbb E[y-z]=m$，$\operatorname{Cov}(y-z,x_t)=(2t-1)I$，再用联合高斯条件均值。

**自查**：在 $t=1/2$，速度场恰为常量 $m$，但条件训练目标 $y-z$ 仍有方差。若 $m=0$，源与目标边缘相同，中间路径方差却先从 1 缩小到 $1/2$、再扩大到 1。这说明独立端点直线插值不一定是最省力的分布路径。

### D. 为什么流损失也不必降到零？

上面的高斯例子中，每坐标条件方差为

$$
\operatorname{Var}(y-z\mid x_t)=2-\frac{(2t-1)^2}{s_t^2}=\frac{1}{s_t^2}.
$$

因此即使预测器等于解析 $v^*$，条件目标 MSE 也为正。在 $t=1/2$ 每坐标残差为 2；这属于目标的条件不确定性，而非数值求解误差。

### E. 研究延伸

设计实验比较独立耦合与小 batch 近似 OT 配对。必须固定数据、网络、训练预算、时间采样与求解器，记录配对本身的计算成本；不要把“更直的轨迹”直接等同于“更好的生成分布”。本仓库尚未实现这一扩展。

## 7. 来源、版本与适用范围

- Lipman et al., **Flow Matching for Generative Modeling**, ICLR 2023：[论文](https://arxiv.org/abs/2210.02747)。阅读重点是边缘与条件 flow matching 目标及 Gaussian probability paths；原论文讨论比本章独立配对直线实验更一般的路径。
- Meta 官方 `facebookresearch/flow_matching`：[固定版本 `affine.py`](https://github.com/facebookresearch/flow_matching/blob/11568d37f8d5a080e12aa7b5305d9c35ae07d136/flow_matching/path/affine.py)。2026-11-10 核对了 $x_t=\sigma_t x_0+\alpha_t x_1$ 与导数目标，以及 `CondOTProbPath` 的 $\alpha_t=t,\sigma_t=1-t$。这是后续官方软件实现，不能当作 2023 论文最初发布的代码版本。
- Liu, Gong, Liu, **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow**, ICLR 2023：[论文](https://arxiv.org/abs/2209.03003)。用于对照 rectification 与耦合更新；本仓库未实现 reflow。

核验边界：2026-11-10 恢复网络后，已下载并核对 `arXiv:2210.02747v2` 的公式 (5)–(15)、(20)–(23)、(26) 及相关定理条件，并补充本仓库零终点噪声与原文正终点噪声的区别。版本、PDF 哈希、页码和未覆盖范围见 [来源审计](../source-audit.md)。这不是整篇论文实验的复现，也未核验本节其它论文的全文。

下一步：[05 评价与实验诊断](05-evaluation.md)，随后完成07/08图像实践与06视频反例，再进入[04研究路线](04-image-video-roadmap.md)。

理论延伸：[从噪声预测到生成速度](../readings/diffusion-flow-bridge.md)沿同一高斯路径推导边缘速度，核对反向时间、概率流1/2系数、VP的v参数化与DDIM/SVD坐标；包含可直接运行的解析高斯检查。
