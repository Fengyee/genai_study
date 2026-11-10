# 01 · 生成模型的共同问题

**先修与依赖：** [00 概率与噪声](00-foundations.md)，梯度优化。**目标：** 把常见模型放进同一个“分布、训练、采样、评价”框架；推导 ELBO；解释为什么某个训练目标不能直接代表全部生成质量。

## 1. 先问四个问题

遇到一个新方法时先写下：

1. 它在什么表示空间建模？像素、离散 token、压缩潜变量还是某种特征？
2. 训练时真正优化什么？精确似然、下界、对抗目标、去噪回归还是速度回归？
3. 生成需要什么过程？祖先采样、一次网络前向、马尔可夫链或 ODE 求解？
4. 论文的实验实际证明了什么？样本观感、分布覆盖、条件遵循、速度或某个任务效用？

这些维度可以组合。使用 Transformer 不意味着模型是自回归；使用 VAE 压缩器也不意味着后续生成器以 VAE 目标训练。

## 2. 似然与自回归：把联合分布拆开

链式法则给出

$$
p_\theta(x_1,\ldots,x_D)=\prod_{i=1}^D p_\theta(x_i\mid x_{<i}).
$$

训练时对观测序列求负对数似然；在适当因果掩码下可以并行计算所有位置的损失。生成时必须按依赖顺序逐个或逐组产生变量。训练并行不等于采样也并行。

对高维图像，可选择像素序列或压缩后的离散 token。变量顺序、表示压缩和条件分解都会影响代价与归纳偏置。连续输入的条件分布还需要指定输出密度族，不是简单把分类 cross-entropy 原样搬过来。

**反例：** 一个模型记住训练集，训练似然很好；面对不同分布的输入或新组合可能失败。似然训练并不免除泛化检查。

## 3. VAE：用变分推断处理难积分

设生成模型 $p_\theta(x,z)=p(z)p_\theta(x\mid z)$，引入近似后验 $q_\phi(z\mid x)$。固定观测 $x$，假设 $0<p_\theta(x)<\infty$，$q_\phi(z\mid x)$ 只在真实后验非零处取正值，且以下期望与 KL 有限，则

$$
\begin{aligned}
\log p_\theta(x)
&=\mathbb E_{q_\phi(z\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
 +D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p_\theta(z\mid x)\right)\\
&=\underbrace{\mathbb E_q[\log p_\theta(x\mid z)]
-D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))}_{\mathcal L_{\mathrm{ELBO}}(x)}
 +D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p_\theta(z\mid x)).
\end{aligned}
$$

最后一项非负，因此 ELBO 是对数似然下界。等号成立当近似后验等于真实后验（几乎处处）。两个 KL 都沿着 $q_\phi$ 到相应生成分布的方向；由 $\log p_\theta(x,z)=\log p_\theta(z\mid x)+\log p_\theta(x)$ 取 $q_\phi$ 期望即可核对。最大化下界既涉及生成模型拟合，也涉及变分近似是否足够表达真实后验；比较两次训练的似然时，还要考虑后验 KL 差距的变化。

若 $q_\phi$ 为对角高斯，可写

$$
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

这将采样的随机性转移到与参数无关的 $\epsilon$，在可交换求导与期望的条件下，便于对 $\mu,\sigma$ 使用路径导数。离散潜变量需选择相应的梯度估计方法。

重建项由观测模型决定：固定方差高斯给出带缩放和常数的平方误差，Bernoulli 给出二元交叉熵。对同一个生成模型，改变重建与 KL 的相对权重会改变标准 ELBO 目标。练习 B 展开维度求和与均值的差异。

## 4. GAN：通过判别信号学习生成器

原始 GAN 的理想化目标为

$$
\min_G\max_D\;\mathbb E_{x\sim p_{\mathrm{data}}}\log D(x)
+\mathbb E_{z\sim p(z)}\log(1-D(G(z))).
$$

在分布具有共同基准测度下的密度、判别器容量无限且可精确优化等条件下，固定 $G$，在 $p_{\mathrm{data}}(x)+p_G(x)>0$ 处的最优判别器为

$$
D^*(x)=\frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x)+p_G(x)}.
$$

两种分布密度都为零的位置，对目标无贡献，$D$ 的取值可任意选择。令 $m=(p_{\mathrm{data}}+p_G)/2$，Jensen–Shannon 散度定义为 $D_{\mathrm{JS}}(p_{\mathrm{data}}\Vert p_G)=\tfrac12D_{\mathrm{KL}}(p_{\mathrm{data}}\Vert m)+\tfrac12D_{\mathrm{KL}}(p_G\Vert m)$。将 $D^*$ 代回原目标得到 $-\log4+2D_{\mathrm{JS}}(p_{\mathrm{data}}\Vert p_G)$。

原论文的收敛分析还假设每轮判别器达到最优，并在分布空间中充分小地更新 $p_G$。有限网络交替更新参数的训练行为需要单独验证。实践常用非饱和生成器损失

$$
L_G^{\mathrm{NS}}=-\mathbb E_{z\sim p(z)}\log D(G(z)),
$$

以增强判别器强烈拒绝生成样本时的梯度；上面的 JS 等式对应原始 minimax 目标。练习 E 直接计算这两个生成器损失的梯度。

GAN 可以快速生成高观感样本，但覆盖率、优化稳定性和精确似然通常需要分别讨论。**“所有 GAN 都会模式崩溃”与“高保真说明没有模式丢失”都不成立。**

## 5. 可逆流与连续流

设 $x,z$ 同维，离散可逆变换 $x=f_\theta(z)$ 及其逆在相关区域连续可微，Jacobian 行列式非零。变量替换给出

$$
\log p_\theta(x)=\log p_Z(f_\theta^{-1}(x))+
\log\left|\det \frac{\partial f_\theta^{-1}(x)}{\partial x}\right|.
$$

等价地，在生成方向写成 $\log p_\theta(f_\theta(z))=\log p_Z(z)-\log|\det J_{f_\theta}(z)|$。绝对值保证反转坐标方向的变换也产生非负密度；练习 C 检查尺度变化造成的体积因子。Real NVP 论文把数据到潜变量的映射命名为 $f$，对应这里的 $f_\theta^{-1}$，阅读时需先对齐方向。

可逆性和可计算 Jacobian 行列式约束了模型结构。连续归一化流用 $\dot x_t=v_\theta(x_t,t)$ 定义演化；例如速度场对状态局部 Lipschitz、对时间连续，并保证相应正反向解在整个积分区间存在唯一时，可构造可逆流。对足够光滑的密度和速度场，沿轨迹有

$$
\frac{d}{dt}\log p_t(x_t)=-\nabla\cdot v_\theta(x_t,t)
=-\operatorname{tr}\!\left(\frac{\partial v_\theta}{\partial x}(x_t,t)\right).
$$

这里是随 $x_t$ 一起移动的总导数；似然计算需要同时积分密度变化并计算或估计散度。唯一性条件很关键：$\dot x=\sqrt{|x|}$ 从 $x(0)=0$ 出发可以一直静止，也可以在任意等待时间后沿抛物线离开，因此该初值不能确定唯一映射。

后续的 flow matching 是训练连续生成流的方法，通常可直接构造训练样本；其采样、密度积分和散度计算各有相应的数值成本。

## 6. 扩散与 flow：把难分布变成随时间的简单任务

扩散模型先定义对数据逐步加噪的过程，训练网络学习与逆过程有关的量，再从噪声出发生成。网络可预测噪声、干净数据或其它等价参数化，但权重、噪声日程和损失定义决定等价性的具体范围。

Flow matching 指定一条从简单分布通往数据的概率路径，通过回归条件速度来学习边缘速度。训练样本往往可直接构造；采样仍需要数值积分。它与扩散在概率路径和 ODE 表达上存在联系，不能仅按“有没有噪声”或“是不是 Transformer”划分。

| 模型族 | 典型训练信号 | 典型生成路径 | 首要核对点 |
| --- | --- | --- | --- |
| 自回归 | 条件似然之和 | 按依赖顺序采样 | 序列顺序、表示和采样延迟 |
| VAE | ELBO | 先验 → 解码条件分布 | 后验近似与重建分布假设 |
| GAN | 对抗目标 | 潜变量 → 生成器 | 优化、覆盖率与评价协议 |
| 可逆流 | 变量替换后的似然 | 可逆映射或 ODE | 结构约束、Jacobian/散度成本 |
| 扩散 | 去噪/score 等目标 | 逆随机过程或相关 ODE | 噪声日程、参数化、采样器 |
| Flow matching | 速度回归 | 数值积分 ODE | 路径、耦合、时间方向、求解误差 |

表格概括典型设计，不是互斥分类。例如潜扩散模型可先使用自编码器，再在潜空间训练扩散模型。

## 7. 评价要对应问题

“好生成模型”没有单一总分。至少分开问：样本是否可信、是否覆盖不同模式、是否遵循条件、是否记忆训练数据、生成成本是多少。二维 toy 分布可以直接可视化模式并研究算法机制，但不能推断人类对图像的审美偏好或视频物理一致性。

FID 等指标比较特征空间中的统计量；即使其估计值下降，也不自动说明文本对齐、时间一致性或安全性改善。使用不同样本数、预处理或特征提取器的数值不能直接放在一张排行榜中。详细研究设计见 [04 图像与视频路线](04-image-video-roadmap.md)。

## 8. 练习与参考答案

**A · ELBO 差距。** 设某样本 ELBO 为 $-5$ nat，后验 KL 差距为 $0.7$ nat，求对数似然。若下一次 ELBO 为 $-4.8$，能否确定似然提高？

<details><summary>答案</summary>

第一次对数似然为 $-4.3$ nat。第二次后验 KL 未知，无法仅凭下界判断；例如差距为 0.1 时似然是 $-4.7$，反而更低。

</details>

**B · 重建项。** 令 $p_\theta(x\mid z)=\mathcal N(\mu_\theta(z),\sigma^2I)$，固定 $\sigma$。写出负对数似然与平方误差的关系。

<details><summary>答案</summary>

$-\log p_\theta(x\mid z)=\|x-\mu_\theta(z)\|^2/(2\sigma^2)+(D/2)\log(2\pi\sigma^2)$。若代码对维度求均值，要补上维数因子；否则 KL 相对权重会改变。

</details>

**C · 可逆变换。** 一维 $Z\sim\mathcal N(0,1)$，$X=2Z$。求 $p_X(x)$，检查遗漏 Jacobian 会发生什么。

<details><summary>答案</summary>

$p_X(x)=p_Z(x/2)/2$，即 $\mathcal N(0,4)$。遗漏 $1/2$ 后密度积分等于 2，无法归一化。

</details>

**D · 指标反例。** 数据均匀分布在两个相距很远的模式。模型只生成其中一个模式，单个样本几乎完美。应该增加什么检查？

<details><summary>答案</summary>

检查模式覆盖与频率，而不只看单样本质量。二维例子可预先定义模式邻域并统计落点；真实图像需说明特征、邻域与评价模型的限制。最近邻检查可辅助发现记忆，但不能单独证明没有训练数据泄漏。

</details>

**E · GAN 的生成器梯度。** 令判别器输出 $D(G(z))=\operatorname{sigmoid}(a)$。分别求单样本 minimax 损失 $\log(1-D(G(z)))$ 和非饱和损失 $-\log D(G(z))$ 对 logit $a$ 的导数。$D(G(z))\to0$ 时各自如何变化？

<details><summary>答案</summary>

两个导数分别为 $-D(G(z))$ 与 $D(G(z))-1$，极限分别为 $0$ 与 $-1$。非饱和目标在这一情况下提供更强的 logit 梯度；传到生成器参数的梯度仍需乘以 $\partial a/\partial\theta_G$。这解释了原论文选择替代损失的动机，也说明损失值与梯度行为要分别检查。

</details>

## 原始来源与版本

- Kingma & Welling, *Auto-Encoding Variational Bayes*, **ICLR 2014**，[arXiv:1312.6114v11](https://arxiv.org/abs/1312.6114v11)，§2–3：ELBO 和重参数化。
- Goodfellow et al., *Generative Adversarial Nets*, **NeurIPS 2014**，[arXiv:1406.2661v1](https://arxiv.org/abs/1406.2661v1)，§3–4：非饱和损失、最优判别器和理想化散度分析。
- Dinh et al., *Density Estimation using Real NVP*, **ICLR 2017**，[arXiv:1605.08803v3](https://arxiv.org/abs/1605.08803v3)，§3.1–3.3：变量替换与可计算的耦合层。
- Chen et al., *Neural Ordinary Differential Equations*, **NeurIPS 2018**，[arXiv:1806.07366v6](https://arxiv.org/abs/1806.07366v6)，§4、§6 和附录 A：连续归一化流、唯一性及密度演化。
- Ho et al., *Denoising Diffusion Probabilistic Models*, **NeurIPS 2020**，[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)：去噪训练与生成过程。
- Lipman et al., *Flow Matching for Generative Modeling*, **ICLR 2023**，[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)：概率路径与条件 flow matching。

2026-11-10 已按固定版本核对以上前四篇的指定公式和条件，PDF 哈希、页码、独立解析检查及 Neural ODE v6 的实验修订说明见 [01 章来源审计](../generative-source-audit.md)。正式发表年份与所读 arXiv 修订日期分别记录；本轮验证范围为数学主张和教学例子。

下一步：[02 扩散](02-diffusion.md)，再读 [03 Flow matching](03-flow-matching.md)。
