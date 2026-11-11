# 少步生成：从积分速度场到学习轨迹端点

**先修：** [02扩散](../chapters/02-diffusion.md)、[03 Flow Matching](../chapters/03-flow-matching.md)、[扩散与flow桥梁](diffusion-flow-bridge.md)中的概率流ODE、条件期望。**学习目标：** 区分数值步数、引导计算与训练目标；推导一致性端点映射；用解析Gaussian例检查边界、轨迹配对和少步误差。

本页连接[04研究路线](../chapters/04-image-video-roadmap.md)与[09图像](../chapters/09-pretrained-image-lab.md)、[10视频](../chapters/10-pretrained-video-lab.md)的推理预算，选取2023年原始一致性模型（Consistency Models, CM）解释一个具体机制。实践只需基础NumPy环境，执行解析公式与小数组运算；真实学生模型的训练、图像质量和硬件加速比需要另外测量。

## 1. “减少调用”可以改变哪些对象？

| 操作 | 改变的对象 | 需要核对的预算 |
| --- | --- | --- |
| 把已有Euler/确定性DDIM（`eta=0`）时间网格变稀 | 同一预测器的数值积分区间 | 每区间求值次数、时间网格、误差 |
| 引导蒸馏（guidance distillation） | 将教师的条件/基线引导组合学入学生输出 | 每区间的条件分支数、batch与网络调用数 |
| 学习一致性端点映射 | 将任意噪声级别的状态映射到固定终点 | 端点预测次数；多步时还包含重新加噪 |

例如[Kontext阅读卡](flux-kontext-2025.md)中固定的dev路径使用guidance-distilled模型，每个区间调用一次学生，默认仍有30个区间。它压缩了引导计算；跨越多少时间区间由采样配置和模型的训练适用范围共同决定。[LTX-2阅读卡](ltx2-video-audio-2026.md)记录的8+3次调用也应按各阶段实际接口解释。下面的CM是独立机制案例，这两张阅读卡中的模型各自沿用其原始训练方案。

## 2. 端点映射沿概率流轨迹保持恒定

采用方差爆炸（variance-exploding, VE）扰动：数据$Y\sim p_{\mathrm{data}}$、独立噪声$Z\sim\mathcal N(0,I)$，单时刻边缘为

$$X_\sigma=Y+\sigma Z,\qquad p_\sigma=p_{\mathrm{data}}*\mathcal N(0,\sigma^2I).$$

本页$\sigma$为噪声标准差，沿数据到噪声方向增加；生成从$\sigma_{\max}$积分到正的$\sigma_{\min}$。原论文用$t,\epsilon,T$分别表示这里的$\sigma,\sigma_{\min},\sigma_{\max}$；02章的离散时间和03章的生成时间沿用各章约定。

对应VE SDE的扩散系数为$\sqrt{2\sigma}$，概率流ODE为

$$\frac{dx}{d\sigma}=w_\sigma(x),\qquad
w_\sigma(x)=-\sigma\nabla_x\log p_\sigma(x).$$

假设该区间内的速度场生成唯一、不爆炸的ODE流。记从状态$(x,\sigma)$沿该流到$\sigma_{\min}$的解为$f^*(x,\sigma)$。在同一条ODE轨迹上，

$$f^*(x_\sigma,\sigma)=x_{\sigma_{\min}},\qquad
f^*(x,\sigma_{\min})=x.$$

若可微，第一式等价于输运关系$\partial_\sigma f^*+J_x f^*\,w_\sigma=0$。它学习的是**整段轨迹的终点**。Euler则利用局部速度推进一个区间，网格变稀后单步跨越更长距离。

从真实$p_{\sigma_{\max}}$出发，精确$f^*$输出$p_{\sigma_{\min}}$。正终点仍含方差为$\sigma_{\min}^2$的扰动；原论文使用很小的正终点近似数据分布。有限$\sigma_{\max}$时，常用起点$\mathcal N(0,\sigma_{\max}^2I)$与真实$p_{\sigma_{\max}}$之间也有近似误差。端点映射、初始边缘和数值求解分别决定不同误差来源。

## 3. 怎样训练端点映射？

原文式(5)用边界参数化

$$f_\theta(x,\sigma)=c_{\mathrm{skip}}(\sigma)x+
c_{\mathrm{out}}(\sigma)F_\theta(x,\sigma),\quad
c_{\mathrm{skip}}(\sigma_{\min})=1,\quad
c_{\mathrm{out}}(\sigma_{\min})=0.$$

它让边界精确等于输入。单独要求两端预测相同会允许全局常数解；边界恒等映射为端点学习提供锚点。

**一致性蒸馏（consistency distillation, CD）。** 取相邻噪声级别$\sigma_n<\sigma_{n+1}$，从带噪数据$x_{n+1}$出发，用固定的score教师求解一个区间，得到$\hat x_n$。以Euler为例，

$$\hat x_n=x_{n+1}+(\sigma_n-\sigma_{n+1})
[-\sigma_{n+1}s_\phi(x_{n+1},\sigma_{n+1})].$$

随后比较两个**端点预测**：

$$\mathcal L_{\mathrm{CD}}=
\mathbb E\left[\lambda(\sigma_n)
d\big(f_\theta(x_{n+1},\sigma_{n+1}),
\operatorname{stopgrad}[f_{\theta^-}(\hat x_n,\sigma_n)]\big)\right].$$

学生参数$\theta$通过梯度更新；目标网络$\theta^-$按指数移动平均（EMA）更新，$\theta^-\leftarrow\mu\theta^-+(1-\mu)\theta$，其输出停止梯度。教师$\phi$负责生成相邻状态，EMA目标网络负责提供端点目标。本文固定的作者代码在CD中使用Heun求解一个区间，调用教师两次，再调用在线学生与目标网络；训练预算与采样预算需分别统计。

原文定理1在Lipschitz、局部求解误差阶、零蒸馏损失及$\theta^-=\theta$等条件下，将离散化加密与教师ODE的端点误差联系起来。它针对教师定义的概率流；实际训练还需评估教师误差、学生拟合与采样起点。

**一致性训练（consistency training, CT）。** 原始CT目标直接复用同一份数据与噪声，比较$f_\theta(Y+\sigma_{n+1}Z,\sigma_{n+1})$与目标网络$f_{\theta^-}(Y+\sigma_nZ,\sigma_n)$。这种配对可以从数据构造，无需score教师求解。两时刻各自有正确的扰动边缘，但一般属于不同的精确概率流轨迹。

原文定理2给出这一步的依据：在精确score教师、Euler、距离函数$d$与目标网络$f_{\theta^-}$具有有界二阶导数、权重有界和score二阶矩有限等条件下，$\mathcal L_{\mathrm{CD}}=\mathcal L_{\mathrm{CT}}+o(\Delta\sigma)$，其中$\Delta\sigma$为最大网格间隔。这是网格加密时的损失关系。下面用两个端点不相等的数值反例帮助区分条件加噪配对与ODE轨迹。

## 4. Gaussian解析例：局部推进与一次端点求值

取一维数据$Y\sim\mathcal N(m,s^2)$，$s>0$，定义$q(\sigma)=s^2+\sigma^2$。此时

$$w_\sigma(x)=\frac{\sigma(x-m)}{q(\sigma)},\qquad
x_\sigma=m+\sqrt{q(\sigma)}z,\qquad
f^*(x,\sigma)=m+\sqrt{\frac{q(\sigma_{\min})}{q(\sigma)}}(x-m).$$

中间表达式描述固定$z$的一条精确轨迹：微分后直接得到上面的速度。代入$f^*$即可验证沿轨迹恒定性与边界。若$z\sim\mathcal N(0,1)$，这些轨迹在每个噪声级别具有正确的Gaussian边缘。

与[DDPM oracle实验](../ddpm-oracle-report.md)中的条件均值联系：当前状态对应的干净数据MMSE去噪器为

$$D_\sigma(x)=\mathbb E[Y\mid X_\sigma=x]
=m+\frac{s^2}{q(\sigma)}(x-m).$$

$D_\sigma$回归可能的干净数据的后验均值；$f^*$沿边缘概率流输运状态。两种训练对象的系数不同，即使令$\sigma_{\min}\to0$也一般不同。对本例$s=0.7,\sigma=2,\sigma_{\min}=0.1$，两个系数分别约为0.10913和0.33370。

下面固定$m=1.5,s=0.7,\sigma_{\min}=0.1,\sigma_{\max}=2$，用9个固定$z$测试相同初始状态和相同终点。数组检查无需随机抽样；方差按完整Gaussian总体的线性变换解析计算，独立于这9个点的样本方差。计算规模小于百次小数组速度求值。

<!-- cpu-check: consistency-endpoint-map -->
```python
import numpy as np

m, data_std, lower, upper = 1.5, 0.7, 0.1, 2.0
z = np.linspace(-2.0, 2.0, 9)

def q(sigma):
    return data_std**2 + sigma**2

def point(sigma):
    return m + np.sqrt(q(sigma)) * z

def velocity(x, sigma):
    return sigma * (x - m) / q(sigma)

def endpoint(x, sigma):
    return m + np.sqrt(q(lower) / q(sigma)) * (x - m)

expected = point(lower)
np.testing.assert_allclose(endpoint(expected, lower), expected, rtol=0, atol=0)
for sigma in (lower, 0.4, 1.0, upper):
    np.testing.assert_allclose(endpoint(point(sigma), sigma), expected,
                               rtol=0, atol=1e-14)

errors = []
print("Euler calls | max endpoint error | population final variance")
for calls in (1, 2, 4, 16, 64):
    x, coefficient = point(upper).copy(), 1.0
    grid = np.linspace(upper, lower, calls + 1)
    for sigma, next_sigma in zip(grid[:-1], grid[1:]):
        h = next_sigma - sigma
        x = x + h * velocity(x, sigma)
        coefficient *= 1 + h * sigma / q(sigma)
    error = float(np.max(np.abs(x - expected)))
    errors.append(error)
    print(f"{calls:11d} | {error:.8f} | {q(upper)*coefficient**2:.8f}")
assert all(a > b for a, b in zip(errors[:-1], errors[1:]))
exact_error = np.max(np.abs(endpoint(point(upper), upper) - expected))
assert exact_error < 1e-14

posterior_mean = m + data_std**2 / q(upper) * (point(upper) - m)
assert not np.allclose(posterior_mean, endpoint(point(upper), upper))
clean, noise = 0.9, 0.8
paired = [float(endpoint(clean + sigma * noise, sigma)) for sigma in (0.4, 1.0)]
assert abs(paired[0] - paired[1]) > 0.1

tau = 1.2
renoised_variance = q(lower) + tau**2 - lower**2
np.testing.assert_allclose(renoised_variance, q(tau), rtol=0, atol=1e-15)
np.testing.assert_allclose(renoised_variance * q(lower) / q(tau), q(lower),
                           rtol=0, atol=1e-15)
print(f"One analytic endpoint evaluation: max error={exact_error:.3e}")
print(f"Target variance={q(lower):.8f}; clean variance={data_std**2:.8f}")
print("Shared-clean/shared-noise pair endpoints:", paired)
```

基础环境float64实际结果：

| Euler速度求值次数 | 最大逐点端点误差 | 输出总体方差 |
| --- | --- | --- |
| 1 | 0.76295130 | 0.10603563 |
| 2 | 0.50084909 | 0.20855866 |
| 4 | 0.27705969 | 0.32327973 |
| 16 | 0.07390368 | 0.44910765 |
| 64 | 0.01882838 | 0.48677495 |

解析端点映射一次求值的最大误差约$2.22\times10^{-16}$；目标$p_{\sigma_{\min}}$方差为0.50，干净数据方差为0.49。这里预先知道完整Gaussian分布和精确解，因此“一次”只统计解析函数求值。学习到相同精度需要另外训练与验证；本表衡量的是指定网格的Euler截断误差。

共用$Y=0.9,Z=0.8$时，$\sigma=0.4,1$两个状态的精确端点分别约为1.25442、1.61586。沿同一概率流轨迹的端点必须相等，这个差值直接检出把CT条件配对当作精确ODE配对的错误。

本例从真实$p_{\sigma_{\max}}=\mathcal N(1.5,4.49)$对应的状态开始。若改用零均值、方差4的高斯起点，精确$f^*$也会输出错误边缘：均值约1.00、方差约0.44543。精确端点公式保留了给定初始分布带来的影响。

## 5. CM的多步采样为何重新加噪？

原文算法1先做一次$x=f_\theta(x_{\sigma_{\max}},\sigma_{\max})$，得到终点预测。要在噪声级别$\tau>\sigma_{\min}$继续精化，就用新独立噪声构造

$$\hat x_\tau=x+\sqrt{\tau^2-\sigma_{\min}^2}\,Z',\qquad
x\leftarrow f_\theta(\hat x_\tau,\tau).$$

若当前$x$精确服从$p_{\sigma_{\min}}$，重新加噪会得到$p_\tau$，精确$f^*$再将它送回$p_{\sigma_{\min}}$。上面的代码用Gaussian方差验证这件事。近似网络可以使用多次预测精化输出，其具体质量收益需按模型实验评估。

作者代码`sample_onestep`调用模型一次；`stochastic_iterative_sampler`按选定索引区间调用。固定源码中`steps=40, ts=(0,22,39)`使用两次模型求值：40定义可选网格，三个索引定义两个实际区间。阅读“steps”配置时，沿循环数清预测器调用、每次batch和重新加噪操作，再与质量和耗时一起记录。

## 6. 练习与参考答案

1. **只有自一致性损失时，全局常数$f_\theta(x,\sigma)=c$有什么问题？** 提示：代入损失和边界。答案：两端距离恒为0，但无法对所有输入满足$f_\theta(x,\sigma_{\min})=x$。边界参数化固定了应当保留的端点信息。
2. **一个30区间的CFG教师每区间分别计算条件与基线；学生已蒸馏引导，但仍沿30区间Euler求解。应记录什么？能否直接改成一次端点预测？** 答案：教师共60次分支求值，若合并batch可为30次网络调用；学生为30次单分支调用。先核对学生输出是否以轨迹端点为训练目标，再验证一次调用的终点误差；引导蒸馏的证据对应对教师引导组合的学习。
3. **把Gaussian的$D_\sigma$当成$f^*$，会同时满足边界与沿轨迹恒定性吗？共噪声CT配对能否替代同轨迹验证？** 答案：正边界处$D_{\sigma_{\min}}$的缩放系数$s^2/(s^2+\sigma_{\min}^2)<1$，已不满足恒等边界；代入$x_\sigma=m+\sqrt{q(\sigma)}z$还得到随$\sigma$变化的$m+s^2z/\sqrt{q(\sigma)}$。CT配对具有另一种跨时间耦合，应按定理条件理解损失关系；上面的端点差值给出逐点反例。

## 7. 固定来源与核验范围

| 原始来源 | 固定版本与定位 | 本页依据 |
| --- | --- | --- |
| Song et al., [Consistency Models](https://arxiv.org/abs/2303.01469v1) | arXiv:2303.01469v1，2023-03-02；PDF第2–6页，式(2)–(10)、算法1–3、定理1–2 | VE概率流、端点定义、边界、CD/CT与多步采样 |
| [作者实现：karras_diffusion.py](https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py) | commit `e32b69ee436d518377db86fb2127a3972d0d8716`；`get_scalings_for_boundary_condition`、`consistency_losses`、`sample_onestep`、`stochastic_iterative_sampler` | 边界系数、教师Heun、CT配对、停止梯度及实际调用 |
| [作者实现：train_util.py](https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/train_util.py) | 同一commit；`_update_target_ema`与`forward_backward` | 目标网络EMA及CD/CT训练分派 |

下载核验的v1论文PDF共48,427,384 bytes，SHA-256为`f2f4dbed7f73d182c0b2dad0747d3df8e2c27a7df05d4b2aa5dee00b6ca94d64`；上述`karras_diffusion.py`为27,865 bytes，SHA-256为`b8fcf9f53e63cff19db676814545ee7644259364236de5c10ac3d69007ee5177`；`train_util.py`为22,745 bytes，SHA-256为`bf387ed2d2d4582e9ab069c14c762e3797878bb2231551a54e5fbe36176b1cfc`。

本页核验原始机制、固定源码和解析性质；源码采样函数另以mock调用计数验证。本文数值表来自上面的CPU检查，范围为一维Gaussian、精确速度与端点；其它分布、训练后的模型及后续CM改进需要各自的证据。

来源身份、原作者函数mock计数与两组独立解析核验见[证据索引](../../records/measurements/consistency-endpoint/README.md)。
