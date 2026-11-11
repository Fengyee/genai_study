# 预测目标与损失权重：从VP旋转读懂训练目标

**先修：** [02扩散](../chapters/02-diffusion.md)的噪声MSE、[扩散与流的理论衔接](diffusion-flow-bridge.md)的VP换算，以及条件期望与加权最小二乘。**学习目标：** 推导clean、noise与VP预测误差的比例；为同一目标换算损失权重；解释时间采样、模型容量与端点对训练目标的影响。

这是一个可选专题。范围限定为归一化的方差保持（variance-preserving，VP）路径，使用一个CPU解析块检查代数和梯度，无需训练、权重下载或数据集。阅读后可以沿[视频训练设计卡](video-training-svd-cogvideox.md)中的CogVideoX接口核对预测目标。

## 1. 先固定同一个状态和预测

沿用衔接专题的数据到噪声时间$u$。令数据$Y$与标准高斯噪声$E$独立，数据有有限二阶矩，并定义

$$X_u=\alpha(u)Y+\sigma(u)E,\qquad
\alpha^2+\sigma^2=1,\qquad
V_{\mathrm{VP}}=\alpha E-\sigma Y.$$

以下先讨论$\alpha,\sigma>0$的内部时间。$\alpha$是幅度，对应DDPM的$\sqrt{\bar\alpha_t}$；单步$\alpha_t=1-\beta_t$沿用02章的定义。VP标签记为$V_{\mathrm{VP}}$、网络预测记为$\widehat v_{\mathrm{VP}}$；全仓库的$v_\theta(x,t)$继续表示flow速度。

这两个等式构成正交旋转，逆变换为

$$Y=\alpha X_u-\sigma V_{\mathrm{VP}},\qquad
E=\sigma X_u+\alpha V_{\mathrm{VP}}.$$

固定同一训练样本$(Y,E,u)$和输入$X_u$，从**同一个**$\widehat v_{\mathrm{VP}}$构造

$$\widehat Y=\alpha X_u-\sigma\widehat v_{\mathrm{VP}},\qquad
\widehat E=\sigma X_u+\alpha\widehat v_{\mathrm{VP}}.$$

此处比较的是这组相互对应的预测。若代码在换算后加入裁剪或阈值操作，需要重新核对变换后的对应关系。固定版本Diffusers的DDIM源码在`step`中先执行上述转换，再处理`clip_sample`或`thresholding`。

## 2. 一次相减，得到三种误差和权重

把预测减去标签，得到

$$\delta Y=-\sigma\delta V_{\mathrm{VP}},\qquad
\delta E=\alpha\delta V_{\mathrm{VP}}.$$

令$L_Y=\|\widehat Y-Y\|^2$、$L_E=\|\widehat E-E\|^2$、$L_V=\|\widehat v_{\mathrm{VP}}-V_{\mathrm{VP}}\|^2$。三个损失采用相同的特征维求和或求平均方式，则逐样本有

$$L_Y=\sigma^2L_V,\qquad L_E=\alpha^2L_V
=S(u)L_Y,\qquad S(u):=\frac{\alpha^2}{\sigma^2}.$$

$S$是噪声调度的**名义信噪比（SNR）**。对某个方差为$s_Y^2$的数据坐标，信号方差与噪声方差之比为$S s_Y^2$；只有$s_Y^2=1$时才等于$S$。VP的系数归一化给出上面的旋转恒等式，实际数据方差仍由数据分布决定。

因此，把三种无权重损失统一写到clean误差坐标，会得到三种时间权重：

| 使用的无权重MSE | 写成clean误差 | 对应时间权重 |
| --- | --- | --- |
| clean预测 | $L_Y$ | $1$ |
| noise预测 | $L_E=S L_Y$ | $S$ |
| VP预测 | $L_V=(1+S)L_Y$ | $1+S$ |

若希望保留指定目标$\lambda_Y(u)L_Y$，其中$\lambda_Y\ge0$仅依赖时间，换预测接口时使用：

| 网络预测对象 | 应乘在该MSE上的权重 |
| --- | --- |
| clean $Y$ | $\lambda_Y$ |
| noise $E$ | $\lambda_Y/S$ |
| VP标签 $V_{\mathrm{VP}}$ | $\lambda_Y/(1+S)=\lambda_Y\sigma^2$ |

这些等式在取batch平均之前成立。实现时先算每个样本的特征MSE，再乘该样本对应的时间权重；先混合不同时间的误差再乘一个平均权重，通常会改变目标。

**梯度也可以逐项对应。** 若$\widehat v_{\mathrm{VP}}$由同一参数$\theta$计算，其余两个预测由上述仿射关系构造，且系数与$\theta$无关，则三个加权标量损失作为$\theta$的函数相等，所以对同一$\theta$的梯度也相等。分别搭建并初始化三套网络时，还需核对它们表示的函数族、参数化与优化过程，才能比较训练轨迹。

02章从固定反向方差的高斯KL推导出另一组时间系数，随后介绍DDPM省略该系数的简化目标。这里的权重来自**预测坐标之间的误差缩放**；指定ELBO或其它目标后，可再用本表换算到所选预测接口。

## 3. 有限容量为何让无权重目标产生不同解？

在不限制函数容量的总体回归中，正的、仅依赖$u$的权重保持各个时间的条件均值最优解，例如$\mathbb E[V_{\mathrm{VP}}\mid X_u,u]$。实际网络通常在不同时间共享有限参数，时间权重会改变它们之间的取舍。某段时间的权重为零时，目标对该段预测不施加约束。

一个可手算的受限拟合例：两个时间等概率出现，系数分别为$(\alpha,\sigma)=(0.6,0.8),(0.8,0.6)$。固定教学样本$Y=2,E=1$，得到两个VP标签$(-1,-0.4)$。限制预测器只能输出一个常数$\widehat v_{\mathrm{VP}}=\theta$，并按第1节得到clean和noise预测。若损失写成$\sum_i\lambda_i(\theta-V_i)^2$，最优值是

$$\theta^*=\frac{\sum_i\lambda_iV_i}{\sum_i\lambda_i}.$$

| 无权重目标 | 写到VP误差坐标后的$\lambda$ | $\theta^*$ |
| --- | --- | --- |
| VP MSE | $(1,1)$ | $-0.700$ |
| noise MSE | $(0.36,0.64)$ | $-0.616$ |
| clean MSE | $(0.64,0.36)$ | $-0.784$ |

这是两项确定标签的有限容量拟合例，用来展示权重如何改变共享参数的最优值。图像训练中的实际收益还取决于数据分布、网络和优化设置，需通过对应实验衡量。

## 4. 时间采样分布与显式权重共同定义目标

令$\ell_\theta$表示选定预测坐标中的逐样本损失，目标为

$$J(\theta)=\int h(u)\,\mathbb E[\ell_\theta\mid u]\,du.$$

$h(u)\ge0$是希望保留的积分权重，可以包含基准时间密度与第2节的误差权重。若代码改从概率密度$\rho(u)$抽时间，并乘显式标量权重$\lambda(u)$，实际目标是

$$\mathbb E_{u\sim\rho}[\lambda(u)\ell_\theta]
=\int\rho(u)\lambda(u)\,\mathbb E[\ell_\theta\mid u]\,du.$$

因此**乘积$\rho\lambda$决定积分权重**。保持指定$h$可用重要性采样（importance sampling）校正$\lambda=h/\rho$。这里假设时间采样独立于数据与噪声，$\rho,\lambda$均与模型参数无关。离散时间把密度换成概率质量、积分换成求和即可。

校正需要覆盖目标的有效区域：在$h>0$且损失可能有贡献的位置，$\rho$必须为正，并且目标期望存在。估计量要有有限方差，还需

$$\int\frac{h(u)^2}{\rho(u)}\,
\mathbb E[\ell_\theta^2\mid u]\,du<\infty.$$

提议分布在重要区域过小时，校正权重可能很大。对随机梯度估计也需相应的梯度矩条件，才能讨论其方差。若直接改变时间分布并保留$\lambda=1$，得到的是新的积分目标。均匀时间与有限区间上的均匀log-SNR涉及不同的坐标密度，换坐标时还需计入Jacobian。

例如目标在两个时间各取一半，损失分别为1和9，目标均值为5。改成概率$(0.1,0.9)$后，无权重均值是8.2；乘权重$(5,5/9)$可恢复5。

## 5. 端点需要单独读换算公式

第2节含$S$倒数的表格用于$\alpha,\sigma>0$。联合旋转$(Y,E)\leftrightarrow(X_u,V_{\mathrm{VP}})$在两个端点仍有定义，但只凭单一预测接口恢复另一对象时会出现退化：

| 理想端点 | 状态和VP标签 | 从同一$\widehat v_{\mathrm{VP}}$得到的预测 | 需要检查的除法 |
| --- | --- | --- | --- |
| $\alpha=0,\sigma=1$ | $X_u=E,\ V_{\mathrm{VP}}=-Y$ | $\widehat E=X_u$，$\widehat Y=-\widehat v_{\mathrm{VP}}$ | 从noise恢复clean的$(X_u-\sigma\widehat E)/\alpha$ |
| $\sigma=0,\alpha=1$ | $X_u=Y,\ V_{\mathrm{VP}}=E$ | $\widehat Y=X_u$，$\widehat E=\widehat v_{\mathrm{VP}}$ | 从clean恢复noise的$(X_u-\alpha\widehat Y)/\sigma$ |

在第一行，通过VP预测换算得到的噪声误差恒为零，而clean回归仍有不确定性：输入是与数据独立的$E$，总体最优clean预测为$\mathbb E[Y]$。条件生成时令噪声与数据及条件独立，固定$c$后最优预测为$\mathbb E[Y\mid c]$。有限的换算公式保留了训练目标的定义，逐样本恢复能力仍受输入信息限制。第二行的clean目标已经等于输入，noise的条件均值则为0。

靠近端点时还需留意含$1/\alpha$或$1/\sigma$的换算对数值误差的放大。具体训练和采样采用哪些时间点，应读取实际调度配置。

## 6. CPU解析检查：损失、梯度和时间平均

在仓库根目录完成README的基础环境安装后，将下面整个代码块保存为`/tmp/prediction_loss_check.py`，运行`.venv/bin/python /tmp/prediction_loss_check.py`。只使用基础环境中的NumPy与PyTorch；没有随机抽样，全部张量采用float64，不读写数据文件。它检查两时间拟合例和一次精确离散求和。

<!-- cpu-check: prediction-and-loss-weighting -->
```python
import numpy as np
import torch

alpha = torch.tensor([0.6, 0.8], dtype=torch.float64)
sigma = torch.tensor([0.8, 0.6], dtype=torch.float64)
clean, noise = 2.0, 1.0
state = alpha * clean + sigma * noise
v_label = alpha * noise - sigma * clean
theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
v_hat = theta.expand_as(alpha)
clean_hat = alpha * state - sigma * v_hat
noise_hat = sigma * state + alpha * v_hat
loss_clean = (clean_hat - clean).square()
loss_noise = (noise_hat - noise).square()
loss_v = (v_hat - v_label).square()
torch.testing.assert_close(loss_clean, sigma.square() * loss_v)
torch.testing.assert_close(loss_noise, alpha.square() * loss_v)

# All three objectives use the same theta and represent mean VP error.
objectives = [
    (loss_clean / sigma.square()).mean(),
    (loss_noise / alpha.square()).mean(),
    loss_v.mean(),
]
values = [loss.item() for loss in objectives]
gradients = [
    torch.autograd.grad(loss, theta, retain_graph=True)[0].item()
    for loss in objectives
]
np.testing.assert_allclose(values, [1.09] * 3, rtol=0, atol=1e-14)
np.testing.assert_allclose(gradients, [2.0] * 3, rtol=0, atol=1e-14)

# Unweighted targets allocate different weights to these two time points.
weights = [torch.ones_like(alpha), alpha.square(), sigma.square()]
optima = [(weight * v_label).sum() / weight.sum() for weight in weights]
np.testing.assert_allclose([x.item() for x in optima],
                           [-0.7, -0.616, -0.784], rtol=0, atol=1e-14)
for weight, optimum in zip(weights, optima):
    derivative = 2 * (weight * (optimum - v_label)).sum()
    assert abs(derivative.item()) < 1e-14

# Exact discrete expectations: target mass, proposed mass, per-time loss.
target_mass = np.array([0.5, 0.5])
proposal_mass = np.array([0.1, 0.9])
per_time_loss = np.array([1.0, 9.0])
target = target_mass @ per_time_loss
uncorrected = proposal_mass @ per_time_loss
corrected = proposal_mass @ ((target_mass / proposal_mass) * per_time_loss)
np.testing.assert_allclose([target, uncorrected, corrected],
                           [5.0, 8.2, 5.0], rtol=0, atol=1e-14)
print("Equivalent VP losses:", [round(x, 6) for x in values])
print("Gradients for shared theta:", [round(x, 6) for x in gradients])
print("Unweighted optima (VP, noise, clean):",
      [round(x.item(), 6) for x in optima])
print("Time means (target, uncorrected, corrected):",
      np.round([target, uncorrected, corrected], 6).tolist())
```

解析预期为：三种等价损失均为1.09，同参数梯度均为2；三个无权重目标的最优参数依次为$-0.7,-0.616,-0.784$；时间均值依次为5、8.2、5。代码中的断言检查这些解析结果；本例的数值只描述所列数组与受限拟合问题。

## 7. 练习与参考答案

1. **取$\alpha=0.6,\sigma=0.8,L_V=4$，求$L_Y,L_E,S$。若目标为$2L_Y$，noise和VP的MSE各乘多少？** 提示：先写误差比例，再换目标。答案：$L_Y=2.56$，$L_E=1.44$，$S=9/16$；权重分别为$32/9$和1.28。代入后三种标量损失都为5.12。
2. **把CPU例子的时间采样改为$(0.2,0.8)$，如何保留等概率目标？如果第二个时间的采样概率变成0呢？** 答案：校正权重为$(2.5,0.625)$，精确期望仍为5。第二个时间有正的目标质量且损失为9；采样概率为0时缺失该区域，无法通过有限权重恢复它的贡献。
3. **在$\alpha=0$处，为什么VP预测可以换成有限的clean预测，同时噪声MSE无法约束这份clean预测？最优clean预测是什么？** 提示：把端点系数直接代入逆旋转。答案：$\widehat Y=-\widehat v_{\mathrm{VP}}$，而$\widehat E=X_u=E$使噪声误差恒为0；输入与$Y$独立，因此无条件MSE最优clean预测为$\mathbb E[Y]$。noise到clean的内部时间公式在此除以0，需要改用端点定义。

## 8. 固定来源与核验范围

| 原始来源 | 固定版本与定位 | 本页依据 |
| --- | --- | --- |
| Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239v2) | arXiv:2006.11239v2；PDF第4–5页，式(12)、式(14) | 时间加权的epsilon MSE与均匀时间简化目标的背景 |
| [Diffusers DDIMScheduler源码](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/schedulers/scheduling_ddim.py) | Diffusers 0.35.2，commit `b71269675ec1b85193107a691dd35c308e46f0a5`；`step`第412–433行，`get_velocity`第500–519行 | VP标签、noise/clean/VP预测接口的变换及其后的裁剪顺序 |

损失系数、同参数梯度、有限容量例与重要性权重由本页给定的假设独立推导，CPU块核验对应恒等式和解析数值。DDIM源码用于核对预测接口；具体训练目标需按训练实现读取。这里只比较VP路径上的平方误差与时间权重，生成质量结论需配套训练实验。固定来源身份、正文运行与另一组参数的独立检查见[核验记录](../../records/measurements/prediction-loss/README.md)；DDPM完整公式定位见[来源审计](../source-audit.md)。
