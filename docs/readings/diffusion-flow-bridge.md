# 从噪声预测到生成速度：扩散、DDIM与Flow Matching的桥梁

**先修：** [02扩散](../chapters/02-diffusion.md)、[03 Flow Matching](../chapters/03-flow-matching.md)、条件期望与常微分方程。**学习目标：** 从同一高斯扰动路径推导噪声、去噪器、score和速度的关系；核对时间方向、系数与端点；解释DDIM和SVD确定更新所用的坐标。

这是可选的理论专题，可配合[09图像推理](../chapters/09-pretrained-image-lab.md)和[10视频推理](../chapters/10-pretrained-video-lab.md)阅读。只需基础NumPy环境即可运行下面的解析检查。随机微积分部分使用一个密度演化公式，完整推导定位到原论文；本页核验公式、原始实现与小数组，模型训练和生成效果沿各实验的实际记录判断。

核心连接是：**同一条高斯扰动路径上，噪声的总体最优预测、去噪器与score可以相互换算；概率路径和时间坐标决定怎样把它们转换为生成速度。**

## 1. 先固定路径和时间方向

设数据$Y\sim p_{\mathrm{data}}$，独立噪声$E\sim\mathcal N(0,I)$，定义

$$X_u=\alpha(u)Y+\sigma(u)E.$$

本页的$u$沿**数据到噪声**方向增加；$\alpha$是数据的幅度系数，对应02章离散时刻的$\sqrt{\bar\alpha_t}$。02章的单步$\alpha_t=1-\beta_t$另有含义。为保持03章的约定，本文把沿$u$的边缘速度记为$w_u$，仍用$v_t$表示沿噪声到数据方向的flow速度。

以下推导在$\alpha,\sigma>0$的内部时间进行，系数可微，数据有有限二阶矩，并允许交换积分和求导。高斯扰动此时给出光滑正密度$p_u$。将连续性方程解释为ODE的边缘演化，还需速度场生成适定的输运流，例如所讨论区间内具有足够正则性、解唯一且不爆炸。下面的Gaussian解析例满足这些要求。

条件生成时可固定条件$c$，把密度改为$p_u(x\mid c)$，后验均值也都包含$c$；为简洁，下文省略这个固定条件。

## 2. 噪声、score与去噪器是怎样连接的？

记$s_u(x)=\nabla_x\log p_u(x)$。给定$Y=y$的条件密度为$\mathcal N(\alpha y,\sigma^2I)$，对$x$求导得到

$$\nabla_x\log p_u(x\mid y)=-\frac{x-\alpha y}{\sigma^2}.$$

对后验$p(y\mid x,u)$求平均，并用$X_u-\alpha Y=\sigma E$，可得

$$s_u(x)=-\frac{\mathbb E[E\mid X_u=x]}{\sigma},\qquad
\bar\epsilon_u(x):=\mathbb E[E\mid X_u=x]=-\sigma s_u(x).$$

这解释了噪声MSE的最优预测为何能提供边缘score。每个训练样本的随机噪声$E$是监督标签，网络给定$(x,u)$后的最优回归为条件均值$\bar\epsilon_u$。

进一步对$X_u=\alpha Y+\sigma E$取条件期望，得到去噪器

$$D_u(x):=\mathbb E[Y\mid X_u=x]
=\frac{x-\sigma\bar\epsilon_u(x)}{\alpha}
=\frac{x+\sigma^2s_u(x)}{\alpha}.$$

最后一种表达通常称为高斯扰动下的 **Tweedie关系**。它还原的是数据的后验均值；不同原始样本仍可能产生同一个带噪状态。

## 3. 从条件导数到边缘速度

固定同一对$(Y,E)$，路径导数是$\alpha'Y+\sigma'E$。依照03章的条件期望与连续性方程推导，边缘速度可取

$$\begin{aligned}
w_u(x)
&=\mathbb E[\alpha'Y+\sigma'E\mid X_u=x]\\
&=\alpha'D_u(x)+\sigma'\bar\epsilon_u(x)\\
&=\frac{\alpha'}{\alpha}x+
\left(\frac{\alpha'}{\alpha}\sigma^2-\sigma\sigma'\right)s_u(x).
\end{aligned}$$

于是$\partial_up_u=-\nabla\cdot(p_uw_u)$。同一密度演化有时可以由多个速度场实现；上式是这条条件路径的后验平均所选出的场。

**时间翻转。** 若设$t=1-u$来匹配03章的生成方向，速度变为$v_t(x)=-w_{1-t}(x)$。也可以保留$u$坐标，从大$u$向小$u$积分，让负的步长承担方向变化。选择一种坐标后同时核对路径、速度与积分区间。

## 4. 为什么probability-flow ODE出现一个1/2？

考虑一个线性漂移、标量扩散系数的前向随机微分方程（SDE）：

$$dX_u=f(u)X_u\,du+g(u)\,dW_u.$$

其中$W_u$为标准Brownian运动，短时间随机增量的协方差为$du\,I$。线性SDE要具有上面的高斯条件边缘，应满足

$$f=\frac{\alpha'}{\alpha},\qquad
g^2=\frac{d\sigma^2}{du}-2f\sigma^2
=2\sigma\sigma'-2\frac{\alpha'}{\alpha}\sigma^2.$$

这里需要$g^2\ge0$，等价于$\sigma^2/\alpha^2$沿$u$不减。若从$u=0$的$X_0=Y$开始，还需$\alpha(0)=1$、$\sigma(0)=0$；若从内部时间$u_a$开始，匹配给定$Y$的条件边缘需要初始化为$X_{u_a}\mid Y\sim\mathcal N(\alpha(u_a)Y,\sigma(u_a)^2I)$。只要求整体边缘一致时，初始化为真实$p_{u_a}$即可。条件路径的速度恒等式本身可以在更一般的系数下成立。

该SDE的密度满足Fokker–Planck方程。对仅依赖时间的$g$，利用$\nabla p_u=p_us_u$可重写为

$$\begin{aligned}
\partial_up_u
&=-\nabla\cdot(fxp_u)+\frac12g^2\Delta p_u\\
&=-\nabla\cdot\left[p_u\left(fx-\frac12g^2s_u\right)\right].
\end{aligned}$$

因此 **概率流常微分方程（probability-flow ODE）** 使用

$$\frac{dx}{du}=fx-\frac12g^2s_u(x)=w_u(x).$$

系数$1/2$来自扩散项的密度演化。反向SDE保留随机增量，其漂移在同一$u$坐标下为$fx-g^2s_u$，按递减$u$求解；将随机项删除后，密度演化也随之改变。Song等人的式(6)与式(13)分别给出这两个过程。状态相关扩散矩阵还涉及额外的散度项，详见原文附录D.1。

**一个能检出错误的平稳例子。** 取$f=-1$、$g^2=2$、$p_u=\mathcal N(0,1)$，则$s_u(x)=-x$，正确概率流速度为$-x-(-x)=0$。若在无噪声ODE中漏掉$1/2$，速度变成$x$，初始方差为1时的方差导数为2；若把score符号反过来，速度为$-2x$，方差导数为$-4$。这些错误都可能保留零均值，检查均值之外的性质很重要。

这个例子也区分了三种跨时间关系：固定$(Y,E)$构造的Gaussian条件路径通常会移动；平稳SDE的样本受到持续随机扰动；概率流ODE在此例中保持每个样本不动。它们在每个时间都具有同一个标准正态分布，跨时间的联合分布和逐样本配对各有不同。

精确边缘对应使用正确score、匹配的初始边缘和精确ODE解。实际系统还包含网络误差、先验近似及有限步求解误差，需按实验的真实配置分析。

## 5. 三种常见接口如何接入？

### VP的v参数化与flow速度

在$\alpha^2+\sigma^2=1$的方差保持（VP）路径上，训练目标可定义为

$$V_{\mathrm{VP}}=\alpha E-\sigma Y.$$

这是对$(Y,E)$的旋转组合。对单位圆约束求导可得

$$\alpha'Y+\sigma'E
=(\alpha\sigma'-\sigma\alpha')V_{\mathrm{VP}},\qquad
w_u=(\alpha\sigma'-\sigma\alpha')\,
\mathbb E[V_{\mathrm{VP}}\mid X_u=x].$$

前面的系数由时间参数化决定。例如$\alpha=e^{-u}$、$\sigma=\sqrt{1-e^{-2u}}$时，它等于$\alpha/\sigma$。03章的直线flow路径使用$tY+(1-t)E$，其条件速度为$Y-E$；阅读叫作`v`的输出时，先找到路径和目标的完整定义。

| 预测对象 | 定义或总体最优预测 | 换算时要保留的量 |
| --- | --- | --- |
| 噪声$\epsilon$ | $\bar\epsilon_u=\mathbb E[E\mid X_u=x]$ | $s_u=-\bar\epsilon_u/\sigma$ |
| 数据/去噪器 | $D_u=\mathbb E[Y\mid X_u=x]$ | $D_u=(x-\sigma\bar\epsilon_u)/\alpha$ |
| VP的$v$ | $\mathbb E[\alpha E-\sigma Y\mid X_u=x]$ | VP归一化与上面的时间系数 |
| 边缘速度 | $w_u=\mathbb E[\alpha'Y+\sigma'E\mid X_u=x]$ | 路径导数、时间方向及坐标单位 |

### DDIM：换坐标后的一步Euler

对VP幅度$\alpha,\sigma$，定义$\bar x=x/\alpha$、$\rho=\sigma/\alpha$。09章无裁剪、`eta=0`的DDIM一步可直接重写为

$$\bar x_{\mathrm{prev}}
=\bar x+(\rho_{\mathrm{prev}}-\rho)\widehat\epsilon(x,u).$$

这在$(\bar x,\rho)$坐标上呈现Euler形式。其条件边缘为$\bar X=Y+\rho E$，正确的边缘导数是$\mathbb E[E\mid\bar X]=-\rho\nabla_{\bar x}\log p_\rho(\bar x)$。在连续VP坐标中，噪声网络实际接收$\bar x/\sqrt{1+\rho^2}$及相应噪声级别。DDIM原文v4第6页式(13)–(15)也强调：在不同时间坐标上取有限Euler步，离散更新通常会有差别。

端点按实际scheduler读取。09章固定的`set_alpha_to_one=False`使最终累计量为$a_0=0.99915$，因此$\rho_{\mathrm{end}}=\sqrt{(1-a_0)/a_0}>0$；这份配置的最后坐标与理想$\rho=0$分别记录。

### SVD：沿加性噪声sigma积分

10章采用加性坐标$Z_\sigma=Y+\sigma E$。此时$D_\sigma(z)=z+\sigma^2s_\sigma(z)$，所以

$$\frac{dz}{d\sigma}
=\frac{z-D_\sigma(z)}{\sigma}
=-\sigma s_\sigma(z).$$

SVD先把VP预测量还原为$D_\theta$，再以递减sigma执行Euler，网格追加0作为终点。负的$\Delta\sigma$已经编码生成方向；首末步的实现与数值核验见[10章](../chapters/10-pretrained-video-lab.md#3-沿sigma积分vp预测量去噪器与euler)。应用条件引导后按组合后的网络预测执行更新；引导速度的目标分布另需分析。

在$\alpha=0$或$\sigma=0$处，本文含除法的换算需要单独分析极限。原路径与条件回归目标有时仍良好定义，例如03章直接使用$Y-E$作为训练标签。

## 6. CPU解析检查：均值、方差和坐标都要检查

取$Y\sim\mathcal N(2,4)$、$\alpha=e^{-u}$、$\sigma=\sqrt{1-e^{-2u}}$，则

$$p_u=\mathcal N(m_u,q_u),\quad m_u=2e^{-u},\quad
q_u=1+3e^{-2u},\quad s_u(x)=-\frac{x-m_u}{q_u},\quad w_u=-x-s_u.$$

这个VP路径把数据方差4逐渐送向1，标准正态是其平稳参照。概率流的解析轨迹为$x_u=m_u+\sqrt{q_u/q_{u_0}}(x_{u_0}-m_{u_0})$。下面用固定标准化坐标核对轨迹，以及Gaussian总体均值/方差的导数；有限时间端点仍含数据成分，初始化使用真实$p_{0.2}$。

<!-- cpu-check: diffusion-flow-bridge -->
```python
import numpy as np

def moments(u):
    a = np.exp(-u)
    b = np.sqrt(-np.expm1(-2*u))
    return a, b, 2*a, 4*a*a + b*b

def velocity(x, u):
    a, b, mean, var = moments(u)
    score = -(x-mean)/var
    return -x-score

errors = []
for u in (.2, .5, 1.2):
    a, b, mean, var = moments(u)
    x = mean + np.sqrt(var)*np.linspace(-2, 2, 9)
    score = -(x-mean)/var
    clean = 2 + 4*a*(x-mean)/var
    noise = b*(x-mean)/var
    np.testing.assert_allclose(noise, -b*score, atol=1e-14)
    np.testing.assert_allclose(clean, (x+b*b*score)/a, atol=1e-14)
    w = -a*clean + a*a/b*noise
    analytic = -mean - 3*a*a/var*(x-mean)
    np.testing.assert_allclose(w, velocity(x,u), atol=1e-14)
    np.testing.assert_allclose(w, analytic, atol=1e-14)
    # Gaussian population mean/variance derivatives for affine drift.
    np.testing.assert_allclose(velocity(mean,u), -mean, atol=1e-14)
    slope = -1 + 1/var
    np.testing.assert_allclose(2*slope*var, -6*a*a, atol=1e-14)

u0, u1 = .2, 1.2
_, _, m0, q0 = moments(u0)
_, _, m1, q1 = moments(u1)
z = np.linspace(-2, 2, 9)  # Fixed standardized coordinates, no Monte Carlo estimate.
x0, exact = m0+np.sqrt(q0)*z, m1+np.sqrt(q1)*z
for steps in (50, 200, 800):
    x = x0.copy()
    h = (u1-u0)/steps
    for k in range(steps):
        x += h*velocity(x,u0+k*h)
    errors.append(float(np.max(np.abs(x-exact))))
assert errors[2] < errors[1] < errors[0]
assert errors[2] < .002
# Stationary N(0,1): score=-x, f=-1, g²=2.
# Affine ODE dX/du=kX has population variance derivative 2*k when variance=1.
x = np.array([-2., 0., 2.]); score = -x
fields = {'correct': -x-.5*2*score, 'missing_half': -x-2*score,
          'wrong_score_sign': -x-.5*2*(-score)}
variance_derivatives = {name: float(2*np.dot(x,w)/np.dot(x,x))
                        for name,w in fields.items()}
assert variance_derivatives == {'correct': 0., 'missing_half': 2., 'wrong_score_sign': -4.}
# eta=0 DDIM, data amplitudes .6 -> .8, noise amplitudes .8 -> .6.
x, eps = np.array([-.2, .7]), np.array([.5, -.8])
clean = (x-.8*eps)/.6
previous = .8*clean+.6*eps
reparameterized = x/.6 + (.6/.8-.8/.6)*eps
np.testing.assert_allclose(previous/.8,reparameterized,atol=1e-14)
print({'Euler_endpoint_errors': errors, 'stationary_variance_derivatives': variance_derivatives,
       'DDIM_coordinate_error': float(np.max(np.abs(previous/.8-reparameterized)))})
```

本轮实际执行的Euler终点最大差约为`0.02340、0.005827、0.001455`，随这组网格细化而减小；DDIM坐标重写残差为`2.22e-16`。这些是解析场的数值核验，预训练网络的效果由对应实践单独测量。

## 7. 练习与参考答案

1. 把03章的噪声到数据路径$X_t=tY+(1-t)E$直接代入前向SDE系数式，为什么得到负的$g^2$？

   <details><summary>提示与答案</summary>

   对$t\in(0,1)$，$f=1/t$，$g^2=-2(1-t)/t<0$，该时间方向不能按此式作为实值前向扩散系数。改用数据到噪声坐标$u=1-t$，$\alpha=1-u$、$\sigma=u$，得到$g^2=2u/(1-u)>0$。条件路径的速度推导在两种坐标下都成立，端点$u=1$另有奇异性。

   </details>

2. 两个确定性求解器沿同一连续概率流积分，可以要求有限步输出逐元素相同吗？

   <details><summary>提示与答案</summary>

   先核对初始状态、预测函数、坐标、网格和数值方法。相同连续ODE的有限离散方案仍有不同误差；DDIM在$\rho$上的Euler与在其它时间上的Euler就能产生不同更新。仅在代数等价且全部输入对应时要求一致；实际误差应与独立解析解或可信参考解比较。

   </details>

3. 平稳例子的概率流速度为0，为什么仍能用它生成标准正态样本？

   <details><summary>提示与答案</summary>

   初始边缘已经是标准正态，保持样本不动即可保持正确分布。改变初始分布会改变结果。这个例子同时说明正确边缘与“每个样本都有运动”是两个可分别核查的问题。

   </details>

## 8. 固定来源与核验范围

核验日期为2026-11-10，PDF按从第一页开始的页码定位。来源文件保存在临时审计目录，[下载/源码身份及独立检查记录](../../records/measurements/probability-flow/README.md)随仓库保存，正文与代码块由本仓库独立组织。

| 来源 | 固定身份与本轮核对范围 |
| --- | --- |
| Song等，*Score-Based Generative Modeling through Stochastic Differential Equations* | [2011.13456v2](https://arxiv.org/abs/2011.13456v2)，2021-02-10，ICLR2021；[PDF](https://arxiv.org/pdf/2011.13456v2)共36页、26,918,820字节，SHA-256 `b244b740e60628d01bc3d4b98d721c2343be1fb9b67ad22593c8555e66f4d4a6`。第4页式(6)/(7)：反向SDE与条件score回归；第7页式(13)：概率流ODE；第17页式(35)–(37)：密度方程与一般扩散矩阵推导。 |
| 原作者JAX实现 | [`yang-song/score_sde` commit `0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c`](https://github.com/yang-song/score_sde/tree/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c)，2021-08-26。[`sde_lib.py`](https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/sde_lib.py)的`SDE.reverse`在probability-flow分支使用0.5并将扩散系数设零；`VPSDE.sde/marginal_prob`核对线性漂移与高斯系数。[`sampling.py`](https://github.com/yang-song/score_sde/blob/0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c/sampling.py)的`get_ode_sampler`从`T`积分到正的`eps`。本轮只读源码，未安装或运行该JAX训练项目。 |
| Song、Meng、Ermon，*Denoising Diffusion Implicit Models* | [2010.02502v4](https://arxiv.org/pdf/2010.02502v4)，2022-10-05；22页、10,857,218字节，SHA-256 `07cb8796d7c100bf7c02bc575f16aab72c32630a8f85abd54ad45dc4f63e80fb`。重新核对第6页式(13)–(15)的坐标变换、ODE及离散方法差别；09章保留实际scheduler边界。 |
| Flow Matching与SVD连接 | [既有FM原文核验](../source-audit.md#3-flow-matching正终点方差与零方差教学路径)固定2210.02747v2，核对后验平均速度与连续性方程；[10章固定来源](../chapters/10-pretrained-video-lab.md#原始来源与核验范围)定位SVD去噪器、VP还原和Euler实现。本文的统一代数与Gaussian反例另经独立推导和NumPy检查。 |

这些证据支持所列公式与接口；本轮未执行图像/视频训练、作者benchmark或质量排名比较。
