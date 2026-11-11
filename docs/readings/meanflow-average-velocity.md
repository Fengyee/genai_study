# MeanFlow：用两时间平均速度连接一次生成与训练导数

**先修：** [03 Flow Matching](../chapters/03-flow-matching.md)的条件期望与ODE，以及[少步生成专题](consistency-endpoint-map.md)的轨迹端点。**学习目标：** 推导平均速度恒等式；核对两个时间输入的JVP切向量；解释停止目标梯度和一次采样各自计算什么。

MeanFlow（Geng等，2025）把一段轨迹的平均速度作为预测对象。本页固定首发论文与作者实现，先研究无类别引导、平方回归这一基础形式。实践使用已有CPU PyTorch进行解析检查；图像网络训练、作者FID和运行加速比均沿其原始实验范围解释。

## 1. 从一条轨迹的位移定义平均速度

为与全仓库的生成时间区分，本页用$`\tau`$表示**数据到噪声**的时间，生成沿$`1\to0`$；原论文把它记为$`t`$。令$`Y`$为数据、$`E\sim\mathcal N(0,I)`$与数据独立：

```math
X_\tau=(1-\tau)Y+\tau E,\qquad V=E-Y,\qquad
w_\tau(x)=\mathbb E[V\mid X_\tau=x].
```

$`V`$是可从训练样本构造的条件速度；$`w`$是生成ODE的边缘速度。假设所讨论区间的$`w`$足够光滑并产生唯一、不爆炸的ODE流，记从$`(x,\tau)`$积分到$`\rho\le\tau`$的状态为$`F_{\rho\leftarrow\tau}(x)`$。对$`\rho<\tau`$定义

```math
U^*(x,\rho,\tau)
=\frac{x-F_{\rho\leftarrow\tau}(x)}{\tau-\rho}
=\frac{1}{\tau-\rho}\int_\rho^\tau w_s(x_s)\,ds.
```

积分中的$`x_s`$沿同一条边缘ODE轨迹变化。原文用$`u`$表示这里的$`U^*`$。输入有两个时间：当前位置$`\tau`$和目标时间$`\rho`$。因此，一次平均速度求值给出的位移为

```math
F_{\rho\leftarrow\tau}(x)=x-(\tau-\rho)U^*(x,\rho,\tau).
```

在$`\rho=\tau`$处用连续延拓定义$`U^*(x,\tau,\tau)=w_\tau(x)`$。连续且有限的平均速度也使$`(\tau-\rho)U^*\to0`$，固定了零长度区间的位移边界。

[CM专题](consistency-endpoint-map.md)学习到固定终点的映射；这里学习两个时间之间的平均速度。两者都涉及整段轨迹信息。普通Euler则使用起点的瞬时速度$`w_\tau(x)`$乘区间长度，误差来自区间内速度变化。

## 2. 固定目标时间，展开总导数

在同一条轨迹上，把$`\rho`$固定，对位移恒等式求$`\tau`$导数。记$`h=\tau-\rho`$，有

```math
U^*+h\frac{d}{d\tau}U^*=w_\tau(x_\tau),\qquad
\frac{d}{d\tau}U^*
=J_xU^*\,w_\tau(x_\tau)+\partial_\tau U^*.
```

整理后得到论文的MeanFlow恒等式（式6、8）：

```math
U^*=w_\tau-h\big(J_xU^*\,w_\tau+\partial_\tau U^*\big).
```

总导数同时包含状态沿ODE移动和显式时间变化。它可用Jacobian-vector product（JVP）计算，具体切向量取决于函数的输入顺序：

| 接口 | 固定$`\rho`$、沿轨迹改变$`\tau`$的切向量 |
| --- | --- |
| 论文顺序$`U(x,\rho,\tau)`$ | $`(w_\tau,0,1)`$ |
| 作者包装函数`u_fn(z_t, t, r)` | `(w, 1, 0)` |
| 直接调用区间接口$`N(x,\tau,h)`$，$`h=\tau-\rho`$ | $`(w_\tau,1,1)`$ |

第三行的$`dh/d\tau=1`$。作者代码将`(z_t,t,r)`包装成底层网络的`(z_t,t,t-r)`后再调用JVP，自动微分会处理这个链式法则。把间隔$`h`$当成固定输入，会计算另一个方向的导数。

## 3. 怎样构造可训练的目标？

精确$`F`$与平均速度定义包含轨迹积分，训练样本本身只给出$`Y,E,\tau`$和条件速度$`V`$。原文式9–11用网络$`U_\theta`$构造停止梯度的目标：

```math
H_\theta(V)=V-h\big(J_xU_\theta V+\partial_\tau U_\theta\big),\qquad
\mathcal L=\mathbb E\left[\left\|U_\theta-
\mathop{\mathrm{stopgrad}}\nolimits[H_\theta(V)]\right\|^2\right].
```

JVP计算目标所需的输入方向导数，`stopgrad`使优化器只通过左侧的$`U_\theta`$更新参数。对目标也做参数求导，会增加穿过JVP的导数并改变更新。作者实现还使用按误差构造、停止梯度的自适应权重；本节先分析未加权平方回归。

为何条件速度可放进JVP？给定$`(x,\rho,\tau)`$与当前参数时，$`H_\theta(V)`$关于$`V`$是仿射函数，所以

```math
\mathbb E[H_\theta(V)\mid x,\rho,\tau]
=w_\tau-h\big(J_xU_\theta w_\tau+\partial_\tau U_\theta\big).
```

这里假设时间对的选择独立于数据/噪声抽样。未加权、停止目标梯度的期望更新因而与使用这个条件均值目标相符。随机条件目标仍有方差；在精确平均速度处，单个训练样本的MSE通常仍为正。带残差相关权重的更新需连同其权重分析，参见[损失权重专题](prediction-and-loss-weighting.md)。

在$`h=0`$的对角线上，目标修正消失，回到条件flow matching。固定作者代码先抽两个时间并排序，再让一个batch中的部分样本取$`\rho=\tau`$；非零区间和对角线样本共同训练两时间接口。实际学习还依赖网络容量、时间采样、损失权重与优化过程。

## 4. Gaussian反例：两个“一次求值”产生不同的输运

取一维$`Y\sim\mathcal N(m,s^2)`$、$`s>0`$，定义

```math
a_\tau=(1-\tau)m,\qquad q_\tau=(1-\tau)^2s^2+\tau^2,\qquad
w_\tau(x)=-m+\frac{(1+s^2)\tau-s^2}{q_\tau}(x-a_\tau).
```

固定$`z`$的轨迹为$`x_\tau=a_\tau+\sqrt{q_\tau}z`$，其导数直接等于$`w_\tau(x_\tau)`$。从而

```math
F_{\rho\leftarrow\tau}(x)
=a_\rho+\sqrt{\frac{q_\rho}{q_\tau}}(x-a_\tau).
```

本例$`q_\tau\ge s^2/(1+s^2)>0`$，整个闭区间的公式良好。代码中的平均速度采用上述差商的有理化形式，使对角线也可直接求值与求导。

从噪声$`x_1=z`$跨到$`0`$：精确平均速度输出$`m+sz`$，一步反向Euler使用$`w_1(z)=z-m`$，会将所有起点送到$`m`$。当$`z\sim\mathcal N(0,1)`$时，两者输出总体方差分别为$`s^2`$和0。这来自已知Gaussian边缘的精确解析场；学习一个图像网络需要单独的训练和质量证据。

下面代码中的`r,t`分别对应本页$`\rho,\tau`$。固定9个点检查逐点端点误差；输出总体方差由解析变换给出。还会在对角线比较两种接口下的JVP，并检出错误地固定间隔的实现。

<!-- cpu-check: meanflow-average-velocity -->
```python
import torch

torch.set_num_threads(1)
m, s = 1.5, 0.7
z = torch.linspace(-2, 2, 9, dtype=torch.float64)

def q(t):
    return (1-t)**2*s*s+t*t

def a(t):
    return (1-t)*m

def w(x, t):
    return -m+((1+s*s)*t-s*s)/q(t)*(x-a(t))

def mean_velocity(x, r, t):
    c = ((1+s*s)*(r+t)-2*s*s)/(q(t).sqrt()*(q(t).sqrt()+q(r).sqrt()))
    return -m+c*(x-a(t))

for lower, upper in ((0., 1.), (.2, .8), (.5, .5)):
    r, t = torch.tensor([lower, upper], dtype=torch.float64)
    x = a(t)+q(t).sqrt()*z
    u, du = torch.func.jvp(mean_velocity, (x, r, t),
                           (w(x, t), torch.zeros_like(r), torch.ones_like(t)))
    target = a(r)+(q(r)/q(t)).sqrt()*(x-a(t))
    torch.testing.assert_close(u, w(x, t)-(t-r)*du, rtol=0, atol=1e-14)
    torch.testing.assert_close(x-(t-r)*u, target, rtol=0, atol=1e-14)
    if lower == upper:
        torch.testing.assert_close(u, w(x, t), rtol=0, atol=1e-14)

def by_interval(x, t, h):
    return mean_velocity(x, t-h, t)

_, right = torch.func.jvp(by_interval, (x, t, t-r),
                          (w(x, t), torch.ones_like(t), torch.ones_like(t)))
_, wrong = torch.func.jvp(by_interval, (x, t, t-r),
                          (w(x, t), torch.ones_like(t), torch.zeros_like(t)))
torch.testing.assert_close(right, du, rtol=0, atol=1e-14)
assert float((wrong-du).abs().max()) > 0.1
print("Holding interval fixed changes JVP by:", float((wrong-du).abs().max()))

r, t = torch.tensor([0., 1.], dtype=torch.float64)
expected = m+s*z
average_step = z-mean_velocity(z, r, t)
euler_step = z-w(z, t)
torch.testing.assert_close(average_step, expected, rtol=0, atol=1e-14)
torch.testing.assert_close(euler_step, torch.full_like(z, m), rtol=0, atol=1e-14)
print("Average-velocity max endpoint error:", float((average_step-expected).abs().max()))
print("One-Euler max endpoint error:", float((euler_step-expected).abs().max()))
print("Exact / one-Euler population variance:", s*s, 0.0)
```

CPU float64实际输出：平均速度的最大端点误差约$`2.22\times10^{-16}`$；一步Euler的最大误差为1.4；总体方差分别0.49和0。错误固定间隔的JVP与正确方向最大差约2.15529。独立检查另外覆盖五组非对角时间、五个对角点和沿精确轨迹的中心差分；JVP与差分最大差约$`2.67\times10^{-10}`$。

## 5. 从源码核对“1-NFE”的范围

作者`solver_step`调用一次`u_fn`并返回`z_t-(t-r)*u`；默认时间表为`[1,0]`。本仓库抽取这两个函数及`sample_one_step`，用NumPy和解析平均速度替代JAX/网络，实际核验默认一次调用及端点结果。这个检查覆盖控制流程与接口；完整JAX执行和图像权重尚未运行。

训练每批还需JVP、参数反向传播，以及所选引导/权重的计算。采样NFE统计生成网络求值；ImageNet实验另用预训练VAE将32×32×4潜变量解码到256×256图像。论文的“从头训练”描述生成网络，其图像实验使用既有VAE表示。

论文表2报告MeanFlow-XL/2在ImageNet 256×256、50K生成样本、1-NFE协议下FID为3.43；固定作者README提供的是MF-B/4 checkpoint，其预期FID为11.4。它们对应不同模型与实验配置。以上数值为作者报告，本页没有下载checkpoint或重新计算FID。

## 6. 练习与参考答案

1. **底层网络接口是`net(x,t,h)`，其中`h=t-r`。要固定`r`求总导数，三个切向量是什么？** 答案：`(w,1,1)`；也可对`fn(x,r,t)=net(x,t,t-r)`使用`(w,0,1)`。分别解释状态、当前时间、区间长度的变化，并运行正文的错误方向反例。
2. **只在$`\rho<\tau`$满足MeanFlow恒等式，是否已固定零区间位移？** 提示：给精确平均速度加$`k/(\tau-\rho)`$。答案：常量$`k`$产生的额外项在$`U+h\,dU/d\tau`$中相消，但位移额外增加$`k`$。对角线有限/连续的要求排除了它；边界与恒等式需要共同核对。
3. **Gaussian例的精确平均速度为什么仍有条件训练误差？** 答案：给定$`X_\tau=x`$时，$`V`$的方差是$`s^2/q_\tau`$；精确目标中$`V`$的系数为$`\sqrt{q_\rho/q_\tau}`$，所以目标条件方差为$`q_\rho s^2/q_\tau^2>0`$。目标的条件均值与精确平均速度相同。对角线退化为普通条件flow matching残差。

## 7. 固定来源与本地证据

| 原始来源 | 固定版本与定位 | 本页核验 |
| --- | --- | --- |
| Geng et al., [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447v1) | 2025-05-19 v1；PDF第3–5页式1–12与算法1–2；第15–16页附录B.2–B.4 | 平均速度定义、总导数、基本目标、采样和边界；第7–9/14页核对图像实验身份 |
| [作者核心实现](https://github.com/Gsunshine/meanflow/blob/58d099c0fa197bd31b9a7b51727a76019ae5fdbe/meanflow.py) | 2025-07-08 commit `58d099c0fa197bd31b9a7b51727a76019ae5fdbe`；`sample_tr`、`forward`、`solver_step` | 两时间包装、切向量、目标停止梯度、误差权重与一次调用 |
| [作者README](https://github.com/Gsunshine/meanflow/blob/58d099c0fa197bd31b9a7b51727a76019ae5fdbe/README.md)及[B/4配置](https://github.com/Gsunshine/meanflow/blob/58d099c0fa197bd31b9a7b51727a76019ae5fdbe/configs/run_b4.yml) | 同一commit，官方JAX/TPU实现，`sampling.num_steps: 1` | 发布模型和论文实验的配置身份 |

固定PDF为6,192,153 bytes，SHA-256 `322ba9244ad2c72382038e421fe0ffa92510c51d4f9c740bd7282b36bfc25206`；核心源码为8,029 bytes，SHA-256 `b8dcee1033e6ba01de0019d14fad846a399d356d67e549ee311a5e19d2a7c3a4`。源码按作者MIT许可引用；教学代码和Gaussian推导为本仓库独立示例。固定来源身份、源码函数替身检查、独立推导与实际CPU输出见[证据索引](../../records/measurements/meanflow-average-velocity/README.md)。
