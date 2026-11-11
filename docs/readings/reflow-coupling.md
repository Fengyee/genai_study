# Reflow：用生成端点更新耦合

**先修：** [03 Flow Matching](../chapters/03-flow-matching.md)的条件期望、线性插值与ODE，特别是[C. 一个能完全推导的边缘速度](../chapters/03-flow-matching.md#c-一个能完全推导的边缘速度)。**完成后应能：** 写出reflow的新配对数据；解释每轮保持哪些边缘；用两个Gaussian例区分时间仿射轨迹与全局最优传输。

本页沿用03章的时间方向：0为源、1为目标。局部符号$`X_0,X_1`$分别表示这两个端点，$`\pi`$表示其联合分布，$`c`$表示位移成本。Rectified Flow（RF）预测瞬时速度；reflow通过生成端点更新训练耦合。[MeanFlow卡](meanflow-average-velocity.md)研究给定轨迹的两时间平均速度与JVP目标。两张卡分别回答配对怎样改变、网络预测什么。

## 1. 从一次rectification到下一轮训练

设$`(X_0,X_1)\sim\pi`$，令$`X_t=(1-t)X_0+tX_1`$。平方回归的总体最优场是

```math
v^\pi_t(x)=\mathbb E[X_1-X_0\mid X_t=x].
```

在条件速度足够正则、ODE存在唯一解时，从$`Z_0=X_0`$积分$`\dot Z_t=v^\pi_t(Z_t)`$，得到新端点耦合$`(Z_0,Z_1)`$。原文将这一耦合变换称为rectification：同一轮内$`\mathop{\mathrm{Law}}\nolimits(Z_t)=\mathop{\mathrm{Law}}\nolimits(X_t)`$，端点配对可改变。

一次reflow以生成的配对$`(Z_0,Z_1)`$构造新监督：抽取时间$`t`$，输入$`(1-t)Z_0+tZ_1`$，目标为$`Z_1-Z_0`$，训练下一条瞬时速度场。源与目标端点边缘在理想迭代中保持；下一轮的中间边缘由新耦合决定。

[作者固定实现](https://github.com/gnobitab/RectifiedFlow/blob/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6/ImageGeneration/run_lib_reflow.py#L141-L221)先用已有模型生成`z0.npy`和`z1.npy`，再对两者使用同一索引抽batch。对两端各自独立重排会改变耦合。其[uniform与L2损失配置](https://github.com/gnobitab/RectifiedFlow/blob/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6/ImageGeneration/losses.py#L55-L133)取$`t\in[0.001,1)`$，传给网络的时间标签为$`999t`$；[离线配对](https://github.com/gnobitab/RectifiedFlow/blob/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6/ImageGeneration/sde_lib.py#L38-L72)用RK45在$`[0.001,1]`$上近似积分。下面采用精确$`[0,1]`$流，将数学操作逐项算清楚。

## 2. 同边缘Gaussian：先收缩再扩张，第二场为零

让两端独立且均为$`\mathcal N(0,I_d)`$，其中$`I_d`$是$`d`$维单位矩阵。复用03章的

```math
q(t)=(1-t)^2+t^2,\qquad
v_t(x)=\frac{2t-1}{q(t)}x.
```

因为$`q'(t)=2(2t-1)`$，精确ODE解为

```math
Z_t=\sqrt{q(t)}Z_0,\qquad Z_1=Z_0.
```

$`q(t)\ge1/2`$，速度为有界时间系数乘状态，ODE在整个区间良好。轨迹沿径向先收缩再扩张，时间上并非仿射；两端之间的弦为常值$`Z_0`$。原场从0跨到1的一步Euler给出$`Z_0+v_0(Z_0)=0`$。

精确生成端点为$`(Z_0,Z_0)`$，故reflow后的插值恒为$`Z_0`$，监督位移恒为0，新场为0。第二场的精确ODE与一步Euler都给出$`Z_0`$。本例第一轮中间边缘为$`\mathcal N(0,q(t)I_d)`$，下一轮则为恒定的$`\mathcal N(0,I_d)`$。

这是总体精确回归与精确ODE下的解析特殊例。学习模型会另有有限数据、拟合和生成配对的数值误差；一般高维任务需要测量这些误差和采样质量。

## 3. 已经时间仿射的流，仍可有较高传输成本

在二维固定逆时针90°旋转矩阵$`J`$，并令$`I=I_2`$：

```math
J=\begin{bmatrix}0&-1\\1&0\end{bmatrix},\qquad
X_0\sim\mathcal N(0,I),\qquad X_1=JX_0,\qquad
M_t=(1-t)I+tJ.
```

$`\mathop{\mathrm{det}}\nolimits M_t=q(t)>0`$，所以$`X_t=M_tX_0`$唯一确定起点。条件速度与精确轨迹为

```math
v_t(x)=(J-I)M_t^{-1}x=\frac{(2t-1)I+J}{q(t)}x,
\qquad Z_t=M_tZ_0.
```

沿每条轨迹，速度恰为常量$`(J-I)Z_0`$；一次Euler精确到达$`JZ_0`$。rectification和理想reflow均保持这一耦合。但它的总体平方位移成本为

```math
\mathbb E\|(J-I)X_0\|^2
=\mathop{\mathrm{tr}}\nolimits\big[(J-I)^\top(J-I)\big]=4.
```

本页成本采用坐标平方和$`c(a)=\|a\|^2`$，不含$`1/2`$系数，单位为坐标单位的平方。源与目标均为$`\mathcal N(0,I)`$，恒等耦合可达到成本0；平方成本非负，故0就是全局最优值。该例给出时间仿射RF固定点与全局二次最优传输（optimal transport，OT）之间的明确差距。

取第二节$`d=2`$时，两个例子还具有相同的中间边缘，因为$`M_tM_t^\top=q(t)I`$；旋转例多出$`Jx/q(t)`$速度分量。同一个边缘路径允许不同速度，端点耦合提供了决定条件回归的额外信息。

## 4. 一个CPU检查

下面代码使用现有NumPy，可直接复制到Python运行。实际核验版本为Python 3.12.14、NumPy 2.1.3、CPU float64；只处理两个二维确定性点和五个时间点，运行无需训练、随机抽样或下载权重。本云环境一次命令墙钟约0.23秒，包含Python启动，属于单次观测。代码核对轨迹导数、一整步Euler和端点；输出的总体成本由Gaussian协方差解析计算。

<!-- cpu-check: reflow-coupling -->
```python
"""Exact Gaussian coupling example. CPU, float64, NumPy; no learning or randomness."""
import numpy as np

I = np.eye(2, dtype=np.float64)
J = np.array([[0., -1.], [1., 0.]], dtype=np.float64)
z0 = np.array([[1., 0.], [.3, -1.2]], dtype=np.float64)  # Rows are particles.

def q(t):
    return (1-t)**2 + t*t

def independent_v(t, x):
    return ((2*t-1)/q(t))*x

def rotation_v(t, x):
    return x @ (((2*t-1)*I+J)/q(t)).T

for t in [0., .25, .5, .75, 1.]:
    M = (1-t)*I+t*J
    # Independent and rotation couplings have the same intermediate covariance.
    np.testing.assert_allclose(M@M.T, q(t)*I, atol=1e-15)
    radial_path = np.sqrt(q(t))*z0
    rotation_path = z0@M.T
    np.testing.assert_allclose(independent_v(t, radial_path),
                               (2*t-1)/np.sqrt(q(t))*z0, atol=1e-15)
    np.testing.assert_allclose(rotation_v(t, rotation_path), z0@(J-I).T, atol=1e-15)

np.testing.assert_array_equal(np.sqrt(q(1.))*z0, z0)  # Exact independent-field ODE.
np.testing.assert_array_equal(z0+independent_v(0., z0), np.zeros_like(z0))
np.testing.assert_allclose(z0+rotation_v(0., z0), z0@J.T, atol=1e-15)
# Reflow of the exact independent-field endpoint pairs (z0, z0): zero target.
np.testing.assert_array_equal(z0+np.zeros_like(z0), z0)
# Population costs use E[X0 X0^T]=I, not the two teaching points' sample average.
print("independent input / exact rectified cost:", float(np.trace(2*I)), 0.)
print("rotation input / exact rectified cost:", float(np.trace((J-I).T@(J-I))), 4.)
print("identity coupling achieves global minimum squared cost:", 0.)
print("independent-field one-step Euler endpoint:", z0+independent_v(0., z0))
```

CPU实际输出：独立原耦合与rectification成本分别为4.0和0.0，旋转原耦合与rectification成本均为4.0，恒等耦合达到全局最低平方成本0.0；独立例原场的一步Euler将两个测试点都映到零。独立检查还用联合协方差、线性求解与中心差分核对这些公式；内部时间点的轨迹导数最大差分误差约为$`2.40\times10^{-10}`$。这些检查验证解析例与实现之间的一致性。

## 5. 论文结论的适用范围

固定v1第13–14页定理3.3、3.5在rectifiable条件下给出边缘保持和凸位移成本非增。原文定义要求条件速度局部有界，积分ODE的解存在且唯一。对相关期望有限的凸函数$`c:\mathbb R^d\to\mathbb R`$，有

```math
\mathbb E[c(Z_1-Z_0)]\le\mathbb E[c(X_1-X_0)].
```

这里成本的输入是端点位移。二维旋转例中的等号符合这一结论；达到全局OT还需要进一步条件。

第15页定理3.7在每轮耦合均可rectify且初始二阶位移矩有限时，控制生成轨迹偏离时间仿射的程度与输入耦合条件回归残差的累计量，并给出最佳相邻轮组合的$`O(1/K)`$界，其中$`K`$是rectification轮数。该界控制迭代中的最佳值；逐轮直度单调性、给定有限轮后一步精确性需另行证明。平方成本分解和一致的相邻轮索引见[推导证据](../../records/measurements/reflow-coupling/squared-cost-derivation.md)。

固定v1第15页式(13)左侧两项在原PDF中漏写平方，随后递推式带有平方；同页best-iterate文字与求和式的索引也不一致。本页采用独立平方展开与明确的相邻轮索引；[来源核对](../../records/measurements/reflow-coupling/README.md)记录了原文定位和处理依据。

## 6. 练习与参考答案

1. **第二轮是否保持第一轮所有中间边缘？** 提示：比较独立Gaussian例的$`q(t)I_d`$与第二轮的$`I_d`$。答案：每轮保持其输入耦合定义的插值边缘；耦合更新后，中间边缘可变化。
2. **把旋转换成180°旋转$`-I`$，上述推导哪里失效？** 答案：$`M_{1/2}=0`$，中点无法恢复起点；90°例中全区间可逆和良好ODE的论证在这里失去前提。
3. **一步Euler精确是否决定平方成本最小？** 答案：90°例一步到达$`JZ_0`$的成本为4，恒等耦合成本为0。数值积分精度与端点配对成本有不同评价对象。

## 7. 固定来源与本地证据

- Liu、Gong、Liu，[Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003v1)，arXiv:2209.03003v1，2022-09-07。[固定PDF](https://arxiv.org/pdf/2209.03003v1)第6页算法1、式(2)，第7页直度式(3)，第13–14页边缘与凸成本定理，第15–17页递归与OT关系。
- [作者仓库](https://github.com/gnobitab/RectifiedFlow/tree/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6)固定commit `5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6`，日期2024-06-20，为后续作者实现快照。[配对与训练](https://github.com/gnobitab/RectifiedFlow/blob/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6/ImageGeneration/run_lib_reflow.py#L141-L221)、[损失](https://github.com/gnobitab/RectifiedFlow/blob/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6/ImageGeneration/losses.py#L55-L133)与[配对求解器](https://github.com/gnobitab/RectifiedFlow/blob/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6/ImageGeneration/sde_lib.py#L38-L72)分别对应本页流程。
- [证据索引](../../records/measurements/reflow-coupling/README.md)保存来源SHA、独立推导和实际CPU输出。作者损失函数经固定SHA校验、AST原样抽取后，使用小张量与模型替身验证配对插值、位移目标、时间缩放及均方损失；执行范围覆盖该函数。本文Gaussian数学代码为独立实现。
