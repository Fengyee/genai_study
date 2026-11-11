# 理想 Gaussian rectified flow：两种耦合的独立数学审查

## 结论及约定

二维 90° 旋转耦合给出所要求的反例：全部条件插值路径和精确 ODE 路径均为恒速端点连线；每个时刻的插值映射可逆，速度场在闭区间两端均良定；rectification 和 reflow 保持原耦合。它的平方距离运输成本是 4，而同分布恒等耦合的成本是 0。

下文使用列向量，时间从 0 到 1，运输成本取 $`\mathbb E\|X_1-X_0\|^2`$，不带 $`1/2`$。采用带 $`1/2`$ 的成本约定时，所有非零成本相应减半。

为避免迭代次数的歧义，将两个步骤写开。给定初始端点联合分布 $`\pi=\mathcal L(X_0,X_1)`$，设


```math
X_t=(1-t)X_0+tX_1,\qquad V=X_1-X_0,\qquad v_t(x)=\mathbb E[V\mid X_t=x].
```


1. **Rectification**：精确求解 $`\dot Z_t=v_t(Z_t)`$、$`Z_0\sim p_0`$，得到新的端点联合分布 $`\pi^R=\mathcal L(Z_0,Z_1)`$。
2. **Reflow 回归**：使用 $`\pi^R`$ 的端点对及其线性插值，重新求总体平方损失最优速度场；随后可再精确积分。

这里使用精确总体条件期望和精确 ODE，结论不依赖神经网络拟合或有限步数值求解。本文中“轨迹直”明确指自身端点间的仿射、恒速路径。仅落在某条空间直线上是较弱性质。

## 一、独立标准 Gaussian 端点

令 $`X_0,X_1`$ 在 $`\mathbb R^d`$ 中相互独立且均为 $`\mathcal N(0,I)`$。定义


```math
q_t=(1-t)^2+t^2\ge\tfrac12.
```


线性插值的边缘分布为 $`p_t=\mathcal N(0,q_t I)`$，并且


```math
\mathop{\mathrm{Cov}}\nolimits(V,X_t)=(2t-1)I.
```


联合 Gaussian 条件期望给出


```math
v_t(x)=\frac{2t-1}{q_t}x=\frac{q'_t}{2q_t}x.
```


条件目标方差为 $`\mathop{\mathrm{Cov}}\nolimits(V\mid X_t)=I/q_t`$；条件路径虽然各自是直线，总体回归通常会平均不同端点对的速度。

ODE 的精确传播算子是


```math
Z_t=\sqrt{q_t/q_s}\,Z_s,\qquad Z_t=\sqrt{q_t}\,Z_0\quad(s=0).
```


它满足 $`\dot Z_t=v_t(Z_t)`$，且 $`Z_1=Z_0`$。对任何非零起点，轨迹始终位于同一径向直线上，先收缩至 $`Z_{1/2}=Z_0/\sqrt2`$，再展开回起点。由于两端相同，其端点连线为恒定路径，因此这条 ODE 路径并非自身端点间的恒速仿射插值。避免将这种径向往返简单称作空间弯曲轨迹。

端点和全区间均良定：写 $`y=2t-1`$，则 $`q_t=(1+y^2)/2`$，从而


```math
\left|\frac{2t-1}{q_t}\right|=\frac{2|y|}{1+y^2}\le1.
```


系数连续且对空间全局一致 Lipschitz，具有唯一完整流。

原始独立耦合的成本为 $`2d`$。Rectification 得到恒等耦合 $`(Z_0,Z_0)`$，成本为 0。对该端点对 reflow：所有插值均等于起点，目标速度为 0，新速度场处处为 0，精确 ODE 为恒定路径，后续迭代保持不变。

## 二、二维 90° 旋转端点

取逆时针旋转


```math
R=\begin{pmatrix}0&-1\\1&0\end{pmatrix},\qquad R^\top=-R,\qquad R^2=-I,
```


令 $`X_0\sim\mathcal N(0,I_2)`$、$`X_1=RX_0`$，故两端分布均为标准 Gaussian。写


```math
A_t=(1-t)I+tR,\qquad X_t=A_tX_0.
```


直接计算得


```math
A_tA_t^\top=q_tI,\qquad \det A_t=q_t\ge\tfrac12,\qquad A_t^{-1}=\frac{(1-t)I-tR}{q_t}.
```


因此每个 $`t\in[0,1]`$ 的插值映射均可逆，两个端点也包括在内，且边缘分布同样为 $`p_t=\mathcal N(0,q_tI)`$。给定 $`X_t=x`$，起点唯一确定为 $`A_t^{-1}x`$，所以


```math
v_t(x)=(R-I)A_t^{-1}x=\frac{(2t-1)I+R}{q_t}x.
```


条件回归残差为零。令 $`B_t=[(2t-1)I+R]/q_t`$；由于 $`B_tA_t=R-I`$，精确解为


```math
Z_t=A_tZ_0=(1-t)Z_0+tRZ_0,
```


并且


```math
\dot Z_t=(R-I)Z_0,\qquad \ddot Z_t=0,\qquad Z_1=RZ_0.
```


原始条件路径与 ODE 路径逐条一致，都是恒速端点连线；原点对应退化的恒定路径。不同起点的轨迹在同一时刻不会交叉，因为 $`A_t`$ 可逆。不同时间上的几何相交与 ODE 唯一性相容。

### 闭区间良定性

系数没有端点奇异性：$`B_0=R-I`$、$`B_1=I+R`$。利用 $`((2t-1)I+R)^\top((2t-1)I+R)=2q_tI`$，有


```math
\|B_t\|_2=\sqrt{2/q_t}\le2.
```


因此速度场在时间闭区间上连续，对空间全局一致 Lipschitz，线性增长；任意有限起点均有唯一覆盖整个区间的解。Gaussian 端点具有有限二阶矩，所有上述成本均有限。90° 选择保证了 $`A_t`$ 的全区间可逆性；若直接换成 180° 旋转，$`A_{1/2}=0`$，本论证所用的条件不再成立。

### Reflow 固定点及全局运输成本

Rectification 产生 $`\mathcal L(Z_0,Z_1)=\mathcal L(Z_0,RZ_0)=\pi`$。再次对该端点对回归，插值分布、目标速度和边缘速度场均保持一致；所有后续精确迭代也相同。

其平方距离成本为


```math
\mathbb E\|RX_0-X_0\|^2=\mathop{\mathrm{tr}}\nolimits((R-I)^\top(R-I))=\mathop{\mathrm{tr}}\nolimits(2I)=4.
```


由于两个边缘分布相同，恒等耦合可行且成本为 0。成本非负，因此全局最优值 $`W_2^2(p_0,p_1)=0`$。旋转耦合严格高于该值；它同时满足恒速直轨迹、闭区间良定性和 reflow 固定点性质。这些性质足以否定“轨迹直或 reflow 不再变化就保证全局平方距离 OT”的一般性断言。

## 相同边缘路径并不确定相同耦合

上述两种初始耦合恰好都生成 $`p_t=\mathcal N(0,q_tI)`$。两速度场的差为 $`Rx/q_t`$。由于 $`\mathop{\mathrm{tr}}\nolimits R=0`$ 且 $`x^\top Rx=0`$，有


```math
\nabla\cdot\left(p_t(x)\frac{Rx}{q_t}\right)=0.
```


故同一边缘密度路径可以对应不同速度场，积分后的端点耦合也可以不同。独立例经 rectification 变为恒等耦合；旋转例保留旋转耦合。

## 可复核范围与运行

本次推导独立完成；没有读取 `/tmp/genai-reflow-candidate` 或联系其作者。另请同级代理独立复核了旋转部分，得到相同的公式、全局 Lipschitz 界和成本。

`check_gaussian.py` 使用已有 PyTorch、float64 和单个 CPU 线程，对固定小数组及含两个端点的时间网格验证逆矩阵、边缘协方差、JVP 与 ODE 一致性、有限差分、旋转恒速/零物质加速度、端点、reflow 场、成本和 Lipschitz 界。Gaussian 条件期望来自本文件的解析推导。脚本中的四点规则只用于精确一二阶矩与二次成本核验；这些固定点不承担 Gaussian 条件期望估计。

在仓库根目录运行（输出路径须尚不存在）：

```bash
.venv/bin/python records/measurements/reflow-coupling/check_gaussian.py \
  --output artifacts/reflow-independent/gaussian.json
```

解析结论适用于连续闭区间与全部起点；数值核验覆盖脚本显式列出的有限输入，不单独承担全局证明。所有实际数值残差、阈值、版本和脚本 SHA-256 记录于 [gaussian-check.json](gaussian-check.json)。
