# Reflow的平方成本分解与相邻轮索引

本页独立推导[阅读卡](../../../docs/readings/reflow-coupling.md)所用的成本关系。假设每轮端点耦合可rectify，单轮边缘保持结论成立，且初始平方位移矩有限。成本采用坐标平方和。

单轮令$`\Delta=X_1-X_0`$、$`D=Z_1-Z_0`$，并定义

```math
X_t=(1-t)X_0+tX_1,\qquad
v_t(x)=\mathbb E[\Delta\mid X_t=x],\qquad
D=\int_0^1v_t(Z_t)\,\mathrm dt.
```

输入耦合$`\pi`$的条件回归残差与精确流的时间仿射偏离分别为

```math
V(\pi)=\int_0^1\mathbb E\|\Delta-v_t(X_t)\|^2\,\mathrm dt,
\qquad
S(Z)=\int_0^1\mathbb E\|v_t(Z_t)-D\|^2\,\mathrm dt.
```

条件期望的正交分解给出

```math
V(\pi)=\mathbb E\|\Delta\|^2
-\int_0^1\mathbb E\|v_t(X_t)\|^2\,\mathrm dt.
```

展开$`S`$中的平方，再使用$`D=\int_0^1v_t(Z_t)\,\mathrm dt`$，得到

```math
S(Z)=\int_0^1\mathbb E\|v_t(Z_t)\|^2\,\mathrm dt
-\mathbb E\|D\|^2.
```

单轮中$`X_t`$与$`Z_t`$边缘相同，两个积分能量项相等。因此记$`C(\pi)=\mathbb E\|\Delta\|^2`$、$`\pi^R=\mathop{\mathrm{Law}}\nolimits(Z_0,Z_1)`$，有

```math
C(\pi)-C(\pi^R)=S(Z)+V(\pi).
```

令$`\pi^k`$为第$`k`$次更新前的端点耦合，$`Z^{k+1}`$为其诱导的精确流，$`\pi^{k+1}=\mathop{\mathrm{Law}}\nolimits(Z_0^{k+1},Z_1^{k+1})`$。初始只定义端点耦合$`\pi^0`$。连续执行$`K\ge1`$次rectification，望远镜求和得到

```math
\sum_{k=0}^{K-1}\left[S(Z^{k+1})+V(\pi^k)\right]
=C(\pi^0)-C(\pi^K)\le C(\pi^0).
```

各项非负，所以最佳相邻轮组合满足

```math
\min_{0\le k\lt K}\left[S(Z^{k+1})+V(\pi^k)\right]
\le\frac{C(\pi^0)}{K}.
```

这里控制累计量及最佳组合；逐轮$`S`$的单调性需要额外论证。独立Gaussian例在二维给出$`S=4-\pi`$、$`V=\pi`$，平方成本下降4；此句中的$`\pi`$为圆周率。旋转例的$`S=V=0`$，成本保持4。数值积分记录见[candidate-math-check.json](candidate-math-check.json)与[root-check.json](root-check.json)。

本推导对应arXiv:2209.03003v1第15页定理3.7及式(13)后的平方成本递推。实际PDF式(13)左侧省略了平方；best-iterate文字的$`S`$索引与上方求和不同。本页采用独立平方展开及明确相邻索引，原文位置见[来源核对](source-audit.md)。
