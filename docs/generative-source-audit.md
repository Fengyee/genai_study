# 第 01 章来源核验：ELBO、对抗目标与变量替换

核验日期：2026-11-10。对应 [01 生成模型的共同问题](chapters/01-generative-models.md)。本轮核对指定数学主张、适用条件和符号方向，并用独立解析例子检查等式；论文中的图像实验和总体性能留待相应实践验证。

## 固定的原始版本

| 论文 | 本次阅读版本与提交日期 | PDF 页数 / 字节数 | SHA-256 |
| --- | --- | --- | --- |
| Auto-Encoding Variational Bayes | [1312.6114v11](https://arxiv.org/pdf/1312.6114v11)，2022-12-10 | 14 / 3,926,758 | `bdcb63b79ed88f64f8ee2d43a721930d01dde5441a4dba20ecaa2342b34e017c` |
| Generative Adversarial Nets | [1406.2661v1](https://arxiv.org/pdf/1406.2661v1)，2014-06-10 | 9 / 530,482 | `ff5819e3a7b713c3bd3107b7de3d51fe0a347aa5d8444f0efdcf2345ef0a8b63` |
| Density Estimation using Real NVP | [1605.08803v3](https://arxiv.org/pdf/1605.08803v3)，2017-02-27 | 32 / 6,681,321 | `b80d619f41d7e7fe537907bdb6029d8b6c20e6b454e1331f3271c9550e4424d6` |
| Neural Ordinary Differential Equations | [1806.07366v6](https://arxiv.org/pdf/1806.07366v6)，2026-10-09 | 19 / 3,800,013 | `32195e2c86f3886279395e93b8b99f2e5d242cb6306eaf2b239ed68b66bce151` |

日期来自 arXiv 版本历史，页码按 PDF 第一页起算。哈希记录本次下载的字节；以后若 arXiv 重新生成同版本 PDF，应复核内容并记录哈希变化。下载后使用环境已有的 `pdftotext -layout` 阅读相关页面；GAN PDF 的路径绘制警告和 Neural ODE 的字体警告未中断正文提取，公式按上下文与独立推导核对。

复核下载与文本提取（输出至 `/tmp`，无需把 PDF 加入仓库）：

```bash
mkdir -p /tmp/genai-generative-audit
for id in 1312.6114v11 1406.2661v1 1605.08803v3 1806.07366v6; do
  curl -fSL --max-time 60 "https://arxiv.org/pdf/$id" \
    -o "/tmp/genai-generative-audit/$id.pdf"
  sha256sum "/tmp/genai-generative-audit/$id.pdf"
  pdftotext -layout "/tmp/genai-generative-audit/$id.pdf" \
    "/tmp/genai-generative-audit/$id.txt"
done
```

## 从原文到教材的映射

| 教材主张 | 固定 PDF 定位 | 核对结果与本轮补充 |
| --- | --- | --- |
| 对数似然 = ELBO + 后验 KL | VAE v11，第 3 页，式 (1)–(3) | 两个 KL 均以近似后验 $q_\phi$ 为第一个分布。教材补充有限边缘密度、支持集和有限期望条件，保留完整恒等式 |
| 高斯重参数化与重建项 | VAE v11，第 3–5 页，式 (4)–(7)、(9)–(10)；附录 B，第 10–11 页 | 重参数化噪声与参数无关；固定先验下对角高斯 KL 可解析求值。第 5 页明确观测分布可为 Gaussian 或 Bernoulli，重建项由观测模型决定 |
| GAN minimax 与非饱和损失 | GAN v1，第 3 页，式 (1) 及其后段落 | 原文已提出用最大化 $\log D(G(z))$ 增强初期梯度。教材补上对应最小化损失及 logit 导数练习 |
| 最优判别器 | GAN v1，第 4 页，Proposition 1、式 (2)–(4) | 在两密度之和为正处有 $D^*=p_{\mathrm{data}}/(p_{\mathrm{data}}+p_G)$；两者均零的区域对目标无贡献 |
| JS 等式与收敛条件 | GAN v1，第 5 页，Theorem 1、式 (5)–(6)、Proposition 2 | 代入最优判别器得到 $-\log4+2D_{\mathrm{JS}}$。收敛论证要求充分容量、每轮最优判别器及分布空间的小幅更新；原文随后说明参数化网络引入多个临界点 |
| 可逆变换的密度 | Real NVP v3，第 3 页，§3.1、式 (2)–(3) | 原文 $f$ 是数据→潜变量，教材 $f_\theta$ 是潜变量→数据。已明确二者互为逆，补同维、可微、Jacobian 非零条件及生成方向的负 log-det 写法 |
| 可计算的耦合层 | Real NVP v3，第 4–5 页，式 (4)–(8) | 仿射耦合层的三角 Jacobian 使 determinant 易算；逆映射只需重新计算 $s,t$。这是正文“结构约束”的具体依据 |
| 连续流的密度演化 | Neural ODE v6，第 4 页，Theorem 1、式 (8)；第 14 页，附录 A | 沿轨迹的 log-density 导数为负散度。原文使用 Lipschitz/时间连续条件，证明中还使用有界性和微分操作；教材以足够光滑及区间内解存在唯一作条件，并用总导数 $d/dt$ 明确沿轨迹含义 |
| 初值唯一性 | Neural ODE v6，第 8 页，§6 “Uniqueness” | 可逆演化依赖解的唯一性。教材的 $\dot x=\sqrt{\lvert x\rvert}$ 等待后离开的反例为独立教学构造 |

核对结果：原有 ELBO、GAN-JS 和离散变量替换公式的方向与系数正确。本轮补足支持集、映射方向和目标适用条件；连续流部分增加负散度公式，使“密度计算的成本”有可追踪的数学对象。

**版本修订值得保留。** Neural ODE v6 的第 4 页脚注明确更正 Figure 3c 的实验日志错误：原先“反向 NFE 约为正向一半”的表述更新为该模型、容差 $10^{-5}$ 到 $10^{-1}$ 下中位比约 0.8。本章没有使用旧性能数字；本轮将所读版本固定为 v6，后续介绍 adjoint 成本时应从修订后的图和实验条件出发。该 NFE 比值是论文对特定实验的报告。

## 独立解析检查

以下例子使用仓库依赖中的 NumPy，无训练、随机采样或外部模型。它们分别检查 KL 方向、JS 系数、Jacobian 绝对值与散度符号。

1. 取 $Z\sim\mathcal N(0,1)$，$X\mid Z\sim\mathcal N(Z,0.49)$，观测 $x=1.3$，近似后验 $q=\mathcal N(-0.2,0.8)$。真实后验均值为 $x/1.49$，方差为 $0.49/1.49$，可独立计算边缘密度与两个 KL。
2. 在三个离散结果上取 $p=(0.6,0.3,0.1)$、$q=(0.1,0.4,0.5)$，直接求和验证最优判别器目标与 JS 表达式。
3. 取 $X=-2Z+3$，密度为 $\varphi((x-3)/(-2))/2$。数值积分应为 1，均值 3，方差 4；负缩放用于检查绝对值。
4. 取 $\dot x_t=a x_t$，$a=0.7$，则 $x_t=e^{at}x_0$，沿轨迹 $\log p_t(x_t)-\log p_0(x_0)=-at$。此解析解直接检验负散度符号。

在仓库根目录激活环境后运行：

```bash
python - <<'PY'
import math
import numpy as np

x, sigma2, mu, qvar = 1.3, 0.49, -0.2, 0.8
postvar, postmu = sigma2 / (1 + sigma2), x / (1 + sigma2)
elbo = -0.5 * (math.log(2 * math.pi * sigma2)
               + ((x - mu)**2 + qvar) / sigma2)
elbo -= 0.5 * (qvar + mu**2 - 1 - math.log(qvar))
gap = 0.5 * (math.log(postvar / qvar)
             + (qvar + (mu - postmu)**2) / postvar - 1)
logp = -0.5 * (math.log(2 * math.pi * (1 + sigma2))
               + x*x / (1 + sigma2))

p, q = np.array([.6, .3, .1]), np.array([.1, .4, .5])
d, m = p / (p + q), (p + q) / 2
value = np.sum(p * np.log(d) + q * np.log1p(-d))
js = .5 * np.sum(p * np.log(p / m) + q * np.log(q / m))

grid = np.linspace(-20, 26, 100001)
density = np.exp(-.5 * ((grid - 3) / -2)**2) / (2 * math.sqrt(2 * math.pi))
moments = [np.trapezoid(f * density, grid)
           for f in [np.ones_like(grid), grid, (grid - 3)**2]]

a, t, z0 = .7, .8, 1.2
xt = math.exp(a*t) * z0
logpt = -.5 * (math.log(2 * math.pi) + 2*a*t + xt**2 / math.exp(2*a*t))
logp0 = -.5 * (math.log(2 * math.pi) + z0**2)
errors = [abs(logp - elbo - gap), abs(value + math.log(4) - 2*js),
          abs(logpt - logp0 + a*t)]
assert max(errors) < 1e-12
assert np.allclose(moments, [1, 3, 4], rtol=0, atol=1e-12)
print({"ELBO": elbo, "posterior_KL": gap, "log_marginal": logp,
       "identity_errors": errors, "affine_moments": moments})
PY
```

当前环境 Python 3.12.14 / NumPy 2.1.3 的实际结果：ELBO `-3.7060802629` nat，后验 KL `2.0206395757` nat，对数边缘密度 `-1.6854406871`；三个恒等式绝对误差最大 `8.9e-16`，仿射密度积分/均值/方差分别为 `1.0 / 3.0 / 4.000000000000001`。这组检查使用解析高斯分布和离散求和，便于读者独立复算。

GAN 梯度练习另外可由 $d\,\operatorname{sigmoid}(a)/da=D(1-D)$ 直接得到两个导数 $-D$ 和 $D-1$。连续流等待反例为：任取 $\tau\ge0$，令 $x(t)=0$（$t\le\tau$），之后 $x(t)=(t-\tau)^2/4$，左右导数在 $\tau$ 都为 0，且满足相同初值。

后续审计继续覆盖自回归的训练/采样接口及研究路线中的具体架构主张；本页固定四篇论文的上述公式和段落。
