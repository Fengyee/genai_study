# 从 SVD 到 CogVideoX：视频表示、训练与公开模型接口

**先修：** [02 扩散](../chapters/02-diffusion.md)、[04 图像与视频路线](../chapters/04-image-video-roadmap.md)、[05 评价](../chapters/05-evaluation.md)、[06 视频时序反例](../chapters/06-video-temporal-lab.md)。[08 引导实验](../chapters/08-image-guidance-lab.md)提供条件丢弃与配对噪声的实践背景。

**学习目标：** 画出两个系统从视频到潜变量再到生成的路径；区分视频帧索引、扩散噪声级别和播放帧率；根据具体权重核对条件输入、预测量、token 数和评价协议。

**范围与核验日期：** 2026-11-10。本文核验固定论文、官方代码和模型配置，运行小型 CPU 代数与尺寸检查。视频权重推理、训练复现与生成质量测量仍待独立实验。SVD 论文首次发表于 2023 年；CogVideoX 于 2024-08-12 首发，本卡采用其 2025 年修订版。

## 1. 先选定要比较的对象

| 对象 | 本卡使用的版本 | 要回答的问题 |
| --- | --- | --- |
| SVD 训练研究 | [2311.15127v1](https://arxiv.org/abs/2311.15127v1)，2023-11-25 | 图像预训练、视频预训练和高质量微调如何衔接？数据筛选怎样影响结果？ |
| SVD / SVD-XT 公开权重 | 下文固定的两个 `stabilityai` 模型 revision | 单张参考图像怎样进入网络？14 帧与 25 帧版本如何配置？ |
| CogVideoX 训练研究 | [2408.06072v3](https://arxiv.org/abs/2408.06072v3)，2025-03-26，ICLR 2025 | 时空压缩、文本与视频联合注意力、混合长度训练怎样配合？ |
| CogVideoX-2B 公开权重 | `THUDM/CogVideoX-2b` 的固定 revision，见来源表 | 49 帧、480×720 的文本到视频接口与论文描述怎样对应？ |

论文覆盖一组训练实验，模型卡和配置对应具体发布物。CogVideoX v3 摘要写最高 768×1360、10 秒、16 fps；本卡所选 2B 模型卡写 480×720、49 帧、8 fps，标为约 6 秒。记录实验时应同时写论文版本与模型 revision。

## 2. SVD：先学图像，再学运动与任务条件

### 三阶段训练与数据证据

SVD 第 3–6 页 §3–4、附录 D 第 19–20 页给出下面的训练路径。尺寸均写为高×宽，更新次数和样本数均为**作者报告**。

| 阶段 | 输入与主要操作 | 论文中的具体配置 |
| --- | --- | --- |
| 图像初始化 | 从 Stable Diffusion 2.1 的空间网络出发，适配连续噪声及预条件函数 | 先用 1k 更新适配时间嵌入，再对全部网络训练 30k 更新 |
| 视频预训练 | 在空间卷积/注意力后加入时间卷积/注意力，训练整个网络 | 14 帧；256×384、150k 更新，再到 320×576、100k 更新 |
| 高质量与任务微调 | 使用约 1M 高质量视频，并选择文本或图像条件 | T2V 微调到 576×1024；I2V 还经过 320×576 的条件适配，再训练 14/25 帧版本 |

原始视频先做多帧率切镜检测，再生成中间帧 CoCa 描述、VideoBLIP 视频描述及二者的 LLM 汇总。筛选特征包括光流、OCR 文本面积、CLIP 图文相似度与美学分数。第 4 页 Table 1 报告从 577M clips 筛到 152M；正文把原始规模近似写为 580M。

作者在 9.8M 子集上分别改变筛选阈值，保持相应实验的训练配置，再用人工偏好选择阈值。第 5 页 Figure 4 还比较后续高质量微调中的差异。这支持在其数据与配置下研究筛选的价值。删去低运动或大量文字的视频，同时改变了训练分布：后续若研究静止镜头或文字动画，需要重新考虑任务覆盖。

### 噪声级别与网络输出

附录 D 第 18 页式(2)写加权干净样本回归：

$$
\mathbb E_{z_0,c,\sigma,\epsilon}\!\left[
\lambda(\sigma)\|D_\theta(z_0+\sigma\epsilon;\sigma,c)-z_0\|_2^2
\right],\qquad \epsilon\sim\mathcal N(0,I).
$$

这里用 $z_0$ 强调生成发生在潜空间；原文用 $x_0$。$\sigma$ 是噪声标准差。预条件去噪器 $D_\theta$ 由底层网络 $F_\theta$ 构造，式(3)、(10)–(13)为

$$
D_\theta(z_\sigma;\sigma,c)
=\frac{z_\sigma}{1+\sigma^2}
-\frac{\sigma}{\sqrt{1+\sigma^2}}
F_\theta\!\left(\frac{z_\sigma}{\sqrt{1+\sigma^2}};
\tfrac14\log\sigma,c\right).
$$

官方配置使用 `VScalingWithEDMcNoise`；后文将它与 `v_prediction` 的代数关系独立核对。采样沿较大噪声级别走向较小噪声级别。此处的 $\sigma$、网络时间嵌入、视频中的第几帧各有不同含义。

### 一张参考图像怎样控制整段视频

第 7 页 §4.3、附录 D.4 第 20 页描述两条图像条件路径：

1. 用参考图的 CLIP image embedding 提供交叉注意力条件。
2. 对参考图加少量噪声，经图像编码器取得分布均值，沿视频帧轴复制，再与带噪视频潜变量按通道拼接。

固定 `svd.yaml` 的去噪输入有 8 通道，输出有 4 通道；参考 latent 和视频 latent 各占 4 通道。空间下采样为 8。视频生成分支的 `scale_factor` 为 0.18215；参考图条件编码器沿用 `VideoPredictionEmbedderWithEncoder` 默认的 `scale_factor=1.0`。复现时应逐条核对两条编码路径。对 14×576×1024 的输出，视频 latent 有 14×4×72×128 个标量。这里保留 14 个潜帧；发布模型另用带时间层的解码器改善重建的时间一致性。

SVD 的引导强度还随**视频帧索引**变化。论文 Figure 15 与固定代码 `LinearPredictionGuider` 都在 $F$ 帧上构造

$$
s_f=s_{\min}+\frac{f}{F-1}(s_{\max}-s_{\min}),\qquad
\widehat D_f=D_{u,f}+s_f(D_{c,f}-D_{u,f}).
$$

同一次去噪调用中，不同帧使用不同系数；这组系数会在采样步骤间重复使用。固定 SVD 配置从 1 增至 2.5。采样脚本把基准分支的两条图像 embedding 置零，帧率、运动分桶与条件噪声强度仍保留，所以 $D_u$ 在这里表示空图像条件的预测。作者报告这一设计平衡其样例中的参考图一致性与过饱和，效果应在目标任务上实测。

### 帧率条件与输出视频分别记录

固定官方脚本的默认值为 `fps_id=6`、`motion_bucket_id=127`、`cond_aug=0.02`；SVD 默认 14 帧/25 步，SVD-XT 默认 25 帧/30 步。`fps_id` 被直接送入条件编码，最终 `imageio.mimwrite(video_path, vid)` 调用没有显式提供播放 fps。

因此复现记录需要同时保存模型的帧率条件和最终视频文件的实际 fps，可用媒体元数据检查播放值。对 $F$ 帧、播放率 $r$，相邻采样时刻间隔为 $1/r$，首尾帧时刻相差 $(F-1)/r$；按每帧显示一个间隔计，文件播放时长为 $F/r$。这与 [06 章](../chapters/06-video-temporal-lab.md)的速度单位直接相连。

## 3. CogVideoX：压缩时间，再联合建模文本与视频

### 3D causal VAE 与 token

论文第 4–5 页 §2.1、Table 1 的预训练变体 B 使用空间 8×8、时间 4 倍压缩，潜通道数 16。VAE 先在 17 帧、256×256 上训练，再以时间上下文并行微调到 161 帧；目标包含 L1 重建、LPIPS、KL，并在训练开始一段时间后加入 3D 判别器的 GAN 损失。

时间因果卷积把 padding 放在序列开始，使局部编码不读取未来帧。去噪 Transformer 使用文本与整个带噪视频上的联合注意力；应分别核对编码器的因果性与去噪网络的注意力范围。

对所选 2B 配置，第一帧单独保留时间位置，其余帧按 4 倍压缩；对 $F=4k+1$，潜帧数为

$$F_z=1+\frac{F-1}{4}.$$

因此 49 帧对应 13 个潜帧。480×720 视频的潜空间为 $16\times13\times60\times90$（本式采用通道、帧、高、宽顺序）。时间 patch 为 1、空间 patch 为 2×2，得到

$$L_v=13\cdot(60/2)\cdot(90/2)=17\,550$$

个视觉 token；配置的文本序列长度为 226，拼接后共 17,776 个 token。每个视觉 patch 在投影前有 $16\cdot1\cdot2\cdot2=64$ 个数。2B 配置的 latent scaling factor 为 1.15258426，应与其 VAE 配套使用。

一张显式的密集注意力矩阵有 $17\,776^2=315\,986\,176$ 个元素/头。FP16 存放它需要约 632 MB/头，这是**张量尺寸分析**；FlashAttention 等实现改变实际中间存储，模型总内存还包括参数、激活、文本编码与解码等部分。

### 文本与视觉如何交互

第 5 页 §2.2 将 T5 文本表示与视频 token 沿序列拼接，在 Expert Adaptive LayerNorm 中分别调制两种模态，再进行联合 3D attention。模态专属的归一化参数帮助处理两类表示的尺度与分布差异。

所选 2B 发布配置采用 sinusoidal position embedding，`use_rotary_positional_embeddings=false`。论文正文详细介绍的 3D RoPE 对应另一些配置；Table 6 同样将 2B 与 5B 的位置编码分开列出。

### 实际训练目标、训练批次和数据

第 6 页 §3 明确使用 **v-prediction 与 zero terminal SNR**。第 7 页式(1)回顾 DDPM 的噪声预测损失，用来解释时间采样策略；复现所选模型时，固定 scheduler 配置中的 `prediction_type=v_prediction` 与 `rescale_betas_zero_snr=true` 是直接依据。

训练混合图像与视频，将图像视为单帧样本。Multi-Resolution Frame Pack 处理不同长度/分辨率；分辨率逐步提高，最后做高质量微调。Explicit Uniform Sampling 把扩散时间范围分给不同数据并行 rank，每个 rank 在自己的区间内抽样。等宽区间、等量样本下，聚合估计仍对应均匀时间目标；分层采样降低时间分布偶然失衡的程度，方差变化还取决于各时间区间的损失分布。

附录 A 第 15 页将最后微调数据约占原数据 20%，并报告视觉质量略有提高、语义能力略有下降。这提醒我们同时评估视觉观感与条件表达。附录 D 第 17 页还描述 T2V 到 I2V 的微调：参考图经 VAE 后沿通道拼接到带噪视频输入，训练中给参考图条件加噪；本卡所选 2B T2V 权重仍使用文本入口。

第 7–8 页 §3.4 报告约 35M 个筛后单镜头视频，平均约 6 秒，并混入经美学筛选的约 2B 张图像。20k 个人工标注视频用于训练六类 Video-LLaMA 过滤器；筛选还使用光流与美学阈值。描述生成先结合简短视频 caption、CogVLM 密集图像 caption 和 GPT-4 汇总，再训练 Llama 2 加速汇总，进一步训练 CogVLM2-Caption。本文记录这些作者方法，实际执行仅用固定论文与配置。

## 4. CPU 检查：把 `v_prediction`、帧轴与尺寸算清楚

在本仓库已安装的 CPU 环境运行下面代码。它只创建小向量和标量，几秒内完成，无需下载权重。模型总推理时间仍需单独测量。

令 $a^2+b^2=1$，带噪输入为 $z_t=az_0+b\epsilon$，常见 VP 扩散的目标为 $v_{\mathrm{VP}}=a\epsilon-bz_0$。于是

$$z_0=az_t-bv_{\mathrm{VP}},\qquad
\epsilon=bz_t+av_{\mathrm{VP}}.$$

取 $a=(1+\sigma^2)^{-1/2}$、$b=\sigma a$，SVD 的预条件式也满足这一还原关系。`v_prediction` 的定义随噪声级别变化；[03 章](../chapters/03-flow-matching.md)直线路径的速度目标则为数据减噪声，使用参数前先展开对应公式。

<!-- cpu-check: video-vp-shapes -->
```python
import numpy as np

z0 = np.array([-2.0, 0.3, 1.2], dtype=np.float64)
eps = np.array([0.7, -1.0, 0.2], dtype=np.float64)
for sigma in [0.25, 1.0, 4.0]:
    a = 1.0 / np.sqrt(1.0 + sigma**2)
    b = sigma * a
    z_sigma = z0 + sigma * eps
    z_t = a * z_sigma
    v = a * eps - b * z0
    denoised = z_sigma / (1.0 + sigma**2) - b * v
    np.testing.assert_allclose(denoised, z0, atol=1e-14)
    np.testing.assert_allclose(a * z_t - b * v, z0, atol=1e-14)
    np.testing.assert_allclose(b * z_t + a * v, eps, atol=1e-14)

# VP 目标在零终点 SNR 处的极限：a=0, b=1。
a, b = 0.0, 1.0
z_t, v = a * z0 + b * eps, a * eps - b * z0
np.testing.assert_array_equal(z_t, eps)
np.testing.assert_array_equal(v, -z0)
np.testing.assert_array_equal(a * z_t - b * v, z0)

frames, height, width = 49, 480, 720
latent_frames = 1 + (frames - 1) // 4
# 独立模拟两次时间下采样的边界：保留首帧，其余按相邻两帧平均。
frame_indices = np.arange(frames, dtype=np.float64)
for _ in range(2):
    frame_indices = np.concatenate([
        frame_indices[:1], frame_indices[1:].reshape(-1, 2).mean(axis=1)
    ])
assert len(frame_indices) == latent_frames and frame_indices[0] == 0
visual_tokens = latent_frames * (height // 8 // 2) * (width // 8 // 2)
total_tokens = visual_tokens + 226
assert (latent_frames, visual_tokens, total_tokens) == (13, 17550, 17776)
assert total_tokens**2 == 315986176

# 视频帧轴上的引导；同一帧系数在两个去噪步骤中保持相同。
scales = np.linspace(1.0, 2.5, 14)
guided = np.zeros((2, 14)) + scales[None, :] * np.ones((2, 14))
np.testing.assert_array_equal(guided[0], guided[1])
assert guided[0, 0] == 1.0 and guided[0, -1] == 2.5
print("v reconstruction passed; latent frames/tokens:",
      latent_frames, visual_tokens, total_tokens)
```

## 5. 从论文评价回到自己的研究问题

**先固定任务。** SVD 第 24 页附录 E.4 的 FVD=242.02 属于基础 T2V 模型在 UCF101 上的零样本协议：13,320 个生成视频、16 帧、25 fps、引导系数 7，按类别频率采样，并进行指定缩放/裁剪。公开 I2V 模型需要参考图像协议。SVD 的 I2V 人工比较还使用了插帧、时间裁剪和水印区域处理（附录 E.3），复核结果时要一起记录。

**保留真实运动。** CogVideoX 第 4 页 Table 1 把相邻帧 L1 差称为 flickering。沿用 [06 章](../chapters/06-video-temporal-lab.md)的静止反例，这个量会同时响应真实运动与闪烁。研究 VAE 时，应并列检查重建误差、原片运动与重建运动；直接最小化帧差可能削弱真实变化。

**保留评价的覆盖范围。** CogVideoX 第 9–10 页选择部分 VBench 维度，并加入 Dynamic Quality 与 GPT4o-MTScore。后者由时间推移视频的变化幅度评价发展而来，使用到其它视频类别时应核对适用性。其人工评分分别考察感官质量、指令遵循、物理模拟与封面质量；特定提示集合上的优势仍需要由目标任务的失败类型复核。

可起步的问题：固定参考图/提示、种子、帧率和时长，测量“身份保持、符合条件的运动、帧间伪影”三个维度，加入静止输出与乱序帧对照。先写判定规则，再看样例，保存完整输出集合。

## 6. 版本差异、许可与运行预算

| 核对项 | 实际观察与记录方式 |
| --- | --- |
| SVD 训练研究与公开条件 | 论文训练 T2V 基础模型并微调多种任务；这里固定的 SVD/SVD-XT 模型卡声明 I2V，图像条件入口不接收文本 prompt。 |
| CogVideoX v3 与固定 2B | v3 覆盖更高分辨率/更长视频；所选 2B 模型卡与配置为 49 帧、480×720、8 fps。 |
| CogVideoX 官方 CLI 的默认值 | 所读代码已含后续系列，`inference/cli_demo.py` 默认 81 帧、16 fps；其中注释区分 1.0 与 1.5。使用 2B 时显式填写对应帧数、帧率与模型路径。 |
| CogVideoX 结构表与代码 | 第 16 页 Table 6 写 2B 有 32 个 attention heads、time embedding 256；固定官方 SAT YAML 与 HF config 一致写 30 heads、hidden size 1920、time embedding 512。这里的尺寸推导使用发布配置，保留差异供追踪。 |
| CogVideoX 训练精度 | 同页 Table 6 概括为 BF16；所选 2B 模型卡写 FP16 训练。加载/运行精度需连同具体 revision 记录。 |
| SVD 代码与权重许可 | 固定官方代码的 `LICENSE-CODE` 为 MIT。所选权重的 `LICENSE.md` 为 2024-07-05 的 Stability AI Community License，规定研究/非商业用途及附条件的商业使用；商业使用条款包含注册及 100 万美元年度收入门槛，另有归属和用途条款。模型卡保留“intended for research purposes only”的使用定位，二者分别记录。 |
| CogVideoX-2B 许可 | 下文固定模型仓库的 `LICENSE` 为 Apache-2.0。模型家族中的其它规模与变体需逐项核对许可。 |

**作者报告的成本：** SVD 模型卡附录写 A100 80GB 上发布默认配置约 100 秒，SVD-XT 约 180 秒；CogVideoX v3 第 16 页 Table 7 写 H800、50 步、BF16 下，2B 的 480×720/6 秒配置约 49 秒、18 GB。它们来自不同硬件、软件和任务，不能直接比较速度。本文未测量视频模型的 CPU/GPU 时间或峰值内存。

开始权重实验时，先固定模型、代码、依赖和许可，清点全部组件文件与下载大小；再运行一个样本，记录加载与采样时间、进程/设备峰值内存、实际帧数与文件 fps。根据这次测量决定种子和条件扫描预算。

## 7. 练习与参考答案

**A · 两个时间轴。** 将 SVD 的 25 帧输出以 5 fps 和 10 fps 播放，潜变量与引导计算完全相同，哪些量改变？

<details><summary>参考答案</summary>

图像序列及帧索引上的引导系数保持；相邻帧时间间隔从 0.2 秒变为 0.1 秒。首尾采样时刻跨度从 4.8 秒变为 2.4 秒，按每帧占一个间隔的播放时长从 5 秒变为 2.5 秒。每秒运动量随时间尺度改变。模型帧率条件应作为另一个独立字段保留。

</details>

**B · 时间压缩边界。** 按本卡的 2B VAE 约定，将 49 帧改成 81 帧且保持 480×720，求潜帧数、视觉 token 数和含 226 文本 token 的注意力矩阵元素数比例。

<details><summary>参考答案</summary>

潜帧为 $1+(81-1)/4=21$；视觉 token 为 $21\cdot30\cdot45=28\,350$，总 token 为 28,576。相对原来的 17,776，每头密集注意力元素数比例为 $(28\,576/17\,776)^2\approx2.584$。这是尺寸外推；所选权重对新增时长的质量与推理支持仍需独立验证。

</details>

**C · 帧差反例。** 模型 A 的相邻帧 L1 比模型 B 小，能从这个量判断哪个模型更好地重建运动吗？

<details><summary>参考答案</summary>

需要查看目标视频及其它诊断。静止重复帧可以把该量降到零，同时丢失目标运动。先比较目标与重建的轨迹/速度，再检查纹理闪烁、身份和遮挡；使用同一帧率和像素范围。

</details>

**D · 看见 `v_prediction`。** 对 $a=b=1/\sqrt2$ 写出 VP 目标，并与本仓库直线路径的目标比较。

<details><summary>参考答案</summary>

$v_{\mathrm{VP}}=(\epsilon-z_0)/\sqrt2$；本仓库 flow 目标为 $z_0-\epsilon$。此噪声级别下它们差一个负的缩放系数，其他级别的系数继续变化。要转换网络输出，应同时转换输入参数化、噪声级别与采样更新。

</details>

## 8. 固定来源与验证边界

论文页码均按 PDF 页序。原始 PDF、固定配置和许可在核验时保存于 `/tmp/genai-video-paper-audit/` 与 `/tmp/genai-cog-paper-review/`；仓库保留原创说明和文件身份。

| 文件 | 固定入口与身份 |
| --- | --- |
| SVD 论文 | [2311.15127v1 PDF](https://arxiv.org/pdf/2311.15127v1)，30 页，15,373,867 字节；SHA-256 `654ef597e183c0544cd753494cb442125bc42b17dd2477481b6799114482a92e` |
| SVD 官方代码 | commit `e8cd657656fa5d61688191730d0e03242bf4ed44`：[采样脚本](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/scripts/sampling/simple_video_sample.py)、[配置](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/scripts/sampling/configs/svd.yaml)、[输出预条件](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/sgm/modules/diffusionmodules/denoiser_scaling.py)、[引导](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/sgm/modules/diffusionmodules/guiders.py)、[代码许可](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/LICENSE-CODE) |
| SVD 编码路径 | 同一 commit：[参考图编码器](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/sgm/modules/encoders/modules.py)、[生成 latent 缩放与解码](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/sgm/models/diffusion.py)、[编码均值](https://github.com/Stability-AI/generative-models/blob/e8cd657656fa5d61688191730d0e03242bf4ed44/sgm/models/autoencoder.py) |
| SVD 模型 | revision `9cf024d5bfa8f56622af86c884f26a52f6676f2e`：[模型卡](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/9cf024d5bfa8f56622af86c884f26a52f6676f2e/README.md)、[权重许可](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/9cf024d5bfa8f56622af86c884f26a52f6676f2e/LICENSE.md) |
| SVD-XT 模型 | revision `9e43909513c6714f1bc78bcb44d96e733cd242aa`：[模型卡](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/9e43909513c6714f1bc78bcb44d96e733cd242aa/README.md)、[权重许可](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/9e43909513c6714f1bc78bcb44d96e733cd242aa/LICENSE.md) |
| CogVideoX 论文 | [2408.06072v3 PDF](https://arxiv.org/pdf/2408.06072v3)，30 页，14,790,587 字节；SHA-256 `699795d3cfdf7acef250e0e28043b001e31ae7ab101f35fd079370c43f6473f3` |
| CogVideoX 官方代码 | commit `7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2`：[2B 配置](https://github.com/THUDM/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/sat/configs/cogvideox_2b.yaml)、[时间压缩实现](https://github.com/THUDM/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/sat/vae_modules/cp_enc_dec.py)、[CLI](https://github.com/THUDM/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/inference/cli_demo.py)、[README](https://github.com/THUDM/CogVideo/blob/7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2/README.md) |
| CogVideoX-2B 模型 | revision `1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01`：[模型卡](https://huggingface.co/THUDM/CogVideoX-2b/blob/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01/README.md)、[Transformer](https://huggingface.co/THUDM/CogVideoX-2b/blob/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01/transformer/config.json)、[VAE](https://huggingface.co/THUDM/CogVideoX-2b/blob/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01/vae/config.json)、[scheduler](https://huggingface.co/THUDM/CogVideoX-2b/blob/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01/scheduler/scheduler_config.json)、[许可](https://huggingface.co/THUDM/CogVideoX-2b/blob/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01/LICENSE) |

模型卡与权重许可 SHA-256：

```text
SVD README       82c5dbd2907c56e2991dde78bfb657c7e8414e47e611f14e516caf9c485081de
SVD-XT README    0513cc8b8da0aff30c91dbc899babe4d411e2d0ac68b8a0aff40103e1489c6b1
SVD LICENSE      d6f6b1a4dce5c852bd6d7d9482d002baf0ccdb71e662250b73be9eec8764ee8d
SVD-XT LICENSE   d6f6b1a4dce5c852bd6d7d9482d002baf0ccdb71e662250b73be9eec8764ee8d
CogVideoX-2B README  63167d42d73fd434096761b693142a69a96275b367c6fda32347d4b9dbce262d
CogVideoX-2B LICENSE 4c3f6aeb71e12415a59117dfd35cfc177514424f0356c81798fde1a239d0b6a8
```

本卡 CPU 检查验证预条件恒等式、帧轴引导与尺寸计算。来源中的生成效果、GPU 用量和训练成本均保留为作者报告；视频权重运行另建实验记录。
