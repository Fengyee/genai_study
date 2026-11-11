# GenAI 中英术语索引

**使用前提：** 能读懂 PyTorch 张量与损失函数；概率基础可随时返回[00章](chapters/00-foundations.md)。**目标：** 从一个术语定位其定义、模型接口和可执行例子。**范围：** 本页索引00–10章及已有研究阅读卡中的38组高频术语，简释以链接处的假设和具体配置为准。

按主题查词，再到右栏阅读推导或运行练习。时间方向、符号和单位统一由[符号约定](notation.md)维护；系统学习顺序见[学习路线](learning-path.md)。原始论文、实现版本和验证范围保留在对应章节及[来源索引](../SOURCES.md)。

## 概率与训练目标

| 术语 | 本仓库中的含义 | 深入入口 |
| --- | --- | --- |
| 似然 / 负对数似然 · likelihood / negative log-likelihood（NLL） | 将观测固定后，模型为其赋予的概率或密度及其负对数。比较NLL须统一表示、预处理和归约；自然对数给出nat，除以$\log2$得到bit。 | [01：离散图像的似然与单位](chapters/01-generative-models.md#2-似然与自回归把联合分布拆开) |
| 自回归 · autoregressive（AR） | 按指定顺序把联合分布分解为各位置在前缀条件下的分布。逐token与分组生成采用不同的条件依赖结构。 | [01：链式分解](chapters/01-generative-models.md#2-似然与自回归把联合分布拆开)；[ResTok：分组分解](readings/restok-image-ar-2026.md#3-128个token为什么只需9轮) |
| 教师强制 · teacher forcing | 训练时用观测到的前缀计算各位置的似然项；生成时使用已经采样的前缀。训练并行度由网络架构和因果约束共同决定。 | [01：训练输入与标签](chapters/01-generative-models.md#2-似然与自回归把联合分布拆开) |
| 因果掩码 · causal mask | 限制预测位置能读取的信息；输入移位、掩码及其他跨位置运算共同保证只使用合法前缀。 | [01：因果性与标签泄漏反例](chapters/01-generative-models.md#用八个结果检查因果性与归一化) |
| 证据下界 · evidence lower bound（ELBO） | 对数边缘似然的下界。VAE中的常见形式是期望重建对数似然减去近似后验与先验的KL；与真实对数似然的差由后验KL给出。 | [01：ELBO恒等式及条件](chapters/01-generative-models.md#3-vae用变分推断处理难积分) |
| 分数函数 · score | 可微密度的对数对数据坐标求梯度，即$\nabla_x\log p(x)$；它描述对数密度的局部变化。 | [符号表：数据梯度](notation.md#术语符号与约定)；[score与噪声/去噪器](readings/diffusion-flow-bridge.md#2-噪声score与去噪器是怎样连接的) |

## 图像与视频表示

| 术语 | 本仓库中的含义 | 深入入口 |
| --- | --- | --- |
| 潜变量 / 潜空间 · latent variable / latent space | 模型内部用于表示数据或生成数据的变量及其取值空间。图像生成常先在潜空间建模，再用配套解码器恢复像素。 | [04：潜空间生成路径](chapters/04-image-video-roadmap.md#2-潜空间先压缩再生成) |
| 变分自编码器 · variational autoencoder（VAE） | 使用近似后验进行变分学习的潜变量模型。具体图像压缩器还可加入感知、对抗等目标；阅读发布模型时核对实际训练目标与编解码接口。 | [01：VAE目标](chapters/01-generative-models.md#3-vae用变分推断处理难积分)；[04：图像压缩器](chapters/04-image-video-roadmap.md#2-潜空间先压缩再生成) |
| Token / tokenizer · 序列单元 / 分词或编码器 | Token是序列中的表示单元，tokenizer规定输入到这些单元的转换。文本与ResTok可用离散索引，图像patch也可表示为连续向量。 | [09：文本token](chapters/09-pretrained-image-lab.md#2-从文本与噪声到图像)；[ResTok：两类模型与token](readings/restok-image-ar-2026.md#1-先把两个模型分开) |
| 向量量化 / 码本 · vector quantization（VQ）/ codebook | 把连续向量映射到码本条目，用索引表示。8192项码本的固定长度索引需要13bit；码本向量维数与模型存储另计。 | [ResTok：离散索引预算](readings/restok-image-ar-2026.md#1-先把两个模型分开) |
| 图块token · patch token | 将像素或潜张量按图块组织，展平并投影为网络序列。Token数量由所在空间、图块尺寸及时间轴决定。 | [04：DiT的latent patches](chapters/04-image-video-roadmap.md#4-从-u-net-到-diffusion-transformer)；[Wan：表示与压缩](readings/wan-video-2025.md#表示压缩率不是一个脱离通道数的数字) |
| 潜变量缩放 · latent scaling factor | 改变潜变量数值坐标的乘法系数。SD1.4使用$z=0.18215z_{\mathrm{VAE}}$，解码前除以该系数；各latent路径须遵循训练坐标；SVD的参考条件编码保持未缩放，生成latent在解码前除以系数。 | [09：latent形状与缩放](chapters/09-pretrained-image-lab.md#2-从文本与噪声到图像)；[10：参考与生成路径](chapters/10-pretrained-video-lab.md#2-参考图走两条条件路径) |
| 时空压缩 · spatial / temporal compression | 编码器沿空间或时间下采样，改变潜张量尺寸。首帧处理影响实际潜帧数，例如所选CogVideoX配置把49帧编码为13个潜帧。 | [SVD/CogVideoX：时间压缩与token](readings/video-training-svd-cogvideox.md#3d-causal-vae-与-token) |
| 帧率 / 时长 · frames per second（fps）/ duration | fps给出每秒显示帧数。等间隔的$F$帧以$r$ fps播放，首尾采样时刻跨度为$(F-1)/r$；每帧显示一个间隔时，播放时长为$F/r$。 | [06：帧率与时间单位](chapters/06-video-temporal-lab.md#帧数帧率和时长不等价)；[10：条件与播放fps](chapters/10-pretrained-video-lab.md#4-扩散时间视频帧索引和播放帧率) |

## 条件与引导

| 术语 | 本仓库中的含义 | 深入入口 |
| --- | --- | --- |
| 交叉注意力 · cross-attention | 查询来自当前特征，键和值来自条件；每个查询沿条件键轴归一化，再汇聚值向量。单个有效条件token时权重恒为1，后续响应还取决于值与网络其余路径。 | [条件接口：轴与单token反例](readings/conditioning-attention.md) |
| CLIP · 图文对比预训练编码器 | 可提供位置隐藏态或池化投影向量。固定SD1.4读取77×768文本隐藏态，SVD读取1×1024全局图像向量；投影、归一化与条件轴按实际接口核对。 | [CLIP的不同输出接口](readings/conditioning-attention.md#2-同样使用clip实际取出的表示不同) |
| 条件丢弃 · condition dropout | 训练中以指定概率隐藏条件，使共享网络接触有条件和空条件样本。08章用独立于数据、噪声和时间的丢弃事件训练边缘速度分支。 | [08：条件丢弃的概率前提](chapters/08-image-guidance-lab.md#2-为什么随机丢弃条件能训练边缘速度) |
| 空条件 · null / unconditional condition | 用模型训练时约定的输入表示条件缺省：08为全零类别向量，09为空文本经过CLIP后的表示。编码方式和训练覆盖决定该输入的含义。 | [08：类别与空条件编码](chapters/08-image-guidance-lab.md#1-从类别条件到空条件)；[09：空文本分支](chapters/09-pretrained-image-lab.md#3-文本条件引导对齐两分支及调用计数) |
| 无分类器引导 · classifier-free guidance（CFG） | 在相同带噪状态、时间和输出参数化下，用$f_u+s(f_c-f_u)$组合基准与条件预测。08组合速度，09组合噪声；使用负面提示时，还要记录基准分支对应的条件。 | [04：CFG约定](chapters/04-image-video-roadmap.md#3-条件控制与引导)；[08：速度组合](chapters/08-image-guidance-lab.md#3-在同一状态上组合两次速度预测) |
| 引导蒸馏 · guidance distillation | 通过训练让学生模型承接引导效果。固定Kontext-dev接口把guidance值作为embedding输入，每个积分区间调用一次生成网络；系数含义按该训练与接口解释。 | [Kontext：蒸馏引导接口](readings/flux-kontext-2025.md#guidance-distilled-接口与显式两分支-cfg) |
| 文本到图像/视频、图像到视频 · T2I / T2V / I2V | 按条件输入和目标输出命名任务。具体模型还需声明参考图、文本、帧率等条件如何编码，哪些输入在采样中保持固定。 | [Wan：条件接口](readings/wan-video-2025.md#条件接口是模型的一部分)；[SVD：参考图像条件](readings/video-training-svd-cogvideox.md#一张参考图像怎样控制整段视频) |

## 网络究竟预测什么

| 术语 | 本仓库中的含义 | 深入入口 |
| --- | --- | --- |
| 噪声预测 · epsilon / noise prediction | 以正向过程采样的噪声$\epsilon$作为平方误差监督；总体最优预测是在给定带噪输入、时间（噪声级别）和条件后的噪声条件期望。 | [02：噪声MSE](chapters/02-diffusion.md#4-为什么是噪声均方误差) |
| 方差保持扩散的v参数化 · variance-preserving（VP）v-prediction | 对$z_t=az_0+b\epsilon$、$a^2+b^2=1$，预测目标为$v=a\epsilon-bz_0$，还原关系为$z_0=az_t-bv$。转换输出时同时核对噪声级别与输入参数化。 | [SVD/CogVideoX：VP还原检查](readings/video-training-svd-cogvideox.md#4-cpu-检查把-v_prediction帧轴与尺寸算清楚) |
| 流匹配速度 · flow-matching velocity | 驱动指定概率路径演化的数据或潜空间速度场。本仓库条件路径$x_t=(1-t)\epsilon+tx$从噪声走向数据，目标为$x-\epsilon$；理想网络在给定状态、时间及所用条件时预测此速度的条件期望。 | [03：条件速度与边缘速度](chapters/03-flow-matching.md#2-条件路径很多为什么只学一个速度场) |
| 一致性模型 / 端点映射 · consistency model（CM） | 预测当前状态沿指定概率流轨迹到固定噪声端点的结果，并要求同轨迹预测一致。原始CD利用教师构造相邻状态，CT用数据共噪声配对，其关系按网格极限定理解释。 | [端点映射、边界与解析反例](readings/consistency-endpoint-map.md) |

## 采样与可复现性

| 术语 | 本仓库中的含义 | 深入入口 |
| --- | --- | --- |
| DDPM · denoising diffusion probabilistic model | 用正向加噪过程构造学习任务，再学习反向生成链。噪声表、预测参数化和反向方差共同决定具体实现。 | [02：后验与反向条件](chapters/02-diffusion.md#3-可计算的后验与不可直接计算的反向条件) |
| DDIM · denoising diffusion implicit model | 与扩散噪声预测配套的一类采样更新。09的$\eta=0$配置在给定初始噪声后不再额外采样噪声，并固定时间网格及末步边界。 | [09：DDIM时间点与确定性](chapters/09-pretrained-image-lab.md#4-ddim的步数噪声级别与确定性) |
| ODE求解器 · ODE solver | 沿速度场数值推进状态，例如显式Euler用$x_{k+1}=x_k+h v(x_k,t_k)$。步长敏感性与已学速度场的误差分别检查。 | [03：ODE与Euler采样](chapters/03-flow-matching.md#4-采样为什么需要-ode-求解器)；[概率流的1/2系数](readings/diffusion-flow-bridge.md#4-为什么probability-flow-ode出现一个12) |
| 训练步 / 采样步 · training step / sampling step | 训练步计优化器更新，采样步计状态更新或所用时间点。参数名由CLI定义：`image_flow --steps`计训练更新，09的`--steps`计DDIM时间点。 | [08：两类步数参数](chapters/08-image-guidance-lab.md#4-训练一次再做固定模型的强度扫描)；[09：采样设置](chapters/09-pretrained-image-lab.md#4-ddim的步数噪声级别与确定性) |
| 网络求值次数 · number of function evaluations（NFE） | 需同时记录网络调用、batch大小和每样本分支预算。09生成单张图的20步为20次batch2调用，累计40次样本分支求值；08显式调用两次网络完成一步。 | [09：调用计数](chapters/09-pretrained-image-lab.md#3-文本条件引导对齐两分支及调用计数)；[08：显式两分支](chapters/08-image-guidance-lab.md#3-在同一状态上组合两次速度预测) |
| 检查点 · checkpoint | 保存模型权重及其架构、训练配置等状态。这里的推理checkpoint用于冻结模型做采样对照；完整续训还需优化器等状态。 | [07：固定模型的求解器对照](chapters/07-image-flow-lab.md#第二步冻结模型单独改变采样步数) |
| 配对噪声 / 随机种子 · paired noise / random seed | 固定模型和同一份初始噪声，只改变待研究的配置，减少比较中的无关随机变化。有参考图增强等额外随机流时也应分别固定；Seed还需结合软件、设备和线程数记录。 | [08：配对输入](chapters/08-image-guidance-lab.md#5-如何控制噪声类别和参考数据)；[00：可复现性](chapters/00-foundations.md#5-随机性与可复现性)；[10：两随机流](chapters/10-pretrained-video-lab.md#5-两个随机流与分阶段运行) |

## 评价与证据

本组条目主要比较样本或特征分布；模型似然见第一组。使用指标时先固定参考集、表示、预处理、样本量及要支持的结论。

| 术语 | 本仓库中的含义 | 深入入口 |
| --- | --- | --- |
| FID · Fréchet Inception Distance | 比较真实与生成图像的固定特征统计，把两组均值/协方差对应的高斯作平方2-Wasserstein距离计算。rFID使用重建样本，gFID使用生成样本，具体参考协议分别记录。 | [05：FID定义与偏差](chapters/05-evaluation.md#3-fid-究竟比较了什么)；[ResTok：rFID/gFID](readings/restok-image-ar-2026.md#6-评价重建生成与参数搜索各回答什么) |
| KID · Kernel Inception Distance | 在固定Inception特征与核上估计平方MMD。常用无偏U统计量在有限样本中可为负，裁剪负值会改变估计性质。 | [05：KID及负估计反例](chapters/05-evaluation.md#kid无偏估计与负数结果) |
| IS · Inception Score | 固定分类器下，生成图像与预测标签互信息的指数；偏好单图预测集中、总体类别分布多样。类内多样性要另行检查。 | [05：IS的熵形式](chapters/05-evaluation.md#is-的熵与互信息形式) |
| 生成质量精确率/召回率 · generative precision / recall | 本仓库介绍近邻球版本：precision估计生成特征落入真实特征邻域的比例，recall估计真实特征被生成特征邻域覆盖的比例。结果依赖特征与近邻数。 | [05：两组邻域的构造](chapters/05-evaluation.md#近邻球怎样估计-precisionrecall) |
| FVD · Fréchet Video Distance | 在固定视频特征空间中比较两组高斯统计。帧数、帧率、时序、预处理、特征权重与层共同组成评价协议。 | [05：FVD输入与任务协议](chapters/05-evaluation.md#5-视频fvd-的表示与任务协议) |
| 模式覆盖 / 模式丢失 · mode coverage / mode dropping | 检查目标分布的哪些模式被模型覆盖。覆盖判据需要明确质量条件；二维实验能覆盖全部8个中心区域，同时保留大量偏离目标的点。 | [05：覆盖率反例](chapters/05-evaluation.md#2-本仓库的真实反例覆盖-8-个模式仍然欠拟合) |

## 三个快速定位任务

1. 看见`v_prediction`，先查“网络究竟预测什么”，写出输入和目标，再对照VP还原与flow路径。
2. 看见“20步”，查“训练步/采样步”与“NFE”，补齐更新次数、调用次数、batch及每样本分支预算。
3. 看见“FID降低”，查“评价与证据”，确认参考数据、特征处理、样本量，再选择支持目标结论的额外观察项。
