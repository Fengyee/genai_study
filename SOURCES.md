# 来源与核验边界

本项目引用原始研究支撑概念，教学代码独立实现。原创代码和文档采用Apache-2.0；外部论文、模型与数据分别遵循原作者许可。基础实验的数据由程序构造，09/10章的预训练权重由读者按固定清单下载到本地缓存，仓库保存其身份、配置和运行证据。

章节末尾记录原始论文和作者实现链接。研究路线以 2013–2024 年代表工作建立基础，再通过固定版本阅读卡补充近年图像/视频进展；新增或变化快的结论需重新核验。各份审计明确记录论文版本、具体主张和对应证据。

## 初始实例核验（2026-11-10，历史）

- GitHub HTTPS 读取与推送成功。
- arxiv.org 的直接访问被云代理 403 拒绝，因此不能声称本轮逐篇重新阅读全文。章节依据经典公式、可访问的作者代码及独立解析检查编写；新颖性/最新排名不在本轮结论范围。
- DDPM 作者实现：[hojonathanho/diffusion 固定版本](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils.py)，核对 forward marginal、posterior variance 和 epsilon parameterization。
- Meta Flow Matching：[固定版本 affine path](https://github.com/facebookresearch/flow_matching/blob/11568d37f8d5a080e12aa7b5305d9c35ae07d136/flow_matching/path/affine.py)，核对仿射路径与路径导数。它的 scheduler 命名和本文符号映射需以各章说明为准。
- 图像/视频的 LDM、DiT、SVD、CogVideoX 官方 README 已读取并固定 commit，详见 [04章来源核验](docs/chapters/04-image-video-roadmap.md)。核验范围是接口与定位，不能替代全文效果复现。
- Apache-2.0 许可证正文取自 Apache 官方 GitHub 组织的 `.github/LICENSE`，版本 2.0（2004-01）。

## 后续核验规则

优先 DOI、arXiv 版本号、作者/机构固定 commit；记录核对了什么，避免把“链接存在”当成“结论正确”。方法效果比较须固定数据、计算预算、评价协议和统计不确定性。链接可读性与内容学术有效性分开检查。

## 依赖安全修订

新增checkpoint载入时核对 [GitHub Reviewed Advisory GHSA-53q9-r3pm-6pq6](https://github.com/advisories/GHSA-53q9-r3pm-6pq6)：CVE-2025-32434影响torch<2.6.0，即使weights_only=True也存在风险，公告修复版2.6.0。已固定2.6.0+cpu并重跑验证；不宣称可安全加载任意第三方文件。

## 发布环境后补充的原始来源核验

当前实例已实际恢复arXiv访问；上面的403记录仅是历史。两篇核心论文的固定v2 PDF已下载、核对并由新实例重验哈希：[DDPM与Flow Matching公式审计](docs/source-audit.md)。审计范围为指定公式/条件，不代表整篇论文实验或全部课程都已核验。

新增 [Wan 2025阅读卡](docs/readings/wan-video-2025.md) 固定技术报告、官方代码commit及单一权重模型卡，涵盖视频表示、条件接口、时间采样、许可和评价边界；没有下载权重或实测GPU性能。课程仍非2025–2026全面综述。

[评价来源核验](docs/evaluation-source-audit.md) 对照 Heusel v6、Chong/Forsyth v3、Parmar v3，核验 FID 定义、模型依赖的有限样本偏差、resize与压缩协议，并用矩阵恒等式和均值偏差解析例交叉检查。固定 PDF 的 SHA、页码与适用条件均在索引中记录。

[生成模型来源核验](docs/generative-source-audit.md) 对照 PixelRNN、VAE、GAN、Real NVP 与 Neural ODE 固定版本，核验自回归因果掩码、ELBO与支持集、对抗目标、变量替换及连续流负散度公式；提供八结果联合归一化/标签泄漏和高斯/离散分布解析检查，并记录Neural ODE v6对历史NFE实验报告的修订。

[图像/视频路线来源核验](docs/image-video-source-audit.md) 对照LDM、CFG、DiT固定版本，补齐CFG空条件分支的训练前提、latent patch尺寸推导与成本口径；记录CFG趋势正文与表格的差异，并重新核对SVD/CogVideoX固定官方README的公开接口。

[其余评价指标核验](docs/evaluation-metrics-audit.md) 固定IS、improved precision/recall、KID、FVD原文，核对熵与互信息、近邻球、无偏U统计量和视频特征协议；直接执行原型高IS、负KID估计及裁剪偏差反例。FVD输入布局、缩放、数值范围、批大小和logits身份另与固定作者源码交叉核对。

[08条件丢弃与引导](docs/chapters/08-image-guidance-lab.md) 沿用已核验CFG的双分支训练结构和FM的速度回归目标，明确采用教学速度组合。空条件的总体最优回归、类别相关丢弃的先验变化及高斯终点密度反例为独立推导；[三seed实测](docs/image-guidance-report.md)保存本仓库原始图形模型的条件响应、几何、裁剪和成本。

[FLUX.1 Kontext 2025阅读卡](docs/readings/flux-kontext-2025.md) 固定v2报告、官方代码commit和单一dev模型卡/许可证revision，核对参考图token、时间/速度变换、guidance蒸馏、NFE以及编辑评价。CPU实际验证latent打包/还原与解析积分；24GB权重存储为按作者12B参数估算的理论数量，权重推理成本待测。

[SVD与CogVideoX训练阅读卡](docs/readings/video-training-svd-cogvideox.md) 固定SVD v1、CogVideoX v3、官方源码及三个具体模型revision，核对训练阶段、数据筛选、VP预条件、时间压缩与帧率；保留论文和发布配置的结构/精度/能力差异。CPU检查验证VP还原、帧轴引导与token尺寸；视频性能数字归属作者及其协议。

[09预训练图像实践](docs/chapters/09-pretrained-image-lab.md)使用原作者CompVis的SD1.4 revision `133a221b8aa7292a167afc5127cb63fb5005638b`，逐文件大小/SHA见[manifest](examples/pretrained-sd14/manifest.json)。[LICENSE-model.txt](examples/pretrained-sd14/LICENSE-model.txt)完整保留CreativeML Open RAIL-M原文及原作者通知，适用其自身条款。SD生成样例由[实测报告](docs/pretrained-image-report.md)记录的固定提示、随机种子和配置产生，输出使用同时遵守模型许可第6条。模型权重保存在忽略的本地artifacts目录。

[ResTok 2026阅读卡](docs/readings/restok-image-ar-2026.md)固定2026-01-07首发论文v1、官方代码和两个模型revision，核对128-token层次、训练因果mask、9轮采样、分类CFG及评价协议；保留池化、残差方向、对齐margin和安装版本差异。CPU反例验证分组条件独立的表达约束与索引预算，权重字节数来自Hub元数据。

[LTX-2 2026阅读卡](docs/readings/ltx2-video-audio-2026.md)固定2026-01-06报告v1、同期代码及模型/许可revision，核对音画双流、条件分解、速度参数化、论文跨模态CFG与实际入口的差别。直接执行小张量与原函数控制流，确认蒸馏8+3区间/11次联合调用；保留论文参数量和README步数差异，GPU性能按作者协议归属。

[SD1.4四请求提示探索](docs/sd14-prompt-pilot.md)继续使用上面固定模型和许可，生成前保存完整计划，保留两提示×两种子的全部原图、metrics和逐项观察。图片与推理输出逐字节相同，两个同seed噪声对完全一致。视觉观察者均为AI代码代理，真实读者评阅尚待开展。

[恢复后的线程诊断](records/measurements/sd14-thread-diagnostic/README.md)及[两线程属性实验](records/measurements/sd14-attributes-followup/README.md)沿用上述SD1.4 revision与许可。诊断原图由固定提示/seed101、4步及1/2线程产生，保持完整字节；后续八请求有独立运行前计划。原单线程超时与未执行状态保存在相邻记录，生成图像的使用同样遵守模型许可第6条。

[VAE编码—解码往返](docs/vae-reconstruction-report.md)继续使用上述SD1.4模型及许可，输入固定为scene/seed43生成图。预先声明六案例，实际完成编码、后验采样、缩放对照和完整重复；全部PNG、数组身份与原始/裁剪指标保留。LDM v2的§3.1/D.1/G与固定Diffusers的后验/编解码接口分别核验，来源和实测记录见[VAE证据索引](records/measurements/vae-reconstruction/README.md)。单图重建误差的解释范围在报告中明确。

[10 SVD视频运行与失败分析](docs/chapters/10-pretrained-video-lab.md)固定原作者SVD revision `9cf024d5bfa8f56622af86c884f26a52f6676f2e`，完整下载并核验[11文件清单](examples/pretrained-svd/manifest.json)。保留[Community2024-07-05许可原文](examples/pretrained-svd/LICENSE-model.txt)、[NOTICE](examples/pretrained-svd/NOTICE)及指定归属 **Powered by Stability AI**。参考图来自上述SD1.4 scene/seed43输出；SVD生成结果同时受其模型许可约束。核对Diffusers0.35.2固定实现中的参考latent、双随机流、VP预条件、Euler更新和联合帧调用，独立复算首末Euler步并实际完成smoke/校准/25步/重复。完整视频输出严重失真，全部帧、运行成本和追加的直接官方API短对照见[报告](docs/pretrained-video-report.md)及[原始证据](records/measurements/pretrained-svd/README.md)；具体失败成因待控制实验。

[扩散与Flow Matching理论衔接](docs/readings/diffusion-flow-bridge.md)固定Score-SDE `2011.13456v2`（PDF SHA `b244b740e60628d01bc3d4b98d721c2343be1fb9b67ad22593c8555e66f4d4a6`）与原作者JAX commit `0acb9e0ea3b8cccd935068cd9c657318fbc6ce4c`，核对式(6)/(7)/(13)/(35)–(37)、概率流0.5系数和T→eps积分。重新核对DDIM v4式(13)–(15)；独立推导后验均值、路径速度、VP参数化和坐标变换，并以解析Gaussian边缘/轨迹、符号错误反例与有限Euler误差验证。JAX项目仅做源码核对；实际CPU代码块使用基础NumPy并纳入质量检查。

[DDPM解析oracle对照](docs/ddpm-oracle-report.md)沿用已固定DDPM v2的正向边缘、条件后验及噪声均值公式，独立推导全协方差遗漏项、平稳方差递推与八高斯score。正式三seed运行复用现有采样器，记录21份数组身份与独立复算；无需训练或新增外部权重。原始运行与seed参数边界修复后的来源分别归档于[测量索引](records/measurements/oracle-study/README.md)。

[条件注意力专题](docs/readings/conditioning-attention.md)重新核对LDM v2 §3.3、固定CompVis attention.py与OpenAI CLIP源码，连接SD1.4的last_hidden_state和SVD的全局image_embeds/参考latent。Diffusers0.35.2 commit `b71269675ec1b85193107a691dd35c308e46f0a5`与Transformers4.57.6 commit `753d61104116eefc8ffc977327b441ee0c8d599f`的五份相关源码和本地安装逐字节一致。CPU检查包含SDPA对照、单token零梯度、mask联合置换和两键sigmoid导数；来源身份及独立检查见[记录](records/measurements/conditioning-attention/README.md)。

[少步生成与一致性端点映射](docs/readings/consistency-endpoint-map.md)固定Consistency Models `2303.01469v1`（PDF SHA `f2f4dbed7f73d182c0b2dad0747d3df8e2c27a7df05d4b2aa5dee00b6ca94d64`）和作者commit `e32b69ee436d518377db86fb2127a3972d0d8716`。核对VE概率流、正端点、边界、CD/CT与EMA/stopgrad；独立Gaussian例验证端点恒定性、Euler误差、MMSE与共噪声配对反例。原作者采样函数用mock实际计数，保留初始噪声由外层提供的范围；[核验记录](records/measurements/consistency-endpoint/README.md)包含全部来源身份与两组参数独立检查。

[预测目标与损失权重](docs/readings/prediction-and-loss-weighting.md)沿固定DDPM v2式12/14解释加权与简化目标，核对Diffusers0.35.2固定DDIM源码的VP标签、三种预测转换及裁剪顺序。误差比例、同参数梯度、有限容量与重要性权重按明确假设独立推导；正文CPU块与两组独立参数检查归档于[核验记录](records/measurements/prediction-loss/README.md)。

[空间token与位置专题](docs/readings/spatial-tokens-and-position.md)重新核对DiT `2212.09748v2`第4–5页和原作者commit `ed81ce2229091fd4ecc9a223645f95cf379d582b`的models.py，定位patch入口、冻结二维正弦位置、输出通道及实际NCHW还原。独立推导patch索引、卷积等价和置换关系，用矩形小数组及作者抽取函数核对，保留源码docstring与实际维序差异；[记录](records/measurements/spatial-tokens/README.md)包含固定身份、原样CPU块和独立检查。

2026-11-11对固定提交9f5b3b6中的23份来源入口文档执行一次[外部可达性检查](records/reader-checks/source-links-20261111.json)：98个去重来源URL的最终HTTP状态均为200，记录4个正常重定向；全部使用HEAD，正文下载0字节。PDF/模型资产与本仓PR/Actions链接按计划排除。此记录核验HTTP入口，论文内容、锚点及科学主张沿用各专题的固定来源与独立审查。

[05提示簇重采样反例](docs/chapters/05-evaluation.md#可运行反例复制结果不会增加独立提示)读取Efron/Narasimhan的[作者稿](https://efron.ckirby.su.domains/papers/2018Automatic-Construction-BCIs.pdf)，PDF SHA `117c03df5fecf8954c21a1faeef2eb249f33fa24bd9969c5de91d1b5d5a19184`，第6–7页§3式(3.1)–(3.7)给出IID观测和有放回抽样定义。提示差复制与方差变化为本仓库独立构造；完整27种分布与错误按行抽样经Fraction枚举/卷积核验，[证据](records/reader-checks/prompt-bootstrap/README.md)明确条件分布、权重与置信区间的验证范围。

[MeanFlow两时间平均速度](docs/readings/meanflow-average-velocity.md)固定2025-05-19论文`2505.13447v1`及作者2025-07-08 commit `58d099c0fa197bd31b9a7b51727a76019ae5fdbe`。核对式3–12、两时间JVP换元、停止目标梯度、对角边界与一次采样；Gaussian及独立二维线性场检验身份和端点，作者三个采样函数以解析替身核对调用。ImageNet FID、图像训练与速度数字按作者协议归属；[证据索引](records/measurements/meanflow-average-velocity/README.md)保留SHA、执行范围与MIT许可。
