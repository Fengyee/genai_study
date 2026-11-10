# 来源与核验边界

本项目引用原始研究来支撑概念，所有 toy 代码独立实现。许可证不延伸到外部论文、仓库或数据。本次没有复制论文图片、模型权重或外部训练数据。

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

[生成模型来源核验](docs/generative-source-audit.md) 对照 VAE、GAN、Real NVP 与 Neural ODE 固定版本，核验ELBO与支持集、对抗目标、变量替换及连续流负散度公式；提供可运行的高斯/离散分布解析检查，并记录Neural ODE v6对历史NFE实验报告的修订。

[图像/视频路线来源核验](docs/image-video-source-audit.md) 对照LDM、CFG、DiT固定版本，补齐CFG空条件分支的训练前提、latent patch尺寸推导与成本口径；记录CFG趋势正文与表格的差异，并重新核对SVD/CogVideoX固定官方README的公开接口。

[其余评价指标核验](docs/evaluation-metrics-audit.md) 固定IS、improved precision/recall、KID、FVD原文，核对熵与互信息、近邻球、无偏U统计量和视频特征协议；直接执行原型高IS、负KID估计及裁剪偏差反例。FVD输入布局、缩放、数值范围、批大小和logits身份另与固定作者源码交叉核对。

[08条件丢弃与引导](docs/chapters/08-image-guidance-lab.md) 沿用已核验CFG的双分支训练结构和FM的速度回归目标，明确采用教学速度组合。空条件的总体最优回归、类别相关丢弃的先验变化及高斯终点密度反例为独立推导；[三seed实测](docs/image-guidance-report.md)保存本仓库原始图形模型的条件响应、几何、裁剪和成本。
