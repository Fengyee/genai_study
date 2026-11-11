# GenAI Study · 从概率建模到图像与视频生成

[第一次运行](#第一次成功运行) · [完整路线](docs/learning-path.md) · [课程自测](docs/self-assessment.md) · [实测证据](records/measurements/README.md)

给熟练使用 PyTorch、有研究经验但尚未系统学习生成式 AI 的 CS master / PhD：用清楚的假设、可核对的推导和能在 CPU 上跑通的小实验建立基础，再进入图像与视频生成研究。

**当前版本**：11个学习单元，涵盖二维DDPM / Flow Matching、小图像生成与引导、视频时序诊断，以及可选的SD1.4图像推理和SVD视频运行/失败分析。基础实验在CPU上使用程序构造数据；可选实践共用独立依赖环境，分别下载约5.11GiB和8.40GiB模型文件。各路径均无需付费API。自然图像/视频训练、发布尺寸的视频质量基线和前沿综述继续按[backlog](records/BACKLOG.md)推进。

## 我适合从哪里开始？

| 你的情况 | 路线 | 完成标志 |
| --- | --- | --- |
| 第一次接触生成建模 | [00 基础](docs/chapters/00-foundations.md) → [01 生成模型](docs/chapters/01-generative-models.md) | 能区分密度、采样、条件建模与评价 |
| 想理解扩散和流模型 | [02 扩散](docs/chapters/02-diffusion.md) → [03 Flow Matching](docs/chapters/03-flow-matching.md) | 推导核心训练目标，跑通并解释二维实验 |
| 准备研究图像/视频 | 按[完整路线](docs/learning-path.md#主线与依赖)完成基础、评价与实践 → [04 研究路线](docs/chapters/04-image-video-roadmap.md) | 能选定一个有假设、基线和评价协议的问题 |

图像方向可从 [07 CPU 图像生成实验](docs/chapters/07-image-flow-lab.md) 学习 BCHW、卷积与类别条件，再用保存的 checkpoint 比较 Euler 采样步数。默认训练 1500 步，运行成本与结果分析见 [图像实测](docs/image-experiment-report.md)。

接着通过 [08 条件丢弃与引导](docs/chapters/08-image-guidance-lab.md) 训练空条件分支，在相同噪声上比较引导强度、条件响应、数值范围与计算成本。

希望亲手生成自然图像时，完成潜空间与条件基础后进入 [09 预训练图像实践](docs/chapters/09-pretrained-image-lab.md)。固定权重、文本、初始噪声和 DDIM 配置，运行 CPU 推理并比较引导强度；先对照[实测报告](docs/pretrained-image-report.md)安排下载、内存和时间。

图像/视频读者还可直接运行 [06 视频时序实验](docs/chapters/06-video-temporal-lab.md)，观察帧集合相同而时间顺序错误的反例。

想把预训练接口延伸到视频，可读[10 SVD运行与失败分析](docs/chapters/10-pretrained-video-lab.md)。低分辨率14帧/25步在本机完整运行约16.8分钟、峰值11.21GiB，输出存在严重形变；[报告](docs/pretrained-video-report.md)保留全部帧、官方API对照和后续诊断问题。

完整的先修图、实践任务与时间预算见 [学习路线](docs/learning-path.md)。遇到新词可查 [39组中英术语索引](docs/glossary.md)，再沿具体章节链接阅读推导；全局 [符号约定](docs/notation.md) 与 [来源核验](SOURCES.md) 帮助跨章核对。

读完一段路线后，用[13题课程自测](docs/self-assessment.md)定位下一步：先独立作答，再展开提示和参考答案，按具体通过判据回到对应章节。

## 第一次成功运行

已验证目标是 Linux x86_64、Python 3.12、CPU PyTorch 2.6.0。锁文件对应这一平台；macOS/Windows 需要各自的依赖锁定与验证。云环境首次安装需要 PyPI 与 PyTorch 官方软件源可达。下面命令从仓库根目录运行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m genai_study.experiment --method flow --steps 5000 --output artifacts/first-flow
```

如果使用本任务准备好的云环境，直接 `source .venv/bin/activate` 后运行实验。没有系统 Python 3.12 时可用预装的 `uv venv --python 3.12 --seed .venv` 创建带 pip 的环境；云环境写缓存遇权限问题时给 uv 设置 `UV_CACHE_DIR=/workspace/.cache/uv`。

成功标志是进程退出码为 0，并生成 `metrics.json`、`samples.npz`、`samples.png`、`loss.png` 和 `checkpoint.pt`。推荐5000次更新：已有三个seed实测中，约92%–93%的生成点进入目标模式邻域，八个簇更清楚；继续对照散点图、留出损失和诊断指标检查剩余误差。记录的训练/评估/采样/指标耗时约8–9秒，完整命令还包括启动、导入、绘图与保存。数值和计时范围见 [实测报告](docs/experiment-report.md)。省略`--steps`时CLI默认800次更新，可用作短训练的欠拟合对照。

安装会下载百 MiB 量级的 CPU torch wheel，具体大小以固定版本安装输出为准；连同其他依赖和缓存，需预留数 GB 空间。

输出目录默认保护已有结果。跨章节复用时，先查看`metrics.json`中的`config.steps`确认训练预算。已有800步结果可保留作对照；需要5000步结果时使用新目录，例如`--output artifacts/first-flow-5000`，后续求解器使用`--checkpoint artifacts/first-flow-5000/checkpoint.pt`。已有5000步结果时可直接重用。

接着运行扩散并检查实现：

```bash
python -m genai_study.experiment --method ddpm --steps 5000 --output artifacts/first-ddpm
bash scripts/quality.sh
```

脚本检查公式性质、文档链接，直接执行16个已审查的CPU教学代码块，并运行二维生成、图像生成和视频时序反例。短 smoke 用于检查程序路径；训练质量由完整实验评估。第 03、07 章提供固定 checkpoint 和初始噪声的求解器步数实验。完整实验按 seed 固定 CPU 随机性；跨 PyTorch/平台的位级一致性仍需另行验证。随机种子、多次评估和教学指标解释见 [实验说明](examples/README.md)。

## 专题阅读

完成相关基础与实践后，按研究问题选择阅读卡。各卡固定原始论文、官方实现或模型版本，并列出可以独立核对的公式与接口。

| 方向 | 入口与重点 |
| --- | --- |
| 扩散与流的数学连接 | [噪声、score与概率流ODE](docs/readings/diffusion-flow-bridge.md)：条件期望、时间方向、DDIM与SVD坐标 |
| 扩散训练目标 | [预测量与损失权重](docs/readings/prediction-and-loss-weighting.md)：clean/noise/VP误差、有限容量与时间采样校正 |
| 少步生成 | [从积分到轨迹端点](docs/readings/consistency-endpoint-map.md)：一致性模型、CD/CT配对与解析Euler对照 |
| 条件表示 | [文本与图像怎样进入生成网络](docs/readings/conditioning-attention.md)：Q/K/V、CLIP接口、单token与联合置换反例 |
| 潜空间表示 | [固定VAE编码与解码](docs/vae-reconstruction-report.md)：后验采样、缩放往返、六案例重建及完整重复 |
| 空间网络接口 | [patch、位置与预测输出](docs/readings/spatial-tokens-and-position.md)：重排索引、卷积等价、自注意力与空间还原 |
| 图像编辑 | [FLUX.1 Kontext 2025](docs/readings/flux-kontext-2025.md)：参考图token、时间方向与guidance蒸馏 |
| 视频生成 | [Wan 2025](docs/readings/wan-video-2025.md)：时空表示、条件与公开模型接口 |
| 视频训练基础 | [SVD与CogVideoX](docs/readings/video-training-svd-cogvideox.md)：训练阶段、视频压缩、VP预测量与版本差异 |
| 离散图像生成 | [ResTok 2026](docs/readings/restok-image-ar-2026.md)：128个token、9轮采样与分组条件独立 |
| 音画联合生成 | [LTX-2 2026](docs/readings/ltx2-video-audio-2026.md)：联合分布、双流条件与两阶段采样成本 |

## 实验覆盖范围与运行问题

- 二维八高斯用于检查训练目标、采样算法与分布诊断；图像与视频质量需要对应的数据和评价协议。
- 07/08学习受控圆盘/方形分布，09使用自然图像模型，10实测预训练图生视频的低分辨率运行与失败案例；06以合成序列检查时序指标。各实践明确自己的数据与适用范围。
- 首次安装失败：检查 Python 版本与网络允许的 `pypi.org`、`files.pythonhosted.org`、`download.pytorch.org` 和 `download-r2.pytorch.org`；修复网络后保留 TLS 与 hash 校验重试。
- CPU 很慢：二维/图形训练使用`--threads 1`，先降低`--steps`与`--samples`确认路径，再恢复完整训练预算。09的预训练入口使用2步、256×256的smoke配置，具体命令与成本见该章。

## 参与和维护

优先解决读者真正遇到的问题。贡献方法见 [CONTRIBUTING.md](CONTRIBUTING.md)，当前检查点见 [执行记录](records/PROGRESS.md)，待办见 [质量 backlog](records/BACKLOG.md)。每小时检查的定义和实际部署边界见 [自动化说明](docs/automation.md)。

原创代码和文档使用 [Apache-2.0](LICENSE)，外部模型、许可副本及生成示例的来源见 [SOURCES.md](SOURCES.md)。欢迎通过 issue 提供可复现问题、通过 PR 补充推导和实验。如果这些材料帮助你完成学习或实验，欢迎推荐给同学和研究伙伴。
