# GenAI Study · 从概率建模到图像与视频生成

给熟练使用 PyTorch、有研究经验但尚未系统学习生成式 AI 的 CS master / PhD：用清楚的假设、可核对的推导和能在 CPU 上跑通的小实验建立基础，再进入图像与视频生成研究。

**当前版本**：10 个学习单元，涵盖二维 DDPM / Flow Matching、16×16 条件图像生成、固定模型的求解器与引导实验、视频时序反例，以及可选的 SD 1.4 预训练图像实践。基础实验在 CPU 上运行，数据由程序构造；可选实践另下载约 5.11 GiB 权重，使用独立依赖环境。各路径均无需付费 API。自然图像/视频训练、视频权重实践及完整前沿综述继续按 [backlog](records/BACKLOG.md)推进。

## 我适合从哪里开始？

| 你的情况 | 路线 | 完成标志 |
| --- | --- | --- |
| 第一次接触生成建模 | [00 基础](docs/chapters/00-foundations.md) → [01 生成模型](docs/chapters/01-generative-models.md) | 能区分密度、采样、条件建模与评价 |
| 想理解扩散和流模型 | [02 扩散](docs/chapters/02-diffusion.md) → [03 Flow Matching](docs/chapters/03-flow-matching.md) | 推导核心训练目标，跑通并解释二维实验 |
| 准备研究图像/视频 | 完成上述基础 → [04 研究路线](docs/chapters/04-image-video-roadmap.md) | 能选定一个有假设、基线和评价协议的问题 |

图像方向可从 [07 CPU 图像生成实验](docs/chapters/07-image-flow-lab.md) 学习 BCHW、卷积与类别条件，再用保存的 checkpoint 比较 Euler 采样步数。默认训练 1500 步，运行成本与结果分析见 [图像实测](docs/image-experiment-report.md)。

接着通过 [08 条件丢弃与引导](docs/chapters/08-image-guidance-lab.md) 训练空条件分支，在相同噪声上比较引导强度、条件响应、数值范围与计算成本。

希望亲手生成自然图像时，完成潜空间与条件基础后进入 [09 预训练图像实践](docs/chapters/09-pretrained-image-lab.md)。固定权重、文本、初始噪声和 DDIM 配置，运行 CPU 推理并比较引导强度；先对照[实测报告](docs/pretrained-image-report.md)安排下载、内存和时间。

进入真实系统阅读时，可选 [FLUX.1 Kontext图像编辑](docs/readings/flux-kontext-2025.md) 或 [Wan视频生成](docs/readings/wan-video-2025.md)：两份2025阅读卡均固定原始报告和官方接口，提供可独立核对的条件、token与时间约定。

视频基础还可读 [SVD与CogVideoX训练对照](docs/readings/video-training-svd-cogvideox.md)，核对分阶段训练、视频压缩、VP预测量、帧率与公开模型的版本差异。

2026专题从 [ResTok图像自回归](docs/readings/restok-image-ar-2026.md)进入：用128个token、9轮采样连接离散表示与分组条件独立，并核对论文指标、公开实现和安装依赖。

视频方向可接着读 [LTX-2音画联合生成](docs/readings/ltx2-video-audio-2026.md)，用链式分解、声画配对反例和源码循环核对联合建模、条件引导及两阶段采样成本。

图像/视频读者还可直接运行 [06 视频时序实验](docs/chapters/06-video-temporal-lab.md)，观察帧集合相同而时间顺序错误的反例。

完整的先修图、实践任务与时间预算见 [学习路线](docs/learning-path.md)。全局 [符号约定](docs/notation.md) 与 [来源核验](SOURCES.md) 帮助跨章阅读。

## 第一次成功运行

已验证目标是 Linux x86_64、Python 3.12、CPU PyTorch 2.6.0。锁文件对应这一平台；macOS/Windows 需要各自的依赖锁定与验证。云环境首次安装需要 PyPI 与 PyTorch 官方软件源可达。下面命令从仓库根目录运行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m genai_study.experiment --method flow --steps 800 --output artifacts/first-flow
```

如果使用本任务准备好的云环境，直接 `source .venv/bin/activate` 后运行实验。没有系统 Python 3.12 时可用预装的 `uv venv --python 3.12 --seed .venv` 创建带 pip 的环境；云环境写缓存遇权限问题时给 uv 设置 `UV_CACHE_DIR=/workspace/.cache/uv`。

成功标志是进程退出码为 0，并生成 `metrics.json`、`samples.npz`、`samples.png`、`loss.png` 和 `checkpoint.pt`。接着对照数据与生成分布的并排图、留出损失和诊断指标，判断分布拟合情况。数值和本机耗时见 [实测报告](docs/experiment-report.md)；安装会下载百 MiB 量级的 CPU torch wheel，具体大小以固定版本安装输出为准；连同其他依赖和缓存，需预留数 GB 空间。

输出目录默认保护已有结果。读到学习路线时可直接重用 `artifacts/first-flow`；需要重新训练时换一个目录，例如 `--output artifacts/flow-seed7-run2`。

接着运行扩散并检查实现：

```bash
python -m genai_study.experiment --method ddpm --steps 800 --output artifacts/first-ddpm
bash scripts/quality.sh
```

脚本检查公式性质、文档链接，以及二维生成、图像生成和视频时序反例的实际运行。短 smoke 用于检查程序路径；训练质量由完整实验评估。第 03、07 章提供固定 checkpoint 和初始噪声的求解器步数实验。完整实验按 seed 固定 CPU 随机性；跨 PyTorch/平台的位级一致性仍需另行验证。随机种子、多次评估和教学指标解释见 [实验说明](examples/README.md)。

## 实验覆盖范围与运行问题

- 二维八高斯用于检查训练目标、采样算法与分布诊断；图像与视频质量需要对应的数据和评价协议。
- 07/08学习受控圆盘/方形分布，09使用已训练的自然图像模型；06检查合成序列的时序指标。[Wan 2025 阅读卡](docs/readings/wan-video-2025.md)连接视频论文与模型接口，视频权重运行列入后续实践。
- 首次安装失败：检查 Python 版本与网络允许的 `pypi.org`、`files.pythonhosted.org`、`download.pytorch.org` 和 `download-r2.pytorch.org`；修复网络后保留 TLS 与 hash 校验重试。
- CPU 很慢：二维/图形训练使用`--threads 1`，先降低`--steps`与`--samples`确认路径，再恢复完整训练预算。09的预训练入口使用2步、256×256的smoke配置，具体命令与成本见该章。

## 参与和维护

优先解决读者真正遇到的问题。贡献方法见 [CONTRIBUTING.md](CONTRIBUTING.md)，当前检查点见 [执行记录](records/PROGRESS.md)，待办见 [质量 backlog](records/BACKLOG.md)。每小时检查的定义和实际部署边界见 [自动化说明](docs/automation.md)。

原创代码和文档使用 [Apache-2.0](LICENSE)，外部模型、许可副本及生成示例的来源见 [SOURCES.md](SOURCES.md)。欢迎通过 issue 提供可复现问题、通过 PR 补充推导和实验。如果这些材料帮助你完成学习或实验，欢迎推荐给同学和研究伙伴。
