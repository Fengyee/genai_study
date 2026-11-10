# GenAI Study · 从概率建模到图像与视频生成

给熟练使用 PyTorch、有研究经验但尚未系统学习生成式 AI 的 CS master / PhD：用清楚的假设、可核对的推导和能在 CPU 上跑通的小实验建立基础，再进入图像与视频生成研究。

**当前版本**：8 个学习单元、二维 DDPM / Flow Matching 训练与采样、16×16 条件卷积图像生成、无需权重的视频时序反例、自动质量检查。它是一套可运行的基础课程起点；真实图像/视频模型训练、预训练模型实操及完整前沿综述仍在 [backlog](records/BACKLOG.md)。不需要 API key 或下载模型权重。

## 我适合从哪里开始？

| 你的情况 | 路线 | 完成标志 |
| --- | --- | --- |
| 第一次接触生成建模 | [00 基础](docs/chapters/00-foundations.md) → [01 生成模型](docs/chapters/01-generative-models.md) | 能区分密度、采样、条件建模与评价 |
| 想理解扩散和流模型 | [02 扩散](docs/chapters/02-diffusion.md) → [03 Flow Matching](docs/chapters/03-flow-matching.md) | 推导核心训练目标，跑通并解释二维实验 |
| 准备研究图像/视频 | 完成上述基础 → [04 研究路线](docs/chapters/04-image-video-roadmap.md) | 能选定一个有假设、基线和评价协议的问题 |

图像方向可从 [07 CPU图像生成实验](docs/chapters/07-image-flow-lab.md) 学习BCHW、卷积与类别条件；默认1500步无需下载数据或权重，成本和局限见 [图像实测](docs/image-experiment-report.md)。

图像/视频读者还可直接运行 [06 视频时序实验](docs/chapters/06-video-temporal-lab.md)，观察帧集合相同而时间顺序错误的反例。

完整的先修图、实践任务与时间预算见 [学习路线](docs/learning-path.md)。全局 [符号约定](docs/notation.md) 与 [来源核验](SOURCES.md) 帮助跨章阅读。

## 第一次成功运行

已验证目标是 Linux x86_64、Python 3.12、CPU PyTorch 2.6.0。macOS/Windows 不能直接套用这里的 Linux CPU 锁文件；其他平台尚未验证。云环境首次安装需要 PyPI 与 PyTorch 官方软件源可达。下面命令从仓库根目录运行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m genai_study.experiment --method flow --steps 800 --output artifacts/first-flow
```

如果使用本任务准备好的云环境，直接 `source .venv/bin/activate` 后运行实验。没有系统 Python 3.12 时可用预装的 `uv venv --python 3.12 .venv`；云环境写缓存遇权限问题时给 uv 设置 `UV_CACHE_DIR=/workspace/.cache/uv`。

成功标志是进程退出码为 0，并生成 `metrics.json`、`samples.png`、`loss.png` 和 `checkpoint.pt`。查看数据与生成分布并排图，不要仅看 loss 的曲线。**文件生成只是运行成功，分布质量还需诊断。** 数值和本机耗时见 [实测报告](docs/experiment-report.md)；安装会下载百MiB量级的CPU torch wheel，具体大小以固定版本安装输出为准；连同其他依赖和缓存，需预留数GB空间。

接着运行扩散并检查实现：

```bash
python -m genai_study.experiment --method ddpm --steps 800 --output artifacts/first-ddpm
bash scripts/quality.sh
```

脚本验证公式性质、文档链接和两种算法的小规模实际运行；短 smoke 检查不评估训练是否收敛；视频检查验证程序生成的时序反例。第03章还提供固定 checkpoint、固定初始噪声的求解器步数实验。完整实验按 seed 固定 CPU 随机性，但跨 PyTorch/平台的位级一致性不保证。随机种子、多次评估和 toy 指标解释见 [实验说明](examples/README.md)。

## 内容边界与常见问题

- 二维八高斯用于学习算法，不能证明高维图像/视频质量，也不是 FID benchmark。
- 图像实验生成受控圆盘/方形，视频实验验证时序指标；真实自然图像/视频模型仍是研究路线。新增 [Wan 2025阅读卡](docs/readings/wan-video-2025.md) 帮助对照原始系统，不宣称复现大模型。
- 首次安装失败：检查 Python 版本与网络允许的 `pypi.org`、`files.pythonhosted.org`、`download.pytorch.org` 和 `download-r2.pytorch.org`；不要关闭 TLS 或删掉 hash 校验。
- CPU 很慢：使用 `--threads 1`，先降低 `--steps` 与 `--samples` 确认路径。低步数输出不应当成正常模型质量。

## 参与和维护

优先解决读者真正遇到的问题。贡献方法见 [CONTRIBUTING.md](CONTRIBUTING.md)，当前检查点见 [执行记录](records/PROGRESS.md)，待办见 [质量 backlog](records/BACKLOG.md)。每小时检查的定义和实际部署边界见 [自动化说明](docs/automation.md)。

代码、原创文档和生成图使用 [Apache-2.0](LICENSE)，第三方材料归属见 [SOURCES.md](SOURCES.md)。欢迎通过 issue 提供可复现问题、通过 PR 补充推导和实验。项目以可信与有用为目标；star 数是读者反馈，无法承诺。
