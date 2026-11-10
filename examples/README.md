# 生成实验与固定模型采样

## 二维生成实验

在仓库根目录完成 README 的环境安装后运行：

```bash
python -m genai_study.experiment --method ddpm --steps 800 --output artifacts/ddpm
python -m genai_study.experiment --method flow --steps 800 --output artifacts/flow
```

运行完成后应得到 `metrics.json`、`samples.npz`、`samples.png`、`loss.png` 和 `checkpoint.pt`。800 次更新是 CPU 教学预算；结合数据与生成散点图、留出损失和诊断指标检查拟合情况。若已经完成 README 的首次实验，可重用 `artifacts/first-flow` 等目录中的结果，并相应修改下文 checkpoint 路径。非空输出目录默认受保护；重跑时选择新目录，确认要覆盖旧实验时才使用 `--overwrite`。

- 数据：8 个等权二维高斯分量，中心位于半径 2 的圆，分量协方差为 `0.08² I`；无需下载。
- 模型：三层宽度 128 的 SiLU MLP，用 8 个正弦/余弦频率编码时间。
- 优化：Adam，学习率 `1e-3`，batch size 256，CPU 单线程。
- DDPM：100 个扩散步，线性 beta 从 `0.0001` 到 `0.1`。此短噪声表针对本章的二维数据尺度设计。Python 索引 `k=0` 对应论文 `t=1`，最后一步方差严格为 0。`--steps` 指训练更新数。
- Flow matching：独立采样噪声和数据端点，直线条件路径，速度目标 `data-noise`；采样使用 100 步显式 Euler。
- 默认 seed 7。模型初始化与训练、固定留出评估、生成、参考数据、基线数据分别使用不同随机流。同一依赖版本和 CPU 环境可复现；跨平台/版本不保证位级一致。

`metrics.json` 记录全部命令配置、依赖版本与实测耗时（训练、评估、采样、指标；不含导入和绘图）。`initial_eval_loss` / `final_eval_loss` 使用同一组 4096 个独立留出样本。DDPM 回归噪声，flow 回归速度，**两者损失数值不能直接比较**。

诊断指标均为二维教学工具：

- `energy_distance_v_statistic`：前至多 512 个生成/参考样本的 `2 E||X-Y|| - E||X-X'|| - E||Y-Y'||`，包含自配对，属于有偏 V 统计量；有限样本之间即使同分布也通常大于 0。此处未开平方。
- `mean_distance_to_nearest_center`：越低不一定越好；坍缩到中心也会得到低值。
- `fraction_within_3sigma`：到最近中心距离小于 `3×0.08` 的比例。二维高斯在三倍标准差**半径**内的理论概率约为 `1-exp(-9/2)=98.89%`，不是一维的 99.73%。
- `covered_modes_at_1pct`：一个模式的上述圆内至少有全部生成样本的 1% 时才计入覆盖。范围 0–8；所有模式都覆盖也不能证明分布正确。

这些指标检查二维分布拟合；图像与视频实验需要各自的数据和诊断协议。练习：比较 200/800/2400 次更新；更换 3 个随机种子；寻找“最近中心距离改善但模式覆盖下降”的反例。额外训练成本由更新数近似线性决定，采样成本由样本数决定。

## 如何识别“覆盖全部模式”带来的误导

`baseline_diagnostics` 用独立的真实数据样本（seed+5）对照同一参考样本（seed+4），与模型的 `diagnostics` 采用同样的样本预算。这个 target-vs-target 基线展示有限样本波动，尤其是 energy V 统计量的非零噪声水平；单次基线不是严格阈值或置信区间，也不要求任意一次模型结果一定大于该数值。`data_theory.own_center_3sigma_mass_2d` 记录解析概率 `1-exp(-9/2)`，指单个二维高斯样本距其生成中心的半径事件；最近中心指标在理论上还可能包含落入其他中心圆的极罕见尾部样本。

反例：每个真实模式放 2% 的样本，剩下 84% 放到远离数据的点，仍得到 coverage=8，但圆内比例只有 16%，能量距离很差。因此要同时阅读图、energy 和圆内比例。绘图为两个子图设置覆盖所有参考/生成点的共同坐标范围，不裁掉离群点；离群点导致主数据区域显得很小本身就是失败信号。

## 固定模型，单独研究 Euler 求解步数

复用上面的flow训练结果。`checkpoint.pt`保存推理权重、架构和训练配置；精确续训还需优化器等状态，当前文件未包含这些状态。旧版实验若缺少checkpoint，请在新目录重新训练。首次比较时运行下列命令；若已按03章生成`artifacts/flow-solver`，直接读取结果。重做对照时将`--output`改为新目录。

```bash
python -m genai_study.solver_study --checkpoint artifacts/flow/checkpoint.pt --steps 10 25 100 200 --samples 2048 --seed 123 --output artifacts/flow-solver
```

此命令接受二维 flow checkpoint，固定初始噪声，对冻结的速度场运行不同步数的显式 Euler。步数列表会排序并去重；每步一次网络求值，因此 NFE（number of function evaluations）等于步数。权重以 `torch.load(weights_only=True, map_location="cpu")` 载入。

输出 `metrics.json`、`samples.npz` 和 `samples.png`。JSON 保存 checkpoint SHA-256、训练 metadata、实际采样配置、独立真实数据基线以及每组 `steps` / `nfe` / `elapsed_seconds` / `diagnostics`。耗时计单次采样，未做预热或重复计时；硬件比较需要额外的测量协议。

`runs[i].paired_rms_l2_difference_from_previous` 是当前与上一个较少步数之间，**同一初始噪声对应的终点差**：`sqrt(mean_i ||x_i^(K)-x_i^(K_previous)||²)`。第一组为 `null`。它衡量两个数值离散化之间的差异；200 步仍是有限步近似，计算真解误差需要独立参考解。`samples.npz` 包含 `initial`、`reference`、`baseline` 与 `steps_10` 等配对数组，可独立复核。

练习：观察终点配对差和数据分布诊断是否一起改善。解释为什么求解步数更多可能让模型自身 ODE 的数值解更稳定，却不一定使它更接近目标八高斯；模型误差、离散化误差和有限样本波动是三个不同来源。再固定初始噪声、提高训练预算重复比较，避免同时改变权重与采样种子。

Checkpoint 载入要求：使用本仓库训练命令产生并由自己保管的可信文件，保留 `weights_only=True` 和格式校验。PyTorch `<2.6.0` 的该载入模式存在 [CVE-2025-32434 / GHSA-53q9-r3pm-6pq6](https://github.com/advisories/GHSA-53q9-r3pm-6pq6)，官方公告列出的修复版本为 `2.6.0`；本项目因此要求修复后的固定依赖，并在旧运行时载入前拒绝执行。版本或格式校验失败时，检查环境版本与 checkpoint 来源后重试。

## 小图像：从训练到独立求解器实验

图像模型使用独立的命令和checkpoint格式。首次运行按顺序执行；已有训练checkpoint时执行第二条。若已按07章生成`artifacts/image-solver`，直接复用采样对照结果；重做时将对应`--output`改为新目录。

```bash
python -m genai_study.image_flow --steps 1500 --seed 7 --output artifacts/image-flow
python -m genai_study.image_solver_study \
  --checkpoint artifacts/image-flow/checkpoint.pt \
  --steps 10 20 40 80 --samples 128 --seed 123 \
  --output artifacts/image-solver
```

训练输出包括 `metrics.json`、`arrays.npz`、`samples.png`、`loss.png` 和 `checkpoint.pt`。独立采样固定权重、初始噪声、类别标签、参考数据和独立目标基线，保存 `metrics.json`、`arrays.npz`、`samples.png`。步数升序去重，每组 NFE 等于 Euler 步数。改变输出目录即可保留另一轮结果。

图像实验按 batch 和全部像素计算相邻方案的 RMS 差：`sqrt(mean_BCHW((x_K - x_previous)^2))`，分别记录原始值和各自裁剪到 `[0,1]` 后的值；与上文二维向量 L2 差采用不同归约。原始差反映数值离散化的敏感性，裁剪差说明显示图像的变化。结合每类的原始越界比例、几何模板通过率和位置/大小统计解释结果。完整定义、控制变量、运行成本和可执行练习见 [07 章](../docs/chapters/07-image-flow-lab.md) 与 [图像实测报告](../docs/image-experiment-report.md)。

## 小图像：条件丢弃与引导强度

首次运行以下两步。已有本组训练checkpoint时直接采样，已有采样结果时直接查看；重做时选择新输出目录并相应修改checkpoint路径。

```bash
python -m genai_study.image_flow --condition-dropout 0.2 --steps 1500 \
  --seed 7 --output artifacts/image-cfg
python -m genai_study.image_guidance_study \
  --checkpoint artifacts/image-cfg/checkpoint.pt \
  --scales 0 1 2 4 --sampling-steps 40 --samples 128 --seed 123 \
  --output artifacts/image-guidance
```

训练以全零条件通道表示隐藏的类别，记录有条件/空条件分支实际训练样本数。引导命令读取这些元数据；默认07章训练的dropout为0，需按上面的配置重新训练以学习空条件分支。

每份噪声分别配上圆盘/方形，所有强度复用同一权重与配对输入；强度0忽略请求类别，强度1恢复条件速度。每步显式计算两分支，40步对应每张图像80 NFE。输出为JSON、原始NPZ和固定索引图，数组键由`runs[].array_key`给出。使用 [08章](../docs/chapters/08-image-guidance-lab.md) 的练习复算配对差，并一起解释几何、多样性和裁剪诊断。

## 可选：固定SD1.4权重的自然图像生成

按 [09章](../docs/chapters/09-pretrained-image-lab.md)建立`.venv-pretrained`并使用独立hash锁安装依赖。先核对许可和[实测成本](../docs/pretrained-image-report.md)，再用`pretrained_image prepare`下载清单中的约5.11GiB文件。之后的`run`从本地加载，每次复核全部文件SHA，记录提示、初始噪声、DDIM配置、实际U-Net调用数、耗时和进程峰值内存。

首次完整运行与两强度配对命令统一放在09章。已生成`artifacts/pretrained-sd14`时直接复用结果；重跑使用新目录。基础二维/图形实验继续使用`.venv`，预训练模块的`--help`和完整性检查测试可在基础环境执行。
