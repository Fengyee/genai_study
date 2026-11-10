# 二维生成实验

在仓库根目录完成 README 的环境安装后运行：

```bash
python -m genai_study.experiment --method ddpm --steps 800 --output artifacts/ddpm
python -m genai_study.experiment --method flow --steps 800 --output artifacts/flow
```

首次成功的标准是：命令退出码为 0，产生 `metrics.json`、`samples.npz`、`samples.png`、`loss.png` 和 `checkpoint.pt`。800 次更新是可在 CPU 上尝试的教学预算，不保证收敛。观察数据与生成散点图、留出样本损失和诊断指标；不要只依靠训练损失。已存在的非空输出目录默认拒绝写入；明确需要重跑覆盖时追加 `--overwrite`。

- 数据：8 个等权二维高斯分量，中心位于半径 2 的圆，分量协方差为 `0.08² I`；无需下载。
- 模型：三层宽度 128 的 SiLU MLP，用 8 个正弦/余弦频率编码时间。
- 优化：Adam，学习率 `1e-3`，batch size 256，CPU 单线程。
- DDPM：100 个扩散步，线性 beta 从 `0.0001` 到 `0.1`。此短噪声表为 toy 实验设计，不是图像扩散的推荐配置。Python 索引 `k=0` 对应论文 `t=1`，最后一步方差严格为 0。`--steps` 是训练更新数，不是扩散步数。
- Flow matching：独立采样噪声和数据端点，直线条件路径，速度目标 `data-noise`；采样使用 100 步显式 Euler。
- 默认 seed 7。模型初始化与训练、固定留出评估、生成、参考数据、基线数据分别使用不同随机流。同一依赖版本和 CPU 环境可复现；跨平台/版本不保证位级一致。

`metrics.json` 记录全部命令配置、依赖版本与实测耗时（训练、评估、采样、指标；不含导入和绘图）。`initial_eval_loss` / `final_eval_loss` 使用同一组 4096 个独立留出样本。DDPM 回归噪声，flow 回归速度，**两者损失数值不能直接比较**。

诊断指标均为二维教学工具：

- `energy_distance_v_statistic`：前至多 512 个生成/参考样本的 `2 E||X-Y|| - E||X-X'|| - E||Y-Y'||`，包含自配对，属于有偏 V 统计量；有限样本之间即使同分布也通常大于 0。此处未开平方。
- `mean_distance_to_nearest_center`：越低不一定越好；坍缩到中心也会得到低值。
- `fraction_within_3sigma`：到最近中心距离小于 `3×0.08` 的比例。二维高斯在三倍标准差**半径**内的理论概率约为 `1-exp(-9/2)=98.89%`，不是一维的 99.73%。
- `covered_modes_at_1pct`：一个模式的上述圆内至少有全部生成样本的 1% 时才计入覆盖。范围 0–8；所有模式都覆盖也不能证明分布正确。

这些指标不是 FID、视觉质量或多随机种子置信区间。二维成功不意味着图像/视频能力。练习：比较 200/800/2400 次更新；更换 3 个随机种子；寻找“最近中心距离改善但模式覆盖下降”的反例。额外训练成本由更新数近似线性决定，采样成本由样本数决定。

## 如何识别“覆盖全部模式”带来的误导

`baseline_diagnostics` 用独立的真实数据样本（seed+5）对照同一参考样本（seed+4），与模型的 `diagnostics` 采用同样的样本预算。这个 target-vs-target 基线展示有限样本波动，尤其是 energy V 统计量的非零噪声水平；单次基线不是严格阈值或置信区间，也不要求任意一次模型结果一定大于该数值。`data_theory.own_center_3sigma_mass_2d` 记录解析概率 `1-exp(-9/2)`，指单个二维高斯样本距其生成中心的半径事件；最近中心指标在理论上还可能包含落入其他中心圆的极罕见尾部样本。

反例：每个真实模式放 2% 的样本，剩下 84% 放到远离数据的点，仍得到 coverage=8，但圆内比例只有 16%，能量距离很差。因此要同时阅读图、energy 和圆内比例。绘图为两个子图设置覆盖所有参考/生成点的共同坐标范围，不裁掉离群点；离群点导致主数据区域显得很小本身就是失败信号。

## 固定模型，单独研究 Euler 求解步数

先运行上面的 flow 训练命令。新的训练输出包含 `checkpoint.pt`（推理权重、架构和训练配置；**不保存优化器状态，不能用于严格续训**）。旧版本实验 JSON 若没有 checkpoint 字段，需要重新运行训练生成权重，不应把旧指标改写成新运行结果。然后：

```bash
python -m genai_study.solver_study --checkpoint artifacts/flow/checkpoint.pt --steps 10 25 100 200 --samples 2048 --seed 123 --output artifacts/flow-solver
```

此命令不训练。它只接受 flow checkpoint，先固定一批初始噪声，再对同一个冻结速度场运行不同步数的显式 Euler。步数列表会排序并去重；每步一次网络求值，因此 NFE（number of function evaluations）等于步数。权重载入明确使用 `torch.load(weights_only=True, map_location="cpu")`，不反序列化任意模型对象。

输出 `metrics.json`、`samples.npz` 和 `samples.png`。JSON 保存 checkpoint SHA-256、训练 metadata、实际采样配置、独立真实数据基线以及每组 `steps` / `nfe` / `elapsed_seconds` / `diagnostics`。耗时只计单次采样，没有 warm-up，不能作为严谨硬件排名。

`runs[i].paired_rms_l2_difference_from_previous` 是当前与上一个较少步数之间，**同一初始噪声对应的终点差**：`sqrt(mean_i ||x_i^(K)-x_i^(K_previous)||²)`。第一组为 `null`。它衡量两个数值离散化之间的差异，**不是真解误差**；最细的 200 步也不是解析真解。`samples.npz` 包含 `initial`、`reference`、`baseline` 与 `steps_10` 等配对数组，可独立复核。

练习：观察终点配对差和数据分布诊断是否一起改善。解释为什么求解步数更多可能让模型自身 ODE 的数值解更稳定，却不一定使它更接近目标八高斯；模型误差、离散化误差和有限样本波动是三个不同来源。再固定初始噪声、提高训练预算重复比较，避免同时改变权重与采样种子。

Checkpoint 使用边界：这里只加载本仓库训练命令产生并由自己保管的可信文件；`weights_only=True` 不是对任意第三方 checkpoint 的安全承诺。PyTorch `<2.6.0` 的该载入模式存在 [CVE-2025-32434 / GHSA-53q9-r3pm-6pq6](https://github.com/advisories/GHSA-53q9-r3pm-6pq6)，官方公告列出的修复版本为 `2.6.0`；本项目因此要求修复后的固定依赖，并在旧运行时载入前拒绝执行。不要通过关闭 `weights_only` 绕过格式或版本错误。
