# 实测证据索引

按要回答的实验问题定位报告，再检查对应JSON中的配置、版本、随机性和计时范围。报告负责解释现象；本目录保存测量值、执行协议与核验记录。各轮进展、失败和恢复步骤见[执行记录](../PROGRESS.md)。

## 从实验问题找到证据

| 实验问题 | 阅读与复现入口 | 原始记录及身份核验 |
| --- | --- | --- |
| 二维模型增加训练预算后，哪些分布诊断改善？ | [二维实测报告](../../docs/experiment-report.md)；800/5000更新、两种方法、三个seed | 下表列出全部12份训练JSON；[provenance](provenance.json)记录当时源码与依赖锁SHA |
| 使用精确最优噪声预测器，DDPM还会产生什么偏差？ | [解析oracle报告](../../docs/ddpm-oracle-report.md)：正态方差递推、八高斯径向收缩 | [oracle-study索引](oracle-study/README.md)：事先预算、全部三seed、独立推导/数组复算、seed边界修复验证和图形来源 |
| 固定二维flow后，增加Euler步数会怎样？ | [800步checkpoint的求解器对照](../../docs/experiment-report.md#固定模型的求解器对照)；该节有独立复现命令 | [solver-study.json](solver-study.json)：训练metadata、checkpoint SHA、配对噪声配置、每组NFE与终点诊断 |
| 相同帧集合能否掩盖时序错误？ | [06章：合成视频时序反例](../../docs/chapters/06-video-temporal-lab.md) | [video-counterexample.json](video-counterexample.json)：实际置换、时间戳、像素/质心指标与单位；解析速度参照见正文 |
| 程序圆盘/方形训练能学到哪些结构，还存在哪些误差？ | [小图像训练实测](../../docs/image-experiment-report.md) | [image-flow](image-flow/)的三个seed JSON与[provenance](image-flow/provenance.json)：几何粗判、类别内变化、原始越界与裁剪诊断 |
| 固定同一图像模型，积分步数如何影响原始值与显示值？ | [图像Euler对照](../../docs/image-experiment-report.md#固定模型后euler-步数改变了什么) | [image-solver-study](image-solver-study/)：各seed的`train-*.json`、`study-*.json`及[provenance](image-solver-study/provenance.json)，保留重新训练和配对采样的身份 |
| 条件丢弃与CFG强度怎样改变条件响应、几何和裁剪？ | [图像CFG实测](../../docs/image-guidance-report.md) | [image-guidance](image-guidance/)：有/无条件训练计数、强度扫描、兼容性对照及[provenance](image-guidance/provenance.json)中的独立复算 |
| 固定SD1.4的真实CPU推理成本、引导对照和重复结果是什么？ | [预训练图像实测](../../docs/pretrained-image-report.md)与[09章操作路径](../../docs/chapters/09-pretrained-image-lab.md) | [pretrained-sd14索引](pretrained-sd14/README.md)：准备/缓存复用、smoke、双引导、完整重复、提示修订及源码/软件身份 |
| 同一图像经VAE编码与解码后，后验采样和缩放带来什么变化？ | [VAE重建实践](../../docs/vae-reconstruction-report.md)：六案例、原始/裁剪像素误差与完整重复 | [vae-reconstruction索引](vae-reconstruction/README.md)：事前协议、全部指标/外层日志、26数组复算、源码和媒体身份 |
| 在生成前固定提示与seed集合后，如何保留并比较全部结果？ | [SD1.4四请求探索报告](../../docs/sd14-prompt-pilot.md) | [sd14-prompt-pilot索引](sd14-prompt-pilot/README.md)：事先计划、全部metrics、实际执行顺序、AI视觉观察与配置/噪声/图片核验 |
| 推荐5000步首次路径能否重跑，并复用checkpoint完成采样对照？ | [首次路径复查报告](../../docs/experiment-report.md#推荐首次路径的再次运行) | [first-run-5000索引](first-run-5000/README.md)：两训练和solver原始JSON、完整argv/外层耗时，以及与历史数组的比对记录 |
| 环境恢复后，首次路径仍能运行吗，哪些结果精确或近似一致？ | [恢复实例核验](../../docs/experiment-report.md#恢复实例后的首次路径与浮点差异)：两训练、两次求解器及其实际成本 | [recovered-cpu-reader-path索引](recovered-cpu-reader-path/README.md)：三请求原计划、观察后追加的旧checkpoint控制、逐数组/参数差异、源码身份和独立诊断复算 |
| 固定SVD图生视频能否在声明的CPU预算内完成，画面表现如何？ | [视频实测与失败分析](../../docs/pretrained-video-report.md)及[10章](../../docs/chapters/10-pretrained-video-lab.md)：完整请求16.8分钟、严重形变与闪烁 | [pretrained-svd索引](pretrained-svd/README.md)：事先计划、准备、smoke/校准/full/重复、官方API短对照、全部帧与播放核验；模型身份和许可见[资产说明](../../examples/pretrained-svd/README.md) |

二维训练预算的全部原始JSON：

| 方法 / 优化器更新数 | seed 7 | seed 19 | seed 43 |
| --- | --- | --- | --- |
| DDPM / 800 | [记录](ddpm-800-seed7.json) | [记录](ddpm-800-seed19.json) | [记录](ddpm-800-seed43.json) |
| DDPM / 5000 | [记录](ddpm-5000-seed7.json) | [记录](ddpm-5000-seed19.json) | [记录](ddpm-5000-seed43.json) |
| Flow / 800 | [记录](flow-800-seed7.json) | [记录](flow-800-seed19.json) | [记录](flow-800-seed43.json) |
| Flow / 5000 | [记录](flow-5000-seed7.json) | [记录](flow-5000-seed19.json) | [记录](flow-5000-seed43.json) |

同名算法的历史记录可能采用不同训练预算、源码版本或运行环境。比较前同时核对`config`、版本、provenance和`timing_scope`；solver还需核对checkpoint的训练metadata与SHA。后续复查保留自己的执行记录，历史测量及其时间按原轮次解释。

## 如何判断证据的类型

运行入口的故障恢复单独见[命令记录器验证](run-record/README.md)：包含真实图像/视频CLI缺模型、正常结束和超时的原始argv、日志、状态与外层耗时；这些检查使用轻量子进程。

| 类型 | 在哪里核对 | 阅读时关注什么 |
| --- | --- | --- |
| 本仓库实际运行 | 上表的测量JSON、报告与执行记录 | 实际配置、输出、过滤/失败状态、计时边界和验证范围；具体字段随实验而异 |
| 解析结论与数值核验 | [02/03来源审计](../../docs/source-audit.md)、[概率流衔接核验](probability-flow/README.md)、[预测目标与时间权重](prediction-loss/README.md)、[条件注意力接口](conditioning-attention/README.md)、[空间token与位置](spatial-tokens/README.md)、[一致性端点映射](consistency-endpoint/README.md)、[生成模型审计](../../docs/generative-source-audit.md)、[评价指标审计](../../docs/evaluation-metrics-audit.md)，以及对应章节的推导和反例 | 假设、变量定义、解析参照与有限样本/浮点误差；例如二维高斯邻域概率由分布公式计算，测量比例由生成样本统计 |
| 原论文作者报告 | [来源总表](../../SOURCES.md)及[研究阅读卡](../../docs/readings/) | 固定论文/官方实现版本、表格或页码、原作者硬件与评价协议；这些数值沿用作者证据身份，本地执行范围在各卡单独说明 |
| 计划与预算估计 | 各实验的`plan.json`和执行记录 | 生成前声明的请求、超时与判断门槛；完成后再对照实际执行记录确认结果 |

提示遵循等视觉观察另保留观察者身份和判据；SD1.4提示探索的标注来自两个AI代理，真实读者评价仍待收集。阅读报告时将这些逐图观察与程序统计值分别核对。

## Git中保存了什么，如何重新取得完整产物

已提交的材料包括本目录的小型JSON与说明、报告中使用的[展示图片](../../docs/assets/)，以及各实验已有的源码/依赖/输出身份记录。SD1.4与SVD的模型文件身份分别由[SD1.4清单](../../examples/pretrained-sd14/manifest.json)和[SVD清单](../../examples/pretrained-svd/manifest.json)逐文件固定大小与SHA；许可和获取方式见相邻资产说明及实践章节。

原始浮点数组（如`samples.npz`、`arrays.npz`、`initial_noise.npy`）、推理checkpoint、下载的模型权重和完整运行日志保存在执行环境的`artifacts/`中，该目录由[.gitignore](../../.gitignore)排除。克隆仓库后可先阅读已提交的JSON与图；需要这些完整产物时，按对应报告/章节重新运行，并将预训练权重下载到声明的本地目录。JSON中的历史路径用于定位当时的产物，文件hash用于核对已有或重新取得的内容。

复现时选择新的输出目录，保留原始结果。数值数组与PNG承担不同用途：前者用于重算指标和逐元素比较，后者用于查看报告声明的展示或完整样本。每次核验应记录实际拿到的文件、比较方法和覆盖范围；已有检查状态可从[自动化说明](../../docs/automation.md)与[执行记录](../PROGRESS.md)继续追踪。
