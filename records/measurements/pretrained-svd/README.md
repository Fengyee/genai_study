# SVD CPU实践的原始证据

**Powered by Stability AI**

[计划](plan.json)在首次模型调用前保存为2ed6b0c；[preflight](preflight.json)记录实际入口a3ed4a8、源码/依赖/manifest的SHA。模型资产与许可见[固定清单](../../../examples/pretrained-svd/README.md)，所有请求沿用预先选定的SD1.4水彩输入、视频seed7和增强seed8。

| 阶段 | 原始metrics | 实际执行 | 全部PNG帧 | 播放与媒体核验 |
| --- | --- | --- | --- | --- |
| 4帧/2步smoke | [smoke.json](smoke.json) | [外层命令](smoke-execution.json) | [原帧ZIP](../../../docs/assets/svd-smoke-frames.zip) | [MP4](../../../docs/assets/svd-smoke.mp4)、[导出/探测](smoke-export.json)、[Chromium播放](smoke-playback.json) |
| 14帧/4步校准 | [calibration.json](calibration.json) | [外层命令](calibration-execution.json) | [原帧ZIP](../../../docs/assets/svd-calibration-frames.zip) | [MP4](../../../docs/assets/svd-calibration.mp4)、[导出/探测](calibration-export.json)、[Chromium播放](calibration-playback.json) |
| 14帧/25步完整低分辨率 | [full.json](full.json) | [外层命令](full-low-resolution-execution.json) | [原帧ZIP](../../../docs/assets/svd-full-frames.zip) | [MP4](../../../docs/assets/svd-full.mp4)、[导出/探测](full-export.json)、[Chromium播放](full-playback.json) |
| 4帧/2步重复 | [smoke-repeat.json](smoke-repeat.json) | [外层命令](smoke-repeat-execution.json) | 与首份smoke逐字节相同 | [重复及配对检查](repeat-and-pair-check.json) |

smoke完整wall47.14秒、峰值9.54GiB；校准wall203.80秒、峰值11.29GiB。[资源判断](full-decision.json)先得到25步保守预算1273.77秒，再启动完整请求，实际wall1007.30秒、峰值11.21GiB。smoke重复wall46.53秒、浮点数组及所有PNG完全一致。完整输出出现严重过饱和和形变，全部帧与解释见[实测报告](../../../docs/pretrained-video-report.md)。

查看完整输出后追加一次[直接官方API对照](direct-comparison.json)，[执行记录](direct-execution.json)为exit0、wall47.29秒，[身份核对](direct-identity-check.json)通过。它独立构造两随机流、直接调用固定管线并省去仓库推理函数/hook；4帧/2步的全部浮点帧和PNG均与smoke相同。该检查覆盖封装一致性，失败成因留待控制实验。

[边界值与变化量记录](boundary-observations.json)从保存的全部float32后处理帧复算0/1通道值比例及相邻帧RMS范围；它明确保留裁剪后的统计口径。

[prepare.json](prepare.json)为临时bootstrap实际下载并逐文件/最终复核的记录，11文件共9018094887字节，264.66秒；[prepare-reuse.json](prepare-reuse.json)为最终仓库入口复用11文件、0次下载、9.51秒。权重与完整浮点数组保存在忽略的本地artifacts，Git保存其身份与上述完整PNG帧档案。

[smoke独立复查](smoke-independent-check.json)从保存的input.png和seed8逐元素重建VAE增强输入，并从seed7重建初始视频噪声；同时核对裁剪框、实际条件、联合视频形状、PNG hash及8位量化。校准使用相同参考增强，初始视频噪声按14帧形状单独复算。原始metrics保留后处理float32帧及其hash；PNG为8位量化结果，MP4使用H.264有损编码。

媒体检查由FFmpeg7.1.5执行全部帧解码，ffprobe核对每帧时间戳、7fps、帧数和时长；Chromium156通过临时本地HTTP实际播放到结束，三个不同请求均无丢帧。浏览器播放检查与内容观察分别记录。

[直接API实验原脚本](direct-reference.py)逐字节归档，SHA-256为`6a21b514c8c545b76cddd8a9cce07653e7f1800bd25d877ee183a368d5f639cc`，与对照JSON中的身份一致。脚本保留当时的`/workspace/genai_study`及输出路径，供审阅实际实现；迁移执行需先调整根目录和选择新的输出路径，再按10章准备同一模型、环境和smoke输入。默认驱动入口保护已有输出，直接`--worker`为内部子进程入口。
