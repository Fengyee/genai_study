# SVD CPU实践的原始证据

**Powered by Stability AI**

[计划](plan.json)在首次模型调用前保存为2ed6b0c；[preflight](preflight.json)记录实际入口a3ed4a8、源码/依赖/manifest的SHA。模型资产与许可见[固定清单](../../../examples/pretrained-svd/README.md)，所有请求沿用预先选定的SD1.4水彩输入、视频seed7和增强seed8。

| 阶段 | 原始metrics | 实际执行 | 全部PNG帧 | 播放与媒体核验 |
| --- | --- | --- | --- | --- |
| 4帧/2步smoke | [smoke.json](smoke.json) | [外层命令](smoke-execution.json) | [原帧ZIP](../../../docs/assets/svd-smoke-frames.zip) | [MP4](../../../docs/assets/svd-smoke.mp4)、[导出/探测](smoke-export.json)、[Chromium播放](smoke-playback.json) |
| 14帧/4步校准 | [calibration.json](calibration.json) | [外层命令](calibration-execution.json) | [原帧ZIP](../../../docs/assets/svd-calibration-frames.zip) | [MP4](../../../docs/assets/svd-calibration.mp4)、[导出/探测](calibration-export.json)、[Chromium播放](calibration-playback.json) |

smoke完整wall47.14秒、峰值9.54GiB；校准wall203.80秒、峰值11.29GiB。按计划进行的[资源判断](full-decision.json)得到25步保守预算1273.77秒，已允许启动完整低分辨率请求；其完成状态与后续重复检查以[执行记录](../../PROGRESS.md)最新检查点为准。

[prepare.json](prepare.json)为临时bootstrap实际下载并逐文件/最终复核的记录，11文件共9018094887字节，264.66秒；[prepare-reuse.json](prepare-reuse.json)为最终仓库入口复用11文件、0次下载、9.51秒。权重与完整浮点数组保存在忽略的本地artifacts，Git保存其身份与上述完整PNG帧档案。

[smoke独立复查](smoke-independent-check.json)从保存的input.png和seed8逐元素重建VAE增强输入，并从seed7重建初始视频噪声；同时核对裁剪框、实际条件、联合视频形状、PNG hash及8位量化。校准使用相同参考增强，初始视频噪声按14帧形状单独复算。原始metrics保留后处理float32帧及其hash；PNG为8位量化结果，MP4使用H.264有损编码。

媒体检查由FFmpeg7.1.5执行全部帧解码，ffprobe核对每帧时间戳、7fps、帧数和时长；Chromium156通过临时本地HTTP实际播放到结束，两个阶段均无丢帧。浏览器播放检查与内容观察分别记录。
