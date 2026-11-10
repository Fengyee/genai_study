# SD1.4 CPU测量证据

模型文件身份见[manifest](../../../examples/pretrained-sd14/manifest.json)，运行与评价见[实测报告](../../../docs/pretrained-image-report.md)，操作命令见[09章](../../../docs/chapters/09-pretrained-image-lab.md)。

本目录保留真实CLI输出的配置、版本、调度、成本和输出身份。`prepare.json`是实际Hub准备过程，`prepare-reuse.json`是最终入口对完整缓存的复用验证；下载时间和推理时间分别记录。

`smoke.json`使用256×256/2步；`default.json`与`guidance3.json`使用512×512/20步、同一提示和seed7。后续复跑及提示修订同样保存原始JSON。各过程均保留模型原有Safety checker。

原始权重与initial_noise.npy在忽略的artifacts目录；完整样本PNG收录在docs/assets，输出hash保存在每份JSON。报告保留提示中缺失的小屋及额外边框等具体观察。
