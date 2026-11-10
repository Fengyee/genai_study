# 固定SVD权重与CPU实践

**Powered by Stability AI**

[manifest.json](manifest.json)固定原作者`stabilityai/stable-video-diffusion-img2vid`的revision `9cf024d5bfa8f56622af86c884f26a52f6676f2e`，包含11个文件及其大小/SHA，共9,018,094,887字节（8.398755GiB）。配置、模型卡和许可已按固定版本读取，权重身份来自Hub LFS元数据；实际使用前需下载完整文件并逐个验证SHA。

完整许可见[LICENSE-model.txt](LICENSE-model.txt)，版本为2024-07-05的Stability AI Community License，归属见[NOTICE](NOTICE)。研究与非商业用途适用其第2节；商业用途另有注册和收入条件，完整条款以许可原文为准。模型、许可和生成结果的使用条件分别记录，仓库原创代码和教材采用根目录Apache-2.0许可。

先声明[分阶段CPU实验计划](../../records/measurements/pretrained-svd/plan.json)，再完成全部11文件SHA核验、smoke、成本校准、14帧/25步完整低分辨率请求及smoke重复。完整请求wall约16.8分钟、峰值11.21GiB，画面有严重形变和闪烁；全部帧、视频及原始JSON见[实测报告](../../docs/pretrained-video-report.md)。两次smoke和直接调用官方API的短对照在当前环境下数值相同。

操作路径和公式见[10章](../../docs/chapters/10-pretrained-video-lab.md)，资产准备和运行状态见[测量索引](../../records/measurements/pretrained-svd/README.md)。发布训练尺寸的视频质量基线仍待独立实验。
