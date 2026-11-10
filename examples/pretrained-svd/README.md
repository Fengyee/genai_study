# 固定SVD权重与CPU实践准备

**Powered by Stability AI**

[manifest.json](manifest.json)固定原作者`stabilityai/stable-video-diffusion-img2vid`的revision `9cf024d5bfa8f56622af86c884f26a52f6676f2e`，包含11个文件及其大小/SHA，共9,018,094,887字节（8.398755GiB）。配置、模型卡和许可已按固定版本读取，权重身份来自Hub LFS元数据；实际使用前需下载完整文件并逐个验证SHA。

完整许可见[LICENSE-model.txt](LICENSE-model.txt)，版本为2024-07-05的Stability AI Community License，归属见[NOTICE](NOTICE)。研究与非商业用途适用其第2节；商业用途另有注册和收入条件，完整条款以许可原文为准。模型、许可和生成结果的使用条件分别记录，仓库原创代码和教材采用根目录Apache-2.0许可。

本轮先声明[分阶段CPU实验计划](../../records/measurements/pretrained-svd/plan.json)，再完成权重准备、入口验证和实际运行。计划保留原始输入、全部帧、两类随机流和真实调用/时间/内存记录。当前运行状态以[执行记录](../../records/PROGRESS.md)最新检查点为准。
