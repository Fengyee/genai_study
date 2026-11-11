# 一致性端点映射的核验记录

[专题](../../../docs/readings/consistency-endpoint-map.md)连接速度场积分与端点映射，解释原始Consistency Models的CD/CT目标、EMA/停止梯度、正噪声端点和实际调用预算。

- [论文下载身份](downloads.json)：2303.01469v1摘要与PDF，主代理重新核对大小和SHA。
- [作者源码身份](source-files.json)：固定openai/consistency_models commit e32b69ee436d518377db86fb2127a3972d0d8716的8个文件，主代理重新核对全部身份。
- [正文解析结果](lesson-analytic-check.json)：m=1.5、数据标准差0.7、sigma_min=0.1、sigma_max=2；精确边缘初态下的Euler网格、总体方差、端点恒等式、MMSE与共噪声配对反例。
- [主代理独立检查](independent-check.json)：另用m=2、数据标准差2核对轨迹端点，PyTorch autograd计算输运方程残差约2.78e-17；独立保留错误初始分布的输出均值/方差。该参数组与正文分别记录。
- [作者函数调用检查](author-function-checks.json)：从固定源码AST提取one-step、多步采样与边界系数三个函数，用明确的mock预测器和噪声对象实际运行；主代理审阅后再次执行，one-step内部1次预测，多步steps=40/ts=(0,22,39)为2次预测。初始随机噪声由外层提供，记录的抽噪声次数只覆盖所测函数内部。

正文唯一NumPy块纳入固定教学白名单，实际输出保存在每轮quality日志。本轮没有训练学生、安装作者完整项目或运行预训练权重；解析函数求值数与真实神经网络速度/生成质量分别解释。CD/CT的网格极限主张按原文定理条件使用，逐点共噪声pair与概率流轨迹的差别有独立数值反例。
