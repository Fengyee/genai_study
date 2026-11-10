# DDPM解析oracle实验记录

[报告与可执行命令](../../../docs/ddpm-oracle-report.md)解释标准正态精确方差递推、八高斯解析score、真实反向协方差的遗漏项，以及邻域率与分量离散程度的关系。

| 记录 | 身份与实际范围 |
| --- | --- |
| [计划](plan.json) | 1466bad先保存；选题来自短解析探索，正式预算固定为三个基础seed、每例4096点、100步、120秒上限 |
| [完整执行](execution.json) | 860e7bf的正式入口，exit0、外层wall1.423秒；初次preflight因最终docstring源码hash变化停止，核对已提交身份后才启动 |
| [正式metrics](metrics.json) | 全部六请求、各随机角色seed、100步schedule、解析方差路径、诊断、21数组身份及计时 |
| [独立推导](independent-derivation.json) | 主代理标量方差递推及高斯精度/全协方差核验；输入为实际float32噪声表常数，以float64复算 |
| [独立数组检查](array-check.json) | 全部21数组/SHA、初噪声、三组正态线性传播、协方差/邻域/coverage/径向二阶矩/energy重新计算 |
| [seed边界修复后检查](bounds-fix-check.json)及[对应metrics](bounds-fix-metrics.json) | 发现当前CPU的0与2^32种子产生相同初噪声，基础seed上限收紧为2^29-1；原三个seed在新目录重跑，21数组逐元素相同，NPZ逐字节相同 |
| [图片来源](figure.json) | 图片使用全部已声明seed；[绘图脚本](../../../scripts/plot_oracle_study.py)核对输入archive SHA并保护已有输出 |

原始arrays.npz位于忽略的artifacts/oracle-study-current；边界修复验证位于artifacts/oracle-study-bounds-check。克隆后按报告使用新目录重跑即可得到全部数组。正式指标对应原始运行版本，后续参数校验修复保留自己的源码与metrics身份，未将旧运行重新标记为新版本。

标准正态总体端点方差0.942097，三个经验方差0.939718/0.955174/0.980694均完整保留。八高斯邻域率约99.8%，R2为0.664–0.677，独立目标约0.994–1.008；这些结果共同显示分量内收缩。理论、有限样本测量与假设的含义见报告，真实图像/视频适用性需单独研究。
