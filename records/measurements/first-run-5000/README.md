# 推荐首次运行的5000步复查

2026-11-10在当前CPU环境重新执行Flow 5000、DDPM 5000及Flow固定checkpoint的10/25/100/200步求解器路径。沿用基础hash锁依赖，三个命令依次执行，均退出0；源码hash、完整argv与120秒单命令上限保存在[execution.json](execution.json)。

| 路径 | 原始结果 | 完整命令耗时 |
| --- | --- | ---: |
| Flow 5000，seed7 | [flow.json](flow.json) | 7.9989秒 |
| DDPM 5000，seed7 | [ddpm.json](ddpm.json) | 7.8955秒 |
| 固定Flow checkpoint求解器 | [solver.json](solver.json) | 2.7388秒 |

训练入口的内部计时分别5.4040/5.4036秒，包含训练、评估、采样和指标；外层完整时间还包括启动、导入、绘图及保存。宿主负载与缓存未控制，耗时按本次测量解释。历史12组基准JSON及其计时完整保留。

[validation.json](validation.json)记录两份训练与历史seed7结果的逐项比对：配置除输出路径外相同，损失、诊断、基线和版本相同；本地generated/reference/training_loss数组逐元素相同。新solver核对checkpoint SHA及5000步metadata，再从NPZ独立复算相邻方案的配对RMS。

完整数组、图、checkpoint和stdout位于忽略的`artifacts/first-run-5000-audit`目录。课程推荐预算与历史800步对照的复现方式见[二维实测报告](../../../docs/experiment-report.md)。
