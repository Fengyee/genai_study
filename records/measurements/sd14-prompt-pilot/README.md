# SD1.4小屋提示探索：计划、全部结果与核验

动机见[上一轮实测](../../../docs/pretrained-image-report.md)：seed7的三个请求均未观察到明确小屋。本次在查看四张新图之前固定[plan.json](plan.json)，比较原场景提示与短小屋提示，各用seed19和43，其余模型/噪声网格/引导/精度保持。

提示删减同时改变语义内容、长度和场景关系。主要观察为小屋“清楚可见／模糊／未观察到”，定义见plan；过滤或技术失败单列，保存全部请求。计划在提交1135255中先保存，随后启动生成；单次420秒、总计1800秒上限。

四请求全部退出0，实际总耗时1074.13秒。每图均20次batch2调用，推理259.97–264.56秒，进程峰值5.34–5.44GiB。两个同seed噪声对逐元素相同，两个不同seed噪声不同；实际模型、软件、固定参数与计划匹配。四张的小屋都清楚可见，每条件2/2；进一步评价仍需更多提示、种子与独立读者。

| 文件 | 内容 |
| --- | --- |
| [plan.json](plan.json) | 生成前声明的请求、固定身份、预算和观察标准 |
| [execution.json](execution.json) | 实际执行顺序、UTC起始时间、外层耗时和退出码 |
| [cabin-seed19.json](cabin-seed19.json)、[scene-seed19.json](scene-seed19.json) | seed19配对原始metrics |
| [scene-seed43.json](scene-seed43.json)、[cabin-seed43.json](cabin-seed43.json) | seed43配对原始metrics |
| [observations.json](observations.json) | 匿名ID的AI视觉观察、协调代理复核范围和逐项结果 |
| [audit.json](audit.json) | 实际核验项目、完整PNG路径/hash与计时环境 |

全部原图、配对解释和复现命令见[实验报告](../../../docs/sd14-prompt-pilot.md)。图片复制后保持原字节；权重和初始噪声数组留在忽略的`artifacts/sd14-prompt-pilot`路径，数组hash保存在metrics。所有请求使用原有Safety checker，返回过滤标记false。本次未改动推理源码或依赖。
