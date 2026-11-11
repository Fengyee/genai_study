# SD1.4两线程颜色与布局实验

四提示×seed101/211的八条请求全部成功，驱动耗时2220.00秒（37.00分钟），过滤/技术失败/未执行均为0。完整原图、对象与颜色观察、分歧、成本和练习见[报告](../../../docs/sd14-attributes-layout.md)。两张单蓝立方体主项获一致确认；单红图保留身份分歧，四张关系图均为几何不可评。计数描述本批固定请求。

这是[原单线程八请求](../sd14-attributes-pilot/README.md)首条超时后的独立实验。[4步线程诊断](../sd14-thread-diagnostic/README.md)按预声明成本规则选择2线程，估计20步326.59秒、查询点RSS5.41GiB，满足门槛；决策未查看图像内容。原批1项超时/7项未执行、诊断两条结果与本批分别保留。

## 计划与实际执行

[计划](plan.json)在`0bc702c562f9dad44ad446d448ffcb1a30d0d995`提交并推送后启动。保留四提示、镜像顺序、20步DDIM、420秒/3600秒限制、首条wall≤360秒/RSS≤8GiB，以及全部观察规则。[预检](preflight.json)固定四份源码/锁/manifest、八条parser与16个新目录；[执行状态](execution.json)给出全部启动/结束时间、PID与退出状态。

本次历史执行命令如下；这些路径已有结果：

```bash
.venv/bin/python records/measurements/sd14-attributes-pilot/execute_plan.py \
  --plan records/measurements/sd14-attributes-followup/plan.json \
  --execution artifacts/sd14-attributes-followup-execution/execution.json
```

完整重做时，另存一份计划到仓库中可跟踪的新路径，把八条`output`、`record_output`及各自`argv`中的两个输出目录同步改成未占用路径，`--execution`也使用新的父目录和文件路径。保持模型、源码身份、预算、提示与次序等实验条件，**先将新计划提交到当前Git HEAD**，再调用驱动；驱动要求计划与HEAD中的字节相同，并会拒绝已有记录与结果目录。重新声明时记录新的计划时间/目的和完整预检，保留本次冻结计划。单条同配置命令可直接沿报告使用新路径与已有记录器。

## 数值与文件证据

| 证据 | 内容 |
| --- | --- |
| [final-summary.json](final-summary.json)、[cost-table.csv](cost-table.csv) | 八条退出、阶段时间、完整wall、查询点RSS与总预算 |
| [final.json](final.json)、[audit.py](audit.py) | 固定commit/源码、版本、实际配置、scheduler、20次batch2调用、token、PNG和噪声检查 |
| [partial-1903.json](partial-1903.json) | 19:04独立重hash全部16个缓存文件，共5,482,669,095字节 |
| [root-check.json](root-check.json) | 主协调另读八PNG/NPY/日志，核对复制字节和全部28个噪声对 |
| [cost-consistency.json](cost-consistency.json)、[process-exit-check.json](process-exit-check.json) | 成本与结束状态复查，记录的驱动/记录器PID均已退出 |
| [observations.json](observations.json) | 逐图合并结果、原始观察SHA、固定分母与未解决分歧 |
| [root-observations.json](root-observations.json)、[blind-observations.json](blind-observations.json) | 两位AI观察者独立原始记录；协调者先保存再读取匿名记录 |
| [rubric.json](rubric.json)、[anonymous-map.json](anonymous-map.json) | 生成前观察规则的匿名版与查看前固定的A–H映射 |

每个`request_id`对应本目录的`-metrics.json`、`-run.json`和`-command.txt`完整原始文件。PNG位于报告链接的`docs/assets/sd14-attributes-<request_id>.png`，与推理输出逐字节相同；原始浮点噪声在忽略的`artifacts/sd14-attributes-followup/<request_id>/initial_noise.npy`。28对中12个同seed对精确一致，16个异seed对不同。本批没有完整重复请求，跨线程诊断PNG也有小差异。

`audit.py`保留本次固定路径、计划commit与原始产物的核验方法；有完整本地产物时可传入一个尚不存在的`--output`文件，`--verify-assets`另做全缓存hash。新克隆可以核对已提交的证据与图像；复跑新计划后应让核验脚本显式指向新的计划身份和产物路径。

模型来源与生成图片许可沿用[SD1.4资产说明](../../../SOURCES.md)。两位观察者均为AI代码代理，真实读者标注仍待收集。目录内计划与原始JSON保留当时措辞和源码路径，当前实测状态以上文和报告为准。
