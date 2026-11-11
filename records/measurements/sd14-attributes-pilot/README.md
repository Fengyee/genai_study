# SD1.4颜色与左右关系：八请求预声明

上一组[小屋提示探索](../sd14-prompt-pilot/README.md)覆盖两提示、两种子。本次选择单对象颜色与双对象左右关系，运行前固定[计划](plan.json)、全部八个请求、资源门槛和观察规则。实际执行在首请求420秒超时后按协议停止：1项超时、7项未执行，生成图片0张。原计划保持原样，完整结果见下文。

| 提示ID | 固定英文提示 | seed |
| --- | --- | --- |
| single_red_cube | a watercolor painting of a red cube, plain white background | 101、211 |
| single_blue_cube | a watercolor painting of a blue cube, plain white background | 101、211 |
| red_cube_left_blue_sphere | a watercolor painting of a red cube to the left of a blue sphere, plain white background | 101、211 |
| blue_sphere_left_red_cube | a watercolor painting of a blue sphere to the left of a red cube, plain white background | 101、211 |

固定SD1.4 revision133a221、CPU float32、512×512、DDIM20步、guidance7.5、空负提示、单线程与原检查器。seed101按表中顺序执行，seed211按逆序执行；同seed四条请求使用相同初始噪声。每次独立进程，模型/软件/时间网格不变；[来源预检](preflight.json)已逐个hash核验缓存16文件（5,482,669,095字节），[依赖预检](installed-lock-versions.json)核对43个版本与锁文件，[CLI预检](cli-preflight.json)包含pip check和真实参数解析结果。四提示token数为13/13/20/20，均低于77上限。

预算为每请求420秒、整个驱动过程3600秒，任一时刻最多一个请求。首请求技术失败或超时就停止后续；成功后还需外层wall≤360秒、模型进程峰值RSS≤8GiB且字段可用。RSS在PNG/NPY保存之后、metrics JSON写入之前查询，口径见计划。生成质量与过滤状态按原槽位保留，均不作为替换请求的依据。剩余预算不足430秒时不启动下一条。

历史同配置约4.5分钟/图，因此本次八图约36分钟是**未测量估计**。`run_record`保存实际wall、完整日志、退出/超时及源码身份，模型metrics另记阶段耗时和RSS。当前环境没有`/usr/bin/time`，沿用上述已有记录接口。

观察按完整512×512原图进行，逐项记录立方体/球体身份、颜色、数量、左右关系和不确定状态。关系仅在目标可唯一定位时评估；视觉估计的框与5%图宽中心间隔规则预先固定。计划由一位只收到匿名图片及描述规则的AI观察者和知情协调者分别检查，保存分歧；真实读者评价仍待开展。每条件仅两个seed，报告具体配对案例与计数。

复现时按计划中的argv选择新的整组结果与记录目录，保留已有输出。模型来源、许可和生成图像归属沿用[SD1.4资产说明](../../../SOURCES.md)。本次不安排八图完整重复；历史管线重复的证据与本批次需要重新核验的身份/数组条件在plan的repeat_policy中分别说明。

## 有界执行与恢复

[执行驱动](execute_plan.py)要求计划与当前Git提交逐字节一致，并在启动前核对源码与八组路径。每条结束后校验记录器、实际配置/版本/时间网格、PNG完整解码和SHA、初噪声及配对；状态按条原子写入。运行前的[7项执行测试](driver-tests.json)使用真实记录器与合成推理产物，[9项结果校验](driver-validator-tests.json)覆盖损坏与配置偏离；这些检查没有执行模型。

在仓库根目录、原计划的全部输出目录尚不存在时运行：

```bash
.venv/bin/python records/measurements/sd14-attributes-pilot/execute_plan.py \
  --plan records/measurements/sd14-attributes-pilot/plan.json \
  --execution artifacts/sd14-attributes-pilot-execution/execution.json
```

已开始后先读取execution.json及对应run.json，核对活跃PID和子进程；驱动会拒绝覆盖原记录。机器中断后需保留已运行槽位并制定明确的续作记录，当前驱动没有自动续跑模式。若cleanup_confirmed为false，须先检查并处理记录的进程，确认退出后才能安排后续工作。[恢复预检](recovery-preflight.json)记录本次重连时所有八条仍未启动。

## 本批实际结果

首请求外层wall为420.304991秒，记录器退出124、模型子进程返回-15；驱动总wall为420.498327秒。最后一条日志是第20步的前置hook，表示该步开始，后续完成状态无法据此确定。模型未保存PNG/NPY/metrics；阶段耗时和RSS字段均不可用，未进行图像评分。其余七项按首请求失败规则保留not_attempted，全部槽位见[execution.json](execution.json)。

[原始运行记录](single-red-cube-seed101-run.json)、[完整日志](single-red-cube-seed101-command.txt)与[独立核验](audit.json)确认退出/超时、日志SHA、全部产物目录缺失及驱动/记录器/模型进程均已退出。[硬件快照](recovery-hardware.json)显示恢复后CPU为Intel Xeon，历史SD1.4成本记录来自AMD EPYC；线程与配额保持，其他宿主因素未控制。

后续单独声明[4步线程成本诊断](../sd14-thread-diagnostic/README.md)，根据实际成本决定是否安排新的完整实验。本批八槽位及失败结论保持。
