# SD1.4颜色与左右关系：八请求预声明

上一组[小屋提示探索](../sd14-prompt-pilot/README.md)覆盖两提示、两种子。本次选择单对象颜色与双对象左右关系，运行前固定[计划](plan.json)、全部八个请求、资源门槛和观察规则。当前完成的是预检；生成结果与实际成本将在执行后另行保存，计划文件保持原样。

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
