# 恢复后CPU实例的首次运行核验

2026-11-11，当前CPU型号为Intel Xeon Platinum8573C，历史SD1.4硬件记录为AMD EPYC9V45。按README重新运行Flow5000、DDPM5000与新Flow checkpoint的10/25/100/200步求解器，seed和所有有效配置保持，仅使用新的输出目录。三条wall为19.43、18.09、6.78秒，均退出0；历史文件未覆盖。

两训练参考数组逐元素相同，生成数组与训练权重存在浮点差异。Flow/DDPM生成坐标的最大绝对差分别4.21e-5、6.96e-5，RMS分别1.86e-6、2.05e-6；模式邻域比例仍为0.9248046875、0.9189453125，覆盖模式均为8。初始/最终留出损失的记录值相同，逐步训练损失数组有差异；各字段分别核对。

观察到差异后，另行保存补充计划，追加一次原历史Flow checkpoint的求解器控制，原checkpoint SHA与metadata完全保持，仅替换输出目录。该命令wall6.45秒，初噪声、reference与baseline数组完全一致；10/25/100/200步输出的最大差分别2.50e-6、4.95e-6、6.32e-6、7.27e-6。它将权重固定下来，仍保留实例/后端与loader实现变化的解释范围。

四命令wall合计50.75秒；从最初开始到追加控制退出162.64秒，低于600秒总预算，每命令180秒、读者命令并发1。各实验线程1，运行期间另有SD1.4两线程与文档/浏览器工作；缓存和宿主负载未控制。单次实例比较支持当前读者路径可运行及实际浮点差异的记录。

独立源码核对确认experiment/toy/solver与基础lock和历史bcdeae8逐字节一致；checkpoint新增finite float32验证和CPU/float32重建，有效默认路径的数值计算保持。历史13个文件的SHA全部保持。NumPy float64另从全部10个新结果重算诊断，计数/比例与记录精确一致，energy和最近中心距离的最大差分别5.40e-7、4.18e-7。

summary.json给出简明结论；plan/execution保留最初三请求；fixed-checkpoint-plan明确追加时间顺序；comparison与fixed-checkpoint-comparison分别保存新训练链和旧权重控制的逐数组/逐tensor差异；source-scope给源码依据，independent-diagnostics给独立复算。四份metrics/run JSON完整保存。driver及审查脚本按当时路径原样留存，供核查方法；重新执行时应先准备新的整组路径，保留这些既有产物。原始数组、checkpoint、图片与完整command.log在artifacts/recovered-cpu-reader-path下。
