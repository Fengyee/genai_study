# 命令记录与失败恢复验证

2026-11-10实际执行，基础Python3.12.14，Linux。记录器源码SHA-256为`be545eff1985b8eed096b0663145bade01303685430310d4b199aa727c685697`；当时HEAD为`efbc2c2`、记录器及集成文档尚在工作树，原JSON保留这一身份。

| 实际请求 | 状态 / 记录器退出码 / 子进程返回值 | 证据 |
| --- | --- | --- |
| 简单Python同时写stdout/stderr | succeeded / 0 / 0，wall 0.01547秒 | [run](success/run.json)、[完整日志](success/command.log) |
| 真实SVD入口读取缺失模型目录 | failed / 2 / 2，wall 0.06362秒 | [run](missing-video-model/run.json)、[完整日志](missing-video-model/command.log) |
| 真实SD1.4入口读取缺失模型目录 | failed / 2 / 2，wall 0.06366秒 | [run](missing-image-model/run.json)、[完整日志](missing-image-model/command.log) |
| 打印后等待10秒，设置0.2秒上限 | timed_out / 124 / -15，wall 0.25049秒 | [run](timeout/run.json)、[完整日志](timeout/command.log) |
| Python `-S`排除site-packages后运行记录器及子命令 | succeeded / 0 / 0 | [run](no-site/run.json)、[完整日志](no-site/command.log) |

[独立检查汇总](validation.json)记录实际结果；协调代理重新计算全部日志SHA，并核对JSON中的记录器源码SHA与执行时文件一致。两个缺失模型请求均在校验阶段失败，模型结果目录未创建，没有权重加载、推理或下载。[读者说明](../../../docs/run-records.md)中的无模型示例也按原命令实际完成，日志为`recording works`。

12项pytest另外覆盖已有记录保护、参数拒绝、literal argv、Git不可用、启动失败、TERM-resistant后代的超时清理，以及SIGINT/SIGTERM中断。独立审查还执行了子命令exit7、SIGKILL和抗TERM的直接子进程，确认原始返回值与记录器退出码分别保存。测试范围为Linux前台命令；宿主机中止、磁盘写入失败和记录器被SIGKILL仍可能留下`running`状态。

本目录逐字节复制原始JSON和小型日志；历史绝对路径指向当时环境。完整本地验证产物保留于忽略的`artifacts/run-record-audit`及`artifacts/run-record-demo`。此前SD1.4/SVD实测使用当时的外层记录脚本，保留原有身份，本轮没有改写其测量记录。
