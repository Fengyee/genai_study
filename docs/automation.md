# 自动检查与持续改进

## 本地检查

安装根目录 README 的固定依赖后，在仓库根目录执行：

```bash
bash scripts/quality.sh
```

脚本优先使用 `.venv/bin/python`；可通过 `GENAI_PYTHON=/absolute/path/to/python` 指定解释器。每次结果写入独立的 `artifacts/quality/<UTC 时间>-<随机后缀>/`，不覆盖旧记录，包括：

- 文档检查、pytest、两个20步生成smoke、视频时序反例、2步图像flow训练与checkpoint求解器对照、条件丢弃训练与引导对照的日志及各自退出码；
- pytest JUnit XML、实验图片/指标，以及本轮总退出码 `exit-status.txt`；
- 检查时间、Git commit、Python 版本。

每项最多 300 秒，超时退出码为 124；单项失败后继续后续检查。CI 另有 25 分钟作业总超时（含安装），达到总超时可能中断尚未完成的检查。任何一项失败，总退出码非零。短训练验证训练、采样、保存结果的操作路径；学习质量需按实验文档执行长训练并解释诊断结果。视频smoke运行合成序列反例；图像求解器smoke载入前一检查的checkpoint，对同一模型和初始噪声执行2步/4步Euler采样；引导smoke另训练空条件分支并比较强度0/1/2。解析性质由pytest覆盖。

`check_docs.py` 检查必要文档、Markdown 内联链接/图片、引用式链接定义、本地文件及 Markdown 标题锚点。忽略代码围栏和行内代码。它针对仓库当前 Markdown 子集，支持 ATX 标题（`#`）和 HTML `id`；不是完整 CommonMark 渲染器。它不访问外部网址，不验证来源事实、公式正确性、孤立的未定义引用标签或图示教学质量；这些仍需要人工审查。

## GitHub Actions 每小时检查

配置位于 [quality.yml](../.github/workflows/quality.yml)：

| 项目 | 配置 |
| --- | --- |
| 触发 | PR、推送到 `main`、手动、每小时第 17 分钟（UTC） |
| 运行位置 | GitHub 托管的 `ubuntu-24.04` runner，Python 3.12 |
| 范围 | 文档链接、已有测试、DDPM/flow训练smoke、视频时序诊断、图像训练/求解器、条件丢弃训练/引导对照 |
| 作业总超时 | 25 分钟（含安装） |
| 并发 | 同一 workflow 与 PR/ref 共用 concurrency group，已有运行不取消 |
| 权限 | `contents: read`，检查任务无提交或修改仓库权限 |
| 失败记录 | Actions 日志和 `quality-<run_id>-<attempt>` artifact，保留 14 天 |

定时任务仅在 workflow 已进入仓库**默认分支**且 GitHub Actions 已启用后生效。PR 中的文件不等于已部署定时任务。GitHub cron 可能延迟或丢弃高负载时段的运行，公共仓库长期无活动后也可能停用 schedule；它不是准点服务。正常计划时间为每小时 `:17 UTC`，无需换算宿主机时区。`cancel-in-progress: false` 保留正在执行的 run，但同一 concurrency group 最多保留一个 pending run，新到的 run 可以替换旧 pending run；它不是保证逐个执行的持久队列。

部署后，在仓库 Actions → **Quality checks** 中先手动执行一次，并记录 run URL、commit、开始/完成时间和结论；然后等待并确认一次 `schedule` 事件。这两类观察分别证明手动执行和调度触发，不能互相替代。安装失败时可能尚未产生 artifact，仍可查 Actions 的安装步骤日志。

官方依据：[触发 workflow](https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#schedule)、[并发控制](https://docs.github.com/en/actions/using-jobs/using-concurrency)、[保存构建产物](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts)。

## 自动编码任务与互斥

质量检查不启动编码代理。本次云环境尚未验证到可用的编码代理调度入口、认证、预算或 Bark 绑定，因此不将“每小时代理改进”描述为已配置。运行中的云会话或后台进程也不能证明持久调度成立。

在这些条件落实后，自动编码任务至少需要：

1. 以仓库为单位的调度并发上限 1；同一工作区的所有修改入口（包括人工触发的代理）共同持有同一个工作区锁。
2. 明确单次时长与费用上限、干净工作区检查、恢复执行记录、失败通知与失败重试上限。
3. 每次只解决一个有证据的问题，经验证后保存检查点，通过 PR 提交；禁止强推和绕过审查合并。
4. 保留退出码、运行 ID、分支/commit 和下次恢复动作；通知失败须在执行记录与聊天中明确报告。

GitHub Actions 的 `concurrency` 只约束对应 workflow 的 runs，**不能锁住其他平台的云工作区**。独立工作区可用 `flock` 加共享锁文件实施互斥，但必须由所有修改任务统一遵守；单独放一个未接入调度器的锁脚本不代表互斥已部署。当前仅部署方案写入仓库，具体运行观察应查执行记录。

仓库提供未接入调度器的 [run_agent_locked.sh](../scripts/run_agent_locked.sh) 适配器。用法为 `bash scripts/run_agent_locked.sh 1800 -- EXECUTABLE ARGUMENT...`，只接受命令 argv，不解析 shell 字符串。它在 Git common directory 的 `genai-agent.lock` 上获取非阻塞独占锁，再以指定秒数超时运行命令；所有修改任务必须通过这个入口，或主动持有同一锁。锁占用返回 75，超时通常返回 124，命令失败保留命令退出码（强制杀死时可能为 137）。仅配置检查 workflow 不会调用它。

每次适配器运行保留 `artifacts/agent-runs/` 中的开始时间、起始 commit、超时、输出与退出码；不记录 argv，也不会自动注入认证、检查费用或发送通知。调用方必须禁止输出凭据，事先配置预算，并在实际代理中执行恢复记录和 PR 工作流。`flock` 是协作式锁：绕过适配器且不取锁的进程仍然能够修改文件；独立 clone 的 Git common directory 也不共享这把锁。它无法防止其他平台或非合作进程写入。适配器的互斥/超时本地验证只证明脚本行为，不证明代理调度已经部署。

最近一次静态与本地故障验证见 [自动化验证记录](../records/automation-validation.md)。

图像训练检查使用2次更新、batch4、16张样本、2步Euler；随后从保存的checkpoint运行独立采样对照。默认1500步图形实验与固定权重实测另见07章及实测报告。

引导训练检查使用2次更新、batch8、condition dropout=0.5、seed7；随后比较0/1/2强度，每组2步Euler、16张配对图像。checkpoint记录实际见过的有/无条件训练样本数，引导入口要求两者均大于0；此固定smoke配置已验证包含两种训练样本。
