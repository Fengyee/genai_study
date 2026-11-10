# 自动化验证记录

检查时间：2026-11-10（以下运行目录时间为 UTC）。检查对象是 `feat/reproducible-genai-curriculum` 分支的工作区文件，初始 HEAD 为 `cefe9c12b34a451d4ec9f7545d6ea011807434bf`；该 commit 只是空仓库初始化，**不能当作这些新增脚本已提交的版本号**。后续提交/推送/PR 状态由主执行记录维护。

## 配置与依赖一致性

已核对以下文件：`README.md`、`requirements.in`、`requirements-lock.txt`、`pyproject.toml`、`.github/workflows/quality.yml`、`scripts/quality.sh`、`scripts/run_agent_locked.sh`、`docs/automation.md`。

- README 与 CI 均使用 Python 3.12、`pip install --require-hashes -r requirements-lock.txt`，再用 `--no-build-isolation --no-deps -e .` 安装本地项目。
- 输入依赖与锁文件一致包含 `torch==2.6.0+cpu`、`numpy==2.1.3`、`matplotlib==3.9.3`、`pytest==8.3.4` 和 `setuptools==75.6.0`。锁文件保留 PyPI 与 PyTorch CPU 官方源。
- `pyproject.toml` 的构建后端要求 `setuptools==75.6.0`，已包含在锁文件中；项目声明 `torch==2.6.0` 允许符合 PEP 440 的本地版本 `2.6.0+cpu`，CPU 选择由锁文件固定。
- README 声明 Linux x86_64 CPU 目标，不把该锁文件宣称为 macOS/Windows 通用环境。锁文件包含其他平台 artifact 的 hash 不代表其他平台已验证。
- `artifacts/` 已被忽略；检查和代理适配器每次均创建新目录，因此不覆盖旧日志或把生成结果混入源文件。

这些是配置一致性检查。全新环境的安装验证与最终完整检查由主执行记录另行报告，不以本节代替实际安装。

## 已实际执行的验证

| 检查 | 观察结果 | 验证范围 |
| --- | --- | --- |
| `bash -n scripts/quality.sh scripts/run_agent_locked.sh` | 退出 0 | Bash 语法 |
| `check_docs.py` 临时文档夹 fixture | 通过 | 接受中文与重复标题锚点；忽略围栏/行内代码中的链接；拒绝不存在的文件、标题和越出仓库的路径 |
| 官方仓库 `git ls-remote` 查三个 Actions tag | 与 workflow 固定 SHA 完全匹配 | checkout v4.2.2、setup-python v5.6.0、upload-artifact v4.6.2 |
| actionlint 1.7.7 检查 `quality.yml` | 退出 0 | workflow 静态配置；二进制来自官方 GitHub release，下载后核对同一官方 release 的 SHA256 清单 |
| 同时启动两个代理适配器 | 首个正常退出 0，第二个返回 75；第二个命令未执行 | 同一 Git common directory 的协作式互斥 |
| 适配器设置 1 秒超时，执行 5 秒睡眠 | 返回 124 | 超时终止与退出码 |
| 适配器运行返回 42 的命令 | 返回 42 | 子命令故障码传递 |
| 超时后再启动成功命令 | 返回 0 | 锁正确释放 |
| `GENAI_PYTHON=/bin/false bash scripts/quality.sh` | 返回 1，四项均被运行并记录各自退出码 | 故意故障注入，证明检查失败未被 `tee` 吞掉；**不是模型/依赖回归** |

故意失败的 quality 运行在 `artifacts/quality/20261110T150503Z-zmB1cx/`，只保留于本地忽略目录，不假定 Git 克隆携带它。代理适配器测试记录位于 `artifacts/agent-runs/`。测试脚本使用临时文件作为首进程已取得锁并开始执行的标志，然后发起第二个进程，避免把启动竞态误判成互斥成功。

早期文档检查曾报告并行任务尚未写入的文档缺失；这属于建设中的检查点。最终全量 `bash scripts/quality.sh` 的真实结果需以完成材料后的最新执行记录为准。本节不宣称已有算法测试或训练质量通过。

## 实际部署状态与边界

检查时 `git ls-remote origin HEAD refs/heads/main` 两项均返回 `cefe9c12b34a451d4ec9f7545d6ea011807434bf`。远端 `main` 仍为初始化空提交，workflow 尚未进入默认分支，因此每小时调度**尚未部署**；没有已观察到的 GitHub Actions run URL 或 `schedule` 事件。

- 计划位置：GitHub Actions 的 `ubuntu-24.04` runner；不是当前 ChatGPT 云工作区里的 cron。
- 计划频率：每小时第 17 分钟 UTC；GitHub 排队可能延迟或丢弃运行。`cancel-in-progress: false` 不取消正在运行的 run，但新的 pending run 可替换旧 pending run。
- 失败记录：本地逐项日志、退出码和 JUnit；Actions 中始终尝试上传 artifact。依赖安装先失败时可能没有本地 artifact，但 Actions 安装日志仍可用于诊断。runner 异常终止时也不能保证 artifact 上传成功。
- 检查 workflow 无写仓库权限，也没有调用编码代理。`flock` 只是协作式锁：不取锁的进程仍可改文件；独立 clone 的 Git common directory 不同，不能凭此跨工作区互斥。
- 当前没有经过验证的代理调度入口、运行认证/预算或 Bark 绑定。适配器不自动补齐这些条件，也不发送通知；不能将文件存在解释成持续代理已经运行。

## 下一次恢复的具体操作

1. 完成并审核 PR 后，将 workflow 合入默认分支；核对 Actions 已启用。
2. 手动触发 **Quality checks**，保存实际 run URL、commit、开始/完成时间和结论。
3. 独立观察至少一次事件类型为 `schedule` 的 run；记录延迟及结果，不能用手动执行代替调度验证。
4. 自动编码前配置真实调度器的认证、费用/时长上限和通知绑定；每个修改入口统一走工作区锁。先使用无副作用命令验证调度、互斥、超时与失败通知，再接入一次一个问题的 PR 工作流。

依赖后续变更：checkpoint功能触发官方CVE检查，torch已从2.5.1升级2.6.0；以上依赖条目已同步。根代理对最终版本重跑质量检查，结果以PROGRESS最新检查点为准。
