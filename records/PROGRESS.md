# 执行记录与恢复检查点

日期：2026-11-10。目标：可信、体系化、可复现的中文 GenAI 自学材料；读者是熟悉PyTorch的研究生，进阶方向图像/视频。1k star是愿景，不是承诺或已达成状态。

## 当前状态

已从空仓库建立课程、CPU实验、规范、依赖锁与自动检查。主分支只含空初始化提交 `cefe9c1`，用于建立PR比较基线；项目内容在 `feat/reproducible-genai-curriculum`。具体推送/PR状态在末尾最新检查点更新，不能由文件存在推断远端已运行。

没有用户已有源文件被覆盖。实际图像/视频模型、预训练权重运行、网站部署和持续编码调度尚未完成。二维模型默认800步显著欠拟合，已作为教学结果如实标注。

## 轮次1：建立可学习、可运行的最小完整主线

问题：空仓库没有先修、内容、规范或实践。依据：本地无提交，GitHub `ls-remote`无refs。实施：学习路线/统一符号/00–04讲义，原创DDPM与flow实验，贡献指南、Apache-2.0、来源记录、PR模板、锁依赖、数学与CLI测试。

验证：原作者DDPM代码与Meta flow affine路径固定commit核对；DDPM双精度两种后验均值100×10点最大差1.88e-13；flow高斯条件方差恒等式通过。初版14项pytest全部通过。论文网站403限制记录在SOURCES.md。

## 轮次2：修正指标造成的错误学习印象

问题：800步时两方法模式覆盖均为8，但默认seed近中心比例仅23%–27%，图像仍是宽圆环。实施：增加独立target-target基线、二维半径3sigma概率 `1-exp(-9/2)`、coverage8但大部分样本错误的反例测试；自适应共用图轴显示离群点；文档明确MSE目标不同、不可跨方法排名。

验证：16项pytest通过；基线新增随机流不改变原训练/采样数值。完整3seeds×2budgets×2methods结果及配置见 `records/measurements/` 和 `docs/experiment-report.md`，不推断模型优劣。

## 轮次3：复现、评价和维护闭环

问题：只在现成环境成功不足以证明读者能运行；缺少评价协议与长期质量检查。实施：从新建venv执行README的hash安装与editable安装，新增评价章与论文模板，Actions每小时检查、超时、失败artifact、协作互斥适配器。

验证范围：干净venv依赖安装与`pip check`通过；读者默认flow命令成功。自动化静态检查/故障注入证据见 `records/automation-validation.md`；全量最新检查记录在下面。GitHub Actions真实执行未观察到。

## 轮次4：把求解器练习变成可执行的受控实验

问题：训练不保存权重，无法用独立命令冻结同一模型比较积分步数。实施：推理checkpoint（不含优化器状态）、显式weights_only载入、solver_study记录权重SHA与固定初始噪声，对10/25/100/200步作配对比较。检查官方CVE-2025-32434公告后，将torch从2.5.1升级至2.6.0+cpu；旧版本loader拒绝读取的回归测试已加入。

验证：最终21项测试全部通过；训练→保存→载入→求解器完整路径成功。新依赖下重跑全部12组benchmark和干净读者环境，records/measurements记录的都是2.6.0实测，旧2.5.1仅在本地artifacts保留；没有重贴版本标签。求解器终点差随细分减小，但目标邻域比例不单调改善，已记录这一教学反例。

## 恢复方式

1. `git status --short --branch`，保留未提交工作；`git log -5 --oneline` 与 `git ls-remote origin`核对实际本地/远端状态。
2. 激活 `.venv`，依赖变化时按README重新hash安装；运行 `bash scripts/quality.sh`，区分故障注入目录和真实质量运行。
3. 阅读BACKLOG，优先P0；网络配置应用后重试GitHub API而不是重复申请token。Git HTTPS认证已验证可读写，不等于API已可用。
4. 按PR交付，不强推、不直接合并；自动修改统一经 `scripts/run_agent_locked.sh`，该锁只约束合作入口。
5. 每轮更新本文件并保存提交；不要把未验证的调度、网站或环境草稿当成已发布能力。

## 外部阻塞与沟通

运行实例的网络策略没有api.github.com/arxiv.org；已保存包含api.github.com、arxiv.org、export.arxiv.org的环境草稿，需用户在环境设置应用。GitHub API直接请求报Forbidden，arXiv CONNECT 403；Git HTTPS读写成功。

未发现Bark变量/secret或通知工具；因此不能发送所要求的Bark阻塞通知，也未声称发送成功。未发现编码代理调度入口/预算配置，环境无crontab。GitHub Actions配置需PR合并后才生效。平台Codex推送只能由平台提供的订阅/调度能力接入，当前没有可调用接口。

## 最新检查点

- `bash scripts/quality.sh`：退出0；21 passed（23.51秒）、文档链接0错误、两算法20更新smoke通过，日志目录 `artifacts/quality/20261110T151506Z-bTjSrl`。
- 随后仅更新报告与链接，`.venv/bin/python scripts/check_docs.py`：70个本地链接、0错误。
- 独立 `/tmp/genai-reader-venv`：最终2.6.0依赖hash安装、editable安装、pip check、800更新flow成功。
- 本轮检查未执行GPU/图像/视频模型，也未执行GitHub远端Actions。
- 已完成源码/文档检查点，正在推送功能分支；最终远端与PR状态将在下一条记录确认。
