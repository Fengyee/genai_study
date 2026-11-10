# 执行记录与恢复检查点

日期：2026-11-10。目标：可信、体系化、可复现的中文 GenAI 自学材料；读者是熟悉PyTorch的研究生，进阶方向图像/视频。1k star是愿景，不是承诺或已达成状态。

## 当前状态

恢复更新：用户已发布环境，并明确要求继续repo。新实例从已推送的 `3a6bd7d` 恢复，初始工作区干净。GitHub API与arXiv访问实测恢复，已创建 [PR #1](https://github.com/Fengyee/genai_study/pull/1)。环境已发布这一状态替代下文历史检查点的“未发布”；本轮不重复修改或发布环境配置。

恢复后的完整检查 `bash scripts/quality.sh` 退出0：26 tests passed（16.39s）、85本地链接0错误、DDPM/flow/video三条smoke成功；结果目录 `artifacts/quality/20261110T153021Z-IOcPXT`。这验证了新实例恢复后的开发能力。


已从空仓库建立课程、CPU实验、规范、依赖锁与自动检查。主分支只含空初始化提交 `cefe9c1`，用于建立PR比较基线；项目内容在 `feat/reproducible-genai-curriculum`。具体推送/PR状态在末尾最新检查点更新，不能由文件存在推断远端已运行。

没有用户已有源文件被覆盖。自然图像/视频模型、预训练权重运行、网站部署和持续编码调度尚未完成；本轮新增程序绘制图形上的小型条件图像生成器。二维模型默认800步显著欠拟合，已作为教学结果如实标注。

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

## 轮次5：补上可运行的视频时序评价反例

问题：视频部分仅有理论研究路线，读者无法在CPU上亲自观察帧分布与时序联合的差异。实施：移动高斯光点、同帧乱序、静态重复帧三个对照；完整帧拼图、原始数组、像素与质心诊断；新增06章、单位说明、练习答案和每小时检查入口。

验证：同帧乱序的frame-set MSE=0，平均图MSE约3.94e-35，但轨迹质心RMSE=11.553px；静态速度=0而轨迹RMSE=18.194px，不能把少动当任务成功。5项新测试覆盖帧多重集、时序误差、静态陷阱、fps单位、CLI实际输出。全量26项测试及三条实际CLI smoke通过。这是合成评价反例，不是视频生成模型。

## 恢复方式

1. `git status --short --branch`，保留未提交工作；`git log -5 --oneline` 与 `git ls-remote origin`核对实际本地/远端状态。
2. 激活 `.venv`，依赖变化时按README重新hash安装；运行 `bash scripts/quality.sh`，区分故障注入目录和真实质量运行。
3. 阅读BACKLOG，优先P0；网络配置应用后重试GitHub API而不是重复申请token。Git HTTPS认证已验证可读写，不等于API已可用。
4. 按PR交付，不强推、不直接合并；自动修改统一经 `scripts/run_agent_locked.sh`，该锁只约束合作入口。
5. 每轮更新本文件并保存提交；不要把未验证的调度、网站或环境草稿当成已发布能力。

## 原始环境的外部阻塞与沟通（历史）

运行实例的网络策略没有api.github.com/arxiv.org；已保存包含api.github.com、arxiv.org、export.arxiv.org的环境草稿，需用户在环境设置应用。GitHub API直接请求报Forbidden，arXiv CONNECT 403；Git HTTPS读写成功。

未发现Bark变量/secret或通知工具；因此不能发送所要求的Bark阻塞通知，也未声称发送成功。未发现编码代理调度入口/预算配置，环境无crontab。GitHub Actions配置需PR合并后才生效。平台Codex推送只能由平台提供的订阅/调度能力接入，当前没有可调用接口。

## 前五轮检查点（历史）

- `bash scripts/quality.sh`：退出0；21 passed（23.51秒）、文档链接0错误、两算法20更新smoke通过，日志目录 `artifacts/quality/20261110T151506Z-bTjSrl`。
- 随后仅更新报告与链接，`.venv/bin/python scripts/check_docs.py`：70个本地链接、0错误。
- 独立 `/tmp/genai-reader-venv`：最终2.6.0依赖hash安装、editable安装、pip check、800更新flow成功。
- 本轮检查未执行GPU/图像/视频模型，也未执行GitHub远端Actions。
- 检查点 `69ff658` 已推送至 `origin/feat/reproducible-genai-curriculum`；`gh pr create ... --body-file records/PR_BODY.md` 实际返回 `Post https://api.github.com/graphql: Forbidden`，尚无PR。
- 第5轮最终 `bash scripts/quality.sh`：退出0；26 passed（24.48秒）、文档0错误、DDPM/flow/video三条smoke通过。日志：`artifacts/quality/20261110T152101Z-mbVAni`。
- 随后的06章实测表与最终报告链接检查：85 links / 0 errors。`git diff --check`通过。
- 第5轮内容将作为独立提交推送同一功能分支；工作区状态应以恢复时的 `git status` 与 `git ls-remote` 为准。预期无未提交源码，忽略的artifacts为本地复现产物。
- 小时检查只在PR合入默认分支后激活；目前main仍为空初始化提交，未观察远端Actions执行。网站未部署，编码代理持久调度未部署，环境配置未发布。
- 下一轮：应用api.github.com/arXiv网络草稿后优先创建PR并核验原始文献；同时可设计CPU极小图像实验。Bark/Codex主动推送仍缺可用接口/绑定，不能声称将于会话外自动继续。

## 轮次6：恢复验证与补齐PR交付

证据：本地HEAD和远端feature分支均为3a6bd7d，main仍cefe9c1。GitHub API返回repo可读/写，arXiv DDPM摘要返回200；未重新申请或复制凭据。查询无既有PR后创建PR #1，状态open，base main。

Actions观察：`gh run list`为空，workflows API total_count=0；Actions permissions接口返回403 Resource not accessible by integration，因此无法推断Actions开关。PR创建成功不等于CI通过。每小时schedule仍需workflow进入默认分支；本轮未合并PR。

恢复后继续最高价值独立任务：07章CPU小图像flow生成、02/03原文核心公式核验、2025视频论文阅读卡。新代码实测与检查完成后分别保存提交到同一PR。自动通知和持续代理调度未出现可用工具或绑定，不能承诺会话外自动运行。

## 轮次7：CPU图像生成与读者路径

问题：只有二维点生成，缺少BCHW/空间网络/条件图像的可运行桥梁。实施：16×16程序圆盘/方形数据、119345参数的条件卷积flow、明确的模板粗判与多样性/裁剪诊断、07章和四条CLI质量检查。数据不下载，未引入新依赖。

执行器恢复诊断：复用旧代理的状态接口指向新实例，但实际shell仍在旧快照3a6bd7d，新增文件未共享。发现后全部旧代理停止写入，仅由一位协调者把允许的6个新/改文件提交到`transfer/resume-image-audit`（b01f278）；主代理在当前实例fetch/cherry-pick为2492448，并核对文件SHA。没有重置用户代码、强推或修改main。旧实例训练数值不作为当前环境证据。

当前实例重新执行1500更新、seed7/19/43：每次训练+评估33.03–33.76秒；粗几何通过率圆盘96.875%–100%、方形92.1875%–98.4375%。必须同时报告原始像素32.8%–46.2%越界及0.0060–0.0101平均裁剪变化；模板规则在裁剪后计算，不能声称原始分布完全拟合。固定原型可高通过率但无多样性，已有反例测试。完整JSON、哈希、图与限制见`docs/image-experiment-report.md`及`records/measurements/image-flow/`。

全量`bash scripts/quality.sh`退出0：31 passed（20.61秒），117本地链接0错误，DDPM/flow/video/image四条实际smoke通过。日志目录`artifacts/quality/20261110T154212Z-8LmAm0`。后续仅修改记录与PR描述时运行文档检查，不重复长训练。

## 轮次8：原始论文核验与2025阅读卡

已读取固定DDPM2006.11239v2、FlowMatching2210.02747v2 PDF，对照核心公式、训练/采样约定与适用条件。明确DDPM图像离散decoder未在toy中实现，以及FM原论文正sigma_min与本仓库零终点噪声极限的差别。新实例重新下载PDF，SHA与审计表相同；相关算法+图像19tests通过，最终31tests也通过。仅审计指定公式/段落，未声称全篇实验复现。

新增Wan2025定向阅读卡：2503.20314v2、固定官方代码commit和单一模型卡revision；核验VAE/token结构、时间采样、I2V接口、许可及评价范围。记录Table5与正文步数不一致，不擅自确定预算；作者显存/速度报告没有标为本仓库实测。未下载权重或调用外部生成模型。

## 轮次8后的远端检查点（历史）

- PR #1已OPEN：https://github.com/Fengyee/genai_study/pull/1 ，本轮全部修改持续推送此PR，不直接合并。
- 已观察旧HEAD3a6bd7d的Actions run42845432968 completed/success；恢复记录HEAD3ba066b的run42845617763也success。最新代码提交的检查状态应以PR checks为准，不能把旧SHA成功当新SHA成功。
- 每小时检查定义已包含图像smoke，但schedule仍需PR合入默认main；目前没有定时运行证据。网站未部署，持续编码调度/通知入口未配置；本轮没有重写或发布环境草稿。
- 下一项高价值工作：使图像实验支持固定权重/初始噪声的独立求解器分析，量化裁剪与积分误差；之后继续01/04/05主张级原文核验及真实读者自测反馈。自然图像与视频模型仍需算力/许可/评价协议。

## 轮次9：消除远端Actions的真实运行时弃用警告

f95d055对应GitHub run42847424682 completed/success，证明本轮图像/课程集成在远端也通过；但日志提示Node20 Action正被平台强制Node24。依据官方release与固定SHA元数据，更新checkout/setup-python/upload-artifact到声明Node24的稳定版本，并核对runner及上传归档兼容性。仅3行pin变化，未拓宽权限或改变调度触发。详见records/automation-validation.md；新提交的远端运行需在推送后观察。

维护观察：当前gh版本的`gh pr edit`因已退役Projects classic GraphQL字段失败，已通过官方REST PATCH /pulls/1成功更新相同PR标题/正文；没有创建重复PR或发送评论。环境设置保持用户发布状态，本轮没有重新应用旧草稿。

轮次9远端验证结果：提交2592b2a的 [GitHub run42848472192](https://github.com/Fengyee/genai_study/actions/runs/42848472192) 已completed/success（59秒），新Node24 Actions的安装、全部检查及artifact上传均成功，未再出现此前Node20强制迁移提示。随后这次提交仅补充验证记录；未修改代码、依赖或workflow。

## 轮次10：让图像积分实验可以独立复现

恢复证据：本地与远端feature均为519e924，PR #1为OPEN，main仍cefe9c1。已观察519e924对应run42848799642成功。当前实例e181d1bdf5e5；本轮由主代理持有Git common directory中的工作区锁，子任务按文件分工协作。

读者问题：图像训练缺少checkpoint，练习无法固定模型比较Euler步数；README与学习路线重复写同一输出目录，第二次执行会触发已有结果保护。实施：新增版本化图像推理checkpoint、共享安全载入与JSON元数据校验、显式初始噪声积分接口及image_solver_study；保存权重SHA、训练配置、固定条件/参考、逐步数诊断、原始/裁剪像素配对RMS与成本。新增图像训练→独立采样质量检查；07章和实验入口给出实际两步命令及可执行练习，说明如何复用或新建结果目录。按用户偏好精炼重复防御性措辞，并将写作约定加入AGENTS。

实际验证：

- 相关25项测试通过，包括旧二维checkpoint兼容、图像权重往返、旧torch版本前置拒绝、常速度/线性ODE解析递推与一阶收敛、输入与参数保持、CLI固定噪声和NumPy独立RMS复算。
- 改前/改后2更新实验：全部5个数组和7组非耗时旧指标精确相等；checkpoint恢复后重现原生成数组。
- 当前实例再次执行`image_flow --steps 1500 --seed 7/19/43`，输出至`artifacts/image-solver-current/train-seed<seed>`；训练/评估34.17、34.35、33.50秒。全部训练数组与历史三seed实测精确相等；checkpoint分别恢复原采样也精确相等。
- 每个checkpoint执行`image_solver_study --steps 10 20 40 80 --samples 128 --seed 123`，输出至同层`study-seed<seed>`。三模型共享噪声、标签和两组目标数据；相邻RMS均缩小，但seed7方形通过率20→40步从89.0625%降到87.5%。float64独立复算与JSON绝对差小于1e-7，裁剪RMS均不超过原始RMS。
- 读者按07章执行20更新训练→10/20/40/80步采样，并直接提取运行练习E代码，退出0。证据`/tmp/genai-reader-check-xum760_5`。
- 在包含下轮来源文档的整合工作区执行`bash scripts/quality.sh`：退出0，46 passed（24.08秒）、138本地链接0错误、DDPM/flow/video/image/image-solver五条实际CLI smoke通过。日志`artifacts/quality/20261110T160855Z-PqFywv`。`git diff --check`通过。

长期记录：`records/measurements/image-solver-study/`保存新训练与求解器JSON、代码/依赖哈希及逐seed验证结果，历史image-flow记录保持；`docs/image-experiment-report.md`给出表格、固定索引图、测量范围和读者命令。模型与全数组保留于忽略的artifacts，可按命令重建。

本轮检查点先保存图像代码/实践文档；01/04/05来源核验文档已完成、仍在工作区，下一轮记录后提交。功能分支尚待统一推送及对应新SHA的远端检查。

## 轮次11：补齐基础、架构和评价的原文证据

问题：01/04/05若干主张仍依赖概念解释，部分段落保留了早期arXiv访问失败状态；需要把公式和系统设计落实到可核对的原始证据。已读取10份固定版本PDF、重新核对2份官方README，分别建立generative-source-audit、image-video-source-audit、evaluation-source-audit，记录版本日期、PDF字节/SHA、页码/公式和具体核验范围。

- 01：核验ELBO/KL方向、GAN最优判别器/JS目标、Real NVP变量替换、CNF负散度；补支持集条件、非饱和损失与梯度练习、映射方向。独立高斯/离散求和/负尺度仿射/线性CNF检查最大残差8.9e-16；GAN logit梯度autograd误差0。Neural ODE v6修订了特定NFE实验报告，索引保留版本差异。
- 04：核验LDM重建/潜表示目标、CFG训练时随机丢弃条件、两种引导系数、DiT latent patch尺度。CFG原文一处“FID单调下降”与同页Table1/图4不一致，已记录具体列与数字；课程依据表格解释引导权衡。独立CFG向量代数残差2.78e-17，48倍表示数量与16倍注意力矩阵尺寸关系通过。SVD/CogVideoX本轮核验固定README接口，训练论文仍在backlog。
- 05：核验FID平方高斯距离、模型依赖的有限样本偏差、resize/压缩协议；补均值平方项偏差公式与同N排序反转手算例。非对易协方差两种平方根迹写法差4.44e-16，特征缩放a²关系通过。更新评价管线记录要求与当前来源状态。

验证：直接提取01审计中的代码块执行成功；`pytest -q tests/test_toy.py`为14 passed（1.26秒），评价定向性质3 passed；整合后的全量46项测试与五条CLI已在轮次10记录通过。最终仅新增来源索引/执行记录/PR描述，运行文档链接与diff检查，具体结果见本轮后续检查点。

## 最新检查点与恢复入口

- 图像实现已保存为194427a；本轮来源/记录保存后统一推送到现有PR #1：https://github.com/Fengyee/genai_study/pull/1 。远端新SHA与Actions结论在推送后核对。
- 所有原始实测JSON、源码哈希、图像和读者路径均已保存；artifacts模型/数组及/tmp的PDF和临时验证结果留作当前实例的补充证据。
- 每小时质量检查位于GitHub Actions，UTC每小时第17分，单项300秒、作业25分钟；main仍cefe9c1，schedule等待PR合入默认分支，尚无schedule执行记录。PR检查已实际运行；编码代理调度和Bark/Codex主动通知仍缺可调用入口/绑定。本轮没有发布环境配置或部署网站。
- 下一项直接执行：核验05的IS、precision/recall、FVD、KID来源与估计条件，补可手算反例；然后补SVD/CogVideoX训练论文和近年图像阅读卡。先核对本地/远端状态并持有协作锁，再从BACKLOG继续。
