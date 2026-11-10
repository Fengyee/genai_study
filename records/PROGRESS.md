# 执行记录与恢复检查点

日期：2026-11-10。目标：可信、体系化、可复现的中文 GenAI 自学材料；读者是熟悉PyTorch的研究生，进阶方向图像/视频。1k star作为长期建设目标，优先通过读者体验和材料质量推进。

## 当前状态

恢复更新：用户已发布环境，并明确要求继续repo。新实例从已推送的 `3a6bd7d` 恢复，初始工作区干净。GitHub API与arXiv访问实测恢复，已创建 [PR #1](https://github.com/Fengyee/genai_study/pull/1)。环境已发布这一状态替代下文历史检查点的“未发布”；本轮不重复修改或发布环境配置。

恢复后的完整检查 `bash scripts/quality.sh` 退出0：26 tests passed（16.39s）、85本地链接0错误、DDPM/flow/video三条smoke成功；结果目录 `artifacts/quality/20261110T153021Z-IOcPXT`。这验证了新实例恢复后的开发能力。


已从空仓库建立课程、CPU实验、规范、依赖锁与自动检查。主分支只含空初始化提交 `cefe9c1`，用于建立PR比较基线；项目内容在 `feat/reproducible-genai-curriculum`。具体推送/PR状态在末尾最新检查点更新，不能由文件存在推断远端已运行。

没有用户已有源文件被覆盖。课程现有11单元，基础CPU实验、可选SD1.4图像推理及SVD低分辨率视频运行均有实际证据；SVD完整输出严重失真，详见末尾检查点。发布尺寸视频质量基线、网站部署和持续编码调度继续排队。二维首次推荐显式5000步，保留CLI缺省800步的欠拟合对照；SD1.4首轮对象缺失与后续四请求主体可见的结果均完整记录。

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

## 轮次11检查点

- 图像实现194427a、来源审计0a4feab均已推送到PR #1：https://github.com/Fengyee/genai_study/pull/1 。新SHA推送后未观察到自动PR检查，workflow API显示active；显式触发当前feature后，[run42852732071](https://github.com/Fengyee/genai_study/actions/runs/42852732071) 在0a4feab完成success，事件为workflow_dispatch，安装、全部检查、artifact上传均成功。该观察确认手动检查路径；每小时schedule仍待默认分支合入。
- 所有原始实测JSON、源码哈希、图像和读者路径均已保存；artifacts模型/数组及/tmp的PDF和临时验证结果留作当前实例的补充证据。
- 每小时质量检查位于GitHub Actions，UTC每小时第17分，单项300秒、作业25分钟；main仍cefe9c1，schedule等待PR合入默认分支，尚无schedule执行记录。PR检查已实际运行；编码代理调度和Bark/Codex主动通知仍缺可调用入口/绑定。本轮没有发布环境配置或部署网站。
- 下一项直接执行：核验05的IS、precision/recall、FVD、KID来源与估计条件，补可手算反例；然后补SVD/CogVideoX训练论文和近年图像阅读卡。先核对本地/远端状态并持有协作锁，再从BACKLOG继续。

## 轮次12：修正DDPM终点噪声的必要条件表述

全路径复查发现02章写“只有当alpha_bar_T足够小时终点才接近标准正态”，与同章练习D的标准正态数据反例冲突。依据：独立高斯线性组合在任意alpha_bar下均为标准正态；一般数据的终点均值/协方差依赖其原始分布及尺度。

修改：将其改为削弱一般数据信号的设计方式，补有限二阶矩条件并链接标准正态练习；将笼统噪声表反例具体化为本仓库八高斯。其协方差为2.0064I；100步beta从0.0001到0.02时，独立NumPy计算alpha_bar=0.3635632481、终点协方差=1.3658900528I，每坐标超出标准正态36.589%。来源审计追加这次教学表述修订。

验证：独立高斯解析推导与四个alpha_bar值的协方差检查通过；`pytest -q tests/test_toy.py`为14 passed（0.79秒）。未改变算法、共享符号或依赖。文档链接及diff检查随本轮提交执行。

全路径复查还验证三组图像raw/clipped RMS与05 coverage-trap原代码，并发现跨入口视频/solver目录复用遗漏；下一轮直接修复这些命令提示。IS/PR/FVD/KID原文核验正并行实施，未完成文件保留在工作区直到验证后提交。

## 轮次13：消除跨章节命令的重复执行卡点

问题与证据：学习路线→06→04重复使用`artifacts/video-counterexample`；实际首次执行退出0，第二次退出2并报告非空目录。03→examples的`flow-solver`、07→examples的`image-solver`也重复，已有checkpoint的提示尚未说明如何复用已完成的采样。

实施：将首次执行、复用结果、换目录重做的说明放到各命令前；02/03也明确复用README训练及对应checkpoint路径。保持现有输出保护和算法行为。独立只读复查覆盖README、学习路线、00–07、examples及相关实现，DDPM/flow时间方向、二维/图像RMS归约一致；05原代码的coverage-trap得到8模式和12.5%邻域比例，三组图像原始数组RMS复算通过。

验证：触发重复目录故障的实际CLI已确认；本轮仅改文档，执行本地链接与diff检查。无需重复训练或安装。

## 轮次14：把其余评价指标写成可核对、可运行的定义

问题：05章IS/precision-recall仅有概念解释，KID只列在来源中，FVD缺具体特征层和输入协议。已读取四篇固定原文及作者FVD固定源码，新增evaluation-metrics-audit，并细化05章的数学定义与条件。

成果：IS由KL展开成互信息并给出1到类别数的范围；近邻球明确排除自身、两组分别构造，再交叉计覆盖。KID写出三次核及独立IID条件下的无偏U统计量，明确有限样本可负。FVD核对Kinetics-400 I3D logits、BTHWC、224空间缩放、像素范围及batch=16；BAIR的16帧、KTH的20帧各归属具体任务协议。

实际验证：直接从05章提取两段Python代码执行，原coverage例通过；新NumPy例得到IS=3、U估计=-3.5。两组各2个Bernoulli特征的全部16种观测，U均值=0，裁剪为非负后均值=0.875。直接KL与熵形式差1.11e-16；手算近邻precision=.75、recall=1及k单调性核对通过。`pytest -q tests/test_toy.py tests/test_video_diagnostics.py`为19 passed（2.14秒）；子任务文档147链接0错误，diff检查通过。最终仅补来源/记录入口后再查文档。

所有代码修改在轮次10已完成全量46项测试与五条实际CLI；随后轮次11–14为数学、教材及来源文档改动，各自执行独立推导/代码块和相关检查。数学修正e7339b7与目录复用ba79a0d已在本地保存，待本轮一起推送。最新已观察的远端成功为0a4feab的workflow_dispatch run42852732071；新SHA将在推送后实际触发并观察。

下一轮直接任务：CPU小图像CFG训练与固定模型引导实验。先设计空条件分支及checkpoint兼容，独立验证参数约定/NFE，再测量训练和引导强度对几何/裁剪的影响。SVD/CogVideoX全文阅读及近年图像进展继续排队。当前无新增权限需求；每小时schedule仍等待PR进入默认分支，环境配置与网站部署状态保持此前记录。

轮次14交付检查点：e7339b7、ba79a0d、f3e1ab8均已推送至PR #1，正文已同步。最终本地文档148链接0错误，diff检查通过。f3e1ab8对应[手动run42854304402](https://github.com/Fengyee/genai_study/actions/runs/42854304402)和[自动PR run42854514413](https://github.com/Fengyee/genai_study/actions/runs/42854514413)均completed/success；自动PR事件已实际观察到。每小时schedule仍无执行记录。

## 轮次15：CPU图像CFG实践

已在f3e1ab8检查点后启动。设计：现有两个条件通道以全零表示空条件，保持网络与checkpoint架构；训练新增独立随机流的condition dropout，默认0保留既有数值路径。引导对照读取已训练空条件的checkpoint，固定噪声并为圆盘/方形配对，比较速度组合v_u+s(v_c-v_u)，每步显式计算两分支并记录2K NFE。普通条件模型缺少空条件训练证据时拒绝该对照。

实施完成：独立mask使用seed+6随机流，空条件/有条件留出损失与实际训练计数分别保存；引导CLI要求元数据证明两类样本均参与训练，强度列表包含1。s0/1直接返回对应分支，调用预算统一2K。配对噪声、原始数组、四类RMS、几何/裁剪、NFE和计时全部留存。根代理审阅发现默认浮点:g格式会使1与1.0000001覆盖同一NPZ键，已改为可往返的.17g标识和显式array_key，实际CLI回归通过。

教学：新增08章、原始概率路径推导、可执行配对检查与密度反例，连入README/学习路线/04/07/examples；课程现为9单元。两位独立数学审查确认总体回归与独立丢弃条件；精确高斯速度组合的终点方差为1/4，而对应幂乘积密度为4/7，说明确定终点密度需要整条路径依据。将loss归约写为d_img，避免类别与通道数符号重载。

实际测量与验证：

- 相关33项测试通过，修复浮点键后额外运行已有CLI回归并通过；默认p0短训练全部5数组/7指标组与前版精确相等。
- 当前实例完整重训p0、seed7、1500更新：同样精确保持旧实验全部5数组与7组非耗时指标；现有checkpoint架构/schema及依赖保持。
- p=.2、1500更新、seeds7/19/43：训练/评估34.22、33.61、34.12秒；实际空条件计数分别19114、19169、19258（各96000训练样本）。每个模型运行0/1/2/4强度、40Euler步、128张输出（64独立噪声对），每组80NFE、1.23–1.34秒。
- 三模型s0配对逐元素相等，s1与普通条件积分器逐元素相等；所有模型共享噪声/标签/两组目标。四类RMS由float64独立复算，与JSON绝对差<1e-7，裁剪满足非扩张性。
- 几何实测：seed19方形粗通过率从s2的98.4375%降到s4的93.75%；seed43圆盘从95.3125%降到90.625%。类别响应、几何与裁剪变化各自报告，见docs/image-guidance-report.md。原始JSON/哈希/验证见records/measurements/image-guidance。
- 逐字提取08练习C执行：20更新/batch16/p.2的短训练→4步采样成功，有/无条件计数259/61；s0/s1/RMS/NFE检查通过。高斯反例数值方差0.25000000001562506与4/7对应解析结果。证据/tmp/genai-cfg-reader-5n9ztb1n。
- `bash scripts/quality.sh`退出0：54 passed（34.11秒）、167链接0错误、七条实际CLI smoke通过；日志artifacts/quality/20261110T164153Z-3OssNw。新检查包括条件丢弃训练和引导扫描，固定smoke计数6/10。最后仅补记录和来源入口，再运行文档/diff检查。

## 最新恢复入口

轮次15实现与实测已保存为d6105a1并推送到现有PR #1，正文同步为9单元课程。最终本地169链接0错误，diff检查通过；[自动PR run42858071120](https://github.com/Fengyee/genai_study/actions/runs/42858071120)在d6105a1完成success。主代理继续持有协作锁，图像代码与文档子任务均已冻结。

下一项：固定版本的近年图像系统阅读卡，优先核验FLUX.1 Kontext报告/代码/模型卡，连接08的条件机制与真实图像编辑；SVD/CogVideoX全文和自回归接口随后推进。每小时质量检查仍在PR中等待合入默认main，尚无schedule事件；编码代理持久调度与Bark/Codex主动通知仍缺可调用接口/绑定。环境配置未重新发布，网站未部署。当前无新增权限需求。

## 轮次16：近年图像编辑系统阅读卡

已完成FLUX.1 Kontext定向核验并接入README、学习路线、08章与SOURCES：2506.15742v2（2025-06-24，19页）、官方代码802fb471、单一dev模型卡revision24e9dedc及其Non-Commercial License v1.1.1均固定，PDF/模型卡/许可哈希记录于卡片。

重点发现：论文时间1噪声→0数据、目标epsilon−x，对应本仓库tau=1−t和速度取负；Kontext-dev使用guidance distillation，固定官方采样器每区间一次调用，即K NFE，guidance值作为embedding输入。参考图token固定拼接，图像组位置轴与flow时间分开。CLI实际签名与专门编辑文档采用30步/2.5，同版通用docstring的4/50文字已作为接口差异记录。

教学与范围：token布局与打包、pro/max/dev访问方式、编辑完成与内容保留、108基础图像上的1026提示对分组、多轮编辑偏差、代码/权重/输出许可及GB/GiB存储口径均逐项说明。主审修正一处对照表，将数据明确为圆盘/方形两类；空条件属于网络输入状态。未下载外部权重、未执行GPU推理或服务调用。

验证：从文档逐字提取CPU代码块执行，latent=(1,16,128,128)→tokens=(1,4096,64)，两图8192个视觉tokens，pack/unpack精确还原；两时间约定的Euler更新与解析结果一致。相关`pytest -q tests/test_image_guidance.py`为8 passed（7.48秒）；子任务174链接0错误。整合导航与主审修订后再次执行链接/diff检查。

下一项：SVD/CogVideoX训练论文、自回归训练/采样接口和2026代表工作；并先评估预训练图像实践的许可、真实硬件与内存成本。轮次15代码与54项全量检查已经在d6105a1远端成功，本轮仅改教学/来源文档，提交后核对新SHA的检查状态。

## 轮次16最终交付与恢复检查点

- 阅读卡与导航提交9e089bc已推送PR #1，PR正文已同步；本地179链接0错误、diff检查通过。[自动PR run42860490726](https://github.com/Fengyee/genai_study/actions/runs/42860490726)在9e089bc完成success。
- 下一步资源证据（只读，2026-11-10）：当前torch为CPU构建，CUDA不可用；/dev中无nvidia设备，/proc/driver/nvidia/gpus不存在。cgroup memory.max=34359738368（32GiB），cpu.max=400000/100000（约4核配额），进程affinity为5个CPU；当时/workspace空闲59.49GiB。可据此评估小型预训练图像CPU推理，实际权重大小、峰值内存和运行时间仍需选定模型后核对。
- 本次新增源代码、课程、实测、来源和图像均已保存并推送；这次后续提交只补交付/资源记录与backlog。下一轮恢复时先核对git status、HEAD/远端与PR checks，重新取得协作锁，再按BACKLOG推进。
- main仍为初始化cefe9c1。每小时质量检查定义在PR中，实际位置为GitHub Actions，UTC每小时:17；自动PR和手动事件已验证，尚无schedule事件。环境未重新发布，网站未部署，持续编码调度与Bark/Codex主动通知缺可调用入口/绑定。最新工具搜索亦未发现相应入口。

## 轮次17：补齐SVD与CogVideoX训练证据

恢复核对：本地与远端feature均为6931a9c、main仍cefe9c1，工作区起始干净；6931a9c对应自动PR run42861179002已completed/success。已重新取得协作写锁，继续自主迭代。

问题：04章此前仅核对SVD/CogVideoX公开README，缺训练研究与具体权重之间的连接。新增对照阅读卡，固定SVD 2311.15127v1、CogVideoX 2408.06072v3、官方代码与三个模型revision；记录数据筛选、阶段训练、VP预条件、条件路径、时间压缩、许可和评价协议。CogVideoX论文Table6与发布2B配置的heads/time embedding/精度差异，以及后续论文能力与早期权重接口的差异均保留。

验证：主代理逐字执行卡片CPU块，VP还原、零SNR端点、帧轴引导、49→13潜帧与17550视觉tokens通过；独立核对81帧外推注意力元素数比例2.5842515845。重新读取两PDF计算SHA，与卡片相符；SVD Table1的577M/152M单位核对为clips。相关视频测试5 passed（1.48秒）。卡片已接入README、学习路线、04及SOURCES，文档链接和diff检查通过。

并行工作未提交：SD1.4可选CPU实践已固定原作者revision、许可和5.106GiB模型文件；独立venv的43项hash锁依赖安装、pip check、pipeline导入通过，原有24项依赖版本保持。完整权重下载和真实推理继续进行；预训练相关新文件暂存工作区，按后续轮次验证后提交。自回归teacher forcing/因果掩码补充正在实施。环境未重新发布、网站未部署、小时schedule仍等待默认分支合入。

## 轮次18：修复uv备用安装入口

问题：README建议`uv venv --python 3.12 .venv`，后续却直接使用`python -m pip`；uv默认创建的环境未安装pip。实际创建两个临时环境对照：无seed的`python -m pip --version`退出1、提示No module named pip；加入`--seed`后退出0，提供pip26.3/Python3.12。证据保留于/tmp/genai-uv-pip-rnzgonrf/result.json。

修改仅为README备用命令添加`--seed`并说明带pip；标准python venv入口与hash依赖安装保持原协议。此修订覆盖实际首次运行故障，未改变项目算法或依赖。文档/diff检查通过后单独保存检查点。SVD阅读卡已保存a36698a，预训练权重下载与自回归材料仍继续处理。

## 轮次19：自回归输入与因果性

问题：01章已有链式分解，缺训练输入、标签、因果掩码与具体图像接口。固定PixelRNN 1601.06759v3（其中也提出PixelCNN），补充像素/RGB顺序、256类输出及nat/bit每维损失，解释教师强制的链式似然、Mask A/B与Transformer输入移位。

验证：主代理逐字提取01及审计的Python块执行。三bit因果模型枚举八种结果，总质量0.9999999999999999；并行/逐位置NLL最大差0，改变当前/未来bit后对应预测不变。读取自身标签的错误分类器伪联合质量为5.832，独立解析为8×0.9³。原有ELBO/GAN/变量替换解析块继续通过（最大恒等式残差8.9e-16）。主代理重算固定PDF SHA及3015028字节，与审计匹配。仅修改教学与来源文档，算法代码和全局符号保持；链接/diff检查随提交执行。

下一项继续SD1.4实践：官方prepare已完整校验16文件、5482669095字节，耗时289.69秒（含2.77秒校验）；实际256×256/2步smoke成功，推理10.93秒、进程峰值约5.32GiB，完整512×512/20步已启动。以最终JSON为准确测量依据，后续补对照、报告和全量检查。

## 轮次20：固定预训练权重的真实CPU图像实践

问题：课程已有二维与程序图形训练，缺从真实文本条件、潜表示、预训练网络到自然图像的完整可运行路径。选用原作者CompVis SD1.4的公开revision133a221b8aa7292a167afc5127cb63fb5005638b，核验模型卡与CreativeML Open RAIL-M原文，固定16文件及逐文件大小/SHA，总计5482669095字节（5.1061GiB）。四组float32 safetensors包含原有Safety checker。

实施：新增可选pretrained_image的prepare/run，按manifest校验配置、tokenizer与全部权重，损坏文件强制重取、已验证文件复用；CPU本地加载、DDIM eta0、显式双分支batch、初始噪声保存、实际调用计数、分段耗时及进程峰值内存记录。保持已有输出保护，并在长采样中输出稀疏进度。requirements-pretrained锁43项依赖，原有24项版本全部保持；独立新venv的hash安装、editable安装、pip check和pipeline导入通过。

教学：新增09章和实测报告，连接README/学习路线/04/examples；核对DDIM原文v4、公式(12)/(16)、零基数组与末步alpha边界，解释空文本条件、77位置、latent缩放与20次batch2调用。全路径只读复查修正README的--samples适用范围、03下一步导航及长推理状态提示。基础质量检查覆盖本地完整性与CLI测试，可选大权重下载保留为显式实验。

实际运行与验证：

- 完整prepare校验16文件，289.69秒；最终入口重跑复用16文件、下载调用0次、6.09秒。初步双连接HTTP试下载中止后改用Hub下载器，未把中止阶段计入此计时；其部分临时文件在完整SHA确认后清理。
- 256×256/2步smoke：推理10.93秒，进程峰值5.32GiB；完整512×512/20步、seed7、threads1：s7.5推理261.52秒/5.38GiB，s3推理262.49秒/5.38GiB。
- 默认完整重复：259.22秒/5.46GiB，初始噪声逐元素相等、PNG逐字节相等；两强度权重、噪声、实际scheduler参数保持，归一化RGB RMS差0.071988229。内部默认参数名称列表按集合比较，原始JSON保留。
- 根据实际缺失项追加“前景小屋”提示，保持seed与采样参数：262.02秒/5.35GiB，24token未截断，同一初始噪声；最终入口实际输出6条进度消息。所有完整运行均20次UNet调用、每次batch2，即40次单样本分支求值。全部五次运行使用原有检查器，过滤标记均false。
- 已逐张查看三份不同请求的全部图片；水彩湖景、针叶树、白天可见，小屋仍未观察到，额外边框保留。默认重复对应同一PNG。报告记录这些具体观察，扩展提示与seed的评价列入backlog。
- 09原文代码逐字执行，独立NumPy DDIM首/末步与库实现最大差1.24e-7/1.70e-7。69项测试通过（31.74秒）、7条基础CLI smoke全部成功，完整quality退出0，日志artifacts/quality/20261110T173958Z-SW6MRz；后续仅补报告/来源/记录，再查文档与diff。

原始JSON、软件/源文件hash和验证范围见records/measurements/pretrained-sd14；三张PNG保留于docs/assets且与运行输出相同。模型权重、独立venv与原始噪声保留在忽略目录。1fafe03已推送PR #1且自动run42864845461为success；本轮实现提交后再观察新SHA。ResTok 2026单文件卡已经子任务核验并保留在工作区，下一轮整合；LTX-2侦察材料位于/tmp/genai-ltx2-scout。

运行位置和外部状态：小时质量workflow在GitHub Actions的PR中，main仍cefe9c1，尚无schedule事件；编码代理持久调度及Bark/Codex通知未发现可调用绑定。本轮仅发布仓库feature提交，环境未重新发布，网站未部署。主代理继续持有协作锁。

## 轮次21：连接2026离散图像生成研究

问题：当前近年阅读卡侧重扩散/flow，缺2026首发工作与01自回归因果性的连接。新增ResTok定向阅读卡，固定2601.03955v1（2026-01-07）、官方commit7811bf23及两个模型revision；讲解tokenizer/生成器、128索引/9轮采样、分组条件独立、分类CFG和rFID/gFID协议。论文与发布代码的池化、残差、对齐margin、VQ损失差异，以及README中torch2.6.0/torchvision0.17.2精确依赖冲突均保留证据。

验证：两位代码审查核对训练输入/标签与采样路径，prefill1+NTP3+HAR5=9次调用；所选CFG前两轮1.0001、后七轮3.75，全程batch2。主代理逐字执行CPU块，128token/9轮/1664索引bits通过；固定前缀下两个相关bit的组内独立模型额外NLL为log2、错误组合概率0.5，联合仍归一化。另有分类CFG幂乘积恒等式独立检查通过。主代理重新计算9304728字节PDF SHA，与卡片一致。

本轮只增加教学卡和导航，未下载ResTok权重、未更改算法或依赖。文档已接入README、学习路线、01、04与SOURCES，提交前执行链接/diff检查。SD1.4实现14038b2已推送PR #1，自动run42868818164实际启动；需继续观察结论。LTX-2后续卡正在核验，当前仍持有写锁并推进独立工作。

## 轮次22：2026音画联合模型的条件与调用核验

新增LTX-2阅读卡，固定2601.03233v1、首发同期commit7179f0d与模型/许可revision。用链式法则澄清顺序条件生成的联合表达能力；介绍双流、三类时间坐标、noise→data递减sigma、预测量及论文跨模态CFG与实际文本引导入口的版本范围。

主代理逐字执行三段CPU块：顺序条件模型声画错配0、独立边缘模型0.5；121帧/512×768/24fps得到6144视频tokens与126音频tokens；解析网格8+3=11区间，速度取负映射与终点检查通过。重新核对3032343字节PDF SHA及许可原文，与卡片一致。独立从官方源码AST提取原函数并执行小张量计数：蒸馏11次联合forward，普通40区间CFG1/4为40/80，普通两阶段默认83；实际视频/音频预测在同一次联合forward返回。相关视频与引导测试13 passed（8.89秒）。

来源差异保留：README第二阶段“4 steps”对应4端点/3区间；论文结论13B+3B与主体14B+5B不同；H100的18倍速度属于指定单步测量；43.29GB主文件和manual-gated Gemma仅核验元数据，未下载。代码与模型的许可按版本分别定位。阅读卡已连接README、路线、04、SOURCES，后续文档检查覆盖新增入口。

远端观察：14038b2自动run42868818164 completed/success，日志核实69 passed（37.73秒）、7条smoke、230链接0错误；ResTok提交e55cb4b已推送，run42869229837也completed/success。

在途实验检查点：四请求提示探索计划先保存为1135255（本地提交），再启动/tmp/genai-run-prompt-pilot.py。准确协议位于records/measurements/sd14-prompt-pilot/plan.json；执行器session9444，逐请求日志与execution.json位于artifacts/sd14-prompt-pilot，总上限1800秒。恢复时先查看实际进程/文件，保留成功、过滤、超时或失败结果，不重跑覆盖。术语索引与CPU教学块自动化另由协调子任务推进，当前写锁仍由主代理持有。

## 轮次23：给新读者提供术语到实践的定位入口

问题：00–09及研究卡的概念逐渐增多，读者缺从英文术语快速返回定义/例子的入口。新增35组中英术语索引，按概率训练、表示、条件、预测量、采样和评价分组，每行链接到具体标题；README、学习路线及符号表加入导航。

验证：两次术语复审核对epsilon条件期望包含时间/条件、VP v还原、flow速度、连续与离散token、CFG蒸馏、调用/batch/分支计数及评价定义；主代理再对照实际toy coverage与章节定义，确认索引范围。全局符号、单位与算法保持原约定，符号表仅新增导航。提交前运行全部本地链接/diff检查；图示可访问性保留为独立后续任务。

LTX-2卡67f3ec2与实验计划1135255正在推送；四请求探索继续按既定协议运行，尚未查看新生成图片或改变请求。教学代码块自动化正实施，当前质量基线仍为69测试与7条CLI smoke通过。

## 轮次24：让教材中的CPU反例持续执行

问题：章节与阅读卡中的解析代码此前逐次手动核验，后续文档改动缺执行回归。新增check_lesson_examples.py，以固定ID/文件白名单和紧邻围栏的HTML标记运行10个现有块；覆盖01、05、生成模型审计、Kontext、SVD/CogVideoX、ResTok及LTX-2。只新增标记，数学代码保持；08唯一Python块依赖训练产物，09需要可选库/模型配置，继续由对应实践验证。

检查器采用当前解释器，在独立进程中逐块运行并保留文档行号、stdout/stderr及状态；默认30秒/块，quality入口20秒/块，外层300秒保持。缺失、重复、未知ID、不支持的shell包装、代码非零或超时均失败，继续处理其余允许块。当前白名单无下载、权重、训练或写文件；贡献与自动化文档写明范围及新增规则。

验证：10个真实代码块全通过（独立运行约2.4秒）；8个独立CLI故障fixture涵盖baseline、missing、duplicate、unknown且不执行、拒绝shell、runtime失败后继续、0.2秒超时后继续/保留输出、.venv heredoc，均符合预期，证据/tmp/genai-lesson-review-us6ma7ns。主代理审阅实现后执行完整quality：10/10教学块、69 passed（32.31秒）、7条CLI smoke及307链接0错误，退出0；日志artifacts/quality/20261110T180918Z-YFx3BU。后续只补文档与记录，再查链接/diff。

当前进行中：四请求pilot前两项cabin/scene seed19均退出0，各约267秒，另外两项继续原计划。尚未查看四张新图或据结果改提示。图示替代说明审查已完成只读阶段，7个Mermaid与9张PNG有明确修订建议；随后单独实施。

## 轮次25：让图示具有完整的文字替代路径

问题：7个Mermaid此前缺少可访问标题/描述，学习依赖主要通过图表达，实验任务也要求必须看图。为全部7图添加accTitle/accDescr，在学习路线加入完整文字依赖，将二维任务改为结合散点图或文字说明与诊断表。既有9张PNG均有描述性alt，关键数值与观察在邻近正文/表格中可读；贡献规范加入后续图示要求。

验证：对照Mermaid官方accessibility语法，用临时Node24.22.1、Mermaid11.12.2和jsdom26.1.0实际解析7图，从diagram.db取回的标题/描述与文档逐字一致，另审阅其与图中路径及正文的含义。证据/tmp/genai-mermaid-accessibility，包括固定依赖锁、官方页面、检查脚本和JSON。此次未执行浏览器布局或屏幕阅读器测试；未改变数学/算法，文档和diff检查随提交执行。

恢复核对：2632875已推送，PR #1仍OPEN/可合并，对应自动PR run42872874791为completed/success；67f3ec2对应run42871500536也成功。四请求pilot已全部退出0，总耗时1074.13秒；已经按预先固定协议完成配置/噪声核验与全部图片观察，原始记录和报告正在单独保存为下一轮。主代理继续持有原协作锁，环境未重新发布。

## 轮次26：完成预先声明的提示与种子探索

问题：首轮seed7三种请求均未观察到明确小屋，需要按固定集合扩大观察。计划1135255先于生成保存，再按cabin19、scene19、scene43、cabin43执行；四次全部退出0，每次低于420秒，驱动程序总1074.13秒低于1800秒预算。未替换请求、删除失败或据中途图片改变配置。

全部生成后检查原图。协调AI代理和仅收到匿名图片/标准的独立上下文AI代理均将四张小屋判为清楚可见，每提示2/2；原场景两张均有湖与针叶树，短提示主体更大，部分建筑出画，材质/小门窗按可辨范围记录。未观察到额外边框。真实读者标注尚未开展；四张支持具体案例比较，当前没有短提示出现率优势的证据。报告连接首轮失败，明确删减同时改变语义、长度和关系。

验证：四组manifest/模型revision/依赖锁/入口源码hash均与计划一致，实际固定参数、软件和scheduler匹配；内部默认参数名列表按排序比较，保留原始JSON。两个同seed初始噪声对逐元素相等，19/43噪声不同；四图均20次U-Net调用、batch2/40分支，Safety checker启用且标记false。复制原图逐字节SHA一致。推理259.97–264.56秒，峰值5.34–5.44GiB；进程总时间和外层命令时间分别保存。

原始JSON、完整图片、观察身份与核验范围见docs/sd14-prompt-pilot.md及records/measurements/sd14-prompt-pilot。相关pretrained测试15 passed（0.11秒），329本地链接0错误、diff检查通过；本轮无算法/依赖变更，未重复完整质量脚本。最近完整质量基线仍为69测试、10教学块、7条CLI smoke，且2632875远端检查成功。

下一轮正在实施首次运行预算改善：依据已有12组实测将推荐预算改5000，保持800历史对照及CLI缺省，核对跨章checkpoint复用并实际重跑读者路径。真实视频CPU实践先进行小文件/元数据可行性审查，未启动新大权重下载。可访问性提交bcdeae8与本轮结果随后推送同一PR；环境、网站与持久调度状态保持。

## 轮次27：让首次运行获得可诊断的八簇结果

问题：入口800步严重欠拟合，而已有12组实测证明5000步能把约92%–93%的点送入模式邻域，成本适合CPU首次实践。README、路线、02/03和examples共8条推荐训练命令改为显式5000；CLI缺省800保持并说明，保留原欠拟合图和指标。跨章复用先检查metrics中的训练预算，新预算选择新目录。历史800步solver数据现在给出独立训练/采样命令，避免读者用新checkpoint期望旧数字。

验证：在当前实例顺序执行两条5000步训练和固定flow checkpoint求解器，外层wall分别7.9989、7.8955、2.7388秒，全部退出0。两训练损失、诊断、基线、版本和原始generated/reference/training_loss数组与原seed7记录相同；flow邻域92.4805%，DDPM91.8945%。新solver实际读取5000步metadata，独立从NPZ复算三个配对RMS约0.06054043/0.02760249/0.00499758。来源hash、完整argv、两类计时及验证JSON归档到records/measurements/first-run-5000。

相关toy/experiment/checkpoint测试21 passed（14.26秒），文档与diff检查通过；未修改算法、依赖和共享约定。两份历史5000步图继续使用原有逐字节输出，历史12组成本不被新计时覆盖。主代理复核所有源码hash、三份metrics SHA、退出码和预算。

交付：bcb056d已推送且PR正文已更新，自动PR run42875297816实际启动，结论随后核对。本轮随后单独提交。下一项为12题跨章自测及错题回读路径；真实SVD CPU实践已发现匿名可读原作者固定权重，仍在核对许可、尺寸和预算，尚未下载。工作区锁实际非阻塞探测为占用，由当前协调会话继续持有。

## 轮次28：把跨章节自测连接到下一步学习

问题：各章已有练习，但读者缺少统一入口判断概念迁移是否掌握。新增12题课程自测，覆盖密度/单位、因果输入、DDPM还原、条件期望、求解器、覆盖反例、裁剪与多样性、条件丢弃、CFG调用、视频顺序、提示配对与研究协议。每题有可检查产物、隐藏提示/答案、通过判据及精确回读链接；末尾按常见错误组织复习路径，接入README与学习路线。

验证：两次文字/公式审查后，主代理再次执行/tmp/genai-self-assessment-check.py。独立解析与实际diagnostics验证密度缩放、5.832标签泄漏质量、DDPM(1,-2)还原、联合高斯最小MSE2、Euler解析收敛、coverage8但邻域16%、原始RMS0.5/裁剪0、空条件0.1/0.9先验及视频3px/s/一步错误方向。直接核对08/09实际调用口径和预训练配对记录；文档/链接/diff检查随提交执行。尚无真实学习者的耗时或通过率测量。

交付与在途工作：5b0de97已推送；bcb056d对应自动PR run42875297816已completed/success。下一轮SVD固定模型为9cf024d5bfa8f56622af86c884f26a52f6676f2e，11文件总9018094887字节；小文件、匿名可达性、F32 safetensors头及Community2024-07-05许可已核验。manifest/许可/NOTICE暂存examples/pretrained-svd，完整权重由/tmp/genai-prepare-svd.py在1200秒上限内下载，session51980，日志/tmp/genai-prepare-svd.log，目标artifacts/models/svd；恢复时先查状态并复用已核验文件。新入口/测试与10章草稿由分工代理实施，尚未声称视频运行成功。环境未重新发布，网站与持久编码调度保持未部署。

## 轮次29：固定视频实践资产与可验证入口

在任何SVD模型调用前，先保存并推送2ed6b0c中的11文件manifest、原文许可、NOTICE及分阶段计划。固定输入为已有SD1.4 scene/seed43完整图，记录原图SHA，采用EXIF转正/RGB/中心裁剪/Lanczos预处理；视频初噪声seed7与参考图增强seed8独立，保留144×256尺寸、帧轴引导1→2.5、fps输入7/内部6及完整时间解码。

实现pretrained_video独立prepare/run，复用现有模型无关校验辅助函数，未修改共享源码或依赖。入口记录参考图裁剪、单位视频初噪声、实际VAE增强输入、实际joint-video形状/调用、scheduler网格、分段耗时、进程峰值、全部float32后处理帧及逐帧PNG。新增23项故障/性质检查覆盖固定身份、损坏修复、路径逃逸、EXIF/裁剪、双随机流、静止帧差反例、参数及无site-packages帮助/输出保护；两次API只读复核与固定Diffusers实现相符。

实际准备：临时bootstrap下载11文件，逐项及结束校验通过，264.66秒，完整字节9018094887。最终仓库prepare入口重新运行：复用11文件、下载调用0、9.51秒（含两轮校验），记录artifacts/svd-current/prepare*.json。完整quality退出0：92 passed（31.64秒）、10/10教学块、7条CLI smoke、397链接0错误；日志artifacts/quality/20261110T183902Z-oYPMQH。独立NumPy对比真实Euler首末步也通过，首步最大差1.84e-5、末步9.31e-8。

当前仅完成资产、入口与无权重/基础检查。接下来记录此入口的源码hash/commit，按已提交plan执行4帧2步smoke（600秒）、14帧4步校准（900秒），再按预设时间/内存阈值决定25步完整低分辨率运行（1800秒）。全部请求及失败状态保存到artifacts/svd-current；10章草稿未提交，待实际结果整合。最新已观察5b0de97远端run42876490960成功；环境状态查询确认当前实例running、网络enforced且无secret/通知绑定，未更改发布配置。

## 轮次30：集中首次入口与原始证据导航

问题：README把多份前沿阅读卡放在首次命令之前，研究路线简表也省略了评价/实践先修；测量目录已有多组JSON，却缺少统一定位。新增首次运行/完整路线/自测/证据快捷入口，把5份研究卡移入首次运行后的专题表，修正04入口为按完整路线完成基础、评价与实践。新增records/measurements/README.md，按实验问题连接报告、原始记录、源码与资产身份，并说明本地数组/checkpoint/权重的取得方式。

验证：只读审查逐条核对课程依赖和索引中的真实文件；文档/diff检查通过。尝试headless Chromium读取GitHub上固定a3ed4a8的三个页面时，均遇ERR_CERT_AUTHORITY_INVALID；保留TLS校验，没有改系统信任或环境策略。相同运行环境的urllib使用已配置CA读取README返回200。桌面/窄屏GitHub布局、远端Mermaid元数据与details交互因此尚无实际验证证据，未记为通过；本地视频HTTP播放另有成功记录。

SVD在途检查点：已推送a3ed4a8，对应自动PR run42877869115为completed/success。4帧2步smoke完整wall47.14秒，推理38.37秒，峰值9.54GiB；14帧4步校准wall203.80秒，推理194.41秒，峰值11.29GiB。两个阶段实际联合输入/分支数、独立重建的初噪声、相同参考增强、图像裁剪和帧hash均通过。全部PNG已导出MP4，ffprobe核实7fps/4或14帧，ffmpeg完整解码退出0；Chromium156从本地HTTP完整播放，分别4/14帧且无丢帧。file协议被浏览器管理员策略阻止后使用本地HTTP测试，未更改策略。

按计划，校准wall×25/4=1273.77秒且峰值小于24GiB，进入25步完整低分辨率请求。决定先写入artifacts/svd-current/full-decision.json，再于2026-11-10T18:46:35Z启动，session99654，硬上限1800秒；日志full-low-resolution.log，成功输出目录full。源commit和hash在preflight.json中固定。恢复时先查此进程与文件，不重复启动或覆盖；完成后还需smoke-repeat、全帧与播放核验、报告/10章整合和按PR交付。已查看smoke与校准的完整拼图：前者模糊且变暗，后者色彩/形状大幅变化；这些运行继续按接口/成本校准解释并完整保留。

## 轮次31：完成SVD CPU全阶段运行与失败分析

完整14帧/25步请求于2026-11-10T18:46:35Z启动，exit0，外层1007.30秒、推理998.69秒、峰值11.21GiB，符合事前1800秒上限；随后4帧/2步完整重复exit0、wall46.53秒。全部四项计划请求完成，没有替换或删除输出。协调代理查看smoke/校准/full全部帧，完整结果出现严重过饱和、闪烁与主体形变；新10章、视频报告和全路径导航按这一结果编写。14帧4步/25步后处理通道边界比例分别8.59%/40.48%，统计明确基于已经裁剪的浮点帧。分辨率、输入域和采样行为是后续研究因素，具体成因待控制实验。

追加事后诊断：独立原生StableVideoDiffusionPipeline直接调用，省去仓库推理函数与hook，4帧/2步全部浮点帧逐元素相同、差异0，四张PNG逐字节相同；wall47.29秒。实际脚本逐字节归档到records/measurements/pretrained-svd/direct-reference.py并核对JSON中的SHA。smoke重复也有完全相同的初噪声、增强输入、浮点帧与PNG；校准/full的14帧初噪声与增强配对相同。独立复算所有输出hash、PNG量化、边界比例/RMS，三个ZIP各包含完整4/14/14帧和input。

媒体：三份MP4均为7fps，FFmpeg7.1.5完整解码通过；ffprobe逐帧时间戳/时长符合i/7及F/7，Chromium156经临时本地HTTP完整播放到结束，解码4/14/14帧、丢帧0。所有原帧ZIP、接触图、MP4及阶段/重复/API/边界统计JSON已保存。单独只读代理再次核对实际数组、媒体、来源、CLI默认值和条件路径，并实际执行10章Euler首末步代码，最大差1.83574e-5/9.31307e-8；其独立全帧观察与失败描述一致。

验证：本轮没有改模型实现/依赖，相关图像和视频pytest共38 passed（0.92秒），文档539本地链接0错误，git diff --check通过。8个Mermaid用Mermaid11.12.2实际解析并取回全部accTitle/accDescr，元数据一致；未验证屏幕阅读器或GitHub页面布局。最近完整本地质量基线仍为轮次29的92测试/10教学块/7CLI smoke。c0df6e2远端run42880369394已completed/success。

检查点：本轮成果单独提交后随同下一轮推送PR #1；未提交的diffusion-flow-bridge阅读材料进入轮次32，继续补齐概率流ODE与DDIM/SVD坐标关系并接入CPU教学检查。当前无运行中的模型任务；权重、数组和完整日志位于忽略的artifacts/svd-current及artifacts/models/svd。环境未重新发布，网站/编码代理调度/通知绑定未部署；每小时Actions定义仍待PR进入默认分支才生效。
