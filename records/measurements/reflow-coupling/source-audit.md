# Reflow：固定来源与范围核对

本记录来自2026-11-11 UTC的只读来源准备，随专题一并归档。未训练、未安装依赖、未下载权重或运行任何生成模型。PDF/源码仅经正常 HTTPS、继承代理与 CA 获取；每个网络请求有超时。完整下载身份见 `source-identity.json`。

## 固定原始来源

- Liu, Xingchao; Gong, Chengyue; Liu, Qiang. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. **arXiv:2209.03003v1**，提交时间 **2022-09-07 08:59:55 UTC**，41 页。abstract 当前列出的实际版本为 v1；下载 URL `https://arxiv.org/pdf/2209.03003v1`。
- PDF **18,024,983 bytes**；SHA-256 **d81790cb6988ef480002adcc84000d4f44855b11fb868001fd53aad133bc3a74**。PDF页码与本次 text 页分割一致。
- 作者代码：`gnobitab/RectifiedFlow`，固定 **5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6**，commit 日期 **2024-06-20T09:39:05Z**。该版本 README 开头明确自称该论文 official implementation。它是后续作者快照，不把它称作 2022 原始发布版本。
- 固定仓库 tree 未发现根 LICENSE/COPYING；若归档源码，应按具体文件现有声明处理，勿给整仓库臆定 MIT 许可。本文 CPU 例为独立推导与实现。部分配置文件带 Google Research Apache 2.0 头部；`sde_lib.py` 无此头部。

## PDF 精确定位与成立条件

| 位置 | 核对内容 | 教学表达建议 |
| --- | --- | --- |
| 第4页式(1)，第6页式(2)、算法1 | 对线性插值回归端点差；精确最优速度是条件期望；算法明确先训练流、生成其端点，再以生成端点递归训练 | 每一轮网络仍预测瞬时速度；更新的是训练耦合 |
| 第12页定义3.1；第13页定义3.2、式(9) | 路径逐样本 C1，条件速度定义于边缘支持集；rectifiable 要求条件速度局部有界，积分 ODE 解存在且唯一 | 卡片可采用更易检查的充分条件：连续时间、空间全局Lipschitz线性场与有限二阶矩；两个解析例均满足 |
| 第13页定理3.3、式(10)–(11) | 单次 rectification 的 `Law(Z_t)=Law(X_t)`；证明用测试函数连续性方程及解唯一性 | 保持的是同一输入耦合所定义的每时刻边缘。下一轮新耦合会定义新的中间边缘 |
| 第13页定理3.5，第14页 Jensen 证明 | canonical constant-speed linear interpolation、rectifiable，任意凸位移函数 `c:R^d→R`，`E c(Z1−Z0)≤E c(X1−X0)` | 明确 `c` 的输入是位移；教学版可要求相关期望有限以使等式链与积分交换良定义。神经拟合与数值积分另产生误差 |
| 第7页式(3)；第14页定义/定理3.6式(12) | `S(Z)=∫E||dotZ_t−(Z1−Z0)||²dt`；S=0为时间仿射轨迹。`V(coupling)=∫E||ΔX−E[ΔX|Xt]||²dt`；直耦合是 Rectify 固定点 | 非交叉指同一时刻的路径唯一；不同时间经过同一位置不涉及 ODE 唯一性 |
| 第15页定理3.7与紧随其后的望远镜证明 | 每一轮耦合均rectifiable；`Σ_{k=0}^K [S(Z^{k+1})+V(π_k)]≤E||X1−X0||²` | 有限初始二阶矩给best-iterate/平均量O(1/K)。不增大主卡范围，不声称每轮S单调、不声明固定两轮普遍一步精确 |
| 第15–17页§3.4，特别第16页定理3.8 | 严格凸成本最优的rectifiable耦合必为直耦合；d≥2直耦合一般不推出给定成本全局最优 | 用二维90°旋转耦合独立验证反方向失败 |
| 第8页 distillation 段；第6页算法1 | 蒸馏拟合既有端点映射；reflow利用端点耦合重新训练流 | 与 MeanFlow 的两时间平均速度接口分工清楚 |

已按协调代理要求渲染并查看原PDF第15页，确认式(13)左边两项实际均漏写平方，而前文指定 `c=||.||²`、下一行递推式均带平方。定理3.7求和中使用 `S(Z^{k+1})+V(π^k)`，其后best-iterate文字改用 `S(Z^k)+V(π^k)`，索引也不一致。教学稿以独立正交分解得到平方成本恒等式，使用相邻索引形式 `Σ_{k=0}^{K−1}[S(Z^{k+1})+V(π^k)]≤C(π^0)`。原PDF页面渲染留存为 `paper-page15.png`；未查看任何生成图像。

## 作者实现核对（固定commit）

固定链接前缀：`https://github.com/gnobitab/RectifiedFlow/blob/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6/`。

1. `ImageGeneration/run_lib_reflow.py:69–80` 载入已有模型与EMA，以先前权重开始本次训练。
2. 同文件 `141–165` 的 `generate_data_from_z0`：拷入 EMA 参数，抽 `z0`，调用 `sde.ode(z0,score_model)` 得到 `z1`，按同样顺序保存 `z0.npy/z1.npy`。
3. 同文件 `171–195` 载入成对数组；`208–221` 对两数组使用**同一 indices**抽样，交给 `batch=[z0,data]` 的训练步。独立重排两个数组会破坏耦合。
4. `ImageGeneration/losses.py:55–133` 中 `reflow_flag=True` 读取两端；`93–105` 在 `t∼Uniform([eps,1))` 下形成 `t*batch+(1−t)*z0`，监督 `batch−z0`；`108` 模型时间输入为 `t*999`；`112–114,128–130` 采用L2并按特征/批量取均值。`eps` 默认 **1e−3**。变量名 `score` 此处承载速度输出。
5. `configs/rectified_flow/cifar10_rf_gaussian_reflow_train.py` 显式 `reflow_type='train_reflow'`, `reflow_t_schedule='uniform'`, `reflow_loss='l2'`，并指定先前checkpoint与配对数据目录。
6. `ImageGeneration/sde_lib.py:38–72` 的实际离线配对求解器为 SciPy **RK45**，`rtol=atol=1e−5`，正向区间 **[1e−3,1]**，网络求值状态转float32，输出float32。这里是作者实用近似，与解析例的精确闭区间 `[0,1]` 需分别表述。
7. 同一训练驱动 `212–214` 还支持在线配对，固定教师用20步Euler。主卡只讲最简离线流程，不扩展在线质量或成本。
8. `losses.py:89–90` 的 `t0` 蒸馏在代码取eps，并可用LPIPS。保持短卡在uniform+L2 reflow范围即可。

## 教学与核验范围

03第6C–D节已经推导独立Gaussian边缘速度及正回归残差。阅读卡直接引用已有速度，用端点ODE解展示**第二次耦合更新**：第一流始末恒等但中间收缩/扩张，第二流为零场。再用旋转固定点回答“已直为何仍非全局OT”。

MeanFlow 卡研究给定轨迹的两时间平均速度、JVP、停止目标梯度及一次预测。本卡继续使用03的 `t:0源→1目标` 和瞬时速度 `v(x,t)`；核心操作是生成配对与再训练，没有MeanFlow的两时间输入/JVP目标。

专题的三个目标：
- 写出一轮 rectification 和再做一次 reflow 所用的端点对，并说明同一轮/跨轮的边缘关系。
- 用已知Gaussian解核对一次Euler、精确流和第二轮零场；明确精确代数特例与学习近似。
- 计算二维旋转耦合成本4及恒等耦合成本0，解释时间仿射与全局二次OT的区别。

实际核验覆盖解析流与作者纯损失函数；图像模型训练、生成质量和实用加速比需要独立实验记录。

## 实际执行作者纯损失函数

`check_author_loss.py` 先显式校验 losses.py 的固定 SHA，再用 AST 原样抽取 `get_rectified_flow_loss_fn`（55–133行），注入已安装 torch 与 model_utils stub，在CPU、float32、1线程、seed1729运行4个1×2×3小张量。验证同pair插值精确相等、标签=999t精确相等、eps≤t<1、预测端点差时loss=0、已知偏移的均值平方loss=7.5；只独立重排z1后对应loss=0.1728000044822693。全部通过。仅运行该原始工厂函数；没有运行数据加载器、原图像网络、求解器或训练。完整JSON为 `author-loss-original.json`。
