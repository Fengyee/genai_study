# 空间token、投影与位置的核验记录

[专题](../../../docs/readings/spatial-tokens-and-position.md)连接07的卷积图像实践与04的DiT结构。范围为明确索引的patch重排、共享投影、全自注意力位置绑定和输出还原。

- [来源身份](source-identity.json)：固定DiT v2第4–5页与作者models.py。主代理重新下载源码逐字节比较，并重新计算43,824,062字节PDF的SHA；保留作者unpatchify的docstring与实际NCHW返回的差别。
- [正文实际运行](card-cpu-result.json)：原样CPU块约1.17秒，patch/嵌入为(1,6,8)/(1,6,3)，卷积等价误差约2.84e-14，固定位置交换内容差约1.774983。
- [独立正文复核](independent-card-check.json)：六重标量循环另算卷积，误差约5.68e-14，并逐题检查索引/秩/输出规则。
- [独立索引和置换](independent-index-check.json)：明确patch内外排列及相应注意力反例。
- [主代理矩形检查](root-independent-check.json)：另用3通道6×4网格、2×2 patch、固定投影；重排/卷积逐元素相等，均值投影碰撞展示信息损失。三个投影矩阵下，联合置换误差约4.44e-16，固定位置只重排内容差约0.51783。完整小数组与输入/核公式保存。
- [作者函数检查](author-unpatchify-check.json)及[主代理独立抽取](root-author-check.json)：只编译已审阅的unpatchify函数，后者以2批次/3通道/4×4网格和六重索引循环独立比较，实际返回BCHW且逐元素相等。
- [数学格式不变量](format-invariants.json)与[最终API检查](api-check.json)：采用已验证的定界和算子语法，40行内＋6块公式的TeX与类别逐项匹配；完整Python围栏段（含末尾换行）2846字节与冻结原稿逐字节相同，执行正文含末尾换行为2832字节。

上述执行仅使用确定的小数组与float64，没有构造DiT网络或运行生成权重。正文块进入固定质量白名单；实际模型速度、显存和生成质量需另行设计实验。推送后的真实GitHub页面验收在执行记录中补充。
