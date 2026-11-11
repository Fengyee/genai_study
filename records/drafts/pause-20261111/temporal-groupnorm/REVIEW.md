# 第5节教学补充评估

建议作为一轮小型 P2 教学改进纳入。现有报告关于时间卷积/GroupNorm的陈述准确；当前可执行例仅隔离补零卷积机制。拟增加一个可手算、CPU直接运行的GroupNorm例，使读者独立验证块内帧变化和局部halo的局限。

`section5-addition.md`为插入第5节的草稿，含25行CPU块；最后一段是练习编辑建议，整合时移至练习区。`teaching-example.py`是CPU块精确副本。输入为单样本、两个通道、一组、8帧、单像素；仿射参数关闭，float64明确为教学简化。epsilon=1e-5来自VAE实际调用层；无需权重或Diffusers运行。

已执行：教学块、真实torch.nn.GroupNorm与解析值、局部输入身份、远端修改影响、两个通道组与batch独立性、train/eval相同。独立子审查用Fraction和70位Decimal复算三组统计及远端修改，三种输出与PyTorch差≤2.23e-16。证据为torch-check.json、independent-comparison.json以及/tmp/genai-temporal-norm-independent的scalar_check.py/results.json。

来源：PyTorch2.6 CPU安装包的normalization.py与torch.version.git_version所指的官方GitHub原文件逐字节相同；源码SHA在torch-upstream-identity.json。三个Diffusers安装文件与轮次61已核对上游的SHA相同。source-notes.json固定具体commit/行与调用关系。

完整SVD因果归因和质量没有新增测量。拟例证明GroupNorm统计域的机制，原始14/7/1分块结果继续按完整decoder理解。尤其原latent严格重复，某些早期层统计可以相同；卷积边界影响后续激活时仍可能产生统计差。没有仓库/Git修改，没有模型加载或推理。

整合建议：保持原三点卷积CPU块和实测数据/图不变；将新块加入检查白名单；保留原来源并补上VAE实际temporal_eps调用链接；更新教学块计数及轮次记录。需要常规质量检查与GitHub公式/代码增量验收。无需重跑VAE模型。
