# 预训练实验的计时与RSS范围

6份报告/章节的10处说明补齐测量终点。SD1.4与SVD的内部总时间和RSS在结果字典构造中查询；SVD随后还运行帧差诊断与文件/数组hash。VAE在RSS查询之后计算输出文件hash，再记录内部总时间。历史数字与原始JSON保持。

[源码与18份记录](audit.json)、[官方定义](external-sources.json)、[VAE独立复核](vae-independent-check.json)、[文档不变量](draft-validation.json)和[主代理核验](root-verification.json)保存依据与检查范围。新的文字将RSS明确为进程启动至查询点；外层run_record另行测量完整命令wall。

本轮改变说明文字，沿用既有数值与执行证据。冻结推理源码的scope简写由这些报告解释，历史记录原样保留。
