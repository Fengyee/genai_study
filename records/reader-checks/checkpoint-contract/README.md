# Checkpoint数值契约与路线复查

2026-11-11，对默认CPU路径、评价公式、随机流、输出保护和两类checkpoint进行有界审查。完整[路线复查](../learning-path-round48.json)未发现新的先修/操作衔接故障；[核心审查](core-review.json)及[独立指标复算](independent-metrics.json)记录具体范围。二维energy显式求和与实现差约1.78e-7，图像逐像素诊断最大差约1.16e-6，均符合本次float32计算精度。

发现的具体问题在直接调用checkpoint helper时出现：保存float64的`1+2^-25`后，加载得到float32的`1`；加载普通float32文件时，全局默认dtype为float64又会改变返回模型的类型。同形整数及非有限state也被原加载路径接受。[代理原始检查](cli-and-dtype-before.json)、[默认dtype检查](default-dtype-before.json)和[主代理冻结旧提交复现](root-before.json)分别保留观察。声明的默认CPU命令仍正常，非有限生成结果会由后续诊断拒绝；此次按P2数值契约问题处理。

[PyTorch 2.6.0源码记录](pytorch-source.json)给出安装文件SHA、上游固定commit及函数定位：`strict=True`约束键，默认`assign=False`保留当前模块的张量属性，复制路径会执行`param.copy_(input_param)`。因此模型重建时的dtype需要由本仓库明确。

修复保持schema 1的float32范围：保存前及加载后检查非空字符串→张量映射、dtype和有限值；两类模型在CPU上构造并显式使用float32，再执行原有严格键/形状加载。无效保存会在写临时文件之前失败。新增32项跨架构回归，52项相关测试通过；实现SHA与命令见[实现记录](implementation.json)。

[历史工件重放](old-artifact-replay.json)覆盖真实二维flow/DDPM各一份、07三训练seed和08三训练seed，共8份checkpoint。故意在默认dtype=float64时载入，再恢复原采样输入协议；所有state保持CPU float32且逐元素相同。按记录中的一个采样配置重放，每个输出数组都与历史保存结果逐元素相同，8个文件SHA保持。未重新训练或运行预训练模型。

完整`bash scripts/quality.sh`退出0：178 tests passed（36.11秒）、15/15教学块、7条CLI smoke；日志`artifacts/quality/20261111T172929Z-tuGEfg`，执行时779个本地链接0错误。后续记录改动另做文档/diff检查。该验证覆盖现有两种模型和列出的数值/结构反例，未测试任意损坏文件、GPU或其它精度的完整实验链。
