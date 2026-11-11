# 固定SD1.4 VAE往返实验记录

[计划](plan.json)在任何重建forward前提交于`6a662c3`：固定512×512 scene/seed43输入、一个编码和六种解码；主请求300秒上限，满足预设wall/内存条件后才完整重复。2026-11-11以源码`89a64cd`完成全部请求与检查，报告见[VAE重建实践](../../../docs/vae-reconstruction-report.md)。

## 实际执行与复算

| 记录 | 实际内容 |
| --- | --- |
| [preflight](preflight.json) | 执行前源码/输入/计划SHA、CPU4核配额/32GiB、133项完整测试及可选环境pip check |
| [primary-run](primary-run.json)、[primary-metrics](primary-metrics.json)、[完整命令日志](primary-command.txt) | 首次exit0，整条命令93.77011秒，峰值1.64236GiB；编码6.37330秒、六次解码81.05777秒 |
| [repeat-decision](repeat-decision.json) | 首次wall≤120秒、峰值≤8GiB，满足事前门槛后批准完整重复 |
| [repeat-run](repeat-run.json)、[repeat-metrics](repeat-metrics.json)、[完整命令日志](repeat-command.txt) | 重复exit0，wall94.84878秒、峰值1.64788GiB |
| [independent-check](independent-check.json)、[实际核验脚本](independent-check.py) | 全部26数组与10产物身份、独立epsilon、归一化/缩放、6例原始和裁剪指标、PNG量化；不调用网络forward |
| [repeat-check](repeat-check.json) | 全部26数组逐元素相同、10产物逐字节相同；NPZ SHA为`3f9544585ede12071a3066de3a0150b234ed072be4c9f7fe3ae9379de41601e4` |
| [media](media.json) | 原尺寸完整拼图和七张PNG的ZIP身份；ZIP逐项与原输出字节核对 |
| [visual-observations](visual-observations.json) | 两位AI代理实际查看文件的清单、逐组观察与单图范围 |

实际运行目录为`artifacts/vae-current/primary`及`repeat`；原始NPZ保留于忽略的artifacts目录，读者可按报告重跑后执行核验脚本。脚本接受结果目录、显式`--record`及新的`--output`路径，报告给出与读者命令相匹配的调用方式。初次核验后的接口整理又在实际数组上执行了一次，结果JSON逐字节相同。JSON和两份日志逐字节复制，日志仅改扩展名为`.txt`；源码、版本与时间保留原始身份。

mode的原始RGB RMS为0.02105618，三个posterior案例约0.021056；人为尺度错配为0.16017480。正确缩放往返相对mode的RGB配对RMS为4.91131e-8、最大差9.53674e-7。两位AI代理核对完整拼图/全部案例，观察到正常坐标下主体保持和错配时的灰棕模糊纹理；单图误差和这些观察按报告中的范围解释。

## 固定来源与小数组检查

- [source-identities](source-identities.json)：Diffusers0.35.2固定commit的5份文件与本地安装逐字节一致；SD1.4 VAE配置与manifest一致、显式scale0.18215。协调代理独立重算6个本地文件身份通过。
- [interface-claims](interface-claims.json)：posterior logvar裁剪、mode/sample、encode/decode、归一化、显示处理和pipeline除scale的函数/行号映射。
- [ldm-paper-check](ldm-paper-check.json)：45页LDM v2 PDF SHA `46ede043a8dc07ca1f0f445620523fe1ad8b2436bd83856a3835612a47e9f79e`；核对§3.1、D.1与G。协调代理独立重算PDF SHA通过。
- [ldm-author-code-check](ldm-author-code-check.json)：原作者commit `a506df5756472e2ebaf9078affdde2c4f1502cd4`中后验采样、乘scale及解码前除scale；论文归一化估计公式与源码的具体计算分别记录。
- [small-function-checks](small-function-checks.json)：固定实现的对角高斯mode/sample/logvar、NumPy独立重参数化、图像归一化/反归一化，以及50,601组区间投影检查。该记录只运行小函数和解析例，无模型forward。

来源核验日期为2026-11-10。LDM论文族的训练损失和数据缩放解释与SD1.4固定发布配置分别引用；常数0.18215未由本次单图统计重新估计。所有协议案例会完整保留，原始数组和运行日志位于本地忽略目录，历史路径/源码身份按实际执行时记录。
