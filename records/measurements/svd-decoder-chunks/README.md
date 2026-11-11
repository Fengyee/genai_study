# SVD时间VAE分块：固定输入实测证据

对应[分块解码报告](../../../docs/svd-decoder-chunk-report.md)。计划与模型探针在提交`1d97b06613872096c6ab86fc9030be8059a1233a`中冻结，随后按14→7→1运行。三次输入像素、编码mode与scaled latent逐字节一致；全部请求完成，查询点RSS最高2.784 GiB，完整命令wall合计160.75秒。

| 文件 | 内容 |
| --- | --- |
| [plan.json](plan.json) | 问题、固定输入/源码、三个argv、顺序、超时、首条继续门槛与比较口径 |
| [run_probe.py](run_probe.py)、[probe-review.json](probe-review.json) | 冻结的真实VAE探针及执行前只读代码审查；使用原版decode_latents |
| [cli-preflight.json](cli-preflight.json) | 已有输出和错误参考图在模型加载前拒绝 |
| [source-identity.json](source-identity.json)、[asset-identity.json](asset-identity.json)、[source-audit.md](source-audit.md) | 固定库commit/安装文件、391 MB权重、config、参考PNG和时序层接口 |
| [execution.json](execution.json) | 起止时间、三个状态、首条gate和1000秒总预算核验 |
| [chunk14-metrics.json](chunk14-metrics.json)、[chunk7-metrics.json](chunk7-metrics.json)、[chunk1-metrics.json](chunk1-metrics.json) | 原始数组身份、实际调用、内部计时、查询点RSS、版本和原始值域 |
| [chunk14-run.json](chunk14-run.json)、[chunk7-run.json](chunk7-run.json)、[chunk1-run.json](chunk1-run.json) | 外层完整命令、Git状态、wall、退出与日志SHA |
| [chunk14-command.txt](chunk14-command.txt)、[chunk7-command.txt](chunk7-command.txt)、[chunk1-command.txt](chunk1-command.txt) | 原日志字节保留，仅更换归档扩展名 |
| [comparison.json](comparison.json)、[comparison-strict-identity.json](comparison-strict-identity.json)、[compare.py](compare.py) | 验证12个NPY文件与数组SHA后，从raw float32独立转float64重算RMS/maxabs/逐帧差 |
| [independent-numeric-audit.json](independent-numeric-audit.json)、[audit_original_run.py](audit_original_run.py) | 独立代理用逐帧dot与math.fsum重算，核对PNG归一化、scaled latent、资产/源码、调用和整个执行顺序 |
| [strict-identity-check.json](strict-identity-check.json) | C-order字节比较区分+0与−0；三组输入、chunk7重复片段和chunk1重复帧均字节相同，比较器修正前后全部数值保持 |
| [reduction-layout-check.json](reduction-layout-check.json) | 保存前的BFCHW→BCFHW view与重载C连续布局，导致邻帧float64归约最大约3.47e−18差；按原布局精确复现metrics |
| [scalar-check.json](scalar-check.json)、[check_scalar.py](check_scalar.py)、[scalar-integration.json](scalar-integration.json) | 12组三点时间卷积的精确有理数、NumPy及内部切口核验；集成仅更新脚本帮助中的路径 |
| [plot.py](plot.py)、[figure-data.json](figure-data.json)、[figure-data.npz](figure-data.npz) | 从全部原始输出独立重算两面板曲线、固定坐标和输入身份 |
| [figure-original-review.json](figure-original-review.json)、[figure-integration.json](figure-integration.json)、[figure-verification.json](figure-verification.json) | 原图目视与数值审查，仓库路径整合后重绘PNG/SVG/NPZ逐字节相同 |
| [teaching-example.py](teaching-example.py) | 报告CPU代码块的原始字节副本 |

本轮原始`normalized_input.npy`、`latent_mode.npy`、`video_latents.npy`和`decoded_raw.npy`保存在忽略的`artifacts/svd-decoder-chunks/chunk{14,7,1}`，文件及数组字节SHA写入每份metrics。Git中归档协议、测量值和核验器；读者可按报告在新目录生成完整数组，再用compare.py重算。

## 检查分工与复跑

`run_probe.py`仅加载固定VAE、编码参考mode并解码重复latent，`--chunk-size`限定14/7/1。输出要求新目录；输入必须匹配归档PNG SHA，VAE/config/许可按manifest验证。原始输出为`[1,3,14,144,256]`，按decoder坐标解释。

`compare.py`接受任意采用相同`chunk14/7/1`与`record-chunk14/7/1`命名的新结果目录。基础环境即可运行，它同时验证文件/数组身份、跨配置输入相等，以及实际调用帧数。独立核验器`audit_original_run.py`按当时工作区、缓存和预声明Git提交设计，适合原测量恢复审计；新目录复跑优先用compare.py。

无需权重的独立算术核验：

```bash
.venv/bin/python records/measurements/svd-decoder-chunks/check_scalar.py \
  --output artifacts/svd-scalar-repeat.json
```

输出须尚不存在。原独立JSON还包含当时可选的安装源码hash；脚本新增`--diffusers-root`时可再次核对本地安装文件。集成后计算AST与原版本相同；源路径帮助文字的变化在scalar-integration.json保留。

## 测量口径

- 三个独立进程，固定2个PyTorch计算线程、float32、相同输入及确定性算法。各配置只执行一次，成本受顺序、缓存和系统状态影响。
- 原版方法返回的全部帧进入比较。整段解码是差值的参考；像素误差和相邻帧差各自回答数值问题，生成质量另需任务与评价协议。
- 外层wall包括命令启动至退出；内部`post_input_validation_to_metrics`从输入SHA校验后起算；decoder-only计时围绕原版方法。RSS为Linux进程高水位，查询点详见metrics。
- 首条wall/RSS门槛在其完成后核对；三次都低于300秒超时，实际序列198.12秒。该RSS门槛控制是否继续，内核层面没有新增内存限额。
- 第三条启动时存在五份新增教学/证据文件；冻结的探针、依赖锁、输入、manifest及三个共享模块SHA保持。独立核验保留了真实Git状态。

重绘科学图，使用已完成的三组原始数组和比较结果；输出目录须尚不存在：

```bash
.venv/bin/python records/measurements/svd-decoder-chunks/plot.py \
  --comparison artifacts/svd-chunk-repeat/comparison.json \
  --run-root artifacts/svd-chunk-repeat \
  --output-dir artifacts/svd-chunk-repeat/figure
```

图中的所有RMS曲线由原始NPY再次计算，并与comparison核对。原始解析图与整合后PNG/SVG/NPZ逐字节相同，source路径与脚本帮助变化单独记录；图中坐标和数值不依赖输出目录。
