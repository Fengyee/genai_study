# MeanFlow平均速度：固定来源与解析核验

对应[MeanFlow阅读卡](../../../docs/readings/meanflow-average-velocity.md)。核验对象是两时间平均速度、总导数JVP、停止目标梯度和采样接口；实际运算使用已有基础CPU环境，无需模型权重或JAX安装。

| 证据 | 内容与范围 |
| --- | --- |
| [source-identity.json](source-identity.json) | arXiv:2505.13447v1及作者commit58d099c的PDF/README/源码/配置/许可URL、SHA、字节与阅读范围 |
| [draft-check.json](draft-check.json) | 冻结草稿的70行内/9块公式格式、4个本地链接、1924字节CPU块原样运行；接入后正文新增本证据入口 |
| [gaussian-derivation.md](gaussian-derivation.md) | 独立Gaussian完整推导：端点、平均速度、对角线、条件残差与半梯度 |
| [gaussian-check.json](gaussian-check.json) | 五组非对角/五组对角、JVP与中心差分、两种一步输运及条件目标的一二阶矩 |
| [linear-field-check.json](linear-field-check.json) | 主协调独立二维线性场：恒等式、两段位移相加、接口换元、错误切向量和停止梯度反例 |
| [source-call-check.json](source-call-check.json) | 作者三个采样方法的原始AST替身核验，记录实际一次调用与默认时间表 |
| [integration-check.json](integration-check.json) | 将上述两个解析脚本改为显式输出路径后重跑，核对计算结果与历史JSON一致；正文代码块身份保持 |
| [LICENSE-upstream-MIT.txt](LICENSE-upstream-MIT.txt) | 作者仓库MIT许可原样副本；原文件身份见source-identity.json |

Gaussian例的精确平均速度一次得到目标方差0.49，瞬时速度反向Euler一步方差为0；9个固定输入点的最大端点误差分别约2.22e−16和1.4。方差来自Gaussian总体的解析变换。额外五组时间检查中，JVP恒等式残差最大约4.44e−16、中心差分与JVP最大差约2.67e−10。

二维场采用`dz/dτ=diag(0.7,-0.4)z`，与Gaussian例独立。正确两种接口JVP逐元素一致，错误固定区间的恒等式残差约0.07435；两段位移合成差约5.55e−17。两个脚本都保留条件目标期望与停止目标梯度的范围说明。

## 重跑解析检查

从仓库根目录，在已准备的基础环境运行。输出文件必须尚不存在；重复时选择新的父目录，保留既有记录。

```bash
mkdir -p artifacts/meanflow-checks
.venv/bin/python records/measurements/meanflow-average-velocity/check_gaussian.py \
  --output artifacts/meanflow-checks/gaussian.json
.venv/bin/python records/measurements/meanflow-average-velocity/check_linear_field.py \
  --output artifacts/meanflow-checks/linear-field.json
```

这两个脚本直接使用float64小数组和解析函数，没有训练阶段、随机抽样或模型下载。输出包括实际误差、版本（Gaussian脚本）和断言结果。正文的教学块与完整Gaussian脚本分别保存；教学块还包含`(x,t,h)`接口的错误方向对照。

## 重跑作者采样接口检查

[check_author_sampler.py](check_author_sampler.py)在执行前严格核对源码大小与SHA，只抽取已审阅的`sample_one_step`、`solver_step`和`_default_schedule`三个函数。用NumPy替代`jnp`，用解析Gaussian平均速度替代网络，保留原函数的调用流程。完整作者源码仅下载到忽略的本地产物目录：

```bash
mkdir -p artifacts/meanflow-author-check
curl --fail --location --max-time 30 \
  https://raw.githubusercontent.com/Gsunshine/meanflow/58d099c0fa197bd31b9a7b51727a76019ae5fdbe/meanflow.py \
  --output artifacts/meanflow-author-check/meanflow.py
.venv/bin/python records/measurements/meanflow-average-velocity/check_author_sampler.py \
  --source artifacts/meanflow-author-check/meanflow.py \
  --output artifacts/meanflow-author-check/result.json
```

原源码为8029字节，SHA-256 `b8dcee1033e6ba01de0019d14fad846a399d356d67e549ee311a5e19d2a7c3a4`。替身检查确认`[1,0]`时间表、`h=1`、一次网络接口调用和解析端点结果。完整JAX/Flax执行、权重质量、图像训练和作者FID未在这些检查中运行。PDF与其它作者文件按source-identity.json中的固定URL重新获取并核对。

来源论文与作者MIT文件按各自许可引用；本目录教学推导和验证脚本属于本项目的原创材料。GitHub API与实际客户端渲染验收已完成，见下文。

## GitHub阅读验收

固定876647b，新卡/符号/术语三页在1440×1000和390×844两视口全部通过；79/90/17个公式均产生MathML，错误与原始TeX为0。新锚点、三道习题、1924字节代码、公式/表格/代码局部滚动及术语入口实际跳转通过，document/article无整体溢出。[API摘要](api-render-summary.json)、[实际客户端摘要](client-render-summary.json)与[主代理复核](root-reader-check.json)记录身份、数量和范围；[桌面JVP表](meanflow-anchor-desktop.png)和[窄屏习题](meanflow-exercises-narrow.png)保留原截图。其它完整截图位于本地产物目录；屏幕阅读器及真实移动设备尚待读者验证。

## 解析轨迹图

[PNG](../../../docs/assets/meanflow-gaussian.png)与[SVG](../../../docs/assets/meanflow-gaussian.svg)同时展示完整区间位移和瞬时切线更新。图中的虚线弦连接当前与目标端点，中间点按端点连线得到。右图九个标记是固定输入，方差0.49/0来自Gaussian总体。图采用颜色、线型及圆/方标记，中文替代说明和相邻配文解释坐标与范围；窄屏可打开矢量图查看细节。

从仓库根目录重跑：

```bash
.venv/bin/python records/measurements/meanflow-average-velocity/plot_geometry.py \
  --output-dir artifacts/meanflow-geometry-reproduction
```

新目录保存PNG、SVG、全部绘图数组data.npz及verification.json；已有目录会被拒绝。[生成记录](figure-verification.json)、[独立几何审查](figure-geometry-review.json)与[主代理核对](figure-root-check.json)保存版本、身份、解析端点与图像检查。原始数据在本地/tmp/genai-meanflow-figure/final，读者可用该命令重新生成。

固定002dd89的新增图片验收见[实际浏览器](figure-browser-check.json)、[SVG字节核验](figure-svg-reachability.json)和[主代理检查](figure-browser-root-check.json)。两视口均正常加载完整原图、中文alt/配文和79个公式，整体无横向溢出；[窄屏配文截图](figure-caption-narrow.png)保留实际文字呈现。窄屏中的整图为缩略显示，细小标签需放大查看。
