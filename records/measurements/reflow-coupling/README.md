# Reflow耦合更新：来源与独立解析核验

对应[Reflow阅读卡](../../../docs/readings/reflow-coupling.md)。固定论文arXiv:2209.03003v1及作者2024-06-20代码快照，核对每轮端点配对、边缘保持与凸位移成本条件，并用两个Gaussian例计算精确流和下一轮回归。

| 证据 | 内容与范围 |
| --- | --- |
| [source-identity.json](source-identity.json) | 41页PDF及9份页面/源码的大小、SHA、版本；作者commit `5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6` |
| [source-audit.md](source-audit.md) | 原文定义/定理页码，作者生成配对、共同索引、uniform+L2目标、RK45与时间范围的具体源码位置 |
| [gaussian-derivation.md](gaussian-derivation.md)、[gaussian-check.json](gaussian-check.json) | 独立代理从头推导Gaussian与90°旋转，闭区间良定性、27项float64检查与JVP/有限差分 |
| [candidate-math-check.json](candidate-math-check.json) | 另一组协方差、线性求解、中心差分及平方成本分解核验 |
| [root-check.json](root-check.json) | 主协调重hash10份来源，查看PDF第15页，独立64点Gauss–Legendre积分；另用60°旋转与不同端点协方差构造成本5和可行成本1.5的对照 |
| [squared-cost-derivation.md](squared-cost-derivation.md) | 本地独立正交分解、单轮平方成本恒等式与一致的相邻轮索引 |
| [author-loss-original.json](author-loss-original.json)、[author-loss-cli.json](author-loss-cli.json) | 固定作者损失工厂的真实AST执行：配对插值、位移目标、999时间缩放、L2归约与重排端点反例 |
| [integration-check.json](integration-check.json) | 接入后的三份CPU产物与独立原记录核对，36行代码字节保持；MeanFlow的79个公式与Python块保持 |
| [draft-validation.json](draft-validation.json)、[draft-integration.json](draft-integration.json) | 36行CPU块原字节、公式格式、来源锚点、作者函数检查CLI及拒绝覆盖/错误SHA；接入后的检查另行记录 |

独立Gaussian输入耦合的平方位移成本为2d，精确rectification得到恒等耦合，随后reflow场为0。二维90°旋转耦合在每个时间点均可逆，所有轨迹为恒速端点连线，rectification保持其成本4；同边缘的恒等耦合成本为0。两个初始耦合的中间边缘均为Gaussian、协方差q(t)I。成本采用坐标平方和，不含1/2；它们是解析总体量。教学点和数值网格用于核对公式。

## 直接重跑独立检查

在仓库根目录使用已有基础环境，选择尚不存在的输出文件：

```bash
.venv/bin/python records/measurements/reflow-coupling/check_gaussian.py \
  --output artifacts/reflow-checks/gaussian.json
.venv/bin/python records/measurements/reflow-coupling/check_costs.py \
  --output artifacts/reflow-checks/costs.json
```

两个脚本都使用小数组和float64。第一个检查逆矩阵、协方差、ODE/JVP、恒速、端点、Lipschitz界及成本；第二个检查条件协方差、中心差分和成本分解的积分。已有输出会被拒绝。正文36行[教学例子](teaching-example.py)另外进入全仓库教学执行器。

## 重跑固定作者损失函数检查

先在新目录取得固定版本文件；同名目录已有内容时复用已核验的文件或选择新目录。

```bash
mkdir -p artifacts/reflow-author-check
curl --fail --location --max-time 30 \
  https://raw.githubusercontent.com/gnobitab/RectifiedFlow/5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6/ImageGeneration/losses.py \
  --output artifacts/reflow-author-check/losses.py
.venv/bin/python records/measurements/reflow-coupling/check_author_loss.py \
  --source artifacts/reflow-author-check/losses.py \
  --output artifacts/reflow-author-check/result.json
```

源码为7411字节，SHA-256 `f59d42f90c982063498c5c6328d184bebb3258a79c8233a712cf06743b4ca5e8`。检查器先验证SHA，再原样抽取`get_rectified_flow_loss_fn`。运行使用PyTorch 2.6.0+cpu、float32、1线程、seed1729和四个1×2×3张量，以模型替身返回已知向量；正确目标loss为0，已知偏移loss为7.5，只重排目标端点后的loss约0.17280000448。执行范围为该损失工厂，模型结构、数据加载器、完整训练和教师求解器按源码核对。

## 原文版本与记录边界

v1第15页式(13)左侧的两个范数在PDF中省略平方；前文指定平方成本，下一行递推式包含平方。同页定理3.7求和用相邻轮，后面的best-iterate文字采用另一组索引。本仓通过独立展开核对平方成本和一致索引，详见推导证据。PDF为18,024,983字节，SHA-256 `d81790cb6988ef480002adcc84000d4f44855b11fb868001fd53aad133bc3a74`；本次查看的原页保存在本地来源目录。

完整PDF和作者源码位于本地`/tmp/genai-reflow-candidate`，按固定URL及SHA可重新取得。此处收录原创教学和检查器；作者代码的归属、文件声明与固定来源在source-audit中记录，固定tree没有统一根许可证文件。这里的结果覆盖解析例、原始函数接口和显式数值网格，图像模型训练与生成质量继续按相应实践证据评价。
