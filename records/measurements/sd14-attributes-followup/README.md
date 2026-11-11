# SD1.4两线程颜色与布局实验

这是[原八请求批次](../sd14-attributes-pilot/README.md)超时后单独声明的实验。[4步诊断](../sd14-thread-diagnostic/README.md)在当前Intel Xeon实例上测得两线程外层76.69秒、推理62.47秒；按原成本判据保守外推20步为326.59秒，查询点RSS为5.41GiB，满足门槛。全部诊断图片尚未视觉评分，选择依据为技术核验与成本。

新[计划](plan.json)保留四提示×seed101/211、执行顺序、20步DDIM和所有观察规则，统一改用2线程与新输出目录；每条420秒、总3600秒、首图wall≤360秒/RSS≤8GiB的门槛保持。八条实际结果尚待执行；八条外推合计约43.55分钟，作为预算估计。线程诊断的两个PNG有150个RGB标量相差1个8位灰度级，因此本批只核对自身固定条件与配对噪声，不假定跨线程逐像素一致。

[预检](preflight.json)核对固定源码与八条真实parser、16个新目录。复用原来已提交的[有界驱动](../sd14-attributes-pilot/execute_plan.py)：

```bash
.venv/bin/python records/measurements/sd14-attributes-pilot/execute_plan.py \
  --plan records/measurements/sd14-attributes-followup/plan.json \
  --execution artifacts/sd14-attributes-followup-execution/execution.json
```

原批次的1项超时和7项未执行继续保留；本批按新身份分别报告。所有可用完整原图在八请求结束后统一观察。
