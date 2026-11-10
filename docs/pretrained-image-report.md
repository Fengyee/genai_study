# SD 1.4：固定权重的CPU推理实测

本次使用原作者CompVis的SD1.4，在当前云实例完成真实文本到图像推理。默认512×512、20步DDIM、引导7.5的单张输出，推理耗时 **261.52秒**，Linux进程生命周期峰值RSS **5.38GiB**。这是一组固定模型的接口、成本和具体样本观察；扩展到多提示、多种子的质量评价列在后续任务中。

## 配置、环境与成本口径

日期为2026-11-10。Linux x86_64、Python3.12.14、CPU PyTorch2.6.0+cpu；容器memory.max为32GiB，CPU配额约4核、affinity为5个逻辑CPU；实验显式使用1个PyTorch线程，未发现NVIDIA设备。主机报告AMD EPYC9V45，宿主共享负载未受本实验控制。

软件完整锁定于[requirements-pretrained-lock.txt](../requirements-pretrained-lock.txt)，所有24项基础依赖版本保持。独立venv经`--require-hashes`安装43项依赖，editable安装与`pip check`通过。模型revision、16个文件的大小和SHA见[manifest](../examples/pretrained-sd14/manifest.json)，许可见[09章](chapters/09-pretrained-image-lab.md)。

权重与配置合计5,482,669,095字节（5.1061GiB）。实际`prepare`下载/验证段为289.69秒，其中结束校验2.77秒；此前用两个HTTPS连接试下载过部分文件后中止，改用Hub下载器，这段初步尝试未计入prepare数字。准备结束后，最终CLI重跑复用16个已校验文件、下载调用0次，耗时6.09秒（包含复用前逐文件校验和结束全量复核）。网络吞吐、缓存及文件系统会影响这些时间。

所有模型推理在独立进程中顺序执行，未做模型预热。最后一次提示修订期间也执行了基础质量检查；这些耗时用于本实例的预算观察。校验计时包含全部5.11GiB文件的流式SHA；加载计时包含可选库导入与本地模型加载；推理计时包含文本编码、去噪、VAE解码、安全检查和PIL后处理。总时间还包括保存图片与初始噪声，止于metrics写入前。峰值为`resource.ru_maxrss`的进程生命周期高水位，按Linux KiB换算GiB，包含导入与加载。

## 固定实验协议

提示词固定为：

> a watercolor painting of a small cabin beside a lake, pine trees, daytime

原始模型采用PNDM配置，本实验明确换用DDIM，epsilon预测、eta=0、float32、CPU、seed7、空文本基准条件、单张输出。20步网格为951,901,…,1。提示含特殊符号共17个token，最大长度77，本次未截断。VAE空间压缩8倍、latent缩放0.18215。

默认完整生成命令见[09章](chapters/09-pretrained-image-lab.md#5-安装准备权重并生成第一张图)。运行结果保存在`artifacts/pretrained-sd14`；两强度对照与重复运行使用独立输出目录，保留全部请求。每条完整生成命令设置900秒外部超时，smoke设置600秒；完成最小运行后才增加完整实验预算。

| 配置 | 尺寸 / DDIM步数 | 校验秒 | 加载秒 | 推理秒 | 总秒 | 峰值RSS GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 路径smoke，s=7.5 | 256×256 / 2 | 3.48 | 3.29 | 10.93 | 17.72 | 5.32 |
| 默认，s=7.5 | 512×512 / 20 | 2.73 | 3.01 | 261.52 | 267.34 | 5.38 |
| 配对引导，s=3 | 512×512 / 20 | 2.84 | 3.06 | 262.49 | 268.47 | 5.38 |
| 默认完整重复，s=7.5 | 512×512 / 20 | 2.77 | 3.02 | 259.22 | 265.08 | 5.46 |
| 后续提示修订，s=7.5 | 512×512 / 20 | 2.90 | 3.34 | 262.02 | 268.33 | 5.35 |

默认配置在新进程完整重复，初始噪声逐元素相同，输出PNG逐字节相同，SHA-256均为`0a6f7683a50ccf07f1b4208deb72446d8a6110758f511ef1fe2d9c784487d504`。2步smoke的时间点为501,1，同时使用较小分辨率，主要检查加载、推理与输出路径。

## 完整样本与预先声明的观察项

根据提示，观察项预先列为“小屋、湖、松树、白天、水彩风格”。每项使用清楚可见、模糊、未观察到三个描述，由本次教材维护者检查，后续可请独立读者复核。

默认s=7.5，seed7：

![固定SD1.4默认输出：带边框的水彩湖景，松树清晰，未见明确小屋](assets/sd14-guidance7-5-seed7.png)

这张输出清楚显示湖面、针叶树和水彩纹理，光照符合白天；小屋未观察到。画面额外包含类似装裱边框的区域。记录实际缺失项能帮助把“生成成功”与提示遵循分别核对；单张样本的观察范围限于这个提示和初始噪声。

配对s=3，seed7：

![相同初始噪声、引导3的水彩湖景，近处树木更密，仍未见明确小屋](assets/sd14-guidance3-seed7.png)

两份512×512初始噪声逐元素相同，权重、实际scheduler参数、20个时间点和其它请求参数保持。引导3的近处树木更密，湖面部分区域更朦胧；两张图的小屋均未观察到，其余四个检查项可见。RGB归一化到[0,1]后，两图所有像素与通道上的RMS差为0.07198823，描述这对输出的像素变化。这个量与构图/颜色一起解读，质量评价仍需对应任务标准。

复核配置时，Diffusers内部`_use_default_values`字段保存的名称列表在不同进程中次序不同；比较按名称集合进行，原始JSON完整保留，其余实际调度参数逐项一致。

## 缺失项的后续尝试

看过上面两张图之后，针对小屋缺失，追加一次明确前后景关系的提示：

> a watercolor painting of a wooden cabin in the foreground, with a lake and pine trees in the background, daytime

这是基于已见结果的后续尝试。保持默认s=7.5、模型、seed7、初始噪声和全部采样参数，提示长度变为24个token且未截断。完整命令为：

```bash
python -m genai_study.pretrained_image run \
  --model-dir artifacts/models/sd14 \
  --output artifacts/pretrained-sd14-foreground \
  --prompt 'a watercolor painting of a wooden cabin in the foreground, with a lake and pine trees in the background, daytime' \
  --seed 7 --steps 20 --guidance-scale 7.5 \
  --height 512 --width 512 --threads 1
```

![修订为前景小屋后的完整样本：山体更明确，湖景与松树仍在，小屋仍未观察到](assets/sd14-foreground-seed7.png)

这张图的山体轮廓和湖面变化清楚，小屋仍未观察到；松树、白天光照、水彩纹理及额外边框继续出现。本报告保留三张不同请求的全部输出，默认重复的PNG与第一张完全一致。这一次提示修订未补出目标物体，下一步应扩展预先声明的提示/seed集合，区分种子、文本表达和模型能力的影响。

## 可复查的证据

原始JSON保存在[records/measurements/pretrained-sd14](../records/measurements/pretrained-sd14)：模型身份、完整scheduler配置、17-token提示、初始噪声hash、实际UNet调用量、时间及输出图片hash均保留。权重和原始噪声数组保存在忽略的本地artifacts；本报告的图片与对应输出PNG逐字节相同。

默认20步实际记录20次UNet调用，每次batch2，合计40次单样本分支求值；2步smoke对应2次调用和4次分支求值。全部五次运行（smoke、两强度、默认重复、提示修订）均使用原有Safety checker，返回`nsfw_content_detected=false`。

09章的独立NumPy/DDIM更新检查已逐字执行：首个索引951的最大绝对差1.24e-7，末步索引1为1.70e-7。测试独立重建beta/累计alpha，核对末步边界与eta=0的更新。

`bash scripts/quality.sh`退出0：69项测试通过（31.74秒）、7条基础CLI smoke通过，日志为`artifacts/quality/20261110T173958Z-SW6MRz`。新增15项测试覆盖文件损坏、恢复下载、固定revision、路径/参数和输出保护；真实模型运行单独记录于本页。新增稀疏进度提示的最终入口已在提示修订的完整20步运行中验证，实际输出6条步骤状态；采样和指标代码保持。本页给出单实例的实际时间与内存；其他硬件和平台需要重新测量，当前样本量支持具体输出与接口核验。
