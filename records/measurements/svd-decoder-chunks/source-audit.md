# SVD VAE 分块实验：固定来源审计

范围：2026-11-11只读现有文档、归档JSON、已安装源码及固定上游源码；完整校验391,017,740字节VAE权重。没有构造模型、执行编码/解码、查看生成图像、下载权重、安装依赖或修改仓库/Git。后续选定实验为严格重复的14帧latent，chunk顺序14→7→1，无扰动/随机抽样。

## 1. 软件来源身份

GitHub官方tag `v0.35.2`为annotated tag对象 `3257c4bc4582ffe0849f4680b5884bfbefe538e7`，实际指向commit **b71269675ec1b85193107a691dd35c308e46f0a5**（tag日期2025-10-15T03:55:42Z）。tag API原响应保存在本地/tmp/genai-svd-chunk-sources。以下五个固定commit文件已通过正常HTTPS下载到/tmp并与本地Diffusers0.35.2安装逐字节比较，全部一致；详见 `source-identity.json`。

| 文件（相对src/diffusers） | 字节数 | SHA-256 |
| --- | ---: | --- |
| pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py | 32628 | c7a8c4afc7944401a1e79aba21ecb5a48221c5b584cf10f43509b3bd8fd577e9 |
| models/autoencoders/autoencoder_kl_temporal_decoder.py | 14778 | a89e764f032d410cfb42d73312c14b29755c792ed12c3452bcc31923e46c8c86 |
| models/resnet.py | 32254 | 0c81501c54a1f05d0d72b0068efc5a71ad5f2bc66802b9694ed9134cb9b9d155 |
| models/unets/unet_3d_blocks.py | 51676 | 3b2108a2fe4d31afd3f64216cdc32831f98a6557200d16deeaf1f559be1aafda |
| models/autoencoders/vae.py | 32733 | 5d79b5d057d0f702e1a335a89913b695d7dc6faa91f3978f17e4a153162afba6 |

pipeline和temporal autoencoder两份SHA还与已有 `records/measurements/pretrained-svd/direct-comparison.json`完全一致。链接统一使用上述commit；tag与模型配置中记录的早期 `_diffusers_version=0.24.0.dev0`不混为实际运行版本。

## 2. 原样decode_latents的接口与坐标

[固定pipeline第290–316行](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L290-L316)：

- 接收 `[B,F,4,h,w]`，flatten前两轴后**除以vae.config.scaling_factor**。
- 循环切取chunk，对每块取实际 `num_frames_in`，再调用 `vae.decode(chunk,num_frames=num_frames_in)`。`num_frames`不是始终14；chunk14/7/1实际调用应为 `[14]`、`[7,7]`、十四个1。
- 各块结果沿帧轴拼回，reshape/permute后输出原始float32 `[B,3,F,H,W]`。该方法不执行 `(decoded/2+0.5).clamp(0,1)`；后处理发生于完整pipeline第619行的另一步。

[固定VAE第278–303行](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/models/autoencoders/autoencoder_kl_temporal_decoder.py#L278-L333)返回编码后验。[mode第743–744行](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/models/autoencoders/vae.py#L743-L744)直接返回mean，未乘缩放因子。

因此本次固定流程应是：

1. 读取已经裁剪好的RGB `input.png`（256×144），按本轮固定float32运算 `pixels/127.5-1` 映射到 `[-1,1]`，得到 `[1,3,144,256]`；三次输入数组身份按实际输出核对。此实验不再裁剪或加入参考增强噪声。
2. `L=vae.encode(image).latent_dist.mode()`，得到未缩放 `[1,4,18,32]`。
3. `L_seq=L[:,None].repeat(1,14,1,1,1)`，所有14帧严格相同；记录L及L_seq的raw float32 SHA。
4. **`z=vae.config.scaling_factor*L_seq`**，s=0.18215，记录实际交给pipeline方法的z SHA。直接将L_seq传入此方法会额外放大约5.49倍。
5. 原样调用 `StableVideoDiffusionPipeline.decode_latents(holder,z,num_frames=14,decode_chunk_size=C)`；holder只需持有同一VAE即可，无须构造UNet/图像编码器或加载完整pipeline。应在dtype CPU float32、eval、requires_grad False和inference_mode下执行。
6. 保存原始 `[1,3,14,144,256]`全部输出，检查有限性；原始值允许超出[-1,1]，不应以RGB的[0,1]规则拒绝。

建议记录除s后进入VAE的实际chunk输入及与L_seq的最大误差。float32乘s再除s通常产生舍入，不能要求与未缩放L_seq逐字节相同；三个配置的z必须相同。

## 3. chunk改变了哪些计算？

[Autoencoder decode第326–328行](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/models/autoencoders/autoencoder_kl_temporal_decoder.py#L326-L328)用 `batch_size=z.shape[0]//num_frames` 创建零 `image_only_indicator[batch_size,num_frames]`，并将实际块长传入decoder。

[SpatioTemporalResBlock第692–715行](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/models/resnet.py#L692-L715)先做空间resblock，然后按该帧数重排为 `[1,C,k,h,w]`，再运行temporal resblock和空间/时间混合。

[TemporalResnetBlock第566–624行](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/models/resnet.py#L566-L624)：

- Conv3d kernel `(3,1,1)`，padding `(1,0,0)`。每个chunk引入自己的时间边界，卷积无法跨块读输入。
- norm1/norm2是32组GroupNorm，对 `[1,C,k,h,w]` 的每个group在 `(C/32)×k×h×w` 上统计均值和方差。
- GroupNorm在eval仍用当前输入统计，见本地PyTorch2.6.0 `torch/nn/modules/normalization.py:237–247,257–259,312–313`；[版本文档](https://pytorch.org/docs/2.6/generated/torch.nn.GroupNorm.html)。改变k会改变统计域，效应可超过单个kernel的近邻帧。
- [TemporalDecoder第76–83、128–133行](https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/models/autoencoders/autoencoder_kl_temporal_decoder.py#L76-L133)还有输出RGB上的时间Conv3d，同样依据块长reshape。

因此chunk是改变时间上下文的执行参数。三种chunk对统一latent的差值能测量本输入上的整体分块影响；本实验没有独立分离卷积padding与GroupNorm各自贡献。因为输入严格重复，初始temporal GroupNorm某些统计可能相同；时间卷积边界可在后续层引入位置差异。实际差值须由测量决定。

可记录而无需预设非零结论的性质：chunk1每次接收同一latent，14输出帧预期一致；chunk7两块接收相同七帧输入，前后两段预期一致。chunk14与chunk7/1是否、在哪里出现差异，保留全数组和逐帧统计后报告。

## 4. 资产与输入身份（本次实际读取/校验）

模型 `stabilityai/stable-video-diffusion-img2vid`，revision `9cf024d5bfa8f56622af86c884f26a52f6676f2e`：

- `vae/diffusion_pytorch_model.safetensors`：**391017740 bytes**，SHA **9975042d7bee021bd53a72b1af14c8627d624f6547ec9abe661b68b962b88c49**；本次完整流式hash与manifest匹配。
- `vae/config.json`：**609 bytes**，SHA **aab2b8766d6db1bf742fa0e3bf217fae64e67a20e7e0fe78de84f79007100ffb**；latent_channels4、block_out_channels[128,256,512,512]、scaling_factor0.18215、force_upcast true。
- `artifacts/svd-current/smoke/input.png`、`artifacts/svd-current/full/input.png`、`docs/assets/svd-input.png`均为**69412 bytes**，SHA **e76d50ef91a5ece55252b9c867d58927ca8b21a303ee112014491608f79ed962**。后续计划应固定其中一个路径并保持SHA。

无需下载UNet或image_encoder，也无需调用完整prepare；仅载入已校验VAE目录。所有源码/权重从当前本地读取，不修改现有缓存。

## 5. 已选预算与证据边界

主协调选定：CPUfloat32、2线程、确定性算法；3个独立进程，按chunk14→7→1运行并分别重新编码同一PNG；每命令timeout300秒，总序列1000秒；若首条RSS>12GiB或wall>240秒则停止。该预算比原低分辨率完整管线中约40秒decode段留有余量，但本次2线程/VAE-only成本尚未实测。

每请求记录实际load/encode/decode/save阶段、完整外层wall、dtype/版本、源码与资产SHA、mode latent与pipeline-scaled latent SHA、实际decoder块帧数、原始全部输出SHA。跨请求统一输入与latent身份后再比较完整raw数组RMS/最大绝对差及逐帧值；RMS使用float64累计并说明单位为原始decoder输出。RSS标出查询点；若使用ru_maxrss，它是进程启动至查询点的高水位。此实验解释解码参数语义，不归因既有SVD严重失真，不提供生成运动质量结论。
