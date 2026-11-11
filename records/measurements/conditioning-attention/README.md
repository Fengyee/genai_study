# 条件注意力接口的核验记录

[阅读卡](../../../docs/readings/conditioning-attention.md)连接LDM的Q/K/V框架、SD1.4的文本序列条件、SVD的全局图像向量与空间参考latent路径，并提供CPU解析反例和练习。

- [作者源码读取](downloads.json)：固定LDM/CLIP commit的URL、字节数与SHA。
- [本地来源身份](local-source-identities.json)：已安装Diffusers/Transformers、固定模型配置及LDM v2 PDF；主代理重新计算全部11文件的大小与SHA。
- [固定库源码比对](remote-source-identities.json)：五个远端固定commit文件与当前安装逐字节一致，主代理重新核对。
- [主代理独立小数组检查](independent-check.json)：两token的查询响应、encoded KV联合置换、单token常值输出与Q/K零梯度，并与PyTorch SDPA交叉核对。

正文CPU块另检查mask同步、两键sigmoid解析特例和中心有限差分；它进入固定教学块白名单后由质量检查独立执行。假设为有限有效logit、每查询至少一个有效key、固定encoded K/V与零dropout，单token结论定位到注意力汇聚支路。完整网络的残差、空间参考条件和非线性继续保留位置相关响应。

本轮读取固定来源、模型配置并执行小张量；没有新增预训练权重前向或图像/视频生成。既有真实请求见09/10章与各自实测报告。
