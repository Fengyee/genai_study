# 概率流理论衔接的核验记录

[专题正文](../../../docs/readings/diffusion-flow-bridge.md)固定路径、时间方向与端点条件，列出论文版本、公式号、PDF哈希及作者代码commit。

- [下载身份](downloads.json)：当时实际读取的Score-SDE v2 PDF、摘要页面及commit API响应，PDF由主代理再次计算SHA核对。
- [作者源码身份](source-files.json)：固定commit的sde_lib.py、sampling.py与README的实际文件大小及SHA；核对范围为概率流系数、VP边缘和积分方向。
- [独立解析检查](independent-check.json)：主代理用联合高斯条件均值独立复算路径速度、VP角速度系数与09实际末端噪声比，最大代数残差约4.77e-15。参数为数据均值2、方差4、VP beta=2；这些是小数组解析比较。

正文的NumPy代码块另外直接进入scripts/check_lesson_examples.py固定白名单，在每次质量检查的独立进程中执行。它检验条件均值、总体均值/方差导数、精确概率流轨迹、错误符号/系数反例，以及DDIM坐标恒等式。50/200/800步Euler终点最大差实际为0.02339748/0.00582653/0.00145522，DDIM代数残差2.22e-16；每次真实输出保存在对应quality日志。

本目录保存核验记录；PDF与第三方源码可按固定URL重新取得。作者JAX训练项目未执行，实验没有新增模型权重或网络训练。
