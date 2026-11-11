# 贡献指南

欢迎修正数学、实现、教学和可复现性问题。请先读 [学习路线](docs/learning-path.md)、[符号约定](docs/notation.md) 与 [质量 backlog](records/BACKLOG.md)。本项目面向熟悉 PyTorch 的研究生，主文档为中文；英文术语和论文标题保留以便检索。

## 一次贡献的最小闭环

1. 在 issue 或 PR 中说明具体读者问题、当前行为和预期改进。小修可直接 PR。
2. 从主分支创建 feature branch；保留他人的未提交修改。新结论优先引用论文、官方文档、作者代码，注明具体版本和核验范围。
3. 区分数学结论、教学假设、实测和猜想。不要把单次 toy 实验当成大型模型结论。
4. 按 README 安装环境；运行 `bash scripts/quality.sh`。数学改动需附独立推导、解析特例或反例；说明未运行的昂贵检查。
5. 在 records/PROGRESS.md 记录变更、证据、命令、结果和限制；更新 backlog。提交 PR，不直接向 main 添加项目内容。

每章应有目标、先修、范围、核心推导/直觉、实践、误区、练习、提示或参考解答、原始来源。新增示例需写清随机性、CPU/GPU、依赖、运行成本、结果文件与失败诊断。不得提交密钥、付费 API 默认调用、外部数据集复制品或大型模型权重。

部分轻量CPU代码块通过紧邻围栏的`<!-- cpu-check: identifier -->`标记进入[教学例子检查](scripts/check_lesson_examples.py)。修改这些块时保留ID与断言，并直接运行检查；新增ID需同时加入脚本中固定文件的白名单，确认无需下载、权重、训练产物或写文件且能在20秒预算内完成。脚本只接受Python围栏或受支持的单一Python heredoc。昂贵实验继续按专门章节验证并记录范围。

图示应有可独立阅读的文字说明。PNG使用描述关键观察的alt，并在正文或表格中给出重要数值；Mermaid添加`accTitle`和`accDescr`，语法见[官方可访问性说明](https://mermaid.js.org/config/accessibility.html)。依赖图同时提供文字路径，让读者可以直接理解章节关系。验证时分别记录语法解析、浏览器呈现和屏幕阅读器实际测试的范围。

数学排版使用GitHub[官方数学语法](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)：行内统一采用 ``$`TeX`$``，保护中文邻接和Markdown特殊字符；块公式使用独立的 `math` 代码围栏。折叠答案中的短公式放在独立段落，采用同一行内定界；较长推导移到折叠区外。实际GitHub检查发现，折叠区混用行内与块公式时可能保留原始TeX，因此本仓库禁止在 `details` 内放 `math` 围栏。文档检查会验证这些格式约定；公式语义、窄屏布局与MathJax实际显示仍需分别核对。

表格中的范数定界使用 `\Vert`，避免Markdown消耗 `\|` 的转义。在 `aligned`、`cases` 等环境中，TeX换行符 `\\` 后把下一项接在同一物理行；当前GitHub渲染接口会给紧邻物理换行的双反斜杠再加一个反斜杠。这两项也纳入文档检查，核验范围见[数学排版记录](records/reader-checks/math-rendering/README.md)。

当前GitHub客户端拒绝 `\operatorname`；命名算子采用已验证的 `\mathop{\mathrm{Cov}}\nolimits` 形式，保留直立名称、算子间距和侧下标。块公式中的小于号使用 `\lt`，在命令与后续字母或数字之间留空格，避免原始 `<` 被客户端误读。API成功产生数学节点后，还需确认客户端确实生成MathML并且没有错误文字。

## 评审重点

事实/数学错误和运行故障优先，其次是先修断层、误导性图表、复现缺口，最后是展示。要求 CI 通过并不代表理论正确：审稿人还应检查假设、公式端点和概念迁移的边界。外链检查受网络限制时须如实记录，不能假称验证成功。

## 许可与归属

原创文档和代码使用 [Apache-2.0](LICENSE)。提交贡献表示你有权以该许可提供它。引用论文不改变论文或外部代码的许可；复制/改写第三方实现前必须核对许可并增加归属记录。外部模型、许可副本和模型生成示例分别记录资产身份与适用条款，来源关系见 [SOURCES.md](SOURCES.md)。请避免未授权复制论文图表或数据。
