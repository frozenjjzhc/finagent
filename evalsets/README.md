# 评测集

评测集是项目的"规格"：它一变，分数就没法和以前比。所以这里的文件都提交到 git，并由脚本生成、可以复现。

| 文件 | 题数 | 划分 | 来源 |
|---|---|---|---|
| `financebench.jsonl` | 150 | dev 50 / test 100 | FinanceBench 开源子集（见下方署名） |
| `seed.jsonl` | 10 | 全部 dev | L01 手工对照 10-K 核实的种子题 |
| `seed_closed_book.jsonl` | 10 | — | 种子题的原始记录（L01/L02 的实验脚本使用），`seed.jsonl` 由它生成 |
| `finqa.jsonl` | 270 | dev 120 / test 150 | FinQA 子集（L04），每题附带一页财报作为上下文（见下方署名） |
| `MANIFEST.json` | — | — | 数据来源、固定的版本、文件哈希、各划分题数 |

重新生成：`python scripts/build_evalsets.py`（会核对原始文件的哈希，结果应与 MANIFEST 一致）。

## 使用规则

- **dev**：调 prompt、做对比实验时随便用。
- **test**：只在里程碑（v0.3、v0.7、v0.11、v0.13、v1.0）跑，**不能根据 test 的结果去改系统**，否则 test 就变成了 dev，分数不再代表真实水平。
- 种子题在 L01/L02 里被用来挑选 prompt，已经"用过了"，所以只能放在 dev。
- FinQA：我们的 dev 只来自 FinQA 的 dev.json，test 只来自 test.json；FinQA 的 train.json 留给 L10 做训练数据，三者互不重叠。

## 字段（`src/finagent/eval/datasets.py` 的 `EvalItem`）

| 字段 | 含义 |
|---|---|
| `answer_type` | `number`：程序按容差判分；`text`：LLM 裁判判分；`invalid_premise`：陷阱题 |
| `gold_text` | 标准答案原文 |
| `gold_value` / `gold_unit` | 数值题换算后的标准答案。金额统一为百万美元（`$8.70` + "USD billions" → 8700） |
| `gold_decimals` | 标准答案原文的小数位数，决定容差；换算会改变它（$8.70 billion 精度 0.01 billion = 10 million → -1） |
| `tolerance_abs` | 单题额外容差，只在题目允许更粗的四舍五入时设置（如 seed_q06） |
| `category` | FinanceBench 的 `question_type`（metrics-generated / domain-relevant / novel-generated），或种子题的类型 |
| `evidence` | 证据所在的文档名和页码，L05 评测检索时使用；FinQA 是文件名和 gold_inds |
| `context` | 给定的上下文（L04，FinQA）：一页财报的文字 + 表格。有它的题要用带 `{{context}}` 的 prompt |
| `sign_agnostic` | 题目自己说明了方向（decrease、lower……），答 42 和 -42 都算对（L04） |

`context` 和 `sign_agnostic` 是 L04 新增的字段，取默认值时不写进文件，所以 L03 的评测集文件逐字节不变。

## 数据来源与署名

`financebench.jsonl` 改编自 **FinanceBench** 的开源子集：

> Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, Bertie Vidgen.
> *FinanceBench: A New Benchmark for Financial Question Answering.* arXiv:2311.11944, 2023.
> https://github.com/patronus-ai/financebench （commit `6e138b2`）

原数据采用 **CC BY-NC 4.0** 许可：允许在注明出处的前提下用于**非商业**目的。本项目是非商业的学习项目；改动包括：统一字段格式、把数值题的标准答案换算为统一单位、划分 dev/test、只保留证据的文档名和页码。

`finqa.jsonl` 改编自 **FinQA**：

> Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, William Yang Wang.
> *FinQA: A Dataset of Numerical Reasoning over Financial Data.* EMNLP 2021.
> https://github.com/czyssrs/FinQA （commit `0f16e28`）

原数据采用 **MIT** 许可。改动包括：
- 只保留数值题，并且只保留两份标准答案对得上的题（`answer` 文字与 `exe_ans` 程序结果一致，见 `parse_finqa_gold`）：原始 dev + test 共 2030 题，丢掉 175 道对不上的、58 道非数值题、14 道重复题；
- 按计算步数（1 / 2 / 3+）分层抽样：dev 每层 40 题，test 每层 50 题；
- 把一页财报（表格前文字、表格、表格后文字）拼成纯文本上下文；标准答案程序保存在 `justification` 里。

已知问题：一致性检查发现不了"程序本身写错"的题。L04 在 dev 上至少找到 4 道（ETR_2002_page_24_4、IPG_2008_page_21_1、ALXN_2007_page_104_1、GS_2012_page_165_3），见 L04 讲义 §7.6。
