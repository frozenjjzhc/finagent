# 评测集

评测集是项目的"规格"：它一变，分数就没法和以前比。所以这里的文件都提交到 git，并由脚本生成、可以复现。

| 文件 | 题数 | 划分 | 来源 |
|---|---|---|---|
| `financebench.jsonl` | 150 | dev 50 / test 100 | FinanceBench 开源子集（见下方署名） |
| `seed.jsonl` | 10 | 全部 dev | L01 手工对照 10-K 核实的种子题 |
| `seed_closed_book.jsonl` | 10 | — | 种子题的原始记录（L01/L02 的实验脚本使用），`seed.jsonl` 由它生成 |
| `MANIFEST.json` | — | — | 数据来源、固定的版本、文件哈希、各划分题数 |

重新生成：`python scripts/build_evalsets.py`（会核对原始文件的哈希，结果应与 MANIFEST 一致）。

## 使用规则

- **dev**：调 prompt、做对比实验时随便用。
- **test**：只在里程碑（v0.3、v0.7、v0.11、v0.13、v1.0）跑，**不能根据 test 的结果去改系统**，否则 test 就变成了 dev，分数不再代表真实水平。
- 种子题在 L01/L02 里被用来挑选 prompt，已经"用过了"，所以只能放在 dev。

## 字段（`src/finagent/eval/datasets.py` 的 `EvalItem`）

| 字段 | 含义 |
|---|---|
| `answer_type` | `number`：程序按容差判分；`text`：LLM 裁判判分；`invalid_premise`：陷阱题 |
| `gold_text` | 标准答案原文 |
| `gold_value` / `gold_unit` | 数值题换算后的标准答案。金额统一为百万美元（`$8.70` + "USD billions" → 8700） |
| `gold_decimals` | 标准答案原文的小数位数，决定容差；换算会改变它（$8.70 billion 精度 0.01 billion = 10 million → -1） |
| `tolerance_abs` | 单题额外容差，只在题目允许更粗的四舍五入时设置（如 seed_q06） |
| `category` | FinanceBench 的 `question_type`（metrics-generated / domain-relevant / novel-generated），或种子题的类型 |
| `evidence` | 证据所在的文档名和页码，L05 评测检索时使用 |

## 数据来源与署名

`financebench.jsonl` 改编自 **FinanceBench** 的开源子集：

> Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, Bertie Vidgen.
> *FinanceBench: A New Benchmark for Financial Question Answering.* arXiv:2311.11944, 2023.
> https://github.com/patronus-ai/financebench （commit `6e138b2`）

原数据采用 **CC BY-NC 4.0** 许可：允许在注明出处的前提下用于**非商业**目的。本项目是非商业的学习项目；改动包括：统一字段格式、把数值题的标准答案换算为统一单位、划分 dev/test、只保留证据的文档名和页码。
