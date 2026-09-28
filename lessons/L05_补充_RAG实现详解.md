# L05 补充 · RAG 是怎么一步步实现的

> 配合 [L05_RAG基础管线.md](L05_RAG基础管线.md) 使用。讲义讲"为什么"，这份讲"代码怎么做"。
> 配套演示：`python walkthroughs/L05_trace_one_question.py`。它跟着一道真实的题走完 8 步，每一步都打印真实数据（第 7 步调用一次模型，约 $0.0001；加 `--offline` 只跑前 6 步）。
> 下面所有的输出都来自这个脚本的实际运行。

跟踪的题目（dev 集 fb03282）：

```
What is Netflix's year end FY2017 total current liabilities (in USD millions)?
Base your judgments on the information provided primarily in the balance sheet.
标准答案：$5466.00（百万美元）；证据：NETFLIX_2017_10K 第 44 页（从 0 数，PDF 阅读器里是第 45 页）
```

---

## 0. 全景：8 步，两个阶段

```
离线（build-corpus、build-index 各跑一次，结果存在 data/ 下）
  ① ingest   PDF 的一页 → 文字                   data/ingest.py       extract_pages、rows_text
  ② chunk    一页 → 若干块                       rag/chunker.py       chunk_page
  ③ embed    一块 → 一个 384 维向量               rag/embedder.py      Embedder.embed_passages
             所有向量 + 所有块存盘                rag/retriever.py     build_index

在线（每个问题都要做）
  ④ retrieve 问题 → 查询向量 → 最相似的 5 块      rag/retriever.py     Retriever.search
  ⑤ context  5 块 → 所在的整页 → 编号的材料       rag/retriever.py     ContextBuilder、format_context
  ⑥ prompt   材料 + 问题 → messages              prompts.py           answer_rag_v1.yaml
  ⑦ generate 读材料、调用计算器、输出 JSON        pipeline/answer.py   answer_question（L04 原样复用）
  ⑧ grade    判分 + 检索命中 + 引用核对           eval/run.py          _retrieval_fields
```

几个数据结构贯穿全程，先认识它们：

| 结构 | 在哪 | 装的是什么 | 例子 |
|---|---|---|---|
| `Page` | data/ingest.py | 一页的文字 | `Page("NETFLIX_2017_10K", 44, "Table of Contents\n...")` |
| `Chunk` | rag/chunker.py | 一个文本块 | `id="NETFLIX_2017_10K:p44:c0"`，`title="Netflix 10-K 2017"` |
| `Hit` | rag/retriever.py | 一个检索结果 | 块 + 相似度 0.809 + 排名 1 |
| `Source` | rag/retriever.py | 交给模型的一段材料 | 编号 [1] + 第 44 页的全文 + 是哪些块带进来的 |

---

## 1. 离线阶段

### ① ingest：PDF 的一页 → 文字

```python
# data/ingest.py
def extract_pages(pdf_path, layout="rows"):
    with pymupdf.open(pdf_path) as pdf:
        return [
            Page(doc_name, i, clean_text(
                rows_text(page.get_text("words")) if layout == "rows" else page.get_text("text")
            ))
            for i, page in enumerate(pdf)       # i 从 0 开始，和 FinanceBench 的页码一致
        ]
```

Netflix 资产负债表那一页，两种抽取方式的真实对比：

```
layout=text（PyMuPDF 默认，98 行）      layout=rows（按坐标重建，41 行）
Cash and cash equivalents              Cash and cash equivalents $ 2,822,795 $ 1,467,576
$                                      Short-term investments — 266,206
2,822,795                              Current content assets, net 4,310,934 3,726,307
$                                      ...
1,467,576                              Total current assets 7,669,974 5,720,291
```

`rows_text` 的核心只有一个循环：把所有字按"行的中线（y）、横坐标（x）"排序，y 相差不超过 3 磅的算同一行。

```python
ordered = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))   # 先按中线，再按横坐标
for w in ordered:
    y = (w[1] + w[3]) / 2
    if rows and abs(y - center) <= tolerance:      # 和当前行的中线足够近：同一行
        rows[-1].append(w)
        center += (y - center) / len(rows[-1])     # 中线取平均，容忍轻微错位
    else:                                           # 否则开新的一行
        rows.append([w])
        center = y
```

`build_corpus` 对 84 份 PDF 逐一下载、抽取，写成 `data/processed/financebench_pages.rows.jsonl`（每页一行 JSON）。

### ② chunk：一页 → 若干块

建索引时，`chunk_pages` 对每一页调用 `chunk_page`，数 token 用的是向量模型自己的分词器：

```python
# rag/retriever.py 的 build_index
chunks = chunk_pages(pages, titles, cfg.chunk.max_tokens, cfg.chunk.overlap_tokens,
                     embedder.count_tokens)     # ← 传入"怎么数 token"的函数
```

Netflix 第 44 页被切成了 3 块（真实输出）：

```
NETFLIX_2017_10K:p44:c0   24 行，251 个 token   第一行：Table of Contents
                                              最后一行：Total current liabilities 5,466,312 4,586,657
NETFLIX_2017_10K:p44:c1   15 行，260 个 token   第一行：Deferred revenue 618,622 443,472
                                              最后一行：Retained earnings 1,731,117 1,128,603
NETFLIX_2017_10K:p44:c2    6 行， 85 个 token   第一行：Accumulated other comprehensive loss ...
```

- 每块都不超过 256 个 token（加上标题稍多一点）；最后一块是这一页剩下的部分，所以短。
- **重叠**：c1 的开头重复了 c0 末尾的两行（`Deferred revenue ...` 和 `Total current liabilities ...`，合计不超过 32 个 token）。所以答案那一行同时出现在 c0 和 c1 里。
- id 里带着页码 `p44`：之后判断"检索有没有命中证据页"，靠的就是它。

### ③ embed：一块 → 一个向量，全部存盘

送进向量模型的是 `embed_text()`：**标题 + 正文**。

```
Netflix 10-K 2017              ← 标题（doc_title 生成）
Table of Contents
NETFLIX, INC.
CONSOLIDATED BALANCE SHEETS
...
```

```python
# rag/embedder.py
def embed_passages(self, texts, show_progress=False):
    return self.model.encode(texts, batch_size=64, normalize_embeddings=True, ...)
```

- `self.model` 第一次用到时才加载 bge-small（`import torch` 和加载权重要十几秒）；
- `normalize_embeddings=True`：每个向量缩放到长度 1。

c0 的向量（真实输出）：384 个数，前 6 个是 `[0.011 -0.027 0.033 -0.057 0.054 -0.002]`，长度 1.0000。它存在向量矩阵的第 32,046 行。

`build_index` 把所有东西存进 `data/index/fb_bge_small/`：

| 文件 | 内容 |
|---|---|
| `vectors.npy` | 形状 (41766, 384) 的 float32 矩阵，第 i 行是第 i 块的向量 |
| `chunks.jsonl` | 41766 行，第 i 行是第 i 块（id、文档、页码、标题、正文） |
| `meta.json` | 建索引时的完整配置、块数、token 统计、语料文件的哈希、用时、依赖版本 |

**两个文件的行号一一对应**：向量矩阵的第 i 行，就是 `chunks.jsonl` 的第 i 行。检索出"第 32,046 行最相似"，就去 `chunks.jsonl` 拿第 32,046 块。

---

## 2. 在线阶段

### ④ retrieve：问题 → 查询向量 → 最相似的 5 块

`Retriever.load("fb_bge_small")` 读回上面三个文件；它会先核对 `meta.json` 里记录的配置和现在的 `configs/index/fb_bge_small.yaml` 是否一致，不一致就报错（配置改了却没重建索引，结果会悄悄对不上）。

`Retriever.search` 做三件事：

```python
q = embedder.embed_queries([question])        # 1. 前缀 + 问题 → 向量 (1, 384)
scores = vectors @ q[0]                        # 2. 和 41766 块逐一算点积 = 余弦相似度
top5 = argpartition + argsort                  # 3. 取最大的 5 个（NumpyIndex.search）
```

1. `embed_queries` 会在问题前面加上 `"Represent this sentence for searching relevant passages: "`（bge 训练时就这么用），段落不加。
2. 一次矩阵乘法，得到 41766 个相似度：最高 0.809，中位数 0.602。
3. `np.argpartition` 只保证"前 5 个都比其余的大"，不做完整排序，比排序 4 万个数快；然后只对这 5 个排序。

真实的前 5 名：

```
#1  0.809  NETFLIX_2017_10K:p44:c0  ← 证据页
#2  0.790  NETFLIX_2017_10K:p12:c1
#3  0.789  NETFLIX_2017_10K:p40:c0
#4  0.785  NETFLIX_2015_10K:p42:c0     ← 同一家公司，2015 年的报告（干扰项）
#5  0.780  NETFLIX_2017_10K:p18:c3
```

证据页的 3 块分别排第 1、6、92 名。

### ⑤ context：5 块 → 所在的整页 → 编号的材料

```python
# rag/retriever.py
cfg = RetrievalConfig(index="fb_bge_small", top_k=5, context="page")
sources = ContextBuilder(cfg, retriever).retrieve([question])[0]
context = format_context(sources)
```

`context: page` 时走 `_expand_to_pages`：按排名顺序处理 5 个结果，把每块换成**它所在的整页**；同一页的第二块只记下块 id，不重复加页。

```python
for h in hits:
    key = (h.chunk.doc_name, h.chunk.page)
    if key in sources:                       # 这一页已经在了
        sources[key].chunk_ids.append(h.chunk.id)
        continue
    sources[key] = Source(len(sources) + 1, ..., self.pages[key], ...)   # 编号 = 第几个新页
```

这道题的 5 块来自 5 个不同的页，所以得到 5 段材料 [1]~[5]。`format_context` 把它们拼起来，每段开头是编号和出处（页码显示时 +1）：

```
[1] Netflix 10-K 2017, page 45
Table of Contents
NETFLIX, INC.
CONSOLIDATED BALANCE SHEETS
(in thousands, except share and per share data)
...

[2] Netflix 10-K 2017, page 13
...
```

一共 14,418 个字符，约 3,452 个 token。

**small-to-big 什么时候真的起作用？** 这道题不需要它：排第 1 的 c0 本身就含答案那一行。需要它的例子是 dev 的 fb07661（Block FY2020 经营活动现金流）：证据页排第 2 的块只有表头，写着 `Net cash provided by operating activities 381,603` 的块排第 152 名。只给块时模型拒答，给整页时答对（`walkthroughs/L05_retrieval.py` 第 5 步）。

### ⑥ prompt：材料 + 问题 → messages

`answer_rag_v1.yaml` 的 user 模板：

```
Excerpts:

{{context}}

Question: {{question}}
```

`build_messages` 生成两条消息（真实长度）：

| 角色 | 字符数 | 内容 |
|---|---|---|
| system | 2,387 | 角色 + 8 条规则（只用材料、核对出处、单位统一为百万美元、引用编号……）+ 计算器说明 + JSON 格式 |
| user | 14,599 | `Excerpts:` + 5 页材料 + `Question: ...` |

材料放在前面、问题放在最后：先让模型读材料，最后才告诉它要回答什么。system 每题都一样，能命中前缀缓存（第二次运行时，同样的材料也能命中）。

### ⑦ generate：L04 的阅读器原样复用

```python
system = load_config("configs/systems/v0.5.yaml")
result = answer_question(client, prompt, question, context=context, **tool_options(system.tools))
```

这一行和 L04 完全一样：`answer_question` 不知道也不关心 context 是 FinQA 给的，还是检索出来的。它负责工具循环、JSON 解析、pydantic 校验、失败重试。

真实输出（调用模型 1 次，花费 $0.00012，这次没有调用计算器）：

```json
{"reasoning": "The FY2017 consolidated balance sheet reports Total current liabilities of $5,466,312 thousand. Converting to millions: 5,466,312 / 1,000 = 5,466.312.",
 "status": "answered", "value": 5466.312, "unit": "USD_millions",
 "answer_text": "Netflix's total current liabilities at fiscal year end 2017 were $5,466.312 million (reported as $5,466,312 thousand).",
 "sources": ["[1]"], "confidence": "high"}
```

注意模型做对的两件事：
1. 看到表头写着 `(in thousands ...)`，把 5,466,312 千美元换算成 5,466.312 百万美元（prompt 规则 3）；
2. `sources` 写的是 `[1]`，就是第 44 页。

### ⑧ grade：判分 + 检索命中 + 引用核对

```
判分：correct（误差在 ±54.66 以内）          ← 标准答案 5466，容差 1% = 54.66（L03 的规则）
检索命中证据页：True                          ← 给模型的 5 页里有第 44 页
模型的 sources：['[1]'] → [('NETFLIX_2017_10K', 44)]   ← 引用的正是证据页
```

在评测里，这些由 `eval/run.py` 的 `_retrieval_fields` 逐题记进 `results.jsonl`：

```python
refs = {s.ref: (s.doc_name, s.page) for s in sources}       # 编号 → （文档, 页）
cited = {int(n) for s in answer.sources for n in re.findall(r"\[(\d+)\]", s)}   # 只认方括号里的数字
"retrieval_hit": any(k in gold for k in refs.values()),      # 给到证据页了吗
"cited_evidence": any(refs.get(r) in gold for r in cited),   # 引用的是证据页吗
```

`retrieval_stats` 再把全部题目汇总成"命中时 / 没命中时的准确率"，也就是讲义 §7.3 的错误分解。

---

## 3. 评测时的整体流程

`finagent eval --config configs/systems/v0.5.yaml` 时，`run_eval` 多做了一步：

```python
all_sources = prepare_sources(system.retrieval, items)   # ① 先为全部 50 题一次性检索（批量算向量）
with ThreadPoolExecutor(max_workers=workers) as pool:    # ② 再并发调用模型答题
    outcomes = list(pool.map(process, items, all_sources))
```

为什么先检索、再并发？
- **快**：50 个问题一次性算向量，比一题一题算快得多；
- **稳**：向量模型在 GPU 上，多个线程同时用它容易出问题。检索只做一次，线程们只负责调用 API。

---

## 4. 配置是怎么串起来的

```
configs/systems/v0.5.yaml             系统怎么用检索
  retrieval:
    index: fb_bge_small      ─────┐
    top_k: 5                      │
    context: page                 │
                                  ▼
configs/index/fb_bge_small.yaml       索引怎么建
  corpus_layout: rows        → 读 data/processed/financebench_pages.rows.jsonl
  embedder: bge-small ...
  chunk: 256 / 32 / title
                                  │  finagent build-index --index fb_bge_small
                                  ▼
data/index/fb_bge_small/              建好的索引（vectors.npy、chunks.jsonl、meta.json）
```

**系统配置**说"用哪个索引、取几个、给模型块还是整页"；**索引配置**说"索引怎么建"。两者分开，所以：

- 想试试**每块 512 个 token**：新建 `configs/index/fb_bge_small_512.yaml`（改 `max_tokens`）→ `finagent build-index --index fb_bge_small_512` → `python scripts/l05_retrieval_eval.py fb_bge_small fb_bge_small_512` 比较检索（免费）→ 值得的话，新建系统配置把 `index` 指向它，再跑端到端评测。
- 想试试**取 10 个**：只改系统配置的 `top_k`，不用重建索引。
- 想试试**换向量模型**（比如 bge-base）：新建索引配置改 `embedder.model`，重建索引。

---

## 5. 常见问题

**问：为什么不直接用 LangChain、LlamaIndex？**
答：它们把这 8 步都封装好了，几行代码就能跑。但出了问题（比如本课的"一半的题没找到证据页"），你不知道该改哪一步。自己写一遍之后，再用框架，一天就能上手（ADR-6）。

**问：向量都存在内存里，语料再大怎么办？**
答：4 万 × 384 个 float32 约 64MB，内存完全放得下。到几百万块时，就换成 FAISS 的 ANN 索引（讲义 §4.5），或者专门的向量数据库（Milvus、Qdrant、pgvector 等）。它们做的事情本质相同：存向量、找最近的 k 个，外加持久化、过滤、分布式。

**问：新财报发布了，要重建整个索引吗？**
答：不用。对新文档单独做 ingest → chunk → embed，把新向量追加到矩阵末尾、新块追加到 `chunks.jsonl` 末尾（行号保持对应）。本项目为了简单，每次都整体重建（约 2.5 分钟）。

**问：检索用的问题和交给模型的问题一样吗？**
答：本课一样。L06 会讲"查询改写"：先让模型把问题改写成更适合检索的形式（比如补上公司全名和年份），再去检索。

---

## 6. 动手练习

1. 运行 `python walkthroughs/L05_trace_one_question.py --offline`，对照本文每一节看输出。
2. 把脚本开头的 `ITEM_ID` 改成 `fb07661`（Block），再跑一次：第 2 步有几块？第 4 步证据页的每块排第几？第 5 步给模型的材料里有没有答案那一行？
3. 把第 5 步的 `context="page"` 改成 `"chunk"`，看看同一道题交给模型的材料变成了什么样。
4. 用 `finagent search "你的问题" --full` 查几个你关心的问题，看检索到的块长什么样。
