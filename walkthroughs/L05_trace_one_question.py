"""L05 补充演示：跟着一道真实的题，把 RAG 的每一步都打印出来。

配合 lessons/L05_补充_RAG实现详解.md 使用。题目是 dev 集的 fb03282：
    "What is Netflix's year end FY2017 total current liabilities (in USD millions)? ..."
    标准答案 $5466.00（百万美元），证据在 NETFLIX_2017_10K 的第 44 页（从 0 数）

运行（先 conda activate finagent，在项目根目录；需要先 build-corpus 和 build-index）：
    python walkthroughs/L05_trace_one_question.py            # 全部 8 步，第 7 步调用一次模型，约 $0.0002
    python walkthroughs/L05_trace_one_question.py --offline  # 只跑不花钱的第 1~6 步

    离线（建索引时做的，这里只是把结果拿出来看）
      第 1 步  ingest   PDF 的一页 → 文字（两种抽取方式对比）       data/ingest.py
      第 2 步  chunk    这一页被切成了哪几块                        rag/chunker.py
      第 3 步  embed    一块文字 → 一个向量                          rag/embedder.py
    在线（每个问题都要做）
      第 4 步  retrieve 问题 → 查询向量 → 最相似的 5 块              rag/retriever.py Retriever
      第 5 步  context  5 块 → 所在的整页 → 编号的上下文             rag/retriever.py ContextBuilder
      第 6 步  prompt   上下文 + 问题 → messages                     prompts.py + answer_rag_v1.yaml
      第 7 步  generate 模型读材料、调用计算器、输出 JSON            pipeline/answer.py（L04）
      第 8 步  grade    判分 + 检索命中 + 引用核对                   eval/metrics.py、eval/retrieval.py
"""

from __future__ import annotations

import sys

import numpy as np

from finagent.config import RetrievalConfig, load_config
from finagent.data.ingest import load_pages
from finagent.eval.datasets import load_evalset
from finagent.eval.metrics import grade_item
from finagent.eval.retrieval import evidence_pages
from finagent.prompts import load_prompt
from finagent.rag.retriever import ContextBuilder, Retriever, format_context

ITEM_ID = "fb03282"
INDEX = "fb_bge_small"


def banner(title: str) -> None:
    print("\n" + "=" * 76 + f"\n{title}\n" + "=" * 76)


def show_lines(text: str, n: int, indent: str = "    ") -> None:
    lines = text.splitlines()
    for line in lines[:n]:
        print(indent + line[:100])
    if len(lines) > n:
        print(f"{indent}……（共 {len(lines)} 行）")


def main() -> None:
    offline = "--offline" in sys.argv
    item = next(it for it in load_evalset("financebench", "dev") if it.id == ITEM_ID)
    gold = evidence_pages(item)
    doc, page = next(iter(gold))
    print(f"题目：{item.question}")
    print(f"标准答案：{item.gold_text}（换算后 {item.gold_value:g} {item.gold_unit}）")
    print(f"证据：{doc} 第 {page} 页（从 0 数；PDF 阅读器里是第 {page + 1} 页）")

    # ------------------------------------------------------------ 离线
    banner("第 1 步 · ingest：PDF 的一页 → 文字（data/ingest.py）")
    text_page = next(p for p in load_pages("text") if (p.doc_name, p.page) == (doc, page))
    rows_page = next(p for p in load_pages("rows") if (p.doc_name, p.page) == (doc, page))
    print("PyMuPDF 默认输出（layout=text）：每个单元格一行")
    show_lines(text_page.text, 14)
    print("\n按坐标重建表格行（layout=rows，默认）：同一行的单元格拼回一行")
    show_lines(rows_page.text, 14)
    print(
        f"\n同一页：text 版 {len(text_page.text.splitlines())} 行，"
        f"rows 版 {len(rows_page.text.splitlines())} 行"
    )

    banner("第 2 步 · chunk：这一页被切成了哪几块（rag/chunker.py）")
    retriever = Retriever.load(INDEX)  # 读 data/index/fb_bge_small/：vectors.npy + chunks.jsonl
    embedder = retriever.embedder
    page_chunks = [c for c in retriever.chunks if (c.doc_name, c.page) == (doc, page)]
    print(f"整个索引 {len(retriever.chunks):,} 块；这一页切成了 {len(page_chunks)} 块：\n")
    for c in page_chunks:
        lines = c.text.splitlines()
        n_tokens = embedder.count_tokens([c.embed_text()])[0]
        print(f"  {c.id}   {len(lines)} 行，{n_tokens} 个 token（含标题）")
        print(f"      第一行：{lines[0][:80]}")
        print(f"      最后一行：{lines[-1][:80]}")
    if len(page_chunks) >= 2:
        a, b = page_chunks[0].text.splitlines(), page_chunks[1].text.splitlines()
        shared = [line for line in b if line in a]
        print(f"\n块 c0 和 c1 重叠的行（c1 开头重复 c0 末尾，不超过 32 个 token）：{shared[:3]}")

    banner("第 3 步 · embed：一块文字 → 一个向量（rag/embedder.py）")
    c0 = page_chunks[0]
    print("送进向量模型的文字（embed_text = 标题 + 正文）：")
    show_lines(c0.embed_text(), 4)
    position = retriever.chunks.index(c0)
    vec = retriever.index.vectors[position]  # 建索引时已经算好、存在 vectors.npy 里
    print(
        f"\n它在索引里是第 {position:,} 行；向量有 {len(vec)} 个数，前 6 个：{np.round(vec[:6], 3)}"
    )
    print(f"长度（L2 范数）= {np.linalg.norm(vec):.4f}（已归一化）")
    print(f"整个索引的向量矩阵形状：{retriever.index.vectors.shape}")

    # ------------------------------------------------------------ 在线
    banner("第 4 步 · retrieve：问题 → 查询向量 → 最相似的 5 块（Retriever.search）")
    print(f"查询前缀：{embedder.query_prefix!r}")
    q = embedder.embed_queries([item.question])  # 前缀 + 问题 → 向量 (1, 384)
    print(f"查询向量形状 {q.shape}，前 6 个数：{np.round(q[0, :6], 3)}")
    scores = retriever.index.vectors @ q[0]  # 和 4 万多块逐一算点积 = 余弦相似度
    print(
        f"和全部 {len(scores):,} 块的相似度：最高 {scores.max():.3f}，中位数 {np.median(scores):.3f}"
    )
    hits = retriever.search(item.question, 5)  # 上面两步 + 取前 5，就是 search 做的事
    print("\n前 5 名：")
    for h in hits:
        mark = "  ← 证据页" if (h.chunk.doc_name, h.chunk.page) in gold else ""
        print(f"  #{h.rank}  {h.score:.3f}  {h.chunk.id}{mark}")
    rank_of = {
        c.id: int((scores > scores[retriever.chunks.index(c)]).sum()) + 1 for c in page_chunks
    }
    print(f"\n证据页上每一块的排名：{rank_of}")

    banner("第 5 步 · context：5 块 → 所在的整页 → 编号的上下文（ContextBuilder）")
    cfg = RetrievalConfig(index=INDEX, top_k=5, context="page")
    sources = ContextBuilder(cfg, retriever).retrieve([item.question])[0]
    print(f"5 块来自 {len(sources)} 个不同的页（同一页只给一次，保持排名顺序）：")
    for s in sources:
        print(f"  [{s.ref}] {s.doc_name} 第 {s.page} 页，带进来它的块：{s.chunk_ids}")
    context = format_context(sources)
    n_tokens = embedder.count_tokens([context])[0]
    print(f"\n拼好的上下文：{len(context):,} 个字符，约 {n_tokens:,} 个 token。开头：")
    show_lines(context, 6)

    banner("第 6 步 · prompt：上下文 + 问题 → messages（answer_rag_v1）")
    prompt = load_prompt("answer_rag_v1")
    messages = prompt.build_messages(item.question, context=context)
    for m in messages:
        body = str(m["content"])
        print(f"  {m['role']:<6} {len(body):>6,} 个字符  开头：{' '.join(body.split())[:80]}…")
    print(f"  user 消息的结尾：…{' '.join(messages[-1]['content'].split())[-110:]}")

    if offline:
        print("\n（--offline：跳过第 7、8 步，不调用模型）")
        return

    banner("第 7 步 · generate：模型读材料、调用计算器、输出 JSON（L04 的阅读器）")
    from finagent.llm.openai_compat import OpenAICompatClient
    from finagent.pipeline.answer import answer_question, tool_options

    system = load_config("configs/systems/v0.5.yaml")
    client = OpenAICompatClient(system.llm, tag="walkthrough:L05_trace")
    result = answer_question(
        client, prompt, item.question, context=context, **tool_options(system.tools)
    )
    for step in result.tool_steps:
        print(f"  [工具] {step.call.name}({step.call.arguments}) → {step.result}")
    answer = result.answer
    if answer is None:
        print(f"  没有得到合格的答案：{result.errors}")
        return
    print(f"  JSON：{answer.model_dump_json(indent=None)[:400]}")
    cost = result.cost_usd or 0.0
    print(f"  调用模型 {len(result.responses)} 次，花费 ${cost:.5f}")

    banner("第 8 步 · grade：判分、检索命中、引用核对")

    def no_judge(item, answer):  # 数值题按规则判分，用不到裁判
        raise AssertionError("数值题不该交给裁判")

    grade = grade_item(item, answer, no_judge)
    print(f"  判分：{grade.label}（{grade.detail}）")
    refs = {s.ref: (s.doc_name, s.page) for s in sources}
    hit = any(key in gold for key in refs.values())
    print(f"  检索命中证据页：{hit}（给模型的 {len(sources)} 页里有没有第 {page} 页）")
    print(
        f"  模型的 sources：{answer.sources} → 对应 {[refs.get(int(s.strip('[]'))) for s in answer.sources if s.strip('[]').isdigit()]}"
    )


if __name__ == "__main__":
    main()
