"""L05 演示：向量检索是怎么工作的。全部在本地运行，不调用大模型，不花钱。

运行（先 conda activate finagent，在项目根目录）：
    python walkthroughs/L05_retrieval.py          # 第 1~3 步只需要向量模型；第 4~5 步需要先建好索引
第一次运行会下载向量模型 bge-small（约 130MB）。

    第 1 步  5 个玩具文本块：相似度矩阵，语义匹配 vs 字面匹配，以及表格行的麻烦
    第 2 步  向量长什么样：384 个数、长度为 1、点积就是余弦相似度
    第 3 步  查询前缀：bge 要求问题前面加一句话，在真实的 dev 题上加不加差多少
    第 4 步  暴力检索 vs FAISS 近似检索（ANN）：速度和准确度的交换
    第 5 步  一道真实的题：检索到了什么，证据页排第几
（第 3~5 步需要先运行 finagent build-corpus 和 finagent build-index）
"""

from __future__ import annotations

import time

import numpy as np

from finagent.config import load_index_config
from finagent.eval.datasets import load_evalset
from finagent.eval.retrieval import evidence_pages, first_hit_rank, recall_at
from finagent.rag.retriever import INDEX_ROOT, Retriever, make_embedder

# 前 4 块是普通的句子，各自回答一个问题；第 5 块是一行表格（财报里大量存在的样子）
CHUNKS = [
    "The company invested $1.6 billion in new factories and machinery during 2018.",
    "Revenue rose 3.5 percent to $32.8 billion, driven by strong demand in Asia.",
    "The Company paid dividends of $3.2 billion to shareholders during the year.",
    "Research, development and related expenses were $1.8 billion.",
    "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)",
]
QUESTIONS = [  # 第 i 个问题本该匹配第 i 块
    "How much did the company spend on capital expenditures?",
    "How fast did sales grow?",
    "How much cash was returned to shareholders?",
    "What was R&D spending?",
]


def banner(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def word_overlap(a: str, b: str) -> float:
    """字面匹配：两段文字共有的词占全部词的比例（Jaccard）。关键词检索的直觉。"""
    wa, wb = set(a.lower().replace("?", "").split()), set(b.lower().replace(",", "").split())
    return len(wa & wb) / len(wa | wb)


def step1_matrix(embedder) -> None:
    banner("第 1 步 · 相似度矩阵：语义匹配 vs 字面匹配")
    for i, c in enumerate(CHUNKS, 1):
        print(f"  块{i}：{c}")
    q = embedder.embed_queries(QUESTIONS)
    p = embedder.embed_passages(CHUNKS)
    sims = q @ p.T  # (4 个问题, 5 个块)：每个格子是一个余弦相似度
    print("\n行 = 问题，列 = 文本块 1~5。每行最大的值标 *，第 i 个问题本该匹配第 i 块\n")
    print(" " * 46 + "".join(f"  块{j + 1}  " for j in range(len(CHUNKS))))
    best = sims.argmax(axis=1)
    for i, question in enumerate(QUESTIONS):
        cells = "".join(f" {s:.3f}{'*' if j == best[i] else ' '} " for j, s in enumerate(sims[i]))
        print(f"{question[:44]:<46}{cells}  {'✔' if best[i] == i else '✘'}")
    print("\n同样的矩阵，换成字面匹配（共有词的比例）：\n")
    for question in QUESTIONS:
        cells = "".join(f" {word_overlap(question, c):.3f}  " for c in CHUNKS)
        print(f"{question[:44]:<46}{cells}")

    print("\n读法（根据上面的实际结果，逐行解释）：")
    table = len(CHUNKS) - 1
    for i in range(len(QUESTIONS)):
        overlap = word_overlap(QUESTIONS[i], CHUNKS[i])
        if best[i] == i and overlap == 0:
            print(
                f"  - 第 {i + 1} 行 ✔：问题和块{i + 1} 一个词都不重合（字面匹配是 0），向量模型仍然把它排第一。\n"
                "    这就是语义匹配：关键词检索做不到"
            )
        elif best[i] == i:
            print(f"  - 第 {i + 1} 行 ✔：匹配正确（两者也有共同的词，字面匹配同样能找到）")
        elif best[i] == table:
            print(
                f"  - 第 {i + 1} 行 ✘：排第一的是块 5，一行满是数字的表格。表格行几乎没有能理解的文字，\n"
                '    它的向量和什么问题都"有点像"，于是可能抢走第一名。财报里到处是这种表格，\n'
                "    这是财报检索的难点之一（L05 的对策：块首加文档标题、交给模型整页；L06 还有 BM25 和重排）"
            )
        else:
            print(
                f"  - 第 {i + 1} 行 ✘：排第一的是块{best[i] + 1}，不是块{i + 1}。意思相近的概念\n"
                '    （比如都是"花了多少钱"）小模型常常分不清：向量检索的结果是"相似"，不保证"正确"'
            )
    print(
        f"  - 所有相似度都挤在 {sims.min():.2f} ~ {sims.max():.2f} 之间：绝对值没有意义，只看排名。\n"
        '    所以不能用"相似度 > 0.7 才算相关"这种固定阈值'
    )


def step2_vector(embedder) -> None:
    banner("第 2 步 · 向量长什么样")
    v = embedder.embed_passages(CHUNKS[:2])
    print(f"每个文本块 → {v.shape[1]} 个数。第 1 块的前 8 个：{np.round(v[0, :8], 3)}")
    print(
        f"长度（L2 范数）：{np.linalg.norm(v[0]):.4f}，{np.linalg.norm(v[1]):.4f}  ← 都已归一化成 1"
    )
    dot = float(v[0] @ v[1])
    cos = dot / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]))
    print(f"点积 {dot:.4f} = 余弦相似度 {cos:.4f}：长度为 1 时两者相等，检索只需做矩阵乘法")


def step3_prefix(retriever: Retriever) -> None:
    banner("第 3 步 · 查询前缀（在真实的 dev 题上比较）")
    embedder = retriever.embedder
    print(f"bge 的查询前缀：{embedder.query_prefix!r}\n")
    items = load_evalset("financebench", "dev")
    questions = [it.question for it in items]
    for label, queries in [
        ("加前缀（默认）", embedder.embed_queries(questions)),
        ("不加前缀", embedder.embed_passages(questions)),  # 把问题当成普通文本来编码
    ]:
        _, ids = retriever.index.search(queries, 10)
        ranks = [
            first_hit_rank([(retriever.chunks[j].doc_name, retriever.chunks[j].page) for j in row],
                           evidence_pages(it))
            for row, it in zip(ids, items, strict=True)
        ]  # fmt: skip
        print(f"  {label:<10} R@5 = {recall_at(ranks, 5):.0%}   R@10 = {recall_at(ranks, 10):.0%}")
    print(
        "\n训练时问题都带着这句前缀、文本块不带，用的时候也要照做。差距有多大，以上面的实际数字为准；\n"
        "50 道题上差几个百分点，也可能只是运气（L03 的教训）。"
    )


def step4_ann(retriever: Retriever) -> None:
    banner("第 4 步 · 暴力检索 vs FAISS 近似检索（ANN）")
    import faiss

    from finagent.eval.datasets import load_evalset
    from finagent.rag.index import FaissIndex, NumpyIndex

    vectors = np.load(INDEX_ROOT / retriever.cfg.name / "vectors.npy")
    queries = retriever.embedder.embed_queries(
        [it.question for it in load_evalset("financebench", "dev")]
    )
    print(
        f"{len(vectors):,} 个块 × {vectors.shape[1]} 维，{len(queries)} 个问题，每个问题取前 10\n"
    )
    exact = NumpyIndex(vectors)
    _, truth = exact.search(queries, 10)
    print(f"{'索引':<22}{'建索引':>8}{'查 50 题':>10}{'和精确结果的重合':>18}")
    for label, factory in [("numpy 暴力检索", None), ("FAISS Flat（精确）", "Flat"),
                           ("FAISS HNSW32", "HNSW32"), ("FAISS IVF256", "IVF256,Flat")]:  # fmt: skip
        t0 = time.perf_counter()
        index = exact if factory is None else FaissIndex(vectors, factory)
        if factory == "IVF256,Flat":
            faiss.extract_index_ivf(index.index).nprobe = 8  # 只查 256 个簇里最近的 8 个
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        _, ids = index.search(queries, 10)
        query = time.perf_counter() - t0
        overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth, strict=True)])
        print(f"{label:<22}{build:>7.2f}s{query * 1000:>8.1f}ms{overlap:>17.0%}")
    print(
        "\n4 万个向量时，暴力检索本身就只要几毫秒，完全够用，所以项目默认用它。\n"
        'ANN 的价值在几百万、上亿个向量时才显现：HNSW 像"跳表 + 社交网络"，从一个点出发沿着\n'
        "邻居一路走向目标；IVF 先把向量聚成若干簇，只在离问题最近的几个簇里找。\n"
        "代价是可能漏掉一些真正最近的邻居（上表最后一列不到 100%）。"
    )


def step5_example(retriever: Retriever) -> None:
    banner("第 5 步 · 一道真实的题（dev 集，test 的题只在里程碑看）")
    item = next(it for it in load_evalset("financebench", "dev") if it.id == "fb03282")
    gold = {(e["doc_name"], e["page"]) for e in item.evidence}
    print(f"问题：{item.question}\n证据页：{sorted(gold)}（页码从 0 开始）\n")
    hits = retriever.search(item.question, 10)
    for h in hits:
        c = h.chunk
        mark = "  ← 证据页" if (c.doc_name, c.page) in gold else ""
        print(f"  #{h.rank:<2} {h.score:.3f}  {c.doc_name} p{c.page}{mark}")
        if h.rank <= 2 or mark:
            print(f"        {' '.join(c.text.split())[:110]}…")
    gold_ranks = [h.rank for h in hits if (h.chunk.doc_name, h.chunk.page) in gold]
    if len(gold_ranks) >= 2:
        print(
            f"\n证据页上的块出现在第 {gold_ranks} 名。同一页被切成了好几块，各自的排名不同：\n"
            "看看每一块的开头，真正写着答案数字的那一块，排名可能并不靠前。\n"
            "v0.5 的做法（context: page）是：只要这一页的**任何一块**进了前 5，就把整页交给模型，\n"
            "数字和表头都在里面。这就是 small-to-big：用小块检索，用整页回答。"
        )


def main() -> None:
    cfg = load_index_config("fb_bge_small")
    embedder = make_embedder(cfg)
    step1_matrix(embedder)
    step2_vector(embedder)
    if not (INDEX_ROOT / cfg.name / "meta.json").exists():
        print("\n第 3~5 步需要索引：先运行 finagent build-corpus 和 finagent build-index")
        return
    retriever = Retriever.load(cfg.name)
    step3_prefix(retriever)
    step4_ann(retriever)
    step5_example(retriever)


if __name__ == "__main__":
    main()
