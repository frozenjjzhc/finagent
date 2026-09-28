"""检索指标（L05）：检索有没有找到标准答案标注的证据页。

FinanceBench 的每道题都标了证据所在的（文档, 页码）。检索到的文本块也有（文档, 页码），
两者一比，就能自动判断检索对不对，不需要调用模型，也不花钱。

    命中（hit）    前 k 个结果里，至少有一个块来自证据页（有多个证据页时，找到任意一页就算）
    Recall@k      命中的题目 / 全部题目
    文档命中       前 k 个结果里，至少有一个块来自证据所在的文档（页不对也算）。
                  它比页命中宽松，两者的差距说明"找对了报告，但没找对页"的有多少
    MRR           第一个命中的排名的倒数的平均：第 1 名命中得 1 分，第 2 名 0.5 分，没命中 0 分。
                  Recall@k 只管"有没有"，MRR 还管"排得靠不靠前"

例子：证据在 3M_2018_10K 的第 59 页。检索结果前 5 个是
    (3M_2022_10K, 40)、(3M_2018_10K, 58)、(3M_2018_10K, 59)、…
    → Recall@1 = 0，Recall@3 = 1，文档命中@1 = 0（3M 2022 不是这份文档），文档命中@2 = 1，MRR = 1/3
"""

from __future__ import annotations

from collections.abc import Sequence

from finagent.eval.datasets import EvalItem

Key = tuple[str, int]  # （文档名, 页码）


def evidence_pages(item: EvalItem) -> set[Key]:
    return {(e["doc_name"], e["page"]) for e in item.evidence}


def first_hit_rank(retrieved: Sequence[Key], gold: set[Key]) -> int | None:
    """第一个命中的排名（从 1 开始）；没有命中返回 None。"""
    return next((i for i, key in enumerate(retrieved, 1) if key in gold), None)


def first_doc_rank(retrieved: Sequence[Key], gold: set[Key]) -> int | None:
    docs = {doc for doc, _ in gold}
    return next((i for i, (doc, _) in enumerate(retrieved, 1) if doc in docs), None)


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: Sequence[int | None]) -> float:
    return sum(1 / r for r in ranks if r is not None) / len(ranks) if ranks else 0.0
