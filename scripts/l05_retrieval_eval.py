"""L05 实验：只评检索，不调用大模型（免费、几秒钟）。

对 FinanceBench 的每道题：用问题去检索，看前 k 个文本块里有没有证据页（见 eval/retrieval.py）。
可以一次比较多个索引（不同的抽取方式、加不加标题……），只在 dev 上做，test 留给里程碑。

用法（先 conda activate finagent，在项目根目录；索引要先用 finagent build-index 建好）：
    python scripts/l05_retrieval_eval.py                              # 默认索引 fb_bge_small
    python scripts/l05_retrieval_eval.py fb_bge_small fb_bge_small_notitle fb_bge_small_text
    python scripts/l05_retrieval_eval.py --show-misses                # 列出没命中的题，做错误分析

输出：runs/l05_retrieval/<时间戳>/report.md 和 results.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from finagent.config import PROJECT_ROOT
from finagent.eval.datasets import load_evalset
from finagent.eval.metrics import bootstrap_ci
from finagent.eval.retrieval import evidence_pages, first_doc_rank, first_hit_rank, mrr, recall_at
from finagent.rag.retriever import Retriever

OUT_ROOT = PROJECT_ROOT / "runs" / "l05_retrieval"
KS = (1, 3, 5, 10, 20)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("indexes", nargs="*", default=["fb_bge_small"])
    parser.add_argument("--split", default="dev", help="dev（默认）或 test。test 只在里程碑看")
    parser.add_argument("--show-misses", action="store_true", help="列出 top-5 没命中的题")
    args = parser.parse_args()

    items = load_evalset("financebench", args.split)
    out_dir = OUT_ROOT / datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    lines = [
        f"# L05 · 检索评测（FinanceBench {args.split}，{len(items)} 题）",
        "",
        "页命中：前 k 个块里有证据页。文档命中：前 k 个块里有证据所在的文档。",
        "",
        "| 索引 | 块数 | "
        + " | ".join(f"R@{k}" for k in KS)
        + " | 文档 R@5 | MRR | R@5 的 95% 区间 |",
        "|---|---|" + "---|" * (len(KS) + 3),
    ]
    rows = []
    for name in args.indexes:
        retriever = Retriever.load(name)
        results = retriever.search_many([it.question for it in items], max(KS))
        ranks, doc_ranks = [], []
        for it, hits in zip(items, results, strict=True):
            keys = [(h.chunk.doc_name, h.chunk.page) for h in hits]
            gold = evidence_pages(it)
            r, dr = first_hit_rank(keys, gold), first_doc_rank(keys, gold)
            ranks.append(r)
            doc_ranks.append(dr)
            rows.append({
                "index": name, "id": it.id, "category": it.category, "question": it.question,
                "evidence": sorted(gold), "hit_rank": r, "doc_hit_rank": dr,
                "top5": [f"{h.chunk.doc_name} p{h.chunk.page} ({h.score:.3f})" for h in hits[:5]],
            })  # fmt: skip
        _, lo, hi = bootstrap_ci([1.0 if r is not None and r <= 5 else 0.0 for r in ranks])
        lines.append(
            f"| {name} | {len(retriever.chunks):,} | "
            + " | ".join(f"{recall_at(ranks, k):.0%}" for k in KS)
            + f" | {recall_at(doc_ranks, 5):.0%} | {mrr(ranks):.3f} | {lo:.0%} ~ {hi:.0%} |"
        )
        if args.show_misses:
            lines += ["", f"### {name}：top-5 没有命中证据页的题", ""]
            for row in rows:
                if row["index"] == name and (row["hit_rank"] is None or row["hit_rank"] > 5):
                    lines.append(f"- **{row['id']}**（{row['category']}）{row['question'][:120]}")
                    lines.append(
                        f"  - 证据：{row['evidence']}；第一次命中：第 {row['hit_rank']} 名"
                    )
                    lines.append(f"  - 检索到：{'；'.join(row['top5'])}")
    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(report)
    print(f"记录目录：{out_dir}")


if __name__ == "__main__":
    main()
