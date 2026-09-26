"""L01 实验：同一批闭卷问题，对比 4 种设置的回答、token、花费和延迟。

用法（先 conda activate finagent，在项目根目录）：
    python scripts/l01_compare.py                              # 10 题 × 4 种设置，先估算费用再确认
    python scripts/l01_compare.py --limit 1 --variants flash   # 冒烟测试：1 题 × 1 种设置
    python scripts/l01_compare.py --yes                        # 跳过确认

输出：runs/l01_compare/<时间戳>/
    results.jsonl  每题每种设置一行（完整回答、usage、花费、延迟）
    report.md      汇总表 + 逐题对比，你在里面手工标注对错
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import openai

from finagent.config import PROJECT_ROOT, load_config, with_overrides
from finagent.llm.openai_compat import OpenAICompatClient
from finagent.llm.pricing import PEAK_PRICES, is_peak

QUESTIONS_FILE = PROJECT_ROOT / "evalsets" / "seed_closed_book.jsonl"
OUT_ROOT = PROJECT_ROOT / "runs" / "l01_compare"

VARIANTS: dict[str, dict] = {
    "flash": {"model": "deepseek-flash", "thinking": False},
    "flash+think": {"model": "deepseek-flash", "thinking": True, "max_tokens": 16384},
    "pro": {"model": "deepseek-v4-pro", "thinking": False},
    "pro+think": {"model": "deepseek-v4-pro", "thinking": True, "max_tokens": 16384},
}

# 花钱之前先估算。每题的输入输出 token 是拍的经验值，故意往高里估。
EST_INPUT_TOKENS = 60
EST_OUTPUT_TOKENS = {False: 500, True: 3000}  # 关 / 开 thinking


def local_now() -> datetime:
    """带时区的本地时间（用于文件夹名和报告标题）。"""
    return datetime.now(timezone.utc).astimezone()


def estimate_cost(n_questions: int, variant_names: list[str]) -> float:
    discount = 1.0 if is_peak(datetime.now(timezone.utc)) else 0.5
    total = 0.0
    for name in variant_names:
        v = VARIANTS[name]
        _, miss_price, out_price = PEAK_PRICES[v["model"]]
        per_q = EST_INPUT_TOKENS * miss_price + EST_OUTPUT_TOKENS[v["thinking"]] * out_price
        total += n_questions * per_q / 1_000_000
    return discount * total


def ask_one(client: OpenAICompatClient, variant: str, q: dict) -> dict:
    row = {"id": q["id"], "type": q["type"], "variant": variant, "question": q["question"]}
    try:
        r = client.chat([{"role": "user", "content": q["question"]}])
    except openai.OpenAIError as e:  # 一道题失败不影响其他题
        return {**row, "error": f"{type(e).__name__}: {e}"[:300]}
    return {
        **row,
        "answer": r.content,
        "finish_reason": r.finish_reason,
        "usage": r.usage.to_dict(),
        "cost_usd": r.cost_usd,
        "latency_s": round(r.latency_s, 2),
    }


def md_cell(text: str, limit: int = 300) -> str:
    """把一段回答压成一个 markdown 表格单元格：去掉换行、转义竖线、截断。"""
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit] + "…"


def write_report(path: Path, questions: list[dict], variants: list[str], rows: list[dict]) -> None:
    by_key = {(r["id"], r["variant"]): r for r in rows}
    out = [f"# L01 闭卷对比实验 · {local_now():%Y-%m-%d %H:%M}", ""]

    out += [
        "## 汇总",
        "",
        "| 设置 | 成功 | 平均延迟 | 平均输出 token | 其中思考 | 总花费 | 被截断 |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in variants:
        ok = [r for r in rows if r["variant"] == v and "error" not in r]
        n = len(ok) or 1
        out.append(
            f"| {v} | {len(ok)}/{len(questions)} "
            f"| {sum(r['latency_s'] for r in ok) / n:.1f}s "
            f"| {sum(r['usage']['completion_tokens'] for r in ok) / n:.0f} "
            f"| {sum(r['usage']['reasoning_tokens'] for r in ok) / n:.0f} "
            f"| ${sum(r['cost_usd'] or 0 for r in ok):.4f} "
            f"| {sum(r['finish_reason'] == 'length' for r in ok)} |"
        )

    out += [
        "",
        "## 逐题对比",
        "",
        (
            '在"对?"一列手工标注：✅ 正确 / ❌ 错误（编了数字）/ 🤷 承认不知道。'
            "参考答案要你自己从 10-K 里查，填进 evalsets/seed_closed_book.jsonl。"
        ),
        "",
    ]
    for q in questions:
        ref = q["answer"] if q["answer"] else "（待你从 10-K 查证）"
        out += [
            f"### {q['id']} · {q['type']} · {q['question']}",
            "",
            f"参考答案：{ref}　·　出处提示：{q['source_hint']}",
            "",
            "| 设置 | 回答 | 输出 token | 延迟 | 对? |",
            "|---|---|---|---|---|",
        ]
        for v in variants:
            r = by_key.get((q["id"], v))
            if r is None or "error" in r:
                out.append(f"| {v} | 出错：{md_cell(r['error'] if r else '缺失')} | - | - | |")
            else:
                out.append(
                    f"| {v} | {md_cell(r['answer'])} | {r['usage']['completion_tokens']} "
                    f"| {r['latency_s']}s | |"
                )
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="L01 闭卷对比实验")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--workers", type=int, default=5, help="同时进行的请求数")
    parser.add_argument("--yes", action="store_true", help="跳过费用确认")
    args = parser.parse_args()

    base = load_config("configs/systems/v0.1.yaml")
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()][: args.limit]

    est = estimate_cost(len(questions), args.variants)
    print(
        f"{len(questions)} 题 × {len(args.variants)} 种设置 = {len(questions) * len(args.variants)} 次调用，"
        f"预估花费约 ${est:.3f}（往高里估；当前{'高峰' if is_peak(datetime.now(timezone.utc)) else '非高峰'}时段）"
    )
    if not args.yes and input("继续？[y/N] ").strip().lower() != "y":
        print("已取消。")
        return

    out_dir = OUT_ROOT / local_now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for v in args.variants:
        cfg = with_overrides(base.llm, **VARIANTS[v])
        client = OpenAICompatClient(cfg, tag=f"l01_compare:{v}")
        # 同一种设置下，多道题并发请求（主要时间花在等网络，线程池就够用）。
        # 用 partial 把 client 和 v 的"当前值"固定下来。如果写成 lambda q: ask_one(client, v, q)，
        # lambda 记住的是变量名而不是值，一旦它在循环进入下一轮之后才执行，拿到的就是新值。
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(partial(ask_one, client, v), questions))
        rows += results
        ok = [r for r in results if "error" not in r]
        print(
            f"  {v:<12} 完成 {len(ok)}/{len(results)}，花费 ${sum(r['cost_usd'] or 0 for r in ok):.4f}"
        )

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_report(out_dir / "report.md", questions, args.variants, rows)
    total = sum(r.get("cost_usd") or 0 for r in rows)
    print(f"\n总花费 ${total:.4f}。报告：{out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
