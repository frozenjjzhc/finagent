"""L02 实验：同样的 10 道题、同一个模型，只换 prompt 版本（v1 / v2 / v3），比较效果。

控制变量：模型固定 deepseek-flash、关闭 thinking、问题相同，唯一的变量是 prompt。

用法（先 conda activate finagent，在项目根目录）：
    python scripts/l02_prompts.py                          # 3 个 prompt × 10 题
    python scripts/l02_prompts.py --limit 2 --yes          # 冒烟测试
    python scripts/l02_prompts.py --thinking               # 加开 thinking 再比一次

输出：runs/l02_prompts/<时间戳>/results.jsonl 和 report.md

注意：这里的 quick_grade 只是**临时判分**，规则很粗糙（容差拍脑袋、题目只有 10 道、没有置信区间）。
L03 会建立正式的评测流程，替换掉它。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import openai

from finagent.config import PROJECT_ROOT, load_config, with_overrides
from finagent.llm.openai_compat import OpenAICompatClient
from finagent.llm.pricing import PEAK_PRICES, is_peak
from finagent.pipeline.answer import answer_question, today_iso
from finagent.prompts import PromptTemplate, load_prompt
from finagent.schemas import Answer, format_value

QUESTIONS_FILE = PROJECT_ROOT / "evalsets" / "seed_closed_book.jsonl"
OUT_ROOT = PROJECT_ROOT / "runs" / "l02_prompts"
GOLD_UNIT = {"USD millions": "USD_millions", "%": "percent"}  # 评测集里的单位写法 → schema 里的写法
STATUS_ZH = {"answered": "已回答", "unknown": "不知道", "invalid_premise": "前提不成立"}


def local_now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def quick_grade(answer: Answer | None, q: dict) -> tuple[str, str]:
    """临时判分，返回（标记, 说明）。✅ 对　❌ 错　🤷 没回答　💥 输出不合格。"""
    if answer is None:
        return "💥", "输出始终没通过校验"
    if q["type"] == "trap":
        if answer.status == "invalid_premise":
            return "✅", "指出了前提错误"
        return ("🤷", "说不知道") if answer.status == "unknown" else ("❌", "编了一个答案")
    if answer.status != "answered":
        return "🤷", STATUS_ZH[answer.status]
    if answer.unit != GOLD_UNIT[q["unit"]]:
        return "❌", f"单位不对：{answer.unit}"
    gold, value = float(q["answer"]), float(answer.value)
    if answer.unit == "percent":
        ok = (
            abs(value - gold) <= 0.5
        )  # 百分比：差 0.5 个百分点以内（18.5% 和四舍五入的 19% 都算对）
    else:
        ok = abs(value - gold) <= 0.01 * abs(gold)  # 金额：相对误差 1% 以内
    return ("✅", "") if ok else ("❌", f"差 {value - gold:+,.1f}")


def estimate_cost(prompts: list[PromptTemplate], n_questions: int, model: str) -> float:
    """粗估：输入按字符数 / 3.3 折算 token（英文约 0.3 token/字符），输出按 250 token 估。"""
    discount = 1.0 if is_peak(local_now()) else 0.5
    _, miss_price, out_price = PEAK_PRICES[model]
    total = 0.0
    for p in prompts:
        chars = sum(len(str(m["content"])) for m in p.build_messages("x" * 120, today="2026-01-01"))
        total += n_questions * (chars / 3.3 * miss_price + 250 * out_price) / 1_000_000
    return discount * total


def run_one(client: OpenAICompatClient, prompt: PromptTemplate, today: str, q: dict) -> dict:
    row = {"id": q["id"], "type": q["type"], "prompt": prompt.name}
    try:
        result = answer_question(client, prompt, q["question"], today=today)
    except openai.OpenAIError as e:
        return {**row, "grade": "💥", "note": f"API 错误：{type(e).__name__}"}
    a = result.answer
    grade, note = quick_grade(a, q)
    return {
        **row,
        "grade": grade,
        "note": note,
        "answer": a.model_dump() if a else None,
        "attempts": result.attempts,
        "errors": result.errors,
        "raw_outputs": [r.content for r in result.responses],
        "cost_usd": result.cost_usd,
        "latency_s": round(result.latency_s, 2),
        "prompt_tokens": sum(r.usage.prompt_tokens for r in result.responses),
        "cache_hit_tokens": sum(r.usage.cache_hit_tokens for r in result.responses),
    }


def write_report(
    path: Path, meta: dict, questions: list[dict], prompts: list[str], rows: list[dict]
) -> None:
    by_key = {(r["id"], r["prompt"]): r for r in rows}
    out = [
        f"# L02 prompt 对比实验 · {meta['time']}",
        "",
        f"模型 {meta['model']}，thinking={meta['thinking']}，日期 {meta['today']}，每个 prompt 跑 {len(questions)} 题。",
        "判分是**临时规则**（金额相对误差 ≤ 1%，百分比 ≤ 0.5 个百分点），L03 会正式建立评测。",
        "",
        "## 汇总",
        "",
        "| prompt | ✅ 对 | ❌ 错 | 🤷 没答 | 💥 不合格 | 第一次就合格 | 平均尝试 | 输入 token（缓存命中） | 花费 | 平均延迟 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in prompts:
        rs = [r for r in rows if r["prompt"] == p]
        c = Counter(r["grade"] for r in rs)
        ok = [r for r in rs if "attempts" in r]
        first_ok = sum(1 for r in ok if r["attempts"] == 1 and r["answer"] is not None)
        n = len(ok) or 1
        out.append(
            f"| {p} | {c['✅']} | {c['❌']} | {c['🤷']} | {c['💥']} | {first_ok}/{len(rs)} "
            f"| {sum(r['attempts'] for r in ok) / n:.2f} "
            f"| {sum(r['prompt_tokens'] for r in ok):,}（{sum(r['cache_hit_tokens'] for r in ok):,}） "
            f"| ${sum(r['cost_usd'] or 0 for r in ok):.4f} | {sum(r['latency_s'] for r in ok) / n:.1f}s |"
        )

    out += [
        "",
        "## 逐题结果",
        "",
        "| 题 | 标准答案 | " + " | ".join(prompts) + " |",
        "|---|---|" + "---|" * len(prompts),
    ]
    for q in questions:
        gold = q["answer"] if q["type"] == "trap" else f"{q['answer']} {q['unit']}"
        cells = []
        for p in prompts:
            r = by_key.get((q["id"], p))
            if r is None or r.get("answer") is None:
                cells.append(f"{r['grade'] if r else '?'} {r.get('note', '') if r else ''}")
                continue
            a = Answer.model_validate(r["answer"])
            shown = format_value(a) if a.status == "answered" else STATUS_ZH[a.status]
            cells.append(
                f"{r['grade']} {shown}（{a.confidence}）{' ' + r['note'] if r['note'] else ''}"
            )
        out.append(f"| {q['id']} {q['type']} | {gold} | " + " | ".join(cells) + " |")

    # 自评信心和对错的关系：模型说"高信心"的时候，真的更容易对吗？
    out += [
        "",
        "## 自评信心 vs 实际对错（所有 prompt 合计）",
        "",
        "| 自评信心 | ✅ | ❌ | 🤷 |",
        "|---|---|---|---|",
    ]
    for level in ("high", "medium", "low"):
        c = Counter(
            r["grade"] for r in rows if r.get("answer") and r["answer"]["confidence"] == level
        )
        out.append(f"| {level} | {c['✅']} | {c['❌']} | {c['🤷']} |")

    retried = [r for r in rows if r.get("errors")]
    if retried:
        out += ["", "## 校验失败记录", ""]
        for r in retried:
            out.append(f"- {r['prompt']} / {r['id']}：{'；'.join(r['errors'])}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="L02 prompt 对比实验")
    parser.add_argument("--prompts", nargs="+", default=["answer_v1", "answer_v2", "answer_v3"])
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题")
    parser.add_argument("--thinking", action="store_true", help="开启 thinking（默认关闭）")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--yes", action="store_true", help="跳过费用确认")
    args = parser.parse_args()

    base = load_config("configs/systems/v0.2.yaml")
    cfg = with_overrides(base.llm, thinking=args.thinking)
    prompts = [load_prompt(n) for n in args.prompts]
    with QUESTIONS_FILE.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()][: args.limit]

    est = estimate_cost(prompts, len(questions), cfg.model) * (5 if args.thinking else 1)
    print(f"{len(prompts)} 个 prompt × {len(questions)} 题，预估花费约 ${est:.3f}（往高里估）")
    if not args.yes and input("继续？[y/N] ").strip().lower() != "y":
        print("已取消。")
        return

    today = today_iso()  # 整个实验用同一个日期，保证可复现
    out_dir = OUT_ROOT / local_now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for prompt in prompts:
        client = OpenAICompatClient(cfg, tag=f"l02_prompts:{prompt.name}")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(partial(run_one, client, prompt, today), questions))
        rows += results
        c = Counter(r["grade"] for r in results)
        print(f"  {prompt.name}: ✅{c['✅']} ❌{c['❌']} 🤷{c['🤷']} 💥{c['💥']}")

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "time": f"{local_now():%Y-%m-%d %H:%M}",
        "model": cfg.model,
        "thinking": cfg.thinking,
        "today": today,
    }
    write_report(out_dir / "report.md", meta, questions, [p.name for p in prompts], rows)
    print(
        f"\n总花费 ${sum(r.get('cost_usd') or 0 for r in rows):.4f}。报告：{out_dir / 'report.md'}"
    )


if __name__ == "__main__":
    main()
