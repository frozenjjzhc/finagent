"""L04 实验一：纯算术探针。模型心算到底有多不靠谱？计算器能补上多少？

和 FinQA 实验的区别：这里**只考算术**。题目里直接给出所有数字，不需要读财报、不需要理解口径，
答错只可能是算错。所以它能干净地回答"LLM 算不好数"这个问题，而 FinQA 的结果还混着读错数、
理解错题意等其他错误。

五类题（每类 --per-type 道，数字随机生成，固定种子）：
    pct4     两个 4 位数的百分比变化，保留 2 位小数
    pct7     两个 7 位数的百分比变化，保留 2 位小数
    product  5 位数 × 4 位数，要精确到个位
    cagr     5 年复合增长率（要开 5 次方），保留 2 位小数
    multi    (A + B + C) / (D - E)，6 位数，保留 4 位小数

对比的几种做法（同一个模型 deepseek-flash）：
    心算            关闭 thinking，要求在 reasoning 里写出过程
    计算器          关闭 thinking，带 calculate 工具，要求每一步都用它
    心算+thinking   （加 --thinking 时）打开 thinking，让模型先想再答

用法（先 conda activate finagent，在项目根目录）：
    python scripts/l04_arith_probe.py                    # 5 类 × 10 题 × 2 种做法
    python scripts/l04_arith_probe.py --thinking         # 再加 thinking 组
    python scripts/l04_arith_probe.py --per-type 2 --yes # 冒烟测试

输出：runs/l04_arith/<时间戳>/results.jsonl 和 report.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial

import openai
from pydantic import BaseModel, ConfigDict, Field

from finagent.agent.loop import run_tool_loop
from finagent.config import PROJECT_ROOT, load_config, with_overrides
from finagent.llm.openai_compat import OpenAICompatClient
from finagent.llm.structured import call_structured
from finagent.tools import build_registry

OUT_ROOT = PROJECT_ROOT / "runs" / "l04_arith"

SYSTEM = """You are a careful financial calculator. Solve the arithmetic problem exactly.

Respond with a single JSON object with exactly these fields, in this order:
{"reasoning": "show the calculation step by step, briefly", "value": number}
"value" must be a plain number (no commas, no % sign), rounded as the problem asks."""

TOOL_INSTRUCTION = """

Calculator: you have a "calculate" tool. Use it for every arithmetic step instead of computing
in your head. Only after you have all the results, give your final answer."""


class NumericAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str = Field(max_length=3000)
    value: float


# ---------------------------------------------------------------- 出题


def make_problems(per_type: int, seed: int = 0) -> list[dict]:
    """生成题目和精确答案。tol 是允许的误差：题目要求保留 2 位小数，就允许四舍五入带来的 0.005。"""
    rng = random.Random(seed)
    problems = []

    def add(kind: str, question: str, gold: float, tol: float) -> None:
        problems.append(
            {"id": f"{kind}_{sum(p['kind'] == kind for p in problems) + 1:02d}",
             "kind": kind, "question": question, "gold": gold, "tol": tol}
        )  # fmt: skip

    for _ in range(per_type):
        a = rng.randint(1000, 9999)
        b = round(a * rng.uniform(0.7, 1.4))
        add(
            "pct4",
            f"Revenue was {a} last year and {b} this year. What is the percentage change? "
            "Answer in percent, rounded to 2 decimal places.",
            (b - a) / a * 100,
            0.005,
        )
        a = rng.randint(1_000_000, 9_999_999)
        b = round(a * rng.uniform(0.7, 1.4))
        add(
            "pct7",
            f"Revenue was {a} last year and {b} this year. What is the percentage change? "
            "Answer in percent, rounded to 2 decimal places.",
            (b - a) / a * 100,
            0.005,
        )
        units, price = rng.randint(10_000, 99_999), rng.randint(1000, 9999)
        add(
            "product",
            f"A company sold {units} units at {price} dollars each. "
            "What was the total revenue in dollars? Give the exact integer.",
            units * price,
            0.5,
        )
        a = rng.randint(10_000, 99_999)
        b = round(a * rng.uniform(1.2, 3.0))
        add(
            "cagr",
            f"Revenue grew from {a} to {b} over 5 years. What is the compound annual growth rate? "
            "Answer in percent, rounded to 2 decimal places.",
            ((b / a) ** (1 / 5) - 1) * 100,
            0.005,
        )
        x = [rng.randint(100_000, 999_999) for _ in range(5)]
        x[3] = x[3] + 1_000_000  # 保证 D - E 为正
        add(
            "multi",
            f"Compute (A + B + C) / (D - E) where A = {x[0]}, B = {x[1]}, C = {x[2]}, "
            f"D = {x[3]}, E = {x[4]}. Round to 4 decimal places.",
            (x[0] + x[1] + x[2]) / (x[3] - x[4]),
            0.00005,
        )
    return problems


# ---------------------------------------------------------------- 做题


def solve(problem: dict, arm: str, clients: dict, registry) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM + (TOOL_INSTRUCTION if arm == "计算器" else "")},
        {"role": "user", "content": problem["question"]},
    ]
    client = clients[arm]
    generate: Callable | None = None
    if arm == "计算器":
        generate = partial(run_tool_loop, client, registry=registry, json_mode=False)
    try:
        res = call_structured(client, messages, NumericAnswer, max_attempts=2, generate=generate)
    except openai.OpenAIError as e:
        return {**problem, "arm": arm, "value": None, "correct": False, "error": type(e).__name__}
    value = res.value.value if res.value else None
    return {
        **problem,
        "arm": arm,
        "value": value,
        "correct": value is not None and abs(value - problem["gold"]) <= problem["tol"],
        "rel_error": None if value is None else abs(value - problem["gold"]) / abs(problem["gold"]),
        "reasoning": res.value.reasoning if res.value else None,
        "tool_calls": [s.call.arguments for s in res.tool_steps],
        "llm_calls": len(res.responses),
        "cost_usd": res.cost_usd,
        "latency_s": round(res.latency_s, 2),
    }


# ---------------------------------------------------------------- 报告


def render_report(rows: list[dict], arms: list[str]) -> str:
    kinds = list(dict.fromkeys(r["kind"] for r in rows))
    out = [
        "# L04 · 纯算术探针",
        "",
        "准确率（答对 / 题数）。答对 = 误差在题目要求的精度以内（2 位小数题允许 ±0.005，整数题要精确）。",
        "",
        "| 题型 | 题数 | " + " | ".join(arms) + " |",
        "|---|---|" + "---|" * len(arms),
    ]
    for kind in kinds + ["合计"]:
        cells = []
        for arm in arms:
            sel = [r for r in rows if r["arm"] == arm and kind in (r["kind"], "合计")]
            cells.append(f"{sum(r['correct'] for r in sel)}/{len(sel)}")
        n = sum(1 for r in rows if r["arm"] == arms[0] and kind in (r["kind"], "合计"))
        out.append(f"| {kind} | {n} | " + " | ".join(cells) + " |")
    out += [
        "",
        "| 做法 | 相对误差中位数（答错的题） | 平均调用模型 | 用了计算器的题 | 总花费 |",
        "|---|---|---|---|---|",
    ]
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        wrong = [r["rel_error"] for r in sel if not r["correct"] and r.get("rel_error") is not None]
        med = f"{statistics.median(wrong):.2e}" if wrong else "—"
        calls = sum(r.get("llm_calls", 0) for r in sel) / len(sel)
        used = sum(bool(r.get("tool_calls")) for r in sel)
        cost = sum(r.get("cost_usd") or 0 for r in sel)
        out.append(f"| {arm} | {med} | {calls:.2f} | {used}/{len(sel)} | ${cost:.4f} |")
    out += ["", "## 答错的题", "", "| 做法 | id | 标准答案 | 模型答案 |", "|---|---|---|---|"]
    for r in rows:
        if not r["correct"]:
            out.append(f"| {r['arm']} | {r['id']} | {r['gold']:.6g} | {r['value']} |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--thinking", action="store_true", help="加一组：心算 + thinking")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--yes", action="store_true", help="跳过费用确认")
    args = parser.parse_args()

    problems = make_problems(args.per_type)
    arms = ["心算", "计算器"] + (["心算+thinking"] if args.thinking else [])
    # 粗估：每题每种做法约 600 输入 + 300 输出 token；计算器组按 3 次调用算，thinking 组按 2000 输出算
    est = len(problems) * (0.0005 + 0.0015 + (0.0025 if args.thinking else 0))
    print(f"{len(problems)} 题 × {len(arms)} 种做法，预估花费约 ${est:.3f}（往高里估）")
    if not args.yes and input("继续？[y/N] ").strip().lower() != "y":
        return

    base = load_config("configs/systems/v0.4.yaml").llm
    clients = {
        "心算": OpenAICompatClient(base, tag="l04_arith:心算"),
        "计算器": OpenAICompatClient(base, tag="l04_arith:计算器"),
        "心算+thinking": OpenAICompatClient(
            with_overrides(base, thinking=True, max_tokens=8192), tag="l04_arith:thinking"
        ),
    }
    registry = build_registry(["calculate"])
    jobs = [(p, arm) for arm in arms for p in problems]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda job: solve(job[0], job[1], clients, registry), jobs))

    out_dir = OUT_ROOT / datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = render_report(rows, arms)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"记录目录：{out_dir}")


if __name__ == "__main__":
    main()
