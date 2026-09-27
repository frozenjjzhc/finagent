"""评测运行器：系统答题 → 判分 → 汇总 → 写入一个不可变的记录目录。

每次评测生成 runs/eval/<时间>_<系统>_<prompt>_<评测集>-<划分>/，里面有：
  results.jsonl     每道题一行：题目、标准答案、系统回答、判分、花费
  summary.json      汇总指标（含置信区间）、按题型拆分、错误类型、成本
  report.md         给人看的报告
  config.yaml       被评测系统的配置快照（含实际使用的 prompt 名）
  prompt.yaml       prompt 文件的快照（prompt 以后改了，也知道当时用的是什么）
  eval.yaml         评测设置快照（裁判、容差、bootstrap）
  meta.json         git commit、评测集文件哈希、日期参数、依赖版本、耗时
目录一旦生成就不再修改：SCOREBOARD 的每一行都指向这样一个目录，保证可复现。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import openai
import yaml

from finagent.config import PROJECT_ROOT, EvalConfig, LLMConfig, SystemConfig
from finagent.eval.datasets import EVALSETS_DIR, EvalItem, load_evalset
from finagent.eval.judge import Judge
from finagent.eval.metrics import (
    Grade,
    bootstrap_ci,
    grade_item,
    mcnemar_exact,
    paired_bootstrap_diff,
    percentile,
    summarize,
)
from finagent.llm.base import LLMClient
from finagent.llm.openai_compat import OpenAICompatClient
from finagent.llm.pricing import PEAK_PRICES, is_peak
from finagent.pipeline.answer import AnswerResult, answer_question, today_iso
from finagent.prompts import PROMPTS_DIR, load_prompt
from finagent.schemas import Answer, format_value

RUNS_EVAL = PROJECT_ROOT / "runs" / "eval"
ClientFactory = Callable[[LLMConfig, str], LLMClient]
LABEL_ICON = {"correct": "✅", "incorrect": "❌", "abstained": "🤷", "invalid": "💥", "error": "⚠️"}


def local_now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def git_info() -> dict:
    """当前代码的 commit，以及有没有未提交的修改（有的话，结果就不能严格复现）。"""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):  # 没装 git，或者不在 git 仓库里
        return {"commit": None, "dirty": None}


def file_sha256(path: Path) -> str:
    """文件哈希。先把 Windows 换行（CRLF）统一成 LF：git 在 Windows 上可能转换换行，但内容其实没变。"""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def estimate_cost(
    system: SystemConfig, prompt_name: str, items: list[EvalItem], eval_cfg: EvalConfig
) -> float:
    """花钱之前先估算（往高里估）：系统每题约 300 个输出 token；文字题再加一次裁判调用。"""
    discount = 1.0 if is_peak(local_now()) else 0.5
    prompt = load_prompt(prompt_name)
    chars = sum(
        len(str(m["content"])) for m in prompt.build_messages("x" * 200, today="2026-01-01")
    )
    _, s_miss, s_out = PEAK_PRICES.get(system.llm.model, (0, 1.32, 3.96))
    _, j_miss, j_out = PEAK_PRICES.get(eval_cfg.judge.llm.model, (0, 1.32, 3.96))
    per_item = chars / 3.3 * s_miss + 300 * s_out
    per_judge = 1500 / 3.3 * j_miss + 150 * j_out
    n_text = sum(1 for it in items if it.answer_type == "text")
    return discount * (len(items) * per_item + n_text * per_judge) / 1_000_000


def run_eval(
    system: SystemConfig,
    set_name: str,
    split: str,
    eval_cfg: EvalConfig,
    *,
    prompt_name: str | None = None,
    today: str | None = None,
    limit: int | None = None,
    workers: int = 5,
    out_root: Path = RUNS_EVAL,
    make_client: ClientFactory = OpenAICompatClient,
) -> Path:
    """跑一次评测，返回记录目录。make_client 可以替换成假客户端，方便不联网测试。"""
    if system.prompt is None:
        raise ValueError("评测需要结构化输出：系统配置里必须有 prompt（v0.2 起）")
    prompt_name = prompt_name or system.prompt.name
    prompt = load_prompt(prompt_name)
    items = load_evalset(set_name, split)[:limit]
    today = today or today_iso()  # 整次评测用同一个日期，并记录下来，保证可复现

    started, t0 = local_now(), time.perf_counter()
    run_dir = out_root / f"{started:%Y%m%d-%H%M%S}_{system.name}_{prompt_name}_{set_name}-{split}"
    run_dir.mkdir(parents=True, exist_ok=False)  # 目录已存在就报错：绝不覆盖旧记录
    sys_client = make_client(system.llm, f"eval:{run_dir.name}")
    judge = Judge(
        make_client(eval_cfg.judge.llm, f"judge:{run_dir.name}"),
        load_prompt(eval_cfg.judge.prompt),
        eval_cfg.judge.max_attempts,
    )

    def process(item: EvalItem) -> tuple[EvalItem, AnswerResult | None, Grade]:
        try:
            res = answer_question(
                sys_client,
                prompt,
                item.question,
                today=today,
                max_attempts=system.prompt.max_attempts,
            )
        except openai.OpenAIError as e:  # 一道题出错不影响其他题
            return item, None, Grade("error", "none", f"API 错误：{type(e).__name__}", "api_error")
        try:
            grade = grade_item(item, res.answer, judge, eval_cfg.relative_tolerance)
        except openai.OpenAIError as e:
            grade = Grade("error", "judge", f"裁判 API 错误：{type(e).__name__}", "judge_api_error")
        return item, res, grade

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(process, items))

    rows = [_row(item, res, grade) for item, res, grade in outcomes]
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    effective = system.model_copy(
        update={"prompt": system.prompt.model_copy(update={"name": prompt_name})}
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(effective.model_dump(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    shutil.copyfile(PROMPTS_DIR / f"{prompt_name}.yaml", run_dir / "prompt.yaml")
    (run_dir / "eval.yaml").write_text(
        yaml.safe_dump(eval_cfg.model_dump(), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    dataset = EVALSETS_DIR / f"{set_name}.jsonl"
    meta = {
        "run": run_dir.name,
        "system": system.name,
        "prompt": prompt_name,
        "evalset": set_name,
        "split": split,
        "n_items": len(items),
        "dataset_sha256": file_sha256(dataset),
        "today": today,
        "started_at": started.isoformat(),
        "duration_s": round(time.perf_counter() - t0, 1),
        "workers": workers,
        "git": git_info(),
        "python": sys.version.split()[0],
        "packages": {p: metadata.version(p) for p in ("openai", "pydantic")},
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = build_summary(rows, judge, eval_cfg)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(render_report(meta, summary, rows), encoding="utf-8")
    return run_dir


def _row(item: EvalItem, res: AnswerResult | None, grade: Grade) -> dict:
    return {
        "id": item.id,
        "category": item.category,
        "answer_type": item.answer_type,
        "question": item.question,
        "gold_text": item.gold_text,
        "gold_value": item.gold_value,
        "gold_unit": item.gold_unit,
        "answer": res.answer.model_dump() if res and res.answer else None,
        "grade": grade.label,
        "grader": grade.grader,
        "detail": grade.detail,
        "tag": grade.tag,
        "attempts": res.attempts if res else 0,
        "errors": res.errors if res else [],
        "raw_outputs": [r.content for r in res.responses] if res else [],
        "cost_usd": res.cost_usd if res else None,
        "latency_s": round(res.latency_s, 2) if res else None,
    }


def build_summary(rows: list[dict], judge: Judge, eval_cfg: EvalConfig) -> dict:
    labels = [r["grade"] for r in rows]
    correct = [1.0 if lab == "correct" else 0.0 for lab in labels]
    _, lo, hi = bootstrap_ci(correct, eval_cfg.bootstrap_resamples, eval_cfg.bootstrap_seed)
    by_category = {}
    for cat in sorted({r["category"] for r in rows}):
        by_category[cat] = summarize([r["grade"] for r in rows if r["category"] == cat])
    latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    system_cost = sum(r["cost_usd"] or 0.0 for r in rows)
    return {
        "metrics": summarize(labels),
        "accuracy_ci95": [lo, hi],
        "by_category": by_category,
        "by_grader": dict(Counter(r["grader"] for r in rows)),
        "error_tags": dict(Counter(r["tag"] for r in rows if r["tag"])),
        "cost_usd": {
            "system": round(system_cost, 6),
            "judge": round(judge.cost_usd, 6),
            "total": round(system_cost + judge.cost_usd, 6),
        },
        "judge_calls": judge.calls,
        "latency_s": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "mean_attempts": sum(r["attempts"] for r in rows) / len(rows) if rows else 0.0,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _answer_cell(r: dict) -> str:
    if r["answer"] is None:
        return "—"
    a = Answer.model_validate(r["answer"])
    shown = format_value(a) if a.value is not None else a.answer_text
    return " ".join(shown.split()).replace("|", "\\|")[:120]


def render_report(meta: dict, s: dict, rows: list[dict]) -> str:
    m = s["metrics"]
    git = meta["git"]
    commit = (git["commit"] or "?")[:7] + (
        "（有未提交的修改，不能严格复现）" if git["dirty"] else ""
    )
    lo, hi = s["accuracy_ci95"]
    out = [
        f"# 评测报告 · {meta['run']}",
        "",
        (
            f"系统 **{meta['system']}** · prompt **{meta['prompt']}** · "
            f"评测集 **{meta['evalset']}/{meta['split']}**（{meta['n_items']} 题）· "
            f"日期参数 {meta['today']} · commit {commit}"
        ),
        "",
        "## 指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| **准确率** | **{_pct(m['accuracy'])}**（95% 置信区间 {_pct(lo)} ~ {_pct(hi)}） | 答对 / 全部 |",
        f"| 给出答案时的准确率 | {_pct(m['answered_accuracy'])} | 答对 / 给了答案的 |",
        f"| 拒答率 | {_pct(m['abstain_rate'])} | 说不知道 |",
        f"| 答错率 | {_pct(m['hallucination_rate'])} | 很肯定地答错 |",
        (
            f"| ✅ / ❌ / 🤷 / 💥 / ⚠️ | {m['correct']} / {m['incorrect']} / {m['abstained']} / "
            f"{m['invalid']} / {m['error']} | 对 / 错 / 没答 / 输出不合格 / 出错 |"
        ),
        "",
        "## 按题型",
        "",
        "| 题型 | 题数 | 准确率 | ✅ | ❌ | 🤷 |",
        "|---|---|---|---|---|---|",
    ]
    for cat, c in s["by_category"].items():
        out.append(
            f"| {cat} | {c['n']} | {_pct(c['accuracy'])} | {c['correct']} | {c['incorrect']} | {c['abstained']} |"
        )
    out += ["", "## 错误类型", ""]
    out += [
        f"- `{tag}`：{n}" for tag, n in sorted(s["error_tags"].items(), key=lambda kv: -kv[1])
    ] or ["- 无"]
    cost = s["cost_usd"]
    out += [
        "",
        "## 成本与速度",
        "",
        f"- 花费：系统 ${cost['system']:.4f} + 裁判 ${cost['judge']:.4f} = ${cost['total']:.4f}",
        f"- 判分方式：{s['by_grader']}；裁判调用 {s['judge_calls']} 次",
        f"- 延迟：P50 {s['latency_s']['p50']:.1f}s，P95 {s['latency_s']['p95']:.1f}s；平均尝试 {s['mean_attempts']:.2f} 次",
        "",
        "## 逐题结果",
        "",
        "| id | 题型 | 标准答案 | 系统回答 | 判分 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        gold = " ".join(r["gold_text"].split()).replace("|", "\\|")[:80]
        detail = " ".join(r["detail"].split()).replace("|", "\\|")[:140]
        out.append(
            f"| {r['id']} | {r['category']} | {gold} | {_answer_cell(r)} "
            f"| {LABEL_ICON[r['grade']]} {r['grader']} | {detail} |"
        )
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- 比较两次评测


def load_results(run_dir: Path) -> dict[str, dict]:
    with (run_dir / "results.jsonl").open(encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


def _verdict(lo: float, hi: float, p: float) -> str:
    """把"区间是否包含 0"和"p 是否 < 0.05"合成一句结论。"""
    ci_says = lo > 0 or hi < 0
    test_says = p < 0.05
    if ci_says and test_says:
        return "**差异显著**"
    if ci_says or test_says:
        return "边缘（两种方法结论不一致，需要更多题目）"
    return "不显著（可能只是运气）"


def compare_runs(dir_a: Path, dir_b: Path, resamples: int = 2000, seed: int = 0) -> str:
    """比较两次评测（配对）：只用两边都有的题，逐题配对计算差异。返回 markdown 文本。"""
    a, b = load_results(dir_a), load_results(dir_b)
    ids = sorted(a.keys() & b.keys())
    meta_a = json.loads((dir_a / "meta.json").read_text(encoding="utf-8"))
    meta_b = json.loads((dir_b / "meta.json").read_text(encoding="utf-8"))
    warnings = []
    if len(ids) < max(len(a), len(b)):
        warnings.append(f"两次评测的题目不完全相同，只比较共同的 {len(ids)} 道")
    if meta_a["dataset_sha256"] != meta_b["dataset_sha256"]:
        warnings.append("两次评测用的评测集文件不同（哈希不一致），比较可能没有意义")

    out = [
        f"# 比较 · {meta_a['evalset']}/{meta_a['split']}，共同题目 {len(ids)} 道",
        "",
        f"A = {meta_a['system']} / {meta_a['prompt']}　　B = {meta_b['system']} / {meta_b['prompt']}",
        "",
    ]
    out += [f"> ⚠️ {w}" for w in warnings]
    out += [
        "",
        "| 指标 | A | B | A − B（配对） | 95% 置信区间 | McNemar p | 结论 |",
        "|---|---|---|---|---|---|---|",
    ]
    # 两个指标都要看：只看准确率，会忽略"答错"和"不答"的区别；对财报问答来说，很肯定地答错比说"不知道"更糟
    for metric, label in (("准确率", "correct"), ("答错率", "incorrect")):
        xa = [1.0 if a[i]["grade"] == label else 0.0 for i in ids]
        xb = [1.0 if b[i]["grade"] == label else 0.0 for i in ids]
        diff, lo, hi = paired_bootstrap_diff(xa, xb, resamples, seed)
        a_only = sum(1 for x, y in zip(xa, xb, strict=True) if x > y)
        b_only = sum(1 for x, y in zip(xa, xb, strict=True) if y > x)
        p = mcnemar_exact(a_only, b_only)
        out.append(
            f"| {metric} | {_pct(sum(xa) / len(ids))} | {_pct(sum(xb) / len(ids))} "
            f"| {diff * 100:+.1f} 个百分点 | {lo * 100:+.1f} ~ {hi * 100:+.1f} | {p:.3f} "
            f"| {_verdict(lo, hi, p)} |"
        )
    out += [
        "",
        '判断标准：配对置信区间不包含 0，**并且** McNemar p < 0.05，才算显著；两种方法结论不一致时标为"边缘"。',
        "",
        "## 结果不同的题",
        "",
        "| id | A | B |",
        "|---|---|---|",
    ]
    for i in ids:
        if a[i]["grade"] != b[i]["grade"]:
            out.append(
                f"| {i} | {LABEL_ICON[a[i]['grade']]} {a[i]['grade']} | {LABEL_ICON[b[i]['grade']]} {b[i]['grade']} |"
            )
    return "\n".join(out) + "\n"
