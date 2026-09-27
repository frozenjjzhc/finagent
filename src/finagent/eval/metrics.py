"""判分规则与统计方法。

判分原则：能用规则判的，就不用模型判（便宜、可复现、没有偏差）。
  - 陷阱题、拒答：按 status 判
  - 数值题：按容差比较数值（本文件）
  - 文字题：交给 LLM 裁判（judge.py）

每道题的判分结果只有五种：
  correct    答对
  incorrect  答错（很肯定地给了错误答案）
  abstained  没回答（status=unknown）
  invalid    输出始终没通过格式校验
  error      调用 API 出错
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from finagent.eval.datasets import EvalItem
from finagent.schemas import Answer

GradeLabel = Literal["correct", "incorrect", "abstained", "invalid", "error"]


@dataclass
class Grade:
    label: GradeLabel
    grader: Literal["rule", "judge", "none"]  # 谁判的
    detail: str = ""  # 给人看的说明，例如"差 +12.0"
    tag: str = ""  # 错误类型，用于错误分析：unit_mismatch / scale_error / wrong_value / ...


# ---------------------------------------------------------------- 数值题


def allowed_error(item: EvalItem, relative: float) -> float:
    """这道题允许的最大绝对误差，取三者中最大的：
    1. 相对误差：relative × |标准答案|（默认 1%）
    2. 标准答案的精度：末位的一半。题目要求"保留两位小数"时，0.01 代表 [0.005, 0.015)
    3. 单题额外容差 tolerance_abs（比如允许按财报四舍五入到整数百分比）
    """
    assert item.gold_value is not None
    candidates = [relative * abs(item.gold_value)]
    if item.gold_decimals is not None:
        candidates.append(0.5 * 10 ** (-item.gold_decimals))
    if item.tolerance_abs is not None:
        candidates.append(item.tolerance_abs)
    return max(candidates)


def grade_number(item: EvalItem, answer: Answer, relative: float = 0.01) -> Grade:
    """比较数值答案。比率（other）和百分比（percent）可以互换：0.01 和 1% 是同一个数。"""
    assert item.gold_value is not None and answer.value is not None
    gold, value, unit = item.gold_value, answer.value, answer.unit

    if unit == item.gold_unit:
        candidates = [value]
    elif {unit, item.gold_unit} <= {"percent", "other"}:
        # 无量纲的数，单位标签常常标得不准：原值、×100、÷100 都试一下
        candidates = [value, value * 100, value / 100]
    else:
        return Grade(
            "incorrect", "rule", f"单位不对：{unit}（应为 {item.gold_unit}）", "unit_mismatch"
        )

    tol = allowed_error(item, relative)
    if any(abs(c - gold) <= tol for c in candidates):
        return Grade("correct", "rule", f"误差在 ±{tol:g} 以内")
    if gold != 0 and any(abs(c + gold) <= tol for c in candidates):
        # 数值对、符号反了。题目问"下降了多少"时，答 42（下降 42）和 -42（变化 -42）都对；
        # 题目没说方向时（"变化了多少"），符号本身就是答案的一部分，答反了算错
        if item.sign_agnostic:
            return Grade("correct", "rule", f"数值在 ±{tol:g} 以内（题目已说明方向，不看符号）")
        return Grade("incorrect", "rule", "符号反了", "sign_error")
    for factor in (1000, 1 / 1000, 100, 1 / 100):
        if abs(value * factor - gold) <= tol:
            return Grade("incorrect", "rule", f"数量级错了（差 {1 / factor:g} 倍）", "scale_error")
    return Grade("incorrect", "rule", f"差 {value - gold:+,.4g}", "wrong_value")


# ---------------------------------------------------------------- 总的判分入口

JudgeFn = Callable[[EvalItem, Answer], Grade]


def grade_item(
    item: EvalItem, answer: Answer | None, judge: JudgeFn, relative: float = 0.01
) -> Grade:
    """给一道题判分。judge 只在"文字题"或"数值题却只给了文字"时才会被调用。"""
    if answer is None:
        return Grade("invalid", "none", "输出始终没通过格式校验", "invalid_output")

    if item.answer_type == "invalid_premise":  # 陷阱题：识破前提才算对
        if answer.status == "invalid_premise":
            return Grade("correct", "rule", "指出了前提错误")
        if answer.status == "unknown":
            return Grade("abstained", "rule", "说不知道")
        return Grade("incorrect", "rule", "没识破陷阱，给出了答案", "fell_for_trap")

    if answer.status == "unknown":
        return Grade("abstained", "rule", "说不知道")
    if answer.status == "invalid_premise":
        return Grade("incorrect", "rule", "错误地声称前提不成立", "false_invalid_premise")

    if item.answer_type == "number" and answer.value is not None:
        return grade_number(item, answer, relative)
    return judge(item, answer)


# ---------------------------------------------------------------- 汇总指标


def summarize(labels: Sequence[str]) -> dict[str, float]:
    """几个互补的指标。只看一个数会被误导，比如"不回答"能让答错率变低，但准确率也跟着降。"""
    n = len(labels)
    c = {k: labels.count(k) for k in ("correct", "incorrect", "abstained", "invalid", "error")}
    answered = c["correct"] + c["incorrect"]
    return {
        "n": n,
        **c,
        "accuracy": c["correct"] / n if n else 0.0,  # 答对 / 全部（拒答也算没答对）
        "answered_accuracy": c["correct"] / answered if answered else 0.0,  # 答对 / 给了答案的
        "abstain_rate": c["abstained"] / n if n else 0.0,
        "hallucination_rate": c["incorrect"] / n if n else 0.0,  # 很肯定地答错
    }


# ---------------------------------------------------------------- 统计：bootstrap 与 McNemar


def bootstrap_ci(
    values: Sequence[float], resamples: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """均值及其 95% 置信区间（bootstrap 百分位法）。

    做法：从 n 道题里"有放回地"抽 n 道，算一次准确率；重复 resamples 次，
    得到准确率的分布，取中间 95% 的范围。它回答的是："如果换一批同样难度的题，
    准确率大概会在什么范围内波动？"
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)  # 固定种子：同样的数据，每次算出同样的区间
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    lo = means[int(alpha / 2 * resamples)]
    hi = means[int((1 - alpha / 2) * resamples) - 1]
    return sum(values) / n, lo, hi


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """两个系统在**同一批题**上的准确率之差（a - b）及其 95% 置信区间。

    "配对"是关键：每次抽样抽的是题目，两个系统用同一批抽到的题比较。
    题目本身的难易差异被抵消了，比两个系统各算各的区间灵敏得多。
    区间不包含 0，才能说两者有显著差异。
    """
    assert len(a) == len(b)
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(diffs, resamples, seed, alpha)


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """McNemar 精确检验的双侧 p 值。

    只看"两个系统结果不同"的题：a 对 b 错的有 a_only 道，a 错 b 对的有 b_only 道。
    如果两个系统其实一样好，这些题应该一半一半（像抛硬币）。p 值就是"纯靠运气
    出现这么不平衡（或更不平衡）"的概率。p < 0.05 通常认为差异显著。
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def percentile(values: Sequence[float], q: float) -> float:
    """分位数，例如 q=0.95 是 P95。"""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=100, method="inclusive")[round(q * 100) - 1]
