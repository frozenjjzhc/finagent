"""判分规则与统计方法的测试。判分规则一旦写错，之后所有的实验结论都会跟着错，所以测得最细。"""

import pytest

from finagent.eval.datasets import EvalItem, parse_gold_number
from finagent.eval.metrics import (
    Grade,
    allowed_error,
    bootstrap_ci,
    grade_item,
    grade_number,
    mcnemar_exact,
    paired_bootstrap_diff,
    summarize,
)
from finagent.schemas import Answer


def item(value, unit, decimals=None, tol=None, answer_type="number") -> EvalItem:
    return EvalItem(
        id="t", source="seed", split="dev", question="Q?", answer_type=answer_type,
        gold_text=str(value), gold_value=value, gold_unit=unit, gold_decimals=decimals, tolerance_abs=tol,
    )  # fmt: skip


def answer(value=None, unit=None, status="answered") -> Answer:
    return Answer(
        reasoning="r", status=status, value=value, unit=unit, answer_text="t", confidence="high"
    )


def never_called(item, answer) -> Grade:
    raise AssertionError("这道题不该交给裁判")


# ---------------------------------------------------------------- 标准答案换算


@pytest.mark.parametrize(
    ("gold", "question", "expected"),
    [
        ("$1577.00", "... (in USD millions) ...", (1577.0, "USD_millions", 2)),
        ("$8.70", "... Answer in USD billions.", (8700.0, "USD_millions", -1)),
        ("1.9%", "... in units of percents ...", (1.9, "percent", 1)),
        ("24.26", "... ratio ... two decimal places", (24.26, "other", 2)),
        ("-0.02", "... return on assets ...", (-0.02, "other", 2)),
        ("$1,234.5", "... USD millions ...", (1234.5, "USD_millions", 1)),
    ],
)
def test_parse_gold_number(gold, question, expected) -> None:
    value, unit, decimals = parse_gold_number(gold, question)
    assert (value, unit, decimals) == pytest.approx(expected)


@pytest.mark.parametrize("gold", ["$5.00", "Yes, it improved", "about 30%-40%"])
def test_parse_gold_number_refuses_to_guess(gold) -> None:
    # "$5.00" 没说单位；另外两个根本不是一个数：都返回 None，交给裁判
    assert parse_gold_number(gold, "What is X?") is None


# ---------------------------------------------------------------- 容差


def test_tolerance_uses_gold_precision() -> None:
    # 标准答案 0.01（保留两位小数）代表 [0.005, 0.015)，所以容差是 0.005，而不是 1% × 0.01
    assert allowed_error(item(0.01, "other", decimals=2), 0.01) == pytest.approx(0.005)
    # 大数字时相对误差更宽：1577 × 1% = 15.77
    assert allowed_error(item(1577.0, "USD_millions", decimals=2), 0.01) == pytest.approx(15.77)
    # $8.70 billion 的精度是 10 million，换算后容差取 max(87, 5) = 87
    assert allowed_error(item(8700.0, "USD_millions", decimals=-1), 0.01) == pytest.approx(87)


def test_per_item_tolerance() -> None:
    it = item(18.5, "percent", decimals=1, tol=0.5)
    assert grade_number(it, answer(19, "percent")).label == "correct"  # 财报四舍五入的 19% 也算对


@pytest.mark.parametrize(
    ("gold", "gold_unit", "value", "unit", "label", "tag"),
    [
        (1577.0, "USD_millions", 1577, "USD_millions", "correct", ""),
        (1577.0, "USD_millions", 1590, "USD_millions", "correct", ""),  # 差 0.8%，在 1% 以内
        (1577.0, "USD_millions", 1600, "USD_millions", "incorrect", "wrong_value"),  # 差 1.5%
        (
            8700.0,
            "USD_millions",
            8.7,
            "USD_millions",
            "incorrect",
            "scale_error",
        ),  # 按 billions 填了数
        (0.01, "other", 1.3, "percent", "correct", ""),  # ROA：0.01 和 1.3% 在精度内一致
        (65.4, "percent", 0.654, "other", "correct", ""),  # 65.4% 和 0.654 是同一个数
        (1577.0, "USD_millions", 1577, "percent", "incorrect", "unit_mismatch"),
    ],
)
def test_grade_number(gold, gold_unit, value, unit, label, tag) -> None:
    g = grade_number(
        item(gold, gold_unit, decimals=2 if gold_unit != "percent" else 1), answer(value, unit)
    )
    assert (g.label, g.tag) == (label, tag)


# ---------------------------------------------------------------- 判分入口


def test_abstain_and_invalid() -> None:
    it = item(1577.0, "USD_millions")
    assert grade_item(it, None, never_called).label == "invalid"
    assert grade_item(it, answer(status="unknown"), never_called).label == "abstained"
    assert grade_item(it, answer(status="invalid_premise"), never_called).label == "incorrect"


def test_trap_question() -> None:
    trap = EvalItem(
        id="t",
        source="seed",
        split="dev",
        question="Q?",
        answer_type="invalid_premise",
        gold_text="x",
    )
    assert grade_item(trap, answer(status="invalid_premise"), never_called).label == "correct"
    assert grade_item(trap, answer(status="unknown"), never_called).label == "abstained"
    assert grade_item(trap, answer(100, "USD_millions"), never_called).tag == "fell_for_trap"


def test_text_questions_go_to_the_judge() -> None:
    text_item = EvalItem(
        id="t",
        source="financebench",
        split="dev",
        question="Q?",
        answer_type="text",
        gold_text="Yes",
    )
    calls = []

    def judge(item, ans):
        calls.append(item.id)
        return Grade("correct", "judge")

    assert grade_item(text_item, answer(), judge).grader == "judge"
    # 数值题却只给了文字、没给 value：也要交给裁判
    assert grade_item(item(1.0, "other"), answer(), judge).grader == "judge"
    assert len(calls) == 2


def test_summarize() -> None:
    s = summarize(["correct", "correct", "incorrect", "abstained"])
    assert s["accuracy"] == 0.5
    assert s["answered_accuracy"] == pytest.approx(2 / 3)  # 拒答不算进分母
    assert s["abstain_rate"] == 0.25 and s["hallucination_rate"] == 0.25


# ---------------------------------------------------------------- 统计


def test_bootstrap_is_reproducible_and_brackets_the_mean() -> None:
    values = [1.0] * 7 + [0.0] * 3
    mean, lo, hi = bootstrap_ci(values, resamples=2000, seed=0)
    assert mean == 0.7 and lo < 0.7 < hi
    assert bootstrap_ci(values, resamples=2000, seed=0) == (mean, lo, hi)  # 同样的种子，同样的区间
    assert hi - lo > 0.4  # 10 道题的区间非常宽：这就是 L02 结论不可靠的原因


def test_more_questions_give_narrower_intervals() -> None:
    _, lo10, hi10 = bootstrap_ci([1.0] * 7 + [0.0] * 3)
    _, lo100, hi100 = bootstrap_ci([1.0] * 70 + [0.0] * 30)
    assert (hi100 - lo100) < (hi10 - lo10) / 2


def test_paired_diff() -> None:
    a = [1.0] * 10
    b = [1.0] * 9 + [0.0]
    diff, lo, _ = paired_bootstrap_diff(a, b)
    assert diff == pytest.approx(0.1) and lo >= 0.0  # 只有一道题不同：区间下界是 0，不能说显著


@pytest.mark.parametrize(
    ("a_only", "b_only", "p"),
    [(0, 0, 1.0), (1, 0, 1.0), (5, 0, 0.0625), (10, 0, 0.001953125), (6, 4, 0.75390625)],
)
def test_mcnemar(a_only, b_only, p) -> None:
    assert mcnemar_exact(a_only, b_only) == pytest.approx(p)
