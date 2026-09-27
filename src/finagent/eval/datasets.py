"""评测集：统一的题目格式（EvalItem）、读取函数，以及把原始标准答案换算成统一单位的规则。

所有评测集都转成同一种格式存在 evalsets/*.jsonl 里，判分代码只需要认识这一种格式。
evalsets/ 里的文件由 scripts/build_evalsets.py 生成，并提交到 git：
评测集是项目的"规格"，它变了，分数就没法和以前比（ADR-8）。
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finagent.config import PROJECT_ROOT
from finagent.schemas import Unit

EVALSETS_DIR = PROJECT_ROOT / "evalsets"

AnswerType = Literal["number", "text", "invalid_premise"]
Split = Literal["dev", "test"]


class EvalItem(BaseModel):
    """一道评测题。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: Literal["seed", "financebench", "finqa"]
    split: Split  # dev：调 prompt、做实验时用；test：只在里程碑跑，防止对它过拟合
    question: str
    answer_type: AnswerType  # number：程序判分；text：LLM 裁判判分；invalid_premise：陷阱题
    gold_text: str  # 原始的标准答案文字（给裁判看，也方便人核对）
    gold_value: float | None = None  # 数值题：换算成 gold_unit 之后的标准答案
    gold_unit: Unit | None = None
    gold_decimals: int | None = None  # 标准答案写到了小数点后几位，决定容差（见 metrics.py）
    tolerance_abs: float | None = None  # 单题的额外容差（这道题允许更粗的四舍五入时才设）
    category: str = ""  # 题型：FinanceBench 的 question_type，或 seed 的 type
    company: str = ""
    period: str = ""  # 财年，如 "2022"
    doc_type: str = ""  # 10k / 10q / 8k / Earnings
    justification: str = ""  # 标准答案的推导说明（给裁判参考）
    evidence: list[dict] = Field(default_factory=list)  # 证据所在的文档和页码，L05 评测检索时用
    # 给定的上下文（L04，FinQA）：财报一页的文字 + 表格。有它的题是"开卷"题，prompt 必须有 {{context}}
    context: str | None = None
    # 题目自己说明了方向（"下降了多少"），答 42 和 -42 都算对。见 metrics.grade_number（L04）
    sign_agnostic: bool = False

    @model_validator(mode="after")
    def _number_needs_value(self) -> EvalItem:
        if self.answer_type == "number" and (self.gold_value is None or self.gold_unit is None):
            raise ValueError("数值题必须有 gold_value 和 gold_unit")
        return self


def load_evalset(name: str, split: str = "all") -> list[EvalItem]:
    """读取 evalsets/<name>.jsonl。split 为 "dev"、"test" 或 "all"。"""
    path = EVALSETS_DIR / f"{name}.jsonl"
    if not path.exists():
        available = sorted(p.stem for p in EVALSETS_DIR.glob("*.jsonl"))
        raise FileNotFoundError(f"找不到评测集 {path.name}，现有的：{available}")
    with path.open(encoding="utf-8") as f:
        items = [EvalItem.model_validate_json(line) for line in f if line.strip()]
    return items if split == "all" else [it for it in items if it.split == split]


# ---------------------------------------------------------------- 标准答案换算

_NUMBER = re.compile(
    r"^\s*(?P<neg>-)?\s*\$?\s*(?P<num>\d[\d,]*(?:\.(?P<dec>\d+))?)\s*(?P<pct>%)?\s*$"
)
_SCALE = {"usd billions": 1000.0, "usd millions": 1.0, "usd thousands": 0.001}


def parse_gold_number(answer: str, question: str) -> tuple[float, Unit, int] | None:
    """把 FinanceBench 数值题的标准答案换算成（数值, 单位, 小数位数）；看不懂就返回 None。

    规则（来自对 50 道数值题的逐一检查）：
      - 答案以 % 结尾                 → percent，例如 "1.9%" → 1.9
      - 答案以 $ 开头，题目说 USD billions → 换算成百万美元，例如 "$8.70" → 8700
      - 答案以 $ 开头，题目说 USD millions → 百万美元，例如 "$1577.00" → 1577
      - 其余纯数字                    → other（比率、天数等），例如 ROA "-0.02"、DPO "93.86"
    小数位数按标准答案原文计算（"$8.70" 是 2 位），换算只改变数值，不改变精度。
    """
    m = _NUMBER.match(answer)
    if not m:
        return None
    value = float(m.group("num").replace(",", ""))
    if m.group("neg"):
        value = -value
    decimals = len(m.group("dec") or "")
    if m.group("pct"):
        return value, "percent", decimals
    if "$" in answer:
        q = question.lower()
        for phrase, scale in _SCALE.items():
            if phrase in q:
                # 精度随换算放大：$8.70 billion 的精度是 0.01 billion = 10 million
                return value * scale, "USD_millions", decimals - _digits_of(scale)
        return None  # 有 $ 却不知道单位：交给裁判，不瞎猜
    return value, "other", decimals


_FINQA_NUMBER = re.compile(
    r"^\s*(?P<neg>-)?\s*\$?\s*(?P<num>\d[\d,]*(?:\.(?P<dec>\d+))?|\.(?P<dec2>\d+))\s*(?P<pct>%)?\s*$"
)


def parse_finqa_gold(answer: str, exe_ans: object) -> tuple[float, Unit, int] | None:
    """FinQA 的标准答案有两份：answer（人写的文字，如 "1.6%"）和 exe_ans（程序算出的数，如 0.01641）。

    只有两份**对得上**的题才保留，返回（数值, 单位, 小数位数）；对不上就返回 None，丢掉这道题。
    这是一种便宜的标注质量检查：两个独立来源一致，标准答案出错的可能性就小得多。
      - "1.6%" 且 exe_ans = 0.01641  → 对得上（×100）→ (1.641, percent, 1)
      - "24.69%" 且 exe_ans = 24.69   → 对得上（程序里已经 ×100）→ (24.69, percent, 2)
      - "127.40" 且 exe_ans = 127.4   → (127.4, other, 2)
      - "3.4%" 且 exe_ans = -0.0339   → 符号对不上 → None
      - "yes" / exe_ans = "yes"       → 不是数值题 → None
    数值用 exe_ans（更精确），小数位数用 answer（它决定容差，见 metrics.allowed_error）。
    """
    if isinstance(exe_ans, bool) or not isinstance(exe_ans, (int, float)):
        return None
    m = _FINQA_NUMBER.match(answer)
    if not m:
        return None
    value = float(m.group("num").replace(",", "")) if m.group("num") else float("." + m.group("dec2"))
    if m.group("neg"):
        value = -value
    decimals = len(m.group("dec") or m.group("dec2") or "")
    tol = max(0.01 * abs(value), 0.5 * 10**-decimals)
    exe = float(exe_ans)
    if m.group("pct"):
        for candidate in (exe * 100, exe):  # 程序可能算的是比率，也可能已经乘过 100
            if abs(candidate - value) <= tol:
                return candidate, "percent", decimals
        return None
    return (exe, "other", decimals) if abs(exe - value) <= tol else None


def _digits_of(scale: float) -> int:
    """1000 → 3，1 → 0，0.001 → -3：换算后小数位数要减去的量。"""
    return {1000.0: 3, 1.0: 0, 0.001: -3}[scale]


# L04 新增的字段。取默认值时不写进文件：这样 L03 的评测集文件逐字节不变，
# 旧记录里的 dataset_sha256 仍然对得上，新旧结果还能直接比较
_ADDED_FIELDS = {"context": None, "sign_agnostic": False}


def dump_items(items: list[EvalItem]) -> str:
    lines = []
    for it in items:
        data = it.model_dump()
        for name, default in _ADDED_FIELDS.items():
            if data[name] == default:
                del data[name]
        lines.append(json.dumps(data, ensure_ascii=False) + "\n")
    return "".join(lines)
