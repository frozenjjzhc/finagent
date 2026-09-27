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
    source: Literal["seed", "financebench"]
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


def _digits_of(scale: float) -> int:
    """1000 → 3，1 → 0，0.001 → -3：换算后小数位数要减去的量。"""
    return {1000.0: 3, 1.0: 0, 0.001: -3}[scale]


def dump_items(items: list[EvalItem]) -> str:
    return "".join(json.dumps(it.model_dump(), ensure_ascii=False) + "\n" for it in items)
