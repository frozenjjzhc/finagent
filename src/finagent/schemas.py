"""结构化答案的 schema（L02）。

模型的回答不再是一段自由文本，而是一个固定结构的 JSON，用 pydantic 校验。
有了固定结构，程序才能自动判分（L03）、自动检查引用（L05）、自动生成训练数据（L10）。

字段顺序是有意安排的：reasoning 排第一，和 prompt 里的格式说明、few-shot 示例保持一致。
注意：模型看不到这个 Python 类，真正让它"先写推理、再写数值"的是 prompt 和示例；
这里保持同样的顺序，是为了让 model_dump_json 的输出（--json 显示、L10 的训练数据）也是推理在前。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Status = Literal["answered", "unknown", "invalid_premise"]
Unit = Literal["USD_millions", "percent", "count", "other"]
Confidence = Literal["high", "medium", "low"]


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 多出来的字段也算错：说明模型没按格式来

    reasoning: str = Field(max_length=1500)  # 简短推理；计算题写出算式
    status: Status  # answered：给出答案；unknown：不知道；invalid_premise：问题的前提不成立
    value: float | None = None  # 数值答案，按 unit 表示；status 不是 answered 时必须为 null
    unit: Unit | None = None
    answer_text: str = Field(max_length=500)  # 给人看的一句话回答
    # 依据的文件。闭卷时是模型"自称"的出处，无法验证；L05 起换成真实的检索片段编号
    sources: list[str] = Field(default_factory=list)
    confidence: Confidence  # 模型的自我评估。它准不准，要到 L03 用数据检验

    @model_validator(mode="after")
    def _check_consistency(self) -> Answer:
        if self.status != "answered" and self.value is not None:
            raise ValueError("status 不是 answered 时，value 必须是 null")
        if self.value is not None and self.unit is None:
            raise ValueError("给了 value 就必须给 unit")
        return self


def format_value(answer: Answer) -> str:
    """把数值和单位格式化成人看的样子，例如 130497 + USD_millions -> "$130,497 million"。"""
    if answer.value is None:
        return "—"
    v = answer.value
    if answer.unit == "USD_millions":
        return f"${v:,.0f} million" if v == int(v) else f"${v:,.2f} million"
    if answer.unit == "percent":
        return f"{v:g}%"
    if answer.unit == "count":
        return f"{v:,.0f}"
    return f"{v:g}"
