"""单次问答流水线（v0.2）：prompt 模板 → 调模型（JSON mode）→ 解析并校验 → 失败时带着错误反馈重试。

为什么要自己校验：DeepSeek 的 JSON mode 只保证输出是"合法的 JSON"，
不保证字段齐全、类型正确、取值合法；而且官方说明偶尔会返回空内容。
所以 JSON mode 只是第一道防线，pydantic 校验才是真正的把关。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import ValidationError

from finagent.llm.base import LLMClient, LLMResponse, Message
from finagent.prompts import PromptTemplate
from finagent.schemas import Answer


class AnswerParseError(ValueError):
    """模型输出无法变成一个合法的 Answer。"""


def parse_answer(text: str) -> Answer:
    """把模型的原始输出变成 Answer。三种失败分开报告，方便排查和反馈给模型。"""
    if not text.strip():
        raise AnswerParseError("输出是空的")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise AnswerParseError(f"不是合法的 JSON：{e}") from None
    try:
        return Answer.model_validate(data)
    except ValidationError as e:
        # 只保留每个错误的"字段 + 原因"，去掉 pydantic 的链接等噪音，反馈给模型时更清楚
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or '整体'}: {err['msg']}" for err in e.errors()
        )
        raise AnswerParseError(f"JSON 不符合要求的格式：{problems}") from None


@dataclass
class AnswerResult:
    answer: Answer | None = None  # 校验通过的答案；所有尝试都失败时为 None
    responses: list[LLMResponse] = field(default_factory=list)  # 每次尝试的原始响应
    errors: list[str] = field(default_factory=list)  # 每次失败的原因

    @property
    def attempts(self) -> int:
        return len(self.responses)

    @property
    def cost_usd(self) -> float | None:
        costs = [r.cost_usd for r in self.responses]
        return None if any(c is None for c in costs) else sum(costs)

    @property
    def latency_s(self) -> float:
        return sum(r.latency_s for r in self.responses)


def today_iso() -> str:
    """本地日期，例如 "2026-09-26"。模型不知道今天是哪天，要由我们告诉它（L01 的发现）。"""
    return datetime.now(timezone.utc).astimezone().date().isoformat()


def repair_message(error: str) -> Message:
    return {
        "role": "user",
        "content": (
            f"Your previous reply was rejected: {error}. "
            "Reply again with ONLY the JSON object in the required format, and nothing else."
        ),
    }


def answer_question(
    client: LLMClient,
    prompt: PromptTemplate,
    question: str,
    today: str | None = None,
    max_attempts: int = 2,
) -> AnswerResult:
    """回答一个问题，返回结构化结果。最多尝试 max_attempts 次（含第一次）。"""
    messages = prompt.build_messages(question, today=today or today_iso())
    result = AnswerResult()
    for _ in range(max_attempts):
        resp = client.chat(messages, json_mode=True)
        result.responses.append(resp)
        try:
            result.answer = parse_answer(resp.content)
            return result
        except AnswerParseError as e:
            result.errors.append(str(e))
            if resp.content.strip():
                # 把它的错误输出原样放回去（assistant），再告诉它错在哪（user）：带反馈的重试。
                # 比原样再问一遍更有效，因为模型知道了上一次具体错在哪里。
                messages = [
                    *messages,
                    {"role": "assistant", "content": resp.content},
                    repair_message(str(e)),
                ]
            # 空输出没什么可反馈的，messages 不变，直接重试
    return result
