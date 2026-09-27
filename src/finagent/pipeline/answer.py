"""单次问答流水线（v0.2 起）：prompt 模板 → 调模型（JSON mode）→ 解析并校验 → 失败时带着错误反馈重试。

为什么要自己校验：DeepSeek 的 JSON mode 只保证输出是"合法的 JSON"，
不保证字段齐全、类型正确、取值合法；而且官方说明偶尔会返回空内容。
所以 JSON mode 只是第一道防线，pydantic 校验才是真正的把关。

L03 起，"调用 → 校验 → 重试"这套通用逻辑挪到了 llm/structured.py，裁判也在用它；
这里只保留和"回答问题"相关的部分。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from finagent.llm.base import LLMClient
from finagent.llm.structured import (
    StructuredParseError,
    StructuredResult,
    call_structured,
    parse_structured,
)
from finagent.prompts import PromptTemplate
from finagent.schemas import Answer

# 保留 L02 的名字，旧代码和测试不用改
AnswerParseError = StructuredParseError


def parse_answer(text: str) -> Answer:
    return parse_structured(text, Answer)


@dataclass
class AnswerResult(StructuredResult[Answer]):
    @property
    def answer(self) -> Answer | None:
        """校验通过的答案；所有尝试都失败时为 None。"""
        return self.value


def today_iso() -> str:
    """本地日期，例如 "2026-09-26"。模型不知道今天是哪天，要由我们告诉它（L01 的发现）。"""
    return datetime.now(timezone.utc).astimezone().date().isoformat()


def answer_question(
    client: LLMClient,
    prompt: PromptTemplate,
    question: str,
    today: str | None = None,
    max_attempts: int = 2,
) -> AnswerResult:
    """回答一个问题，返回结构化结果。最多尝试 max_attempts 次（含第一次）。"""
    messages = prompt.build_messages(question, today=today or today_iso())
    result = call_structured(client, messages, Answer, max_attempts, result=AnswerResult())
    assert isinstance(result, AnswerResult)
    return result
