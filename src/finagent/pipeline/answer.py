"""单次问答流水线（v0.2 起）：prompt 模板 → 调模型（JSON mode）→ 解析并校验 → 失败时带着错误反馈重试。

为什么要自己校验：DeepSeek 的 JSON mode 只保证输出是"合法的 JSON"，
不保证字段齐全、类型正确、取值合法；而且官方说明偶尔会返回空内容。
所以 JSON mode 只是第一道防线，pydantic 校验才是真正的把关。

L03 起，"调用 → 校验 → 重试"这套通用逻辑挪到了 llm/structured.py，裁判也在用它；
这里只保留和"回答问题"相关的部分。

L04 加了两样东西：
  - context：给定的财报片段（FinQA 评测、ask --context-file）；L05 起由检索提供
  - tools：可用的工具。有工具时，每次尝试是一整个工具循环（agent/loop.py），而不是一次调用
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any

from finagent.agent.loop import run_tool_loop
from finagent.config import ToolsConfig
from finagent.llm.base import LLMClient
from finagent.llm.structured import (
    StructuredParseError,
    StructuredResult,
    call_structured,
    parse_structured,
)
from finagent.prompts import PromptTemplate
from finagent.schemas import Answer
from finagent.tools.registry import ToolRegistry, build_registry

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
    *,
    context: str | None = None,
    tools: ToolRegistry | None = None,
    max_rounds: int = 6,
    tool_json_mode: bool = False,
) -> AnswerResult:
    """回答一个问题，返回结构化结果。最多尝试 max_attempts 次（含第一次）。

    context：给定的上下文。prompt 必须有 {{context}}，否则报错（不能让上下文被悄悄丢掉）。
    tools：可用的工具；None 表示不带工具。max_rounds：每次尝试最多几轮工具调用。
    tool_json_mode：工具循环里要不要开 JSON mode（默认不开，原因见 config.ToolsConfig）。
    """
    variables: dict[str, object] = {"today": today or today_iso()}
    if context is not None:
        if "context" not in prompt.variables:
            raise ValueError(f"prompt {prompt.name} 没有 {{{{context}}}}，给定的上下文会被丢掉")
        variables["context"] = context
    messages = prompt.build_messages(question, **variables)
    generate = None
    if tools is not None:
        # partial 先填好除 messages 以外的参数，得到一个"只差 messages"的函数，交给 call_structured
        generate = partial(
            run_tool_loop, client, registry=tools, json_mode=tool_json_mode, max_rounds=max_rounds
        )
    result = call_structured(
        client, messages, Answer, max_attempts, result=AnswerResult(), generate=generate
    )
    assert isinstance(result, AnswerResult)
    return result


def tool_options(cfg: ToolsConfig | None) -> dict[str, Any]:
    """把系统配置里的 tools 段变成 answer_question 的参数。没有工具时返回空 dict。"""
    if cfg is None or not cfg.enabled:
        return {}
    return {
        "tools": build_registry(cfg.enabled),
        "max_rounds": cfg.max_rounds,
        "tool_json_mode": cfg.json_mode,
    }
