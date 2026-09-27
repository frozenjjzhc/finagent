"""通用的结构化调用：JSON mode → pydantic 校验 → 失败时带着错误反馈重试。

L02 里这套逻辑只服务于"回答问题"；L03 的裁判也需要完全一样的流程，所以抽成通用函数，
传入任意 pydantic 模型类（Answer、Verdict……）都能用。

L04：一次"生成"不一定只是一次调用。带工具时，模型可能先调用几次计算器，最后才输出 JSON。
所以把"生成"做成可以替换的参数 generate：默认是调用一次模型；
带工具时由 agent/loop.py 的工具循环代替。校验和重试的逻辑不变。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from finagent.llm.base import LLMClient, LLMResponse, Message, ToolStep

T = TypeVar("T", bound=BaseModel)


class StructuredParseError(ValueError):
    """模型输出无法变成一个合法的结构化对象。"""


def parse_structured(text: str, model_cls: type[T]) -> T:
    """把模型的原始输出解析成 model_cls。三种失败分开报告，方便排查，也方便反馈给模型。

    故意分两步（先 json.loads，再 model_validate），而不用 model_validate_json 一步完成：
    这样才能区分"根本不是 JSON"和"是 JSON 但格式不对"。
    """
    if not text.strip():
        raise StructuredParseError("输出是空的")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise StructuredParseError(f"不是合法的 JSON：{e}") from None
    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        # 只保留每个错误的"字段 + 原因"，去掉 pydantic 的链接等噪音
        problems = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or '整体'}: {err['msg']}" for err in e.errors()
        )
        raise StructuredParseError(f"JSON 不符合要求的格式：{problems}") from None


@dataclass
class Generation:
    """一次"生成"的结果。没有工具时就是一次调用；有工具时是一整个工具循环（L04）。"""

    final: LLMResponse  # 最后一条回复，要解析的 JSON 在它的 content 里
    messages: list[Message]  # 生成结束时完整的对话，最后一条是 final 对应的 assistant 消息
    responses: list[LLMResponse]  # 期间所有的调用（算花费、数调用次数用）
    tool_steps: list[ToolStep] = field(default_factory=list)  # 期间执行过的工具调用
    hit_max_rounds: bool = False  # 工具循环因为轮数用完而停下（此时 final 里没有最终答案）


Generate = Callable[[list[Message]], Generation]


def single_call(client: LLMClient, messages: list[Message]) -> Generation:
    """默认的生成方式：JSON mode 调用一次模型。"""
    resp = client.chat(messages, json_mode=True)
    # 只放回 content：不带工具时，没有必要把 reasoning_content 传回去
    reply: Message = {"role": "assistant", "content": resp.content}
    return Generation(final=resp, messages=[*messages, reply], responses=[resp])


@dataclass
class StructuredResult(Generic[T]):
    value: T | None = None  # 校验通过的对象；所有尝试都失败时为 None
    responses: list[LLMResponse] = field(default_factory=list)  # 所有调用的原始响应
    errors: list[str] = field(default_factory=list)  # 每次失败的原因
    attempts: int = 0  # 尝试了几次。带工具时，一次尝试里可能有好几次调用（L04 前两者相等）
    tool_steps: list[ToolStep] = field(default_factory=list)  # 所有执行过的工具调用

    @property
    def cost_usd(self) -> float | None:
        """所有尝试的总花费。只要有一次花费未知，总花费就是未知（不能当成 0）。"""
        costs = [r.cost_usd for r in self.responses]
        return None if any(c is None for c in costs) else sum(costs)

    @property
    def latency_s(self) -> float:
        return sum(r.latency_s for r in self.responses)


def repair_message(error: str) -> Message:
    return {
        "role": "user",
        "content": (
            f"Your previous reply was rejected: {error}. "
            "Reply again with ONLY the JSON object in the required format, and nothing else."
        ),
    }


def call_structured(
    client: LLMClient,
    messages: list[Message],
    model_cls: type[T],
    max_attempts: int = 2,
    result: StructuredResult[T] | None = None,
    generate: Generate | None = None,
) -> StructuredResult[T]:
    """调用模型并得到一个校验通过的 model_cls 对象，最多尝试 max_attempts 次（含第一次）。

    result 参数允许调用方传入一个子类实例（比如 AnswerResult），结果会写进去。
    generate 为 None 时，每次尝试调用一次模型；带工具时传入工具循环（见 pipeline/answer.py）。
    """
    result = result if result is not None else StructuredResult()
    for _ in range(max_attempts):
        gen = generate(messages) if generate is not None else single_call(client, messages)
        result.attempts += 1
        result.responses.extend(gen.responses)
        result.tool_steps.extend(gen.tool_steps)
        if gen.hit_max_rounds:
            # 模型一直在调用工具，没给出答案。从头再来一次（不把这段历史带进下一次尝试）
            result.errors.append("工具调用达到轮数上限，没有给出最终答案")
            continue
        try:
            result.value = parse_structured(gen.final.content, model_cls)
            return result
        except StructuredParseError as e:
            result.errors.append(str(e))
            if gen.final.content.strip():
                # 它的错误输出已经在 gen.messages 的最后（assistant），再告诉它错在哪（user）：
                # 带反馈的重试。带工具时，之前的工具调用和结果也都保留，不用重算
                messages = [*gen.messages, repair_message(str(e))]
            # 空输出没什么可反馈的，messages 不变，直接重试
    return result
