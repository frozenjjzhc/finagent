"""通用的结构化调用：JSON mode → pydantic 校验 → 失败时带着错误反馈重试。

L02 里这套逻辑只服务于"回答问题"；L03 的裁判也需要完全一样的流程，所以抽成通用函数，
传入任意 pydantic 模型类（Answer、Verdict……）都能用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from finagent.llm.base import LLMClient, LLMResponse, Message

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
class StructuredResult(Generic[T]):
    value: T | None = None  # 校验通过的对象；所有尝试都失败时为 None
    responses: list[LLMResponse] = field(default_factory=list)  # 每次尝试的原始响应
    errors: list[str] = field(default_factory=list)  # 每次失败的原因

    @property
    def attempts(self) -> int:
        return len(self.responses)

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
) -> StructuredResult[T]:
    """调用模型并得到一个校验通过的 model_cls 对象，最多尝试 max_attempts 次（含第一次）。

    result 参数允许调用方传入一个子类实例（比如 AnswerResult），结果会写进去。
    """
    result = result if result is not None else StructuredResult()
    for _ in range(max_attempts):
        resp = client.chat(messages, json_mode=True)
        result.responses.append(resp)
        try:
            result.value = parse_structured(resp.content, model_cls)
            return result
        except StructuredParseError as e:
            result.errors.append(str(e))
            if resp.content.strip():
                # 把它的错误输出原样放回去（assistant），再告诉它错在哪（user）：带反馈的重试
                messages = [
                    *messages,
                    {"role": "assistant", "content": resp.content},
                    repair_message(str(e)),
                ]
            # 空输出没什么可反馈的，messages 不变，直接重试
    return result
