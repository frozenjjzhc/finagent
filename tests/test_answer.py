"""结构化答案的测试：schema 校验规则，以及"解析失败 → 带反馈重试"的流程。全程不联网。"""

import json

import pytest

from finagent.llm.base import LLMResponse, Usage
from finagent.pipeline.answer import AnswerParseError, answer_question, parse_answer
from finagent.prompts import load_prompt

GOOD = {
    "reasoning": "From the income statement.",
    "status": "answered",
    "value": 130497,
    "unit": "USD_millions",
    "answer_text": "Revenue was $130,497 million.",
    "sources": ["NVIDIA 10-K FY2025"],
    "confidence": "high",
}

# ---------------------------------------------------------------- parse_answer 与 schema


def test_parse_good_answer() -> None:
    a = parse_answer(json.dumps(GOOD))
    assert a.value == 130497 and a.unit == "USD_millions"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "空"),
        ('```json\n{"status": "answered"}\n```', "不是合法的 JSON"),  # 外面包了 markdown 代码块
        (json.dumps({**GOOD, "status": "maybe"}), "status"),  # 取值不在允许范围内
        (json.dumps({**GOOD, "value": "130,497"}), "value"),  # 数字写成了带逗号的字符串
        (json.dumps({**GOOD, "unit": None}), "unit"),  # 给了 value 却没给 unit
        (json.dumps({**GOOD, "status": "unknown"}), "null"),  # 说不知道，却又给了数
        (json.dumps({**GOOD, "extra": 1}), "extra"),  # 多出来的字段
    ],
)
def test_parse_rejects_bad_output(text: str, reason: str) -> None:
    with pytest.raises(AnswerParseError, match=reason):
        parse_answer(text)


def test_unknown_answer_is_valid() -> None:
    a = parse_answer(json.dumps({**GOOD, "status": "unknown", "value": None, "unit": None}))
    assert a.status == "unknown"


# ---------------------------------------------------------------- answer_question：重试流程


class ScriptedClient:
    """按顺序返回预先写好的输出，并记下每次收到的 messages 和 json_mode。"""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def chat(self, messages, on_delta=None, json_mode=False) -> LLMResponse:
        self.calls.append({"messages": list(messages), "json_mode": json_mode})
        return LLMResponse(
            content=self.outputs.pop(0),
            finish_reason="stop",
            model="fake",
            usage=Usage(),
            latency_s=0.1,
            cost_usd=0.001,
        )


PROMPT = load_prompt("answer_v2")


def test_first_try_success() -> None:
    client = ScriptedClient([json.dumps(GOOD)])
    result = answer_question(client, PROMPT, "Q?", today="2026-09-26")
    assert result.answer is not None and result.attempts == 1
    assert client.calls[0]["json_mode"] is True  # 结构化模式一定要开 JSON mode


def test_retry_with_error_feedback() -> None:
    client = ScriptedClient(["not json at all", json.dumps(GOOD)])
    result = answer_question(client, PROMPT, "Q?", today="2026-09-26")
    assert result.answer is not None and result.attempts == 2
    assert result.cost_usd == pytest.approx(0.002)  # 两次尝试的花费都要算上
    retry_messages = client.calls[1]["messages"]
    # 重试时：先把它的错误输出放回去（assistant），再告诉它错在哪（user）
    assert retry_messages[-2] == {"role": "assistant", "content": "not json at all"}
    assert "rejected" in retry_messages[-1]["content"]


def test_empty_output_retries_without_feedback() -> None:
    client = ScriptedClient(["", json.dumps(GOOD)])
    result = answer_question(client, PROMPT, "Q?", today="2026-09-26")
    assert result.answer is not None
    assert client.calls[1]["messages"] == client.calls[0]["messages"]  # 空输出没什么可反馈的


def test_gives_up_after_max_attempts() -> None:
    client = ScriptedClient(["bad", "still bad"])
    result = answer_question(client, PROMPT, "Q?", today="2026-09-26", max_attempts=2)
    assert result.answer is None
    assert len(result.errors) == 2
