"""工具循环的测试（L04）：用"按剧本回复"的假模型，检查每一轮发出去的 messages 是否正确。全程不联网。"""

import json
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from finagent.agent.loop import run_tool_loop
from finagent.config import LLMConfig
from finagent.llm.base import LLMResponse, ToolCall, Usage
from finagent.llm.openai_compat import OpenAICompatClient
from finagent.pipeline.answer import answer_question
from finagent.prompts import load_prompt
from finagent.tools import build_registry

REGISTRY = build_registry(["calculate"])
GOOD = {
    "reasoning": "(5829 - 5735) / 5735 * 100 = 1.639",
    "status": "answered",
    "value": 1.639,
    "unit": "percent",
    "answer_text": "Revenue grew 1.639%.",
    "sources": ["table: revenue | 5829 | 5735"],
    "confidence": "high",
}


def calls(*expressions: str) -> list[ToolCall]:
    return [
        ToolCall(f"call_{i}", "calculate", json.dumps({"expression": e}))
        for i, e in enumerate(expressions)
    ]


def reply(content: str = "", tool_calls: list[ToolCall] | None = None, reasoning=None):
    return LLMResponse(
        content=content,
        finish_reason="tool_calls" if tool_calls else "stop",
        model="fake",
        usage=Usage(),
        latency_s=0.1,
        cost_usd=0.001,
        reasoning=reasoning,
        tool_calls=tool_calls or [],
    )


class ScriptedClient:
    """按顺序返回剧本里的回复，并记下每次收到的参数（messages 复制一份，防止之后被修改）。"""

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.requests: list[dict] = []

    def chat(self, messages, on_delta=None, json_mode=False, tools=None) -> LLMResponse:
        self.requests.append({"messages": list(messages), "json_mode": json_mode, "tools": tools})
        return self.script.pop(0)


START = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q"}]

# ---------------------------------------------------------------- to_message


def test_to_message_in_openai_format() -> None:
    msg = reply(tool_calls=calls("1 + 1")).to_message()
    assert msg["role"] == "assistant"
    assert msg["tool_calls"] == [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "calculate", "arguments": '{"expression": "1 + 1"}'},
        }
    ]
    assert "reasoning_content" not in msg  # 没开 thinking 时不带这个字段


def test_to_message_keeps_reasoning_for_thinking_mode() -> None:
    msg = reply(tool_calls=calls("1 + 1"), reasoning="let me compute").to_message()
    assert msg["reasoning_content"] == "let me compute"  # DeepSeek 要求带工具时原样传回


# ---------------------------------------------------------------- run_tool_loop


def test_loop_executes_parallel_calls_and_returns_results() -> None:
    client = ScriptedClient(
        [
            reply(tool_calls=calls("(5829 - 5735) / 5735 * 100", "(5735 - 5102) / 5102 * 100")),
            reply(content="final"),
        ]
    )
    gen = run_tool_loop(client, START, REGISTRY, json_mode=True)

    assert gen.final.content == "final" and not gen.hit_max_rounds
    assert len(gen.responses) == 2 and len(gen.tool_steps) == 2
    # 第 2 轮发出去的 messages：原来的 2 条 + assistant 的调用请求 + 两条 tool 结果
    second = client.requests[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "tool"]
    assert [m["tool_call_id"] for m in second[3:]] == ["call_0", "call_1"]  # 用编号对上号
    assert second[3]["content"] == "1.63905841325"
    assert second[4]["content"] == "12.4068992552"
    # 每一轮都带着 tools 和 json_mode（API 无状态，每次都要完整发送）
    assert all(r["tools"] == REGISTRY.schemas() and r["json_mode"] for r in client.requests)
    assert gen.messages[-1] == {"role": "assistant", "content": "final"}
    assert START == [{"role": "system", "content": "sys"}, {"role": "user", "content": "Q"}]


def test_loop_feeds_errors_back_so_the_model_can_fix_them() -> None:
    client = ScriptedClient(
        [
            reply(tool_calls=calls("5,829 - 5,735")),  # 第 1 轮：写了千分位逗号
            reply(tool_calls=calls("5829 - 5735")),  # 第 2 轮：看到错误信息后改正
            reply(content="final"),
        ]
    )
    gen = run_tool_loop(client, START, REGISTRY)
    assert [s.ok for s in gen.tool_steps] == [False, True]
    error_message = client.requests[1]["messages"][-1]
    assert error_message["role"] == "tool" and "thousands separators" in error_message["content"]


def test_loop_stops_at_max_rounds() -> None:
    client = ScriptedClient([reply(tool_calls=calls("1 + 1")) for _ in range(3)])
    gen = run_tool_loop(client, START, REGISTRY, max_rounds=3)
    assert gen.hit_max_rounds and len(gen.responses) == 3 and len(client.requests) == 3


def test_loop_without_tool_calls_is_a_single_request() -> None:
    client = ScriptedClient([reply(content="direct answer")])
    gen = run_tool_loop(client, START, REGISTRY)
    assert gen.final.content == "direct answer" and gen.tool_steps == []


# ---------------------------------------------------------------- answer_question + 工具

PROMPT = load_prompt("answer_ctx_v1")


def test_answer_with_tools_end_to_end() -> None:
    client = ScriptedClient(
        [reply(tool_calls=calls("(5829 - 5735) / 5735 * 100")), reply(json.dumps(GOOD))]
    )
    res = answer_question(client, PROMPT, "Growth?", context="revenue | 5829 | 5735", tools=REGISTRY)
    assert res.answer is not None and res.answer.value == 1.639
    assert res.attempts == 1 and len(res.responses) == 2 and len(res.tool_steps) == 1
    assert "revenue | 5829 | 5735" in client.requests[0]["messages"][-1]["content"]
    # 带工具时默认不开 JSON mode：开着时模型几乎不调用工具（L04 实测）
    assert not any(r["json_mode"] for r in client.requests)


def test_repair_keeps_the_tool_history() -> None:
    client = ScriptedClient(
        [
            reply(tool_calls=calls("1 + 1")),
            reply("not json"),  # 第 1 次尝试：算完了，但输出不是 JSON
            reply(json.dumps(GOOD)),  # 第 2 次尝试：看到反馈后改正
        ]
    )
    res = answer_question(client, PROMPT, "Q", context="C", tools=REGISTRY)
    assert res.answer is not None and res.attempts == 2
    retry = client.requests[2]["messages"]
    # 重试时带着完整的历史（包括工具调用和结果），最后是"你的输出不合格"的反馈
    assert [m["role"] for m in retry[-4:]] == ["assistant", "tool", "assistant", "user"]
    assert "rejected" in retry[-1]["content"]


def test_hitting_max_rounds_restarts_from_scratch() -> None:
    client = ScriptedClient(
        [reply(tool_calls=calls("1 + 1")), reply(tool_calls=calls("2 + 2")), reply(json.dumps(GOOD))]
    )
    res = answer_question(client, PROMPT, "Q", context="C", tools=REGISTRY, max_rounds=2)
    assert res.answer is not None and res.attempts == 2
    assert "轮数上限" in res.errors[0]
    assert len(client.requests[2]["messages"]) == 2  # 第 2 次尝试从头开始，不带之前的工具历史


def test_context_requires_a_context_prompt() -> None:
    with pytest.raises(ValueError, match="context"):
        answer_question(ScriptedClient([]), load_prompt("answer_v3"), "Q", context="C")


# ---------------------------------------------------------------- OpenAICompatClient 的工具部分


def fake_completion(message: dict, finish_reason: str) -> ChatCompletion:
    """用 SDK 自己的类型构造一个响应，和真实 API 返回的结构完全一样。"""
    return ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "deepseek-flash",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("finagent.tracing.log_call", lambda record: None)  # 不写调用记录
    return OpenAICompatClient(LLMConfig())


def test_request_includes_tools(client) -> None:
    request = client._build_request(START, json_mode=True, tools=REGISTRY.schemas())
    assert request["tools"][0]["function"]["name"] == "calculate"
    assert "tool_choice" not in request  # 不传就是 auto：由模型决定调不调用


def test_parses_tool_calls_from_the_api(client) -> None:
    completion = fake_completion(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_00_abc",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'},
                }
            ],
        },
        "tool_calls",
    )
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: completion))
    )
    resp = client.chat(START, tools=REGISTRY.schemas())
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls == [ToolCall("call_00_abc", "calculate", '{"expression": "1+1"}')]


def test_streaming_with_tools_is_rejected(client) -> None:
    with pytest.raises(ValueError, match="流式"):
        client.chat(START, on_delta=lambda kind, text: None, tools=REGISTRY.schemas())
