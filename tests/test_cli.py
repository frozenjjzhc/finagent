"""命令行的测试：裸调模式（v0.1）和结构化模式（v0.2）。用假客户端代替真实 API，全程不联网。"""

import json

import pytest
from typer.testing import CliRunner

from finagent import cli
from finagent.cli import build_messages
from finagent.llm.base import LLMResponse, Usage

V01 = "configs/systems/v0.1.yaml"

# ---------------------------------------------------------------- build_messages：纯函数，最好测


def test_without_system_prompt() -> None:
    assert build_messages("Q") == [{"role": "user", "content": "Q"}]


def test_system_prompt_comes_first() -> None:
    messages = build_messages("Q", "只回答数字")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == "只回答数字"


def test_blank_system_prompt_is_ignored() -> None:
    # 只有空格的 system prompt 等于没有，不应该发一条空的 system 消息
    assert build_messages("Q", "   ") == [{"role": "user", "content": "Q"}]


# ---------------------------------------------------------------- 端到端：命令行参数 → messages


class FakeClient:
    """假的 LLM 客户端：不联网，记下收到的 messages，返回 reply 里预设的内容。"""

    last_messages: list | None = None
    last_json_mode: bool | None = None
    reply = "42"

    def __init__(self, cfg, tag: str = "") -> None:
        self.cfg = cfg

    def chat(self, messages, on_delta=None, json_mode=False) -> LLMResponse:
        FakeClient.last_messages = messages
        FakeClient.last_json_mode = json_mode
        return LLMResponse(
            content=FakeClient.reply,
            finish_reason="stop",
            model="fake",
            usage=Usage(),
            latency_s=0.0,
        )


@pytest.fixture
def fake(monkeypatch):
    # 替换的是 cli 模块里的名字（"在哪里用，就在哪里替换"），而不是 openai_compat 模块里的
    monkeypatch.setattr(cli, "OpenAICompatClient", FakeClient)
    FakeClient.reply = "42"
    return FakeClient


def run(*args: str):
    return CliRunner().invoke(cli.app, ["ask", *args])


def test_raw_mode_system_option_reaches_the_model(fake) -> None:
    result = run("What is 6*7?", "--config", V01, "--system", "只回答数字", "--no-stream")
    assert result.exit_code == 0, result.output
    assert fake.last_messages == [
        {"role": "system", "content": "只回答数字"},
        {"role": "user", "content": "What is 6*7?"},
    ]
    assert fake.last_json_mode is False  # 裸调模式不开 JSON mode


GOOD = {
    "reasoning": "From the income statement.",
    "status": "answered",
    "value": 130497,
    "unit": "USD_millions",
    "answer_text": "Revenue was $130,497 million.",
    "sources": ["NVIDIA 10-K FY2025"],
    "confidence": "high",
}


def test_structured_mode_prints_the_answer(fake) -> None:
    fake.reply = json.dumps(GOOD)
    result = run("What was NVIDIA's total revenue for fiscal year 2025?")  # 默认配置就是 v0.2
    assert result.exit_code == 0, result.output
    assert "$130,497 million" in result.output
    assert "已回答" in result.output
    assert fake.last_json_mode is True
    assert fake.last_messages[0]["role"] == "system"  # system prompt 来自 prompt 文件


def test_structured_mode_json_flag(fake) -> None:
    fake.reply = json.dumps(GOOD)
    result = run("Q?", "--json")
    assert json.loads(result.output.split("\n[")[0])["value"] == 130497


def test_structured_mode_rejects_system_option(fake) -> None:
    result = run("Q?", "--system", "hi")
    assert result.exit_code == 2


def test_structured_mode_reports_failure(fake) -> None:
    fake.reply = "sorry, no json"
    result = run("Q?")
    assert result.exit_code == 1
    assert "次尝试都没有得到合格的结构化答案" in result.output
