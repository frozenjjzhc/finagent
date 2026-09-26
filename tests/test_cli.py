"""命令行的测试：build_messages 的组装规则，以及 --system 选项有没有真正传到模型。全程不联网。"""

from typer.testing import CliRunner

from finagent import cli
from finagent.cli import build_messages
from finagent.llm.base import LLMResponse, Usage

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
    """假的 LLM 客户端：不联网，只记下收到的 messages，返回一个固定回答。"""

    last_messages: list | None = None

    def __init__(self, cfg, tag: str = "") -> None:
        self.cfg = cfg

    def chat(self, messages, on_delta=None) -> LLMResponse:
        FakeClient.last_messages = messages
        return LLMResponse(
            content="42", finish_reason="stop", model="fake", usage=Usage(), latency_s=0.0
        )


def test_system_option_reaches_the_model(monkeypatch) -> None:
    # 替换的是 cli 模块里的名字（"在哪里用，就在哪里替换"），而不是 openai_compat 模块里的
    monkeypatch.setattr(cli, "OpenAICompatClient", FakeClient)
    result = CliRunner().invoke(
        cli.app, ["ask", "What is 6*7?", "--system", "只回答数字", "--no-stream"]
    )
    assert result.exit_code == 0, result.output
    assert FakeClient.last_messages == [
        {"role": "system", "content": "只回答数字"},
        {"role": "user", "content": "What is 6*7?"},
    ]
