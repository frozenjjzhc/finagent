"""FinAgent 命令行。安装后在终端直接用：finagent --help

typer 的用法：函数参数就是命令行参数。
  - 普通参数（question）       → 位置参数
  - typer.Option 标注的参数    → --选项
  - "--thinking/--no-thinking" → 一对开关；默认 None 表示"不覆盖配置文件里的值"
"""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Literal

import openai
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from finagent import tracing
from finagent.config import LLMConfig, load_config, with_overrides
from finagent.llm.base import LLMResponse, Message
from finagent.llm.openai_compat import OpenAICompatClient

app = typer.Typer(
    help="FinAgent：美股财报问答助手（学习项目）", no_args_is_help=True, add_completion=False
)

DEFAULT_CONFIG = "configs/systems/v0.1.yaml"


def build_messages(question: str, system_prompt: str | None = None) -> list[Message]:
    """组装发给模型的 messages。

    规则：有 system prompt 时，它必须是第一条；没有（或只有空白）时，就不加 system 消息。
    单独写成一个函数，是为了能直接测试它，不用每次都调 API。
    """
    messages: list[Message] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": question})
    return messages


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="要问的问题")],
    config: Annotated[str, typer.Option(help="系统配置文件")] = DEFAULT_CONFIG,
    model: Annotated[str | None, typer.Option(help="临时换模型，如 deepseek-v4-pro")] = None,
    thinking: Annotated[
        bool | None, typer.Option("--thinking/--no-thinking", help="临时开 / 关 thinking 模式")
    ] = None,
    max_tokens: Annotated[int | None, typer.Option(help="输出 token 上限")] = None,
    temperature: Annotated[
        float | None, typer.Option(help="采样温度，0~2（只在关闭 thinking 时有效）")
    ] = None,
    stream: Annotated[bool, typer.Option("--stream/--no-stream", help="是否流式输出")] = True,
    show_reasoning: Annotated[bool, typer.Option(help="显示 thinking 模式的思考过程")] = False,
    # 参数名不能叫 system：函数体里已经有一个变量 system（整份系统配置），会把它覆盖掉。
    # 所以参数叫 system_prompt，再用 "--system" 指定命令行上的写法。
    system_prompt: Annotated[
        str | None, typer.Option("--system", help="作为 system 消息放在最前面的一段话")
    ] = None,
) -> None:
    """问一个问题。v0.1：闭卷、没有检索和工具；可以用 --system 加一条 system 消息。"""
    system = load_config(config)
    try:
        llm_cfg = with_overrides(
            system.llm,
            model=model,
            thinking=thinking,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except ValidationError as e:
        typer.secho(f"参数组合不合法：\n{e}", fg="red", err=True)
        raise typer.Exit(2) from None

    client = OpenAICompatClient(llm_cfg, tag=system.name)
    messages = build_messages(question, system_prompt)
    printer = _StreamPrinter(show_reasoning) if stream else None
    try:
        resp = client.chat(messages, on_delta=printer)
    except openai.OpenAIError as e:
        typer.secho(_explain_error(e), fg="red", err=True)
        raise typer.Exit(1) from None

    if not stream:
        if show_reasoning and resp.reasoning:
            typer.secho(f"[思考过程]\n{resp.reasoning}\n\n[回答]", fg="bright_black")
        typer.echo(resp.content, nl=False)
    typer.echo()
    typer.secho(_footer(llm_cfg, resp), fg="bright_black")
    if resp.finish_reason == "length":
        typer.secho("警告：输出被 max_tokens 截断了，上面的回答不完整！", fg="yellow")


@app.command()
def usage(
    day: Annotated[str | None, typer.Option(help="日期 YYYY-MM-DD，默认今天")] = None,
) -> None:
    """汇总某一天的调用次数、token 用量和花费（数据来自 runs/calls/）。"""
    records = tracing.read_calls(day)
    if not records:
        typer.echo("这一天没有调用记录。")
        return

    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in records:
        s = stats[r["model"] + (" +thinking" if r.get("thinking") else "")]
        s["calls"] += 1
        if "error" in r:
            s["errors"] += 1
            continue
        u = r["usage"]
        s["prompt"] += u["prompt_tokens"]
        s["hit"] += u["cache_hit_tokens"]
        s["completion"] += u["completion_tokens"]
        s["reasoning"] += u["reasoning_tokens"]
        s["cost"] += r.get("cost_usd") or 0.0
        s["latency"] += r["latency_s"]

    # rich 负责画表格：它能正确计算中文字符的显示宽度，表格不会错位
    table = Table(title=f"调用汇总 · {day or '今天'}（token 数；延迟为平均值）")
    table.add_column("模型", no_wrap=True)  # 模型名不折行，宁可让别的列窄一点
    for col in ("调用", "失败", "输入", "缓存命中", "输出", "思考", "延迟", "花费"):
        table.add_column(col, justify="right", no_wrap=True)
    total_cost = 0.0
    for name, s in sorted(stats.items()):
        ok = s["calls"] - s["errors"]
        total_cost += s["cost"]
        table.add_row(
            name,
            f"{s['calls']:.0f}",
            f"{s['errors']:.0f}",
            f"{s['prompt']:,.0f}",
            f"{s['hit']:,.0f}",
            f"{s['completion']:,.0f}",
            f"{s['reasoning']:,.0f}",
            f"{s['latency'] / ok:.1f}s" if ok else "-",
            f"${s['cost']:.4f}",
        )
    Console().print(table)
    typer.echo(f"合计花费：${total_cost:.4f}")


class _StreamPrinter:
    """流式回调：边收边打印。思考过程默认不显示，只提示一次"思考中"。"""

    def __init__(self, show_reasoning: bool) -> None:
        self.show_reasoning = show_reasoning
        self.last_kind: str | None = None

    def __call__(self, kind: Literal["reasoning", "content"], text: str) -> None:
        if kind != self.last_kind:
            if kind == "reasoning":
                typer.secho(
                    "[思考过程]" if self.show_reasoning else "[思考中……]", fg="bright_black"
                )
            elif self.last_kind == "reasoning":
                typer.secho("\n[回答]" if self.show_reasoning else "[回答]", fg="bright_black")
            self.last_kind = kind
        if kind == "reasoning":
            if self.show_reasoning:
                typer.secho(text, fg="bright_black", nl=False)
        else:
            typer.echo(text, nl=False)  # click 的 echo 每次都会 flush，所以能看到"逐字蹦出来"


def _footer(cfg: LLMConfig, r: LLMResponse) -> str:
    u = r.usage
    cost = "未知" if r.cost_usd is None else f"${r.cost_usd:.6f}"
    ttft = "" if r.ttft_s is None else f"，首 token {r.ttft_s:.2f}s"
    return (
        f"[{cfg.model} | thinking={'on' if cfg.thinking else 'off'} | "
        f"输入 {u.prompt_tokens}（缓存命中 {u.cache_hit_tokens}） · "
        f"输出 {u.completion_tokens}（思考 {u.reasoning_tokens}） | {cost} | "
        f"{r.latency_s:.1f}s{ttft} | finish={r.finish_reason}]"
    )


def _explain_error(e: openai.OpenAIError) -> str:
    """把 SDK 异常翻译成人话。注意顺序：子类要写在父类前面。"""
    if isinstance(e, openai.AuthenticationError):
        return "401 认证失败：API key 无效，检查 .env"
    if isinstance(e, openai.RateLimitError):
        return "429 请求太频繁：SDK 已经自动重试过，还是失败，请稍后再试"
    if isinstance(e, openai.APIStatusError):
        hints = {
            400: "请求格式错误",
            402: "余额不足，去 platform.deepseek.com 充值",
            422: "参数不合法",
            500: "服务器故障，稍后重试",
            503: "服务器繁忙，稍后重试",
        }
        return f"HTTP {e.status_code}：{hints.get(e.status_code, '')}（{e.message}）"
    if isinstance(e, openai.APITimeoutError):  # 它是 APIConnectionError 的子类，要先判断
        return "请求超时：可以调大 config 里的 timeout_s"
    if isinstance(e, openai.APIConnectionError):
        return "连不上服务器：检查网络或代理"
    return f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    app()
