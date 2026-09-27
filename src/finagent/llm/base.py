"""LLM 层的统一接口（ADR-1）。

两条约定：
  1. 消息直接用 OpenAI chat 格式：list[dict]，例如
         [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
     DeepSeek、vLLM、Ollama 和 HF 的 apply_chat_template 都认这个格式，所以不自己发明。
  2. 所有后端都返回同一个 LLMResponse，上层（CLI、eval、agent）只认它。

L01 只有纯文本对话；L04 加上了工具调用（ToolCall、LLMResponse.tool_calls、chat 的 tools 参数）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

Message = dict[str, object]

# 流式回调：kind 是 "reasoning"（思考过程）或 "content"（正式回答），text 是这次新到的片段
DeltaCallback = Callable[[Literal["reasoning", "content"], str], None]


@dataclass
class Usage:
    """一次调用的 token 用量。计费就按这几个数（见 llm/pricing.py）。"""

    prompt_tokens: int = 0  # 输入 token 总数 = cache_hit_tokens + cache_miss_tokens
    completion_tokens: int = 0  # 输出 token 总数，包含 reasoning_tokens
    cache_hit_tokens: int = 0  # 输入中命中上下文缓存的部分，按缓存价计费（约为原价 1/50）
    cache_miss_tokens: int = 0  # 输入中未命中缓存的部分，按正常输入价计费
    reasoning_tokens: int = 0  # 输出中属于思考过程的部分，也按输出价计费

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class ToolCall:
    """模型发出的一次工具调用请求（L04）。

    注意：模型只是"说"它想调用哪个函数、传什么参数，真正执行函数的是我们自己的代码。
    """

    id: str  # 调用编号。回传结果时，role=tool 的消息要用 tool_call_id 对上这个编号
    name: str  # 函数名
    arguments: str  # 参数：模型生成的 JSON **字符串**，不是 dict。可能不合法，要自己解析和校验


@dataclass
class ToolStep:
    """工具循环里的一步：模型请求的调用，加上我们执行之后回传给它的结果（L04）。"""

    call: ToolCall
    result: str  # 回传给模型的文字；出错时以 "Error:" 开头
    ok: bool  # 执行是否成功
    latency_s: float = 0.0


@dataclass
class LLMResponse:
    content: str  # 正式回答
    finish_reason: str  # 为什么停下：stop / length / content_filter / tool_calls / ...
    model: str  # 服务端报告的模型名（不一定和请求时写的名字相同）
    usage: Usage
    latency_s: float  # 从发出请求到收完全部输出的总时间
    cost_usd: float | None = None  # 按 pricing.py 算出的花费；不认识的模型为 None
    reasoning: str | None = None  # thinking 模式下的思考过程（reasoning_content）
    ttft_s: float | None = None  # 首 token 延迟（time to first token），只有流式调用才有
    tool_calls: list[ToolCall] = field(default_factory=list)  # 模型要求调用的工具（L04）

    def to_message(self) -> Message:
        """把这次回复变成一条 assistant 消息，追加到 messages 里，下一轮原样发回去。

        工具循环必须这样做：模型要看到"自己刚才请求了哪些调用"，才能理解后面的 role=tool 消息。
        thinking 模式下还要带上 reasoning_content：DeepSeek 文档要求，带 tools 的请求
        必须把之前的 reasoning_content 完整传回（否则可能返回 400）。
        """
        message: Message = {"role": "assistant", "content": self.content}
        if self.reasoning is not None:
            message["reasoning_content"] = self.reasoning
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return message


class LLMClient(Protocol):
    """所有后端都要实现的接口。上层代码只依赖它，不依赖具体后端。"""

    def chat(
        self,
        messages: list[Message],
        on_delta: DeltaCallback | None = None,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """发送一轮对话。

        on_delta 为 None：一次性返回。
        on_delta 不为 None：走流式，每收到一个片段就调用一次 on_delta，
        最后仍然返回完整的 LLMResponse（包括 usage 和 cost）。
        json_mode 为 True：要求模型只输出一个合法的 JSON 对象（L02）。
        tools：可供模型调用的工具（OpenAI 格式的 schema 列表，L04）。模型想调用时，
        返回的 finish_reason 是 "tool_calls"，要调用的内容在 LLMResponse.tool_calls 里。
        """
        ...
