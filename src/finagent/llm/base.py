"""LLM 层的统一接口（ADR-1）。

两条约定：
  1. 消息直接用 OpenAI chat 格式：list[dict]，例如
         [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
     DeepSeek、vLLM、Ollama 和 HF 的 apply_chat_template 都认这个格式，所以不自己发明。
  2. 所有后端都返回同一个 LLMResponse，上层（CLI、eval、agent）只认它。

L01 只有纯文本对话；L04 会在这里加上 tool_calls。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

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
class LLMResponse:
    content: str  # 正式回答
    finish_reason: str  # 为什么停下：stop / length / content_filter / tool_calls / ...
    model: str  # 服务端报告的模型名（不一定和请求时写的名字相同）
    usage: Usage
    latency_s: float  # 从发出请求到收完全部输出的总时间
    cost_usd: float | None = None  # 按 pricing.py 算出的花费；不认识的模型为 None
    reasoning: str | None = None  # thinking 模式下的思考过程（reasoning_content）
    ttft_s: float | None = None  # 首 token 延迟（time to first token），只有流式调用才有


class LLMClient(Protocol):
    """所有后端都要实现的接口。上层代码只依赖它，不依赖具体后端。"""

    def chat(self, messages: list[Message], on_delta: DeltaCallback | None = None) -> LLMResponse:
        """发送一轮对话。

        on_delta 为 None：一次性返回。
        on_delta 不为 None：走流式，每收到一个片段就调用一次 on_delta，
        最后仍然返回完整的 LLMResponse（包括 usage 和 cost）。
        """
        ...
