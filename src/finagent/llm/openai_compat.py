"""OpenAI 兼容后端。

L01：连 DeepSeek（base_url = https://api.deepseek.com）。
L15：同一个类，把 base_url 换成本地 vLLM / Ollama 的地址即可。

DeepSeek 在标准 OpenAI 格式之外的扩展，都集中在这个文件里处理（也就是 ADR-1 说的"漏"）：
  - 请求：thinking 开关放在 extra_body={"thinking": {...}}
  - 响应：思考过程在 message.reasoning_content；缓存用量在 usage.prompt_cache_hit/miss_tokens
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import openai
from openai import OpenAI

from finagent import tracing
from finagent.config import LLMConfig
from finagent.llm.base import DeltaCallback, LLMResponse, Message, Usage
from finagent.llm.pricing import cost_usd, is_peak


class OpenAICompatClient:
    def __init__(self, cfg: LLMConfig, tag: str = "") -> None:
        api_key = os.environ.get(cfg.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"环境变量 {cfg.api_key_env} 为空：检查项目根目录的 .env")
        self.cfg = cfg
        self.tag = tag  # 写进调用记录，标明是哪个系统版本 / 哪个实验发起的调用
        # SDK 自带重试：连接错误、408/409/429/5xx 会按指数退避自动重试，最多 max_retries 次。
        # 400/401/402/422 这类"请求本身有问题"的错误不重试：重试多少次结果都一样。
        self._client = OpenAI(
            api_key=api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )

    def chat(
        self,
        messages: list[Message],
        on_delta: DeltaCallback | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        request = self._build_request(messages, json_mode)
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()  # 测耗时用 perf_counter（单调时钟），不用 time.time()
        try:
            if on_delta is None:
                resp = self._call_once(request, t0)
            else:
                resp = self._call_stream(request, t0, on_delta)
        except openai.OpenAIError as e:
            self._log(messages, started_at, json_mode=json_mode, error=e)  # 失败的调用也要记下来
            raise
        resp.cost_usd = cost_usd(self.cfg.model, resp.usage, started_at)
        self._log(messages, started_at, json_mode=json_mode, resp=resp, stream=on_delta is not None)
        return resp

    # ------------------------------------------------------------------ 请求

    def _build_request(self, messages: list[Message], json_mode: bool = False) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            # thinking 不是 OpenAI 标准参数，SDK 不认识它，所以放进 extra_body：
            # SDK 会把 extra_body 里的键原样合并进请求的 JSON 顶层。
            # （L15 接本地推理服务时，再看要不要去掉这个字段。）
            "extra_body": {"thinking": {"type": "enabled" if self.cfg.thinking else "disabled"}},
        }
        if self.cfg.reasoning_effort is not None:
            request["reasoning_effort"] = self.cfg.reasoning_effort
        if self.cfg.max_tokens is not None:
            request["max_tokens"] = self.cfg.max_tokens
        if self.cfg.temperature is not None:
            request["temperature"] = self.cfg.temperature
        if json_mode:
            # JSON mode：只保证输出是合法 JSON，不保证符合我们的 schema。
            # DeepSeek 还要求 prompt 里出现 "json" 这个词，并给出格式示例。
            request["response_format"] = {"type": "json_object"}
        return request

    # ------------------------------------------------------------------ 非流式

    def _call_once(self, request: dict[str, Any], t0: float) -> LLMResponse:
        completion = self._client.chat.completions.create(**request)
        latency = time.perf_counter() - t0
        choice = completion.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            # reasoning_content 是 DeepSeek 的扩展字段，SDK 的类型定义里没有，只能用 getattr 取
            reasoning=getattr(choice.message, "reasoning_content", None),
            finish_reason=choice.finish_reason or "",
            model=completion.model,
            usage=parse_usage(completion.usage),
            latency_s=latency,
        )

    # ------------------------------------------------------------------ 流式

    def _call_stream(
        self, request: dict[str, Any], t0: float, on_delta: DeltaCallback
    ) -> LLMResponse:
        # include_usage=True：让服务端在最后附上一个带 usage 的 chunk，否则流式拿不到 token 数
        stream = self._client.chat.completions.create(
            **request, stream=True, stream_options={"include_usage": True}
        )
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = ""
        model = self.cfg.model
        usage = Usage()
        ttft: float | None = None

        for chunk in stream:
            model = chunk.model or model
            if chunk.usage is not None:
                usage = parse_usage(chunk.usage)
            if not chunk.choices:  # 收尾的 chunk 只带 usage，没有 choices
                continue
            choice = chunk.choices[0]
            reasoning_piece = getattr(choice.delta, "reasoning_content", None)
            if reasoning_piece:
                ttft = ttft if ttft is not None else time.perf_counter() - t0
                reasoning_parts.append(reasoning_piece)
                on_delta("reasoning", reasoning_piece)
            if choice.delta.content:
                ttft = ttft if ttft is not None else time.perf_counter() - t0
                content_parts.append(choice.delta.content)
                on_delta("content", choice.delta.content)
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        return LLMResponse(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts) or None,
            finish_reason=finish_reason,
            model=model,
            usage=usage,
            latency_s=time.perf_counter() - t0,
            ttft_s=ttft,
        )

    # ------------------------------------------------------------------ 记录

    def _log(
        self,
        messages: list[Message],
        started_at: datetime,
        json_mode: bool = False,
        resp: LLMResponse | None = None,
        stream: bool = False,
        error: Exception | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "ts": started_at.isoformat(),
            "tag": self.tag,
            "base_url": self.cfg.base_url,
            "model": self.cfg.model,
            "thinking": self.cfg.thinking,
            "reasoning_effort": self.cfg.reasoning_effort,
            "json_mode": json_mode,
            "peak": is_peak(started_at),
            "stream": stream,
            "messages": messages,
        }
        if error is not None:
            # 401 的错误信息里可能带有部分 key，不记录原文
            message = "" if isinstance(error, openai.AuthenticationError) else str(error)[:500]
            record["error"] = {
                "type": type(error).__name__,
                "status": getattr(error, "status_code", None),
                "message": message,
            }
        elif resp is not None:
            record.update(
                response_model=resp.model,
                content=resp.content,
                reasoning=resp.reasoning,
                finish_reason=resp.finish_reason,
                usage=resp.usage.to_dict(),
                cost_usd=resp.cost_usd,
                latency_s=round(resp.latency_s, 3),
                ttft_s=None if resp.ttft_s is None else round(resp.ttft_s, 3),
            )
        tracing.log_call(record)


def parse_usage(u: Any) -> Usage:
    """把 SDK 的 usage 对象转成我们自己的 Usage。兼容两种写法：
    - DeepSeek：prompt_cache_hit_tokens / prompt_cache_miss_tokens（扩展字段）
    - OpenAI 标准：prompt_tokens_details.cached_tokens（vLLM 等其他后端用这个）
    """
    if u is None:
        return Usage()
    hit = getattr(u, "prompt_cache_hit_tokens", None)
    if hit is None:
        details = u.prompt_tokens_details
        hit = (details.cached_tokens or 0) if details else 0
    miss = getattr(u, "prompt_cache_miss_tokens", None)
    if miss is None:
        miss = u.prompt_tokens - hit
    completion_details = u.completion_tokens_details
    reasoning = (completion_details.reasoning_tokens or 0) if completion_details else 0
    return Usage(
        prompt_tokens=u.prompt_tokens,
        completion_tokens=u.completion_tokens,
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
        reasoning_tokens=reasoning,
    )
