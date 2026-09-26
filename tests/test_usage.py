"""usage 解析：用 SDK 自己的类型构造出假响应来测，不需要联网。"""

from openai.types import CompletionUsage

from finagent.llm.base import Usage
from finagent.llm.openai_compat import parse_usage


def test_deepseek_extension_fields() -> None:
    u = CompletionUsage.model_validate(
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_cache_hit_tokens": 64,  # DeepSeek 扩展字段
            "prompt_cache_miss_tokens": 36,
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
    )
    r = parse_usage(u)
    assert (r.cache_hit_tokens, r.cache_miss_tokens, r.reasoning_tokens) == (64, 36, 30)


def test_standard_openai_fields_fallback() -> None:
    # vLLM 等后端只给标准字段：缓存命中写在 prompt_tokens_details.cached_tokens
    u = CompletionUsage.model_validate(
        {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 40},
        }
    )
    r = parse_usage(u)
    assert (r.cache_hit_tokens, r.cache_miss_tokens, r.reasoning_tokens) == (40, 60, 0)


def test_missing_usage() -> None:
    assert parse_usage(None) == Usage()
