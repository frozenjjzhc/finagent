"""计费与高峰时段判断。这类纯函数最适合写单元测试：不联网、跑得快、边界清楚。"""

from datetime import datetime, timedelta, timezone

import pytest

from finagent.llm.base import Usage
from finagent.llm.pricing import cost_usd, is_peak

UTC = timezone.utc
# Windows 上没有系统时区数据库，zoneinfo 需要额外装 tzdata；测试里直接用固定偏移就够了
LA_SUMMER = timezone(timedelta(hours=-7))  # 洛杉矶夏令时 UTC-7


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (datetime(2026, 9, 28, 2, 30, tzinfo=UTC), True),  # 周一 02:30，第一个高峰段内
        (datetime(2026, 9, 28, 4, 0, tzinfo=UTC), False),  # 04:00 整：左闭右开，已经结束
        (datetime(2026, 9, 28, 5, 0, tzinfo=UTC), False),  # 两个高峰段之间
        (datetime(2026, 10, 2, 9, 59, tzinfo=UTC), True),  # 周五 09:59
        (datetime(2026, 10, 2, 10, 0, tzinfo=UTC), False),  # 周五 10:00 整
        (datetime(2026, 9, 26, 2, 0, tzinfo=UTC), False),  # 周六不分高峰
    ],
)
def test_is_peak(when: datetime, expected: bool) -> None:
    assert is_peak(when) is expected


def test_is_peak_converts_timezone() -> None:
    # 洛杉矶周日 18:30 = UTC 周一 01:30 → 高峰。只看本地时间会误判成"周末"
    assert is_peak(datetime(2026, 9, 27, 18, 30, tzinfo=LA_SUMMER))


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValueError):
        # 故意传入不带时区的 datetime，验证它会被拒绝。noqa 告诉 ruff：这里是有意为之
        is_peak(datetime(2026, 9, 28, 2, 0))  # noqa: DTZ001


def test_off_peak_is_half_price() -> None:
    u = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, cache_miss_tokens=1_000_000)
    peak = cost_usd("deepseek-flash", u, datetime(2026, 9, 28, 2, 0, tzinfo=UTC))
    off = cost_usd("deepseek-flash", u, datetime(2026, 9, 28, 12, 0, tzinfo=UTC))
    assert peak == pytest.approx(0.30 + 1.20)
    assert off == pytest.approx(peak / 2)


def test_cache_hits_are_50x_cheaper() -> None:
    when = datetime(2026, 9, 28, 2, 0, tzinfo=UTC)
    miss = cost_usd("deepseek-flash", Usage(prompt_tokens=10**6, cache_miss_tokens=10**6), when)
    hit = cost_usd("deepseek-flash", Usage(prompt_tokens=10**6, cache_hit_tokens=10**6), when)
    assert miss / hit == pytest.approx(50)


def test_unknown_model_has_no_price() -> None:
    assert cost_usd("qwen3-1.7b", Usage(), datetime.now(UTC)) is None
