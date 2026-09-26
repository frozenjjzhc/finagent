"""按 token 计费。

价格来源：DeepSeek 官网价格页，核对日期见 PRICES_VERIFIED。价格会变，变了就改这张表；
每条调用记录（runs/calls/*.jsonl）都保存了原始 token 数，事后可以按新价格重算。

计费公式（单位：美元）：
    cost = 折扣 × (缓存命中的输入 × 缓存价 + 未命中的输入 × 输入价 + 输出 × 输出价) / 1,000,000
    折扣：高峰时段 1.0，非高峰时段 0.5
"""

from __future__ import annotations

from datetime import datetime, timezone

from finagent.llm.base import Usage

PRICES_VERIFIED = "2026-09-25"

# 高峰价，单位：美元 / 百万 token。三个数依次是：(缓存命中的输入, 缓存未命中的输入, 输出)
PEAK_PRICES: dict[str, tuple[float, float, float]] = {
    "deepseek-flash": (0.006, 0.30, 1.20),
    "deepseek-v4-pro": (0.044, 1.32, 3.96),
}
OFF_PEAK_DISCOUNT = 0.5

# 高峰时段（UTC，周一到周五），左闭右开：[01:00, 04:00) 和 [06:00, 10:00)
PEAK_HOURS_UTC = ((1, 4), (6, 10))


def is_peak(when: datetime) -> bool:
    """判断某个时刻是不是高峰时段。

    when 必须带时区。不带时区的 datetime（naive datetime）不知道自己是哪个时区的时间，
    是时区 bug 最常见的来源，所以直接拒绝。
    官方还排除了中国法定节假日，这里没处理：节假日会被当成高峰，只会高估成本，不会低估。
    """
    if when.tzinfo is None:
        raise ValueError("when 必须带时区，例如 datetime.now(timezone.utc)")
    utc = when.astimezone(timezone.utc)
    if utc.weekday() >= 5:  # 5 = 周六，6 = 周日
        return False
    return any(start <= utc.hour < end for start, end in PEAK_HOURS_UTC)


def cost_usd(model: str, usage: Usage, when: datetime) -> float | None:
    """算一次调用的花费。不认识的模型（比如以后的本地模型）返回 None，而不是瞎算一个数。"""
    prices = PEAK_PRICES.get(model)
    if prices is None:
        return None
    hit_price, miss_price, out_price = prices
    discount = 1.0 if is_peak(when) else OFF_PEAK_DISCOUNT
    total = (
        usage.cache_hit_tokens * hit_price
        + usage.cache_miss_tokens * miss_price
        + usage.completion_tokens * out_price
    )
    return discount * total / 1_000_000
