"""单位换算工具（L04 补充讲义的完整示例：怎么加一个新工具）。

为什么值得做成工具：FinQA 的表格有的以千为单位、有的以百万为单位，题目却可能问另一种单位，
L03 / L04 的错误里专门有一类 scale_error（差了 1000 倍）。计算器当然也能 ÷1000，
但专用工具逼模型**明确说出**"原来是什么单位、要换成什么单位"，不容易漏掉或弄反。
到底有没有用，要用评测来回答（补充讲义第 2.8 节）。

这个文件演示了新工具的几个要素：
  - Literal 类型的参数 → 说明书里变成 enum，模型只能从几个值里选，传错了 pydantic 会拦下
  - docstring 写清"什么时候用"和"不要用在哪里"
  - 可预见的错误抛 ToolError，错误信息写给模型看
"""

from __future__ import annotations

import math
from typing import Literal

from finagent.tools.calculator import format_number
from finagent.tools.registry import ToolError, tool

Scale = Literal["units", "thousands", "millions", "billions"]
_FACTOR: dict[str, float] = {"units": 1.0, "thousands": 1e3, "millions": 1e6, "billions": 1e9}


@tool
def convert_units(value: float, from_scale: Scale, to_scale: Scale) -> str:
    """Convert a number from one scale to another (units, thousands, millions, billions). Use this when the excerpt states numbers in one scale (for example a table "in thousands") but the question asks for another (for example "in millions"). Do not use it for percentages.

    Args:
        value: The number as written in the source, for example 5829 for "$5,829 thousand".
        from_scale: The scale the value is written in.
        to_scale: The scale the question asks for.
    """
    # JSON 标准里没有 NaN，但 Python 的 json.loads 接受 NaN / Infinity（L02 讲过的坑），
    # pydantic 的 float 默认也放行，所以要自己检查
    if not math.isfinite(value):
        raise ToolError("value must be a finite number")
    return format_number(value * _FACTOR[from_scale] / _FACTOR[to_scale])
