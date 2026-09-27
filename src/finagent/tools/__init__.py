"""模型可以调用的工具（L04 起）。

AVAILABLE_TOOLS 是"工具名 → 工具"的总表，系统配置的 tools.enabled 从这里挑。
加一个新工具的步骤见 lessons/L04_补充_如何添加工具.md。
L07 会加入 search_filings、get_metric。
"""

from finagent.tools.calculator import calculate
from finagent.tools.registry import Tool, ToolRegistry, build_registry, tool
from finagent.tools.units import convert_units

# 注册：把工具放进总表。放进来不等于启用，系统配置的 tools.enabled 里写了名字才会发给模型
AVAILABLE_TOOLS: dict[str, Tool] = {t.name: t for t in (calculate, convert_units)}

__all__ = ["AVAILABLE_TOOLS", "Tool", "ToolRegistry", "build_registry", "tool"]
