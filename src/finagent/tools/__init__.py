"""模型可以调用的工具（L04 起）。

AVAILABLE_TOOLS 是"工具名 → 工具"的总表，系统配置的 tools.enabled 从这里挑。
L07 会加入 search_filings、get_metric。
"""

from finagent.tools.calculator import calculate
from finagent.tools.registry import Tool, ToolRegistry, build_registry, tool

AVAILABLE_TOOLS: dict[str, Tool] = {t.name: t for t in (calculate,)}

__all__ = ["AVAILABLE_TOOLS", "Tool", "ToolRegistry", "build_registry", "tool"]
