"""工具注册表（L04）：把一个普通的 Python 函数变成模型能调用的"工具"。

模型怎么知道有哪些工具可用？靠请求里的 tools 参数：每个工具一段 JSON Schema，
写明函数名、做什么用、有哪些参数、每个参数是什么类型。手写这段 JSON 又长又容易和函数对不上，
所以用装饰器 @tool 从函数本身自动生成：

    @tool
    def calculate(expression: str) -> str:
        '''Evaluate an arithmetic expression ...        ← 第一段 → 工具的 description

        Args:
            expression: A single arithmetic expression  ← Args 段 → 参数的 description
        '''

    参数名、类型、是否必填 ← 来自函数签名（类型注解 + 默认值）

同一个 pydantic 模型还负责**校验模型传来的参数**：模型可能漏参数、传错类型、多传参数，
这些都在执行前拦下来，变成一条错误信息回传给模型，让它自己改正。
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from finagent.llm.base import ToolCall, ToolStep


class ToolError(ValueError):
    """工具执行失败，而且失败原因应该告诉模型（比如表达式写错了），让它改正后重试。"""


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]  # 由函数签名生成的参数模型，既用来生成 schema，也用来校验参数

    def schema(self) -> dict[str, Any]:
        """OpenAI 格式的工具定义，放进请求的 tools 列表里。"""
        parameters = self.args_model.model_json_schema()
        _strip_titles(parameters)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def run(self, arguments: str) -> tuple[str, bool]:
        """执行一次调用，返回（回传给模型的文字, 是否成功）。

        任何失败都**不抛异常**，而是变成一条 "Error: ..." 文字：
        工具出错时，整个程序不应该崩溃；把错误告诉模型，它通常能自己改正（错误回传）。
        """
        try:
            data = json.loads(arguments or "{}")
        except json.JSONDecodeError as e:
            return f"Error: arguments are not valid JSON ({e.msg}).", False
        if not isinstance(data, dict):
            return "Error: arguments must be a JSON object.", False
        try:
            args = self.args_model.model_validate(data)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'arguments'}: {err['msg']}"
                for err in e.errors()
            )
            return f"Error: invalid arguments: {problems}.", False
        try:
            result = self.fn(**args.model_dump())
        except ToolError as e:
            return f"Error: {e}", False
        # 工具代码本身的意外错误：同样回传给模型（带上异常类型方便排查），不让整个循环崩溃
        except Exception as e:  # noqa: BLE001
            return f"Error: {type(e).__name__}: {e}", False
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return text, True


def tool(fn: Callable[..., Any]) -> Tool:
    """装饰器：从函数签名和 docstring 生成一个 Tool。"""
    description, arg_docs = _parse_docstring(inspect.getdoc(fn) or "")
    if not description:
        raise ValueError(f"工具 {fn.__name__} 缺少 docstring：模型要靠描述决定什么时候调用它")
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name not in hints:
            raise ValueError(f"工具 {fn.__name__} 的参数 {name} 缺少类型注解")
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[name] = (hints[name], Field(default, description=arg_docs.get(name)))
    args_model = create_model(
        f"{fn.__name__}_args",
        __config__=ConfigDict(
            extra="forbid"
        ),  # 多传了参数也算错：生成的 schema 里是 additionalProperties: false
        **fields,
    )
    return Tool(name=fn.__name__, description=description, fn=fn, args_model=args_model)


class ToolRegistry:
    """一组可用的工具。工具循环通过它拿到 schema 列表，并按名字执行调用。"""

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolStep:
        t0 = time.perf_counter()
        t = self._tools.get(call.name)
        if t is None:  # 模型编了一个不存在的工具名：同样回传错误，并告诉它有哪些可用
            result, ok = f"Error: unknown tool '{call.name}'. Available: {self.names}.", False
        else:
            result, ok = t.run(call.arguments)
        return ToolStep(call=call, result=result, ok=ok, latency_s=time.perf_counter() - t0)


def build_registry(names: Iterable[str]) -> ToolRegistry:
    """按名字（系统配置里 tools.enabled 列出的）组装注册表。名字写错就报错，而不是悄悄少一个工具。"""
    from finagent.tools import AVAILABLE_TOOLS  # 放在函数里导入，避免循环导入

    unknown = [n for n in names if n not in AVAILABLE_TOOLS]
    if unknown:
        raise ValueError(f"未知的工具：{unknown}，可用的：{sorted(AVAILABLE_TOOLS)}")
    return ToolRegistry(AVAILABLE_TOOLS[n] for n in names)


# ---------------------------------------------------------------- 辅助函数


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Google 风格 docstring → （工具描述, {参数名: 参数描述}）。

    "Args:" 之前的部分是工具描述；"Args:" 之后，每个 "名字: 描述" 开头的行是一个参数，
    缩进更深的行是上一个参数描述的续行。
    """
    head, _, args_part = doc.partition("\nArgs:")
    description = " ".join(head.split())
    arg_docs: dict[str, str] = {}
    current = None
    for line in args_part.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        name, sep, text = stripped.partition(":")
        if (
            sep
            and name.isidentifier()
            and line.startswith(" " * 4)
            and not line.startswith(" " * 8)
        ):
            current = name
            arg_docs[name] = text.strip()
        elif current is not None:
            arg_docs[current] += " " + stripped
    return description, arg_docs


def _strip_titles(schema: dict[str, Any]) -> None:
    """去掉 pydantic 自动加的 "title" 字段：对模型没用，还白占输入 token（每次请求都要发送）。"""
    schema.pop("title", None)
    for sub in schema.get("properties", {}).values():
        _strip_titles(sub)
