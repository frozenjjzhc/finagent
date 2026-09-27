"""L04 补充演示：一个 Python 函数是怎么变成模型能调用的工具的，以及怎么加一个新工具。

配合 lessons/L04_补充_如何添加工具.md 使用。
运行（先 conda activate finagent，在项目根目录）：
    python walkthroughs/L04_add_tool.py            # 全部 4 步
    python walkthroughs/L04_add_tool.py --offline  # 只跑不花钱的第 1~3 步
花费：第 4 步 2~4 次很小的调用（flash），不到 $0.001。

    第 1 步  拆开 @tool：签名、类型注解、docstring → 参数模型 → 说明书（离线）
    第 2 步  Tool.run 的三关：模型传来各种参数时会发生什么（离线）
    第 3 步  注册与启用：AVAILABLE_TOOLS、build_registry、按名字分派（离线）
    第 4 步  真实模型同时拿到两个工具：它会怎么选
"""

from __future__ import annotations

import inspect
import json
import sys
from typing import get_type_hints

from finagent.agent.loop import run_tool_loop
from finagent.config import load_config
from finagent.llm.base import ToolCall
from finagent.llm.openai_compat import OpenAICompatClient
from finagent.tools import AVAILABLE_TOOLS, build_registry
from finagent.tools.registry import _parse_docstring
from finagent.tools.units import convert_units


def banner(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def step1_dissect() -> None:
    banner("第 1 步 · 拆开 @tool（离线）")
    fn = convert_units.fn  # @tool 返回的是 Tool 对象，原来的函数保存在 .fn 里
    print(f"(1) inspect.signature → 参数名和默认值：\n    {inspect.signature(fn)}\n")
    hints = get_type_hints(fn)
    print("(2) get_type_hints → 每个参数的类型：")
    for name, tp in hints.items():
        print(f"    {name:<10} {tp}")
    description, arg_docs = _parse_docstring(inspect.getdoc(fn) or "")
    print(f"\n(3) _parse_docstring → 工具描述（docstring 第一段）：\n    {description[:100]}…")
    print("    参数描述（Args 段）：")
    for name, doc in arg_docs.items():
        print(f"    {name:<10} {doc}")
    print("\n(4) create_model → 参数模型的字段（类型、是否必填、描述）：")
    for name, field in convert_units.args_model.model_fields.items():
        print(f"    {name:<10} {field.annotation}  必填={field.is_required()}")
    print("\n(5) Tool.schema() → 最终发给模型的说明书：")
    print(json.dumps(convert_units.schema(), indent=2, ensure_ascii=False))
    print(
        "\n注意：Literal[...] 变成了 enum，float 变成了 number；pydantic 自动加的 title 已被去掉。"
    )


def step2_run() -> None:
    banner("第 2 步 · Tool.run 的三关（离线）")
    cases = [
        ("正常", '{"value": 5829, "from_scale": "thousands", "to_scale": "millions"}'),
        ("不是 JSON（第 1 关）", "value=5829"),
        ("不是对象（第 2 关）", '[5829, "thousands", "millions"]'),
        ("漏了参数（第 3 关）", '{"value": 5829, "from_scale": "thousands"}'),
        (
            "取值不在 enum 里（第 3 关）",
            '{"value": 1, "from_scale": "thousand", "to_scale": "millions"}',
        ),
        (
            "多了参数（第 3 关）",
            '{"value": 1, "from_scale": "units", "to_scale": "units", "round": 2}',
        ),
        (
            "数字写成字符串（宽松模式会转换）",
            '{"value": "5829", "from_scale": "thousands", "to_scale": "millions"}',
        ),
        (
            "NaN（函数自己检查，抛 ToolError）",
            '{"value": NaN, "from_scale": "units", "to_scale": "units"}',
        ),
    ]
    for label, arguments in cases:
        result, ok = convert_units.run(arguments)
        print(f"  {'✔' if ok else '✘'} {label}\n      参数 {arguments}\n      返回 {result}")


def step3_registry() -> None:
    banner("第 3 步 · 注册与启用（离线）")
    print(f"AVAILABLE_TOOLS（已注册的全部工具）：{list(AVAILABLE_TOOLS)}")
    registry = build_registry(["calculate", "convert_units"])  # 相当于配置里 enabled 写了这两个
    print(f"本次启用：{registry.names}")
    print(f"发给模型的 tools 参数里有 {len(registry.schemas())} 份说明书")
    for name, args in [
        ("calculate", '{"expression": "5829 - 5735"}'),
        ("convert_units", '{"value": 94, "from_scale": "thousands", "to_scale": "millions"}'),
        ("percent_change", '{"old": 1, "new": 2}'),  # 还没写的工具（作业 3）
    ]:
        step = registry.execute(ToolCall("call_x", name, args))
        print(f"  execute({name}) → {step.result}")
    try:
        build_registry(["calculater"])
    except ValueError as e:
        print(f"配置里名字拼错：{e}")


def step4_live() -> None:
    banner("第 4 步 · 真实模型同时拿到两个工具")
    client = OpenAICompatClient(
        load_config("configs/systems/v0.4.yaml").llm, tag="walkthrough:L04_add_tool"
    )
    registry = build_registry(["calculate", "convert_units"])
    messages = [
        {
            "role": "system",
            "content": "Answer using only the table given by the user. Use the tools for every "
            "calculation and unit conversion. Reply in one short sentence.",
        },
        {
            "role": "user",
            "content": "Table (in thousands):\nyear | 2019 | 2018\nnet revenue | 5829 | 5735\n\n"
            "Question: By how much did net revenue increase from 2018 to 2019, in millions?",
        },
    ]
    gen = run_tool_loop(client, messages, registry)
    results = {s.call.id: s.result for s in gen.tool_steps}
    for round_no, resp in enumerate(gen.responses, 1):
        if not resp.tool_calls:
            print(f"  第 {round_no} 轮：不再调用工具，给出最终回答：{resp.content}")
            continue
        print(f"  第 {round_no} 轮：一次请求了 {len(resp.tool_calls)} 个调用")
        for call in resp.tool_calls:
            print(f"      {call.name}({call.arguments}) → {results[call.id]}")
    print(
        f"调用模型 {len(gen.responses)} 次，花费 ${sum(r.cost_usd or 0 for r in gen.responses):.6f}"
    )
    used = {s.call.name for s in gen.tool_steps}
    if "convert_units" not in used:
        print("这次模型没用 convert_units，而是用计算器 ÷1000（或者心算）。选哪个工具由模型决定。")
    elif any(
        {"calculate", "convert_units"} <= {c.name for c in r.tool_calls} for r in gen.responses
    ):
        print(
            "注意：calculate 和 convert_units 出现在同一轮。也就是说，模型没等计算器返回 94，\n"
            "就自己心算出 94 填进了 convert_units。并行调用只适合互不依赖的计算；\n"
            "有依赖的步骤，本该等上一个结果回来再发下一个调用。这次心算碰巧对了，\n"
            "但这正是 L04 讲义说的：给了工具，模型也不一定按你设想的方式用。"
        )
    else:
        print("模型先等计算器返回，再用结果做单位换算：有依赖的步骤分了两轮，这是理想的用法。")
    print("专用工具值不值得留下，要在评测集上比较准确率和工具使用率才知道（补充讲义第 2.8 节）。")


def main() -> None:
    step1_dissect()
    step2_run()
    step3_registry()
    if "--offline" not in sys.argv:
        step4_live()


if __name__ == "__main__":
    main()
