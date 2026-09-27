"""手写的工具循环（L04）。function calling 的全部机制都在这几十行里。

一次完整的工具调用，要和模型来回好几轮：

    第 1 轮  我们 → 模型：messages + tools（工具说明书）
             模型 → 我们：finish_reason="tool_calls"，tool_calls=[calculate("(5829-5735)/5735*100")]
                          （模型只是"说"要调用什么，它自己不会执行任何东西）
    执行     我们在本地执行 calculate，得到 "1.63905841325"
    第 2 轮  我们 → 模型：messages + [assistant 的调用请求] + [role=tool 的执行结果] + tools
             模型 → 我们：finish_reason="stop"，content = 最终答案

    消息列表的样子：
        system    你是财务分析师……
        user      Context: ... Question: 增长率是多少？
        assistant (tool_calls: call_0 → calculate("(5829-5735)/5735*100"))
        tool      (tool_call_id: call_0) 1.63905841325
        assistant {"reasoning": "...", "value": 1.64, ...}

要点：
  - API 是无状态的（L01）：每一轮都要把完整的历史（包括之前的调用和结果）重新发过去
  - 一次回复里可以有多个 tool_calls（并行调用），每个都要执行，每个都要回传一条 role=tool 消息，
    用 tool_call_id 对上号
  - 工具出错不抛异常，而是把错误信息当作结果回传，让模型自己改正（见 tools/registry.py）
  - 必须有轮数上限：模型可能一直调用下去（L07 会专门讲终止条件和成本控制）
"""

from __future__ import annotations

from finagent.llm.base import LLMClient, Message
from finagent.llm.structured import Generation
from finagent.tools.registry import ToolRegistry


def run_tool_loop(
    client: LLMClient,
    messages: list[Message],
    registry: ToolRegistry,
    *,
    json_mode: bool = False,
    max_rounds: int = 6,
) -> Generation:
    """反复调用模型并执行它请求的工具，直到它不再请求工具（给出最终回答），最多 max_rounds 轮。"""
    messages = list(messages)  # 复制一份：不修改调用方传进来的列表
    tools = registry.schemas()
    responses, steps = [], []
    for _ in range(max_rounds):
        resp = client.chat(messages, json_mode=json_mode, tools=tools)
        responses.append(resp)
        messages.append(resp.to_message())  # 模型的回复（含调用请求）要放回历史
        if not resp.tool_calls:  # 不再调用工具：这就是最终回答
            return Generation(final=resp, messages=messages, responses=responses, tool_steps=steps)
        for call in resp.tool_calls:
            step = registry.execute(call)
            steps.append(step)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": step.result})
    # 轮数用完，模型还在调用工具：不再发请求，如实返回"没有最终答案"
    return Generation(
        final=responses[-1],
        messages=messages,
        responses=responses,
        tool_steps=steps,
        hit_max_rounds=True,
    )
