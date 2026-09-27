"""L04 演示：function calling 的完整机制，一轮一轮打印出来。

运行（先 conda activate finagent，在项目根目录）：
    python walkthroughs/L04_tool_loop.py            # 全部 5 步
    python walkthroughs/L04_tool_loop.py --offline  # 只跑不花钱的第 1、2 步
花费：第 3~5 步共 6~8 次很小的调用（flash），合计约 $0.002。

    第 1 步  函数 → 工具说明书（schema）：模型看到的到底是什么（离线）
    第 2 步  为什么不能用 eval()，AST 白名单怎么挡住攻击（离线）
    第 3 步  手写工具循环：不用任何封装，直接用 openai SDK 跑两轮
    第 4 步  错误回传：工具报错之后，模型能不能自己改正
    第 5 步  thinking 模式 + 工具：reasoning_content 要不要传回去

第 3~5 步故意**不用**项目里的 run_tool_loop，而是把每一步摊开写：
看懂了这里，agent/loop.py 的 30 行就没有秘密了。
"""

from __future__ import annotations

import ast
import json
import os
import sys

import openai
from dotenv import load_dotenv
from openai import OpenAI

from finagent.config import PROJECT_ROOT
from finagent.tools import AVAILABLE_TOOLS
from finagent.tools.calculator import evaluate
from finagent.tools.registry import ToolError

CALCULATE = AVAILABLE_TOOLS["calculate"]
TOOLS = [CALCULATE.schema()]
NO_THINKING = {"thinking": {"type": "disabled"}}

# 一段编造的"财报片段"（数字是虚构的，不来自任何评测集，避免泄漏）
CONTEXT = """Table (in millions):
year | 2019 | 2018 | 2017
net revenue | 5829 | 5735 | 5102"""
QUESTION = "What was the percentage growth of net revenue in 2019 and in 2018?"
SYSTEM = (
    "Answer using only the table given by the user. Use the calculate tool for every "
    "arithmetic step. Reply in one short sentence."
)


def banner(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def show_messages(messages: list[dict]) -> None:
    """每条消息打印一行：角色 + 关键内容。看清楚每一轮到底发了什么。"""
    for m in messages:
        role = m["role"]
        if m.get("tool_calls"):
            calls = "; ".join(
                f"{c['id'][:12]}… {c['function']['name']}({c['function']['arguments']})"
                for c in m["tool_calls"]
            )
            print(f"  {role:<9} [请求调用] {calls}")
        elif role == "tool":
            print(f"  {role:<9} [{m['tool_call_id'][:12]}… 的结果] {m['content']}")
        else:
            text = " ".join(str(m["content"]).split())
            print(f"  {role:<9} {text[:90]}{'…' if len(text) > 90 else ''}")


def execute(tool_call) -> str:
    """执行模型请求的一次调用，返回要回传给它的文字。出错时回传错误信息，而不是抛异常。"""
    try:
        args = json.loads(tool_call.function.arguments)  # arguments 是 JSON **字符串**
        return CALCULATE.fn(**args)
    except (json.JSONDecodeError, TypeError, ToolError) as e:
        return f"Error: {e}"


# ---------------------------------------------------------------- 离线


def step1_schema() -> None:
    banner("第 1 步 · 函数 → 工具说明书（离线）")
    print("tools/calculator.py 里的函数：\n")
    print(
        '    @tool\n    def calculate(expression: str) -> str:\n        """Evaluate an ... (docstring)"""'
    )
    schema_text = json.dumps(CALCULATE.schema(), indent=2, ensure_ascii=False)
    print("\n@tool 自动生成的 schema（每次请求都放在 tools 参数里发给模型）：\n")
    print(schema_text)
    print(
        f"\n要点：\n"
        f"  - 模型看不到函数的代码，只看到这段说明书：名字、描述、参数。描述写得好不好，直接决定模型\n"
        f"    会不会用、会不会用对。它本质上也是 prompt 的一部分\n"
        f"  - 这段 JSON 约 {len(json.dumps(TOOLS)) // 4} 个 token，每次请求都要付费（好在能命中缓存）\n"
        f"  - additionalProperties: false + required：多传、漏传参数都会被 pydantic 拦下"
    )


def step2_ast() -> None:
    banner("第 2 步 · 为什么不能用 eval()（离线）")
    attack = "__import__('os').getcwd()"
    print(f"模型如果写出 {attack!r}，eval 会照做：")
    # 演示用：一个无害的命令，只读出当前目录
    print(f"  eval → {eval(attack)}")
    print("  这里只是读出当前目录；换成删除文件、读取 .env，eval 也一样照做。\n")

    expr = "(5829 - 5735) / 5735 * 100"
    print(f"AST 白名单的第一步：ast.parse 只做语法分析，不执行任何东西。{expr!r} 的语法树：\n")
    print(ast.dump(ast.parse(expr, mode="eval").body, indent=2))
    print("\n逐个节点检查：只认识数字、+ - * / **、正负号、abs/min/max，其余一律拒绝：\n")
    for e in [expr, attack, "5,829 - 5,735", "15% * 3", "10 ** 10 ** 10", "1 / 0"]:
        try:
            print(f"  {e!r:<32} → {evaluate(e):.12g}")
        except ToolError as err:
            print(f"  {e!r:<32} → 拒绝：{err}")


# ---------------------------------------------------------------- 联网


def step3_manual_loop(client: OpenAI) -> None:
    banner("第 3 步 · 手写工具循环")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{CONTEXT}\n\nQuestion: {QUESTION}"},
    ]
    for round_no in range(1, 5):
        print(f"\n--- 第 {round_no} 轮：发出 {len(messages)} 条消息 ---")
        show_messages(messages)
        r = client.chat.completions.create(
            model="deepseek-flash", messages=messages, tools=TOOLS, extra_body=NO_THINKING
        )
        msg, finish = r.choices[0].message, r.choices[0].finish_reason
        print(f"模型回复：finish_reason={finish!r}，tool_calls {len(msg.tool_calls or [])} 个")
        if not msg.tool_calls:
            print(f"最终回答：{msg.content}")
            break
        for tc in msg.tool_calls:
            print(
                f"  请求调用 {tc.function.name}，arguments 的类型是 "
                f"{type(tc.function.arguments).__name__}：{tc.function.arguments}"
            )
        # 1) 把模型的回复（含调用请求）原样放回历史
        messages.append(msg.model_dump(exclude_none=True))
        # 2) 逐个执行，每个结果一条 role=tool 消息，用 tool_call_id 对上号
        for tc in msg.tool_calls:
            result = execute(tc)
            print(f"  我们执行 → {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    print(
        '\n要点：模型自己不执行任何东西，它只输出"想调用什么"；执行、回传都是我们的代码。\n'
        "两个互不依赖的计算，模型通常在同一轮里一起请求（并行调用），省掉一轮来回。"
    )


def step4_error_feedback(client: OpenAI) -> None:
    banner("第 4 步 · 错误回传")
    bad = "(5,829 - 5,735) / 5,735 * 100"
    try:
        evaluate(bad)
    except ToolError as e:
        error = f"Error: {e}"
    # 下面这段历史是**我们手工构造的**：假装模型上一轮写出了带千分位逗号的表达式。
    # 模型真的会不会写成这样不重要，这里要看的是：拿到错误信息之后，它会怎么做
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{CONTEXT}\n\nQuestion: What was the growth in 2019?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_demo_0",
                    "type": "function",
                    "function": {"name": "calculate", "arguments": json.dumps({"expression": bad})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_demo_0", "content": error},
    ]
    print("构造的历史（最后一条是计算器真实返回的错误信息）：")
    show_messages(messages)
    r = client.chat.completions.create(
        model="deepseek-flash", messages=messages, tools=TOOLS, extra_body=NO_THINKING
    )
    msg = r.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"\n模型重新请求：{tc.function.arguments} → 执行结果 {execute(tc)}")
        print(
            "它读懂了错误信息，去掉逗号重试。这就是为什么工具出错时要回传错误，而不是让程序崩溃。"
        )
    else:
        print(f"\n模型没有重试，直接回答：{msg.content}")
        print("（这次它选择自己算。错误回传只是给了它改正的机会，不保证它一定用。）")


def step5_thinking(client: OpenAI) -> None:
    banner("第 5 步 · thinking 模式 + 工具：reasoning_content")
    thinking = {"thinking": {"type": "enabled"}}
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{CONTEXT}\n\nQuestion: {QUESTION}"},
    ]
    r = client.chat.completions.create(
        model="deepseek-flash", messages=messages, tools=TOOLS, extra_body=thinking
    )
    msg = r.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    print(
        f"第 1 轮：finish_reason={r.choices[0].finish_reason!r}，思考过程 {len(reasoning)} 个字符"
    )
    if not msg.tool_calls:
        print("这次模型没有调用工具，第 5 步到此为止。")
        return
    full = msg.model_dump(exclude_none=True)
    print(f"assistant 消息里的字段：{sorted(full)}（注意 reasoning_content）")
    results = [
        {"role": "tool", "tool_call_id": tc.id, "content": execute(tc)} for tc in msg.tool_calls
    ]

    print("\n(a) 按文档要求，把 reasoning_content 原样传回：")
    r2 = client.chat.completions.create(
        model="deepseek-flash",
        messages=[*messages, full, *results],
        tools=TOOLS,
        extra_body=thinking,
    )
    print(
        f"    成功，finish_reason={r2.choices[0].finish_reason!r}：{(r2.choices[0].message.content or '')[:100]}"
    )

    print("\n(b) 故意去掉 reasoning_content 再发：")
    stripped = {k: v for k, v in full.items() if k != "reasoning_content"}
    try:
        r3 = client.chat.completions.create(
            model="deepseek-flash",
            messages=[*messages, stripped, *results],
            tools=TOOLS,
            extra_body=thinking,
        )
        print(f"    没有报错（finish_reason={r3.choices[0].finish_reason!r}）。")
        print(
            '    文档说"不正确回传会返回 400"，但这次 flash 接受了。不要依赖这一点：\n'
            "    服务端随时可能开始严格检查，而且去掉思考过程后，模型接不上自己上一轮的思路。\n"
            "    项目里 LLMResponse.to_message() 总是把它带上。"
        )
    except openai.BadRequestError as e:
        print(f"    400 错误：{str(e)[:160]}")
        print("    和文档说的一致：带工具时，之前的 reasoning_content 必须传回。")


def main() -> None:
    step1_schema()
    step2_ast()
    if "--offline" in sys.argv:
        return
    load_dotenv(PROJECT_ROOT / ".env")
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    step3_manual_loop(client)
    step4_error_feedback(client)
    step5_thinking(client)


if __name__ == "__main__":
    main()
