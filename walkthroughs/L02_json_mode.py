"""L02 演示：结构化输出的几个关键机制。

运行（先 conda activate finagent，在项目根目录）：
    python walkthroughs/L02_json_mode.py
花费：3 次很小的调用（flash，关闭 thinking），合计不到 $0.001。

    第 1 步  模板变量为什么不能用 str.format（离线）
    第 2 步  不开 JSON mode，只在 prompt 里要求 JSON：模型会怎么输出？
    第 3 步  开了 JSON mode，但 prompt 里没有 "json" 这个词：会怎样？
    第 4 步  开 JSON mode 且 prompt 写对：合法 JSON ≠ 符合 schema
    第 5 步  few-shot 在 messages 里长什么样，缓存为什么能命中（离线）
"""

from __future__ import annotations

import json
import os
from typing import Literal

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from finagent.config import PROJECT_ROOT
from finagent.prompts import load_prompt, render

load_dotenv(PROJECT_ROOT / ".env")
client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
NO_THINKING = {"thinking": {"type": "disabled"}}


def banner(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def try_parse(text: str) -> bool:
    try:
        print(f"json.loads 成功：{json.loads(text)}")
        return True
    except json.JSONDecodeError as e:
        print(f"json.loads 失败：{e}")
        return False


def step1_format_pitfall() -> None:
    banner("第 1 步 · 模板变量为什么不能用 str.format（离线）")
    template = 'Reply like {"value": 1, "unit": "percent"}. Today: {{today}}'
    try:
        template.format(today="2026-09-26")
    except (KeyError, ValueError) as e:
        print(f"str.format 报错：{type(e).__name__}: {e}")
        print("  原因：format 把 JSON 示例里的花括号当成了变量。")
    print(f"我们的 render：{render(template, today='2026-09-26')}")


def step2_no_json_mode() -> None:
    banner("第 2 步 · 不开 JSON mode，只在 prompt 里要求 JSON")
    r = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "user",
                "content": "What is 15% of 240? Reply in JSON with keys value and unit.",
            }
        ],
        max_tokens=100,
        extra_body=NO_THINKING,
    )
    text = r.choices[0].message.content
    print(f"原始输出：{text!r}")
    if try_parse(text):
        print("  这次碰巧是纯 JSON。但没有任何机制保证：换个问题或多跑几次，")
        print("  它可能在外面包上 ```json 代码块，或者前后夹着解释文字，json.loads 就会失败。")
        print("  另外留意字段的内容（比如 unit 是不是空的）：格式合法，不代表内容有用。")
    else:
        print("  外面包了代码块或夹着解释文字，程序就没法直接解析。这就是需要 JSON mode 的原因。")


def step3_json_mode_without_the_word() -> None:
    banner('第 3 步 · 开了 JSON mode，但 prompt 里没有 "json" 这个词')
    try:
        r = client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {
                    "role": "user",
                    "content": "What is 15% of 240? Reply with an object with keys value and unit.",
                }
            ],
            max_tokens=100,
            response_format={"type": "json_object"},
            extra_body=NO_THINKING,
        )
        text = r.choices[0].message.content
        print(f"这次没有报错。原始输出：{text!r}")
        try_parse(text)
        print(
            '  文档要求 prompt 里出现 "json" 并给出示例；不照做，行为没有保证，可能哪天就失败了。'
        )
    except openai.APIStatusError as e:
        print(f"服务器拒绝了请求：HTTP {e.status_code}：{e.message}")
        print(
            '  所以我们的 prompt 文件都写了 "JSON"，测试 test_prompt_builds_valid_messages 专门检查这一点。'
        )


class TinyAnswer(BaseModel):
    value: float
    unit: Literal["USD_millions", "percent", "count", "other"]


def step4_valid_json_is_not_enough() -> None:
    banner("第 4 步 · 开 JSON mode 且 prompt 写对：合法 JSON ≠ 符合 schema")
    r = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "system",
                "content": 'Reply in JSON like {"value": 12.5, "unit": "percent"}. '
                'unit must be one of "USD_millions", "percent", "count", "other".',
            },
            {"role": "user", "content": "What is 15% of 240?"},
        ],
        max_tokens=100,
        response_format={"type": "json_object"},
        extra_body=NO_THINKING,
    )
    text = r.choices[0].message.content
    print(f"原始输出：{text!r}")
    print(f"pydantic 校验通过：{TinyAnswer.model_validate_json(text)!r}")

    bad = '{"value": "36 dollars", "unit": "dollars"}'  # 合法的 JSON，但类型和取值都不对
    print(f"\n再看一段合法但不合格的 JSON：{bad}")
    try:
        TinyAnswer.model_validate_json(bad)
    except ValidationError as e:
        for err in e.errors():
            print(f"  校验失败：{err['loc'][0]}: {err['msg']}")
    print("  JSON mode 只管语法，管不了内容；内容由 pydantic 把关，不合格就带着错误信息重试。")


def step5_messages_and_cache() -> None:
    banner("第 5 步 · few-shot 在 messages 里长什么样（离线）")
    prompt = load_prompt("answer_v3")
    a = prompt.build_messages("What was Apple's revenue?", today="2026-09-26")
    b = prompt.build_messages("What was Tesla's revenue?", today="2026-09-26")
    for i, m in enumerate(a):
        preview = " ".join(str(m["content"]).split())[:70]
        print(f"  [{i:2d}] {m['role']:<9} {len(str(m['content'])):>5} 字符 | {preview}")
    same = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    print(f"\n两个不同问题的 messages：前 {same} 条完全相同，只有最后一条不同。")
    print("  system 和示例放在前面、每次都一样，所以能命中前缀缓存；变化的日期和问题放在最后。")


if __name__ == "__main__":
    step1_format_pitfall()
    step2_no_json_mode()
    step3_json_mode_without_the_word()
    step4_valid_json_is_not_enough()
    step5_messages_and_cache()
