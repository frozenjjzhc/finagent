"""L01 演示：先不用 SDK，用 Python 标准库直接发 HTTP 请求，看清 API 调用的本来面目。

运行（先 conda activate finagent，在项目根目录）：
    python walkthroughs/L01_raw_http.py
花费：4 次很小的调用（关闭 thinking，每次最多 60 个输出 token），合计不到 $0.001。

你会看到：
    第 1 步  请求长什么样：一个 HTTPS POST，header 里带 key，body 是一段 JSON
    第 2 步  非流式响应：一次性返回一整段 JSON，里面有 choices 和 usage
    第 3 步  流式响应：服务器一行一行地推送 "data: {...}"，最后是 "data: [DONE]"（这叫 SSE）
    第 4 步  用 SDK 做同样的事，顺便验证"API 没有记忆"
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
API_KEY = os.environ["DEEPSEEK_API_KEY"]
BASE_URL = "https://api.deepseek.com"
URL = f"{BASE_URL}/chat/completions"


def banner(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def post(body: dict):
    """发一个 POST 请求，返回响应对象（可以一次读完，也可以逐行读）。"""
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    return urllib.request.urlopen(request, timeout=60)


def step1_and_2() -> None:
    body = {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "用一句话解释什么是 10-K。"}],
        # 直接发 HTTP 时，thinking 就写在 JSON 顶层。
        # 用 SDK 时要写 extra_body={"thinking": ...}，那只是因为 SDK 不认识这个参数，
        # SDK 最终发出去的 JSON 和这里一模一样。
        "thinking": {"type": "disabled"},
        "max_tokens": 60,
    }
    banner("第 1 步 · 请求：一个 HTTPS POST")
    print(f"POST {URL}")
    print("Content-Type: application/json")
    print("Authorization: Bearer sk-****（真实 key 从 .env 读取，绝不打印）")
    print(json.dumps(body, ensure_ascii=False, indent=2))

    banner("第 2 步 · 非流式响应：一次性返回完整 JSON")
    t0 = time.perf_counter()
    with post(body) as resp:
        data = json.loads(resp.read())
    print(f"(耗时 {time.perf_counter() - t0:.2f}s)")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    answer = data["choices"][0]["message"]["content"]
    usage = data["usage"]
    print("\n要点：")
    print(f"  - 回答在 choices[0].message.content：{answer!r}")
    print(
        f"  - finish_reason = {data['choices'][0]['finish_reason']!r}（stop 是自然结束，length 是被截断）"
    )
    print(
        f"  - 输入 {usage['prompt_tokens']} token，输出 {usage['completion_tokens']} token；按这两个数计费"
    )
    print(
        f"  - 回答有 {len(answer)} 个字符，约合 {usage['completion_tokens'] / max(len(answer), 1):.2f} token/字符"
        "（官方经验值：中文约 0.6，英文约 0.3）"
    )


def step3_stream() -> None:
    body = {
        "model": "deepseek-flash",
        "messages": [
            {"role": "user", "content": "列出 10-K 年报里最重要的三个部分，每个不超过十个字。"}
        ],
        "thinking": {"type": "disabled"},
        "max_tokens": 60,
        "stream": True,
        "stream_options": {"include_usage": True},  # 让最后一个 chunk 带上 usage
    }
    banner("第 3 步 · 流式响应：服务器逐行推送（SSE）")
    t0 = time.perf_counter()
    ttft = None
    lines: list[str] = []
    with post(body) as resp:
        for raw in resp:  # HTTP 连接不断开，服务器每生成一点就推送一行
            line = raw.decode("utf-8").rstrip("\n")
            if not line:
                continue  # SSE 用空行分隔事件
            if ttft is None and line.startswith("data:"):
                ttft = time.perf_counter() - t0
            lines.append(line)

    print(
        f"一共收到 {len(lines)} 行。首个数据行在 {ttft:.2f}s 到达（TTFT），全部收完用了 {time.perf_counter() - t0:.2f}s。"
    )
    print("\n前 3 行（原样）：")
    for line in lines[:3]:
        print("  " + line[:200])
    print("  ...")
    print("最后 3 行（原样）：")
    for line in lines[-3:]:
        print("  " + line[:300])

    text = ""
    for line in lines:
        if line.startswith("data:") and line != "data: [DONE]":
            chunk = json.loads(line[len("data:") :])
            if chunk["choices"]:
                text += chunk["choices"][0]["delta"].get("content") or ""
    print(f"\n把所有 delta.content 拼起来，就是完整回答：\n{text}")


def step4_sdk_and_statelessness() -> None:
    from openai import OpenAI

    banner("第 4 步 · 用 SDK 做同样的事，并验证 API 没有记忆")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    no_thinking = {"thinking": {"type": "disabled"}}

    first = [{"role": "user", "content": '记住这个数字：42。只回复"好的"。'}]
    r1 = client.chat.completions.create(
        model="deepseek-flash", messages=first, max_tokens=10, extra_body=no_thinking
    )
    reply = r1.choices[0].message.content
    print(f"第 1 轮  用户：记住这个数字：42  →  模型：{reply}")

    # 新请求里只带这一个问题，不带历史
    question = {
        "role": "user",
        "content": '我刚才让你记住的数字是多少？只回复数字，不知道就回复"不知道"。',
    }
    r2 = client.chat.completions.create(
        model="deepseek-flash", messages=[question], max_tokens=10, extra_body=no_thinking
    )
    print(f"第 2 轮（不带历史）  →  模型：{r2.choices[0].message.content}")

    # 把历史（包括模型自己上一轮的回答）一起发回去
    history = [*first, {"role": "assistant", "content": reply}, question]
    r3 = client.chat.completions.create(
        model="deepseek-flash", messages=history, max_tokens=10, extra_body=no_thinking
    )
    print(f"第 2 轮（带上历史）  →  模型：{r3.choices[0].message.content}")
    print(
        f"\n带历史那次的输入是 {r3.usage.prompt_tokens} token，不带历史只有 {r2.usage.prompt_tokens} token："
    )
    print('  API 是无状态的，"多轮对话"就是每次把全部历史重新发一遍，所以对话越长，每轮越贵。')


if __name__ == "__main__":
    try:
        step1_and_2()
        step3_stream()
        step4_sdk_and_statelessness()
    except urllib.error.HTTPError as e:
        # 出错时服务器也返回 JSON，说明错在哪里
        print(f"\nHTTP {e.code}：{e.read().decode('utf-8', errors='replace')}")
