# L01 · LLM API 基础：让 FinAgent 第一次开口

> ⏱ 约 4 小时（原理 1.5h + 读代码 1h + 实验与作业 1.5h）
> 📍 版本：v0.1（闭卷裸调：没有 system prompt、没有检索、没有工具）
> 📦 本课产出：能用的 `finagent ask` / `finagent usage` 命令、一次 4 种设置的对比实验、你亲手查证的 10 道题答案
> 🧭 **配套阅读指南**：[L01_阅读指南.md](L01_阅读指南.md)，里面有阅读顺序、每一步的知识点和易错点，可以打勾。

---

## 0. 这节课学完你能做到

1. 讲清楚一次 LLM API 调用从发出到返回，中间都发生了什么（HTTP、token、prefill / decode、采样、计费）。
2. 看懂 Chat Completions 的请求和响应，知道每个字段是干什么的。
3. 算清楚 token 怎么计费，说出为什么输出比输入贵、为什么缓存命中便宜。
4. 正确处理 `finish_reason`、流式输出、错误与重试。
5. 理解 thinking 模式的代价，以及"参数被静默忽略"这类坑怎么防。
6. 读懂 FinAgent v0.1 的全部代码，亲手跑一次对比实验。

---

## 1. 一次 API 调用，从头到尾发生了什么

```
你的代码                           DeepSeek 服务器
---------                          ----------------
messages（list[dict]）
   | SDK 把它序列化成 JSON
   v
HTTPS POST /chat/completions  -->  1. 校验 key 和参数
  Authorization: Bearer sk-...     2. chat template：把 messages 拼成一段带特殊标记的文本
  body: {"model", "messages"...}   3. tokenizer：文本 -> token id 序列
                                   4. prefill：整段输入一次性送进模型（并行计算），得到 KV cache
                                   5. decode：一次生成一个 token，循环直到停止
                                      每一步：logits -> 采样 -> 新 token -> 追加到末尾 -> 再算下一个
                                   6. detokenize：token id -> 文本
                              <--  7. 返回 JSON（或者流式地一段一段推回来）
LLMResponse（content, usage...）   8. 按 usage 里的 token 数扣费
```

**和你的 nanolm 对照一下**：第 5 步就是你在 `gpt.py` 里写的 `generate()`：取最后一个位置的 logits，softmax，采样，把新 token 接到序列末尾，重复。区别只有两点：
- 服务器用 **KV cache** 保存已经算过的 key / value，每一步只算新 token，不用把整个序列重算一遍（L08 你会亲手实现）；
- 服务器把很多用户的请求**拼成 batch** 一起算，这样 GPU 才不空转。

### 1.1 为什么输出 token 比输入 token 贵？（面试常考）

- **prefill（处理输入）**：所有输入 token 在**一次**前向计算里并行处理。GPU 的算力被充分利用，属于"计算受限"（compute-bound）。
- **decode（生成输出）**：每生成**一个** token 都要做一次完整的前向计算，而且只能一个接一个地来。每一步都要把全部模型权重和 KV cache 从显存里读一遍，却只算出一个 token，属于"显存带宽受限"（memory-bound）。

所以平摊到每个 token 上，输出的成本高得多。DeepSeek flash 的价格：输出 $1.20 / 百万 token，输入 $0.30，差 4 倍。

### 1.2 为什么缓存命中的输入便宜 50 倍？

如果这次请求的**开头部分**（比如同一段很长的 system prompt，或者同一段对话历史）和之前某次请求完全一样，服务器可以直接复用之前存下来的 KV cache，跳过这部分的 prefill。

DeepSeek 的缓存是**自动**的，不用改代码；但必须是"前缀完全匹配"才能命中。这直接决定了 prompt 该怎么组织：

```
[固定不变的内容：规则、示例、文档]  +  [每次都变的内容：用户的问题]
 ^ 放前面，能命中缓存                     ^ 放最后
```

L02 写 prompt、L10 生成训练数据时都会用到这个原则。

---

## 2. Chat Completions 格式

这是 OpenAI 定义、现在几乎所有厂商和推理框架都兼容的格式（ADR-1 选它做内部格式的原因）。

### 2.1 请求

这是 `walkthroughs/L01_raw_http.py` 第 1 步真实发出去的 JSON：

```json
{
  "model": "deepseek-flash",
  "messages": [{"role": "user", "content": "用一句话解释什么是 10-K。"}],
  "thinking": {"type": "disabled"},
  "max_tokens": 60
}
```

`messages` 里的四种角色：

| role | 是什么 | 在哪一课用到 |
|---|---|---|
| `system` | 给模型定规则：身份、要求、输出格式 | L02 |
| `user` | 用户说的话 | L01 起 |
| `assistant` | 模型**之前**说过的话。多轮对话时要由你放回去 | L01（第 2.2 节） |
| `tool` | 工具执行的结果 | L04 |

注意 `thinking` 这个字段：直接发 HTTP 时，它就写在 JSON 顶层；用 openai SDK 时，要写成 `extra_body={"thinking": ...}`。这只是因为 SDK 的参数列表里没有这一项，SDK 最终发出去的 JSON 和上面一模一样。

### 2.2 API 没有记忆

演示脚本第 4 步的真实结果：

```
第 1 轮  用户：记住这个数字：42  →  模型：好的
第 2 轮（不带历史）  →  模型：不知道
第 2 轮（带上历史）  →  模型：42
带历史那次的输入是 36 token，不带历史只有 20 token
```

服务器不会记住你上一次说了什么。你在聊天产品里看到的"多轮对话"，其实是**每一轮都把全部历史重新发一遍**。所以：
- 第 n 轮的输入 ≈ 前 n-1 轮所有内容的总和，**对话越长，每一轮越贵**；
- 这也是缓存重要的原因：历史部分是前缀，第二次发就能命中缓存。

### 2.3 响应

演示脚本第 2 步收到的真实 JSON（节选）：

```json
{
  "model": "deepseek-flash",
  "choices": [{
    "message": {"role": "assistant", "content": "10-K 是美国上市公司每年向 SEC 提交的详细年度报告，……"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 30,
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 12
  },
  "system_fingerprint": "aeb56401ca74e127821c4f9126dcb669"
}
```

| 字段 | 含义 |
|---|---|
| `choices[0].message.content` | 回答正文 |
| `choices[0].message.reasoning_content` | thinking 模式下的思考过程（DeepSeek 扩展字段，上面这次关了 thinking，所以没有） |
| `choices[0].finish_reason` | 为什么停下（第 4 节） |
| `usage.prompt_tokens` / `completion_tokens` | 输入 / 输出 token 数，**计费就按它** |
| `usage.prompt_cache_hit_tokens` / `miss_tokens` | 输入里命中 / 未命中缓存的部分（DeepSeek 扩展字段） |
| `usage.completion_tokens_details.reasoning_tokens` | 输出里属于思考过程的部分 |
| `system_fingerprint` | 服务端配置的指纹。它变了，说明后端换过版本，同样的输入可能得到不同的输出 |

---

## 3. Token 与计费

### 3.1 token 是什么

你的 nanolm 是**字符级**的：词表 65 个字符，一个字符一个 token。真实的大模型用 **BPE**（子词）：常见的词或词的片段合并成一个 token，词表有十几万个。

DeepSeek 官方给的经验值：**英文约 0.3 token / 字符，中文约 0.6 token / 字符**。演示脚本里那句中英混合的回答测出来是 0.54。

一个后面会用到的现象：数字会被切得很不规则，比如 "130,497" 可能被切成好几段。这是 LLM 算数不可靠的原因之一，L04 会讲到。

### 3.2 计费公式

```
花费（美元）= 折扣 × ( 缓存命中的输入 × 缓存价
                     + 未命中的输入   × 输入价
                     + 输出（含思考） × 输出价 ) ÷ 1,000,000

折扣：高峰时段 1.0，非高峰时段 0.5
高峰：UTC 周一到周五 01:00–04:00、06:00–10:00
```

价格（美元 / 百万 token，高峰价，2026-09-25 核对，写在 `llm/pricing.py`）：

| 模型 | 缓存命中的输入 | 未命中的输入 | 输出 |
|---|---|---|---|
| deepseek-flash | 0.006 | 0.30 | 1.20 |
| deepseek-v4-pro | 0.044 | 1.32 | 3.96 |

**手算验证一次**。`finagent ask` 的真实输出：

```
[deepseek-flash | thinking=off | 输入 22（缓存命中 0） · 输出 18（思考 0） | $0.000014 | ...]
```

当时是非高峰时段：

```
0.5 × (22 × 0.30 + 18 × 1.20) ÷ 1,000,000
= 0.5 × (6.6 + 21.6) ÷ 1,000,000
= 0.0000141 美元   ✓ 和程序算的一致
```

### 3.3 放大到真实场景

以后的 RAG 系统，每道题大约 3000 个输入 token（检索到的财报段落）加 300 个输出 token，跑一遍 100 道题的评测（按高峰价）：

```
flash：100 × (3000 × 0.30 + 300 × 1.20) ÷ 1,000,000 ≈ $0.13
pro  ：100 × (3000 × 1.32 + 300 × 3.96) ÷ 1,000,000 ≈ $0.51
```

都不贵，但差了约 4 倍。日常开发用 flash 反复跑，关键节点再用 pro，就是这么来的。

---

## 4. finish_reason：每次都要检查

| 取值 | 含义 | 该怎么办 |
|---|---|---|
| `stop` | 自然结束，或碰到了停止词 | 正常 |
| `length` | 达到 `max_tokens` 上限，**被截断了** | 回答不完整！调大 max_tokens，或者让回答更短 |
| `tool_calls` | 模型要调用工具 | 执行工具再继续（L04） |
| `content_filter` | 内容被安全过滤 | 记录下来，换个问法 |
| `insufficient_system_resource` | 服务器资源不足，生成被中断（DeepSeek 特有） | 稍后重试 |
| `aborted` | 生成被中断（DeepSeek 特有） | 稍后重试 |

**`length` 最危险**：被截断的回答看上去就像一个完整的回答，特别是数字题，"营收是 $130." 这种半截内容很容易被当真。所以 `finagent ask` 碰到 `length` 会打印黄色警告，对比实验的报告里也专门有"被截断"一列。

---

## 5. thinking 模式

**是什么**：模型先生成一段思考过程（`reasoning_content`），再给出正式回答。还是同一个模型，只是多做了很多步 decode。

**代价**：思考过程的 token 按**输出价**计费，延迟也跟着变长。我们测的两道判断素数的题：

```
Is 17 a prime number?  →  思考 16 token，回答 "Yes"
Is 91 a prime number?  →  思考 28 token，回答 "No"
```

题目简单，思考就很短；遇到难题，思考可以有几千 token。

**三个要注意的地方**：

1. **DeepSeek 默认开启 thinking**。所以 v0.1 的配置里显式写了 `thinking: false`。凡是影响成本和行为的参数，不要依赖厂商的默认值：默认值会变，而且你不写出来，读代码的人也不知道当时是什么设置。
2. **参数被静默忽略**：thinking 模式下 DeepSeek 不支持 `temperature`，但你传了也**不报错，只是不生效**。"API 接受了一个参数"不等于"这个参数起作用了"。我们的做法是在配置层把它变成显式错误：
   ```
   finagent ask "hi" --thinking --temperature 0.5
   → 参数组合不合法：thinking 模式下 DeepSeek 会静默忽略 temperature。要调温度，请设 thinking: false
   ```
3. **多轮对话时的思考内容**：不带工具时，把上一轮的 `reasoning_content` 发回去也会被忽略；**带工具时必须原样发回**。这一点 L04 会遇到。

**什么时候值得开**：需要多步推理或计算的题目。简单的事实查找，开了只是多花钱。值不值，由第 11 节的实验回答。

---

## 6. 流式输出

**格式**：服务器不等全部生成完，而是生成一点推一点。协议叫 SSE（Server-Sent Events），就是一行行的文本：

```
data: {"choices":[{"delta":{"content":"1."}, ...}], "usage":null}
data: {"choices":[{"delta":{"content":" 业务"}, ...}], "usage":null}
...
data: {"choices":[{"delta":{"content":""},"finish_reason":"stop"}], "usage":{"prompt_tokens":20, ...}}
data: [DONE]
```

每行的 `delta.content` 是一小段新文本，拼起来就是完整回答。最后一行 `[DONE]` 表示结束。请求时要带上 `stream_options: {"include_usage": true}`，否则流式调用拿不到 token 数；DeepSeek 把 usage 放在带 `finish_reason` 的那一行里。

**两个延迟指标**：
- **TTFT**（time to first token，首 token 延迟）：从发出请求到第一个字出现，决定了"感觉快不快"。
- **总延迟**：从发出请求到最后一个字出现。

演示脚本测到：TTFT 0.32s，总共 1.07s。**流式不会缩短总延迟**（甚至略增），它缩短的是用户**感知到的**等待时间。回答越长，流式的体验优势越明显。

**我们的接口设计**：`chat(messages, on_delta=None)`。不传 `on_delta` 就一次性返回；传了就走流式，每来一段调用一次回调，**最后仍然返回完整的 `LLMResponse`**。两种模式共用同一个接口、同一种返回值，上层代码不用写两套。

---

## 7. 错误与重试

DeepSeek 的错误码（来自官方文档）：

| 状态码 | 原因 | 该重试吗 |
|---|---|---|
| 400 | 请求体格式错误 | ❌ 改代码 |
| 401 | API key 错误 | ❌ 检查 key |
| 402 | 余额不足 | ❌ 充值 |
| 422 | 参数不合法 | ❌ 改参数 |
| 429 | 请求太快 | ✅ 等一会儿再试 |
| 500 | 服务器故障 | ✅ 稍后重试 |
| 503 | 服务器过载 | ✅ 稍后重试 |
| 连接错误 / 超时 | 网络问题 | ✅ |

**原则**：只重试**暂时性**的错误（429、5xx、网络问题）。4xx 说明请求本身有问题，重试一万次结果都一样。

**指数退避 + 随机抖动（jitter）**：

```
第 k 次重试前等待 ≈ 基础时间 × 2^k + 随机的一小段
```

- **指数**：服务器过载时，大家都立刻重试只会让它更堵，所以越来越慢地重试。
- **随机抖动**：如果 1000 个客户端同一秒失败、又在同一秒重试，会形成新一轮洪峰（thundering herd），加点随机能把它们错开。

openai SDK 已经内置了这套逻辑，我们只需要在配置里设 `max_retries: 2`，不必自己重写（"不要重新实现 SDK 已有的功能"）。

**一个容易忽略的点**：LLM 调用的重试**是要付钱的**。如果服务器其实已经生成完了、只是回来的路上断了网，重试就会再生成一遍、再扣一次费。所以重试次数要有上限。

我们代码里还做了两件事：
- **失败的调用也记录下来**（错误类型和状态码），排查问题时知道失败过几次、失败在哪；
- **401 的错误信息不记录原文**，因为里面可能带有部分 key。

---

## 8. 模型选型：能力、价格、延迟三者的权衡

没有哪个模型三样都占优，只能根据**你自己的任务**实测再选。我们的候选：

| | deepseek-flash | deepseek-v4-pro |
|---|---|---|
| 能力 | 较好 | 更强 |
| 价格 | 基准 | 贵 3~4 倍 |
| 延迟 | 快 | 较慢 |

我们的策略：**日常开发和反复跑评测用 flash；老师模型用哪个，由实验和后面的 eval 决定**。第 11 节的实验就是第一次测量。

---

## 9. 代码导读（按这个顺序读）

| 顺序 | 文件 | 重点看什么 |
|---|---|---|
| 1 | `configs/systems/v0.1.yaml` | 一个"系统版本" = 一个配置文件 |
| 2 | `src/finagent/config.py` | pydantic 校验；`extra="forbid"`；`_reject_silently_ignored`；`with_overrides` 为什么要重新校验 |
| 3 | `src/finagent/llm/base.py` | 统一接口：`Usage`、`LLMResponse`、`LLMClient`、`DeltaCallback` |
| 4 | `src/finagent/llm/pricing.py` | 计费公式；为什么拒绝不带时区的时间 |
| 5 | `src/finagent/llm/openai_compat.py` ★ | `_build_request`（extra_body）、`_call_once`、`_call_stream`、`parse_usage`、`_log` |
| 6 | `src/finagent/tracing.py` | JSONL 追加写入；为什么加锁 |
| 7 | `src/finagent/cli.py` | typer 怎么把函数参数变成命令行参数；流式打印；异常翻译的顺序 |
| 8 | `tests/` | 不联网怎么测：用 SDK 自己的类型构造假的响应 |
| 9 | `scripts/l01_compare.py` | 花钱前先估算；线程池并发；`partial` 和 `lambda` 的区别 |

几个值得专门讲的设计：

**① 用 `Protocol` 定义接口，而不是继承基类。**
`LLMClient` 是个 `Protocol`：任何类只要有签名对得上的 `chat()` 方法，就算实现了这个接口，不需要 `class X(LLMClient)`。这叫"结构化类型"（鸭子类型的静态版本）。L08 写本地模型后端时，两个类之间没有任何继承关系，照样可以互换。

**② DeepSeek 的扩展字段用 `getattr` 取。**
`reasoning_content`、`prompt_cache_hit_tokens` 不在 openai SDK 的类型定义里。SDK 的模型允许额外字段，所以数据其实拿得到，只是类型提示里没有，于是用 `getattr(obj, "字段", None)`：取到就用，取不到（比如以后换成 vLLM）就是 None。所有这类"漏"都集中在 `openai_compat.py` 这一个文件里。

**③ 计价用配置里的模型名，不用响应里的。**
服务端返回的 `model` 字段不一定和你请求时写的一样，而价格表是按你请求的名字建的。两个都记进日志：`model` 是请求的，`response_model` 是服务端报告的。

**④ `with_overrides` 为什么不用 `model_copy(update=...)`？**
pydantic 的 `model_copy` **不会重新运行校验器**。配置文件里合法（`temperature: 0.7`），命令行再加个 `--thinking`，组合起来就非法了；用 `model_copy` 的话，这个非法组合会被悄悄放过去。`tests/test_config.py` 里的 `test_overrides_are_revalidated` 专门测了这一点。

**⑤ 计时用 `time.perf_counter()`，不用 `time.time()`。**
`time.time()` 是墙上时钟，系统对时的时候可能往回跳；`perf_counter()` 是单调时钟，只往前走，专门用来测耗时。

---

## 10. 动手

在你自己的 PowerShell 终端里，按顺序做：

```powershell
conda activate finagent
cd D:\claude\best\finagent
python -m pytest -q                                   # 19 个测试应全部通过
python walkthroughs/L01_raw_http.py                   # 看清 HTTP 请求、JSON、SSE、无状态
finagent ask "What was NVIDIA's total revenue for fiscal year 2025?"
finagent ask "Is 91 a prime number?" --thinking --show-reasoning
finagent ask "Explain what a 10-K is." --max-tokens 5    # 故意触发 finish_reason=length
finagent ask "hi" --thinking --temperature 0.5          # 看配置层怎么拦截非法组合
finagent usage                                          # 今天的调用汇总
```

（`pip install -e .` 我已经替你跑过了，`finagent` 命令可以直接用。）

---

## 11. 实验：闭卷问答能做到什么程度？

```powershell
python scripts/l01_compare.py
```

10 道题 × 4 种设置（flash / flash+think / pro / pro+think）= 40 次调用。脚本会先显示估算费用（往高里估约 $0.1–0.2，实际一般更低），你确认后才开始。跑完生成 `runs/l01_compare/<时间>/report.md`。

**打开 report.md，带着这 6 个问题看：**

1. **编数字**：和你从 10-K 查到的数字比，哪些答错了，语气却很肯定？
2. **知识截止**：q02、q05 问的是 2026 年初才发布的财报，模型是承认不知道，还是编了一个数？
3. **陷阱题 q10**：有没有编出一个"电动车业务收入"？
4. **计算题 q06、q07**：算对了吗？用的原始数字对吗？
5. **thinking 值不值**：开了之后准确率变了多少？成本和延迟翻了几倍？
6. **flash vs pro**：差距主要出现在哪一类题上？

这个实验的预期结论，正好对应后面几课要解决的问题：

| 你会看到的问题 | 由哪一课解决 |
|---|---|
| 回答格式乱，没法自动判分；人工标 40 个回答很累 | L02 结构化输出 + L03 自动评测 |
| 算错 | L04 工具（计算器） |
| 不知道新财报、编数字 | L05 RAG（先检索原文，再回答） |

---

## 12. 作业

1. **查 10-K，填答案**（最重要）：对 q01–q09，在 SEC EDGAR（https://www.sec.gov/edgar/search/ ）按 `source_hint` 的提示找到原始数字，填进 `evalsets/seed_closed_book.jsonl` 的 `answer`、`unit`，并把 `verified` 改成 `true`，在 `notes` 里写上页码或章节。
   你会亲身体会到在 100 多页的 10-K 里找一个数字有多费劲，这正是 L05 的 RAG 要自动化的事。这 10 道题也会成为 L03 评测集的最初几条。
2. **跑实验并写结论**：在 report.md 里标注每个回答的对错，然后写 3–5 句结论，填进 `SCOREBOARD.md` 的 Dev 表。
3. **选定老师模型**：flash 还是 v4-pro？thinking 开还是关？从准确率、成本、延迟三个角度写出理由。
4. **代码练习（自己写）**：给 `finagent ask` 加一个 `--system` 选项，把一段文字作为 system 消息放在 messages 的最前面。
   提示：只改 `cli.py` 里的 `ask` 函数，大约 5 行。写完试试 `--system "只回答数字，不要解释"`，看输出怎么变。这是 L02 的预习。
5. **git**：如果还没做 L00 的 git 步骤，先做。然后：
   ```powershell
   git add .
   git commit -m "L01: LLM API basics (v0.1)"
   git tag v0.1
   ```

做完把第 2、3 题的结论和第 4 题的代码发给我。

---

## 13. 自测题（先自己答，再看第 14 节）

1. 一次 chat completion 请求，服务器端大致经过哪几步？
2. 为什么输出 token 比输入 token 贵？
3. 为什么缓存命中的输入便宜很多？怎样组织 prompt 才能提高命中率？
4. "API 没有记忆"是什么意思？多轮对话的总成本怎么随轮数增长？
5. `finish_reason=length` 意味着什么？为什么危险？
6. 哪些错误该重试，哪些不该？为什么重试要加随机抖动？
7. DeepSeek 在 thinking 模式下会静默忽略 temperature。工程上怎么防范这类问题？
8. TTFT 和总延迟分别影响什么？流式输出能降低总延迟吗？
9. （代码）`with_overrides` 为什么不用 `model_copy(update=...)`？
10. （代码）`tracing.log_call` 为什么要加锁？

---

## 14. 参考答案

**1.** 校验 → chat template 把 messages 拼成一段文本 → tokenize → prefill（并行处理整个输入，建立 KV cache）→ decode（逐个生成 token：logits、采样、追加，直到停止）→ detokenize → 返回，并按 usage 计费。

**2.** prefill 一次前向就能并行处理所有输入 token，计算受限、效率高；decode 每生成一个 token 就要做一次完整的前向，还只能串行，每一步都要把权重和 KV cache 从显存读一遍，属于显存带宽受限。平摊到每个 token，输出的成本高得多。

**3.** 前缀和之前的请求完全一样时，服务器可以直接复用这段前缀的 KV cache，省掉它的 prefill 计算。组织原则：固定不变的内容（规则、示例、文档）放前面，每次都变的内容（用户问题）放最后；不要在开头放时间戳这类每次都变的东西。

**4.** 服务器不保存对话状态，每次请求都要带上全部历史。第 n 轮的输入约等于前面所有轮的内容之和，所以单轮成本随轮数线性增长，总成本大约随轮数的平方增长。缓存能把历史部分的单价压低很多。

**5.** 输出达到了 `max_tokens` 上限，被截断了。危险在于截断的回答看起来像完整的回答，尤其是数字、JSON 这类内容，只剩半截也可能被当真。必须检查并处理。

**6.** 重试暂时性错误：429、500、503、连接错误、超时。不重试 400、401、402、422：请求本身有问题，重试多少次结果都一样。加随机抖动是为了把大量客户端的重试时间错开，避免它们同时重试、再次把服务器打垮。另外要记住，LLM 调用的重试是会再扣费的。

**7.** 在自己的配置层把非法组合变成显式错误（我们的 `_reject_silently_ignored`）；给这条规则写测试；读文档时专门留意"不支持但不报错"的说明；用实验验证参数确实生效，比如 temperature 设 0 和 2 各跑几次，看输出的多样性有没有变化。

**8.** TTFT 决定用户"感觉快不快"，总延迟决定整个任务多久完成（比如跑评测的总时间）。流式不会降低总延迟，它只把等待拆散了，降低的是用户感知到的等待。

**9.** `model_copy` 不会重新运行校验器。配置里合法的值，加上命令行的覆盖，可能组合成非法的配置；只有重新走一遍 `model_validate` 才能发现。

**10.** 对比实验用线程池并发调用，多个线程会同时往同一个文件追加内容。不加锁的话，两行内容可能交错在一起，写出一行坏掉的 JSON。

---

## 15. 下一课预告 · L02 Prompt Engineering + 结构化输出

v0.1 的回答是自由文本：有时带单位，有时不带；有时 million，有时 billion；有时一句话，有时一段话。程序没法自动判断对错，你只能一个个人工看。L02 会做：
- 写 system prompt，定规则和边界（比如"不知道就说不知道"）；
- zero-shot、few-shot、先推理后回答，这几种写法的对比；
- 用 JSON mode 加 pydantic 校验，得到**结构化**的回答：`{value, unit, reasoning, sources, confidence}`；
- prompt 版本化：prompt 放在文件里，不写死在代码里。

作业第 4 题的 `--system` 就是 L02 的起点。
