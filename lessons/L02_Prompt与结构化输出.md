# L02 · Prompt Engineering 与结构化输出

> ⏱ 约 4 小时（原理 1.5h + 读代码 1h + 实验与作业 1.5h）
> 📍 版本：v0.2（仍然闭卷：prompt 模板 + JSON mode + schema 校验 + 带反馈重试）
> 📦 本课产出：三个版本的 prompt 文件、结构化答案 schema、问答流水线、一次 prompt 对比实验
> 🧭 **配套阅读指南**：[L02_阅读指南.md](L02_阅读指南.md)　·　📘 **补充**：[L02_补充_JSON详解.md](L02_补充_JSON详解.md)

---

## 0. 这节课学完你能做到

1. 写出结构清楚的 system prompt：角色、规则、格式、边界各司其职。
2. 分清什么是好规则、什么是空话，避开常见的反模式。
3. 会用 few-shot 示例，并且知道示例怎么选、选错了会怎样。
4. 理解"先推理后回答"为什么有效，以及怎么在 JSON 里实现它。
5. 用 JSON mode + pydantic + 带反馈的重试，得到**程序能直接使用**的答案。
6. 把 prompt 当作代码来管理：写成文件、带版本号、有测试。

---

## 1. 从 L01 的问题出发

| L01 看到的问题 | 本课的对策 | 在哪 |
|---|---|---|
| 回答是自由文本，程序没法自动判对错 | 固定结构的 JSON 答案（schema） | `schemas.py` |
| 单位混乱（billion / million），夹 LaTeX | 规则 3、7：统一用百万美元；一句话、不用 markdown | prompt 规则 |
| 模型不知道今天的日期，以为新财报还没发布 | 每次把日期写进问题里 | `user_template` |
| 口径搞混（Meta、Tesla） | 规则 2：只答题目问的那一行 | prompt 规则 |
| 不懂装懂、编数字 | 规则 5：不确定就答 unknown，绝不猜 | prompt 规则 + `status` 字段 |
| 陷阱题要靠模型自觉 | 把"前提不成立"做成一个明确的状态 | `status = invalid_premise` |

---

## 2. system prompt 的结构

以 `prompts/answer_v2.yaml` 为例，一份好的 system prompt 通常有四块：

```
① 角色与任务   You are a meticulous financial analyst ... answering from memory
② 规则         1. Fiscal years ... 2. Line items ... 3. Units ... （共 9 条）
③ 输出格式     Respond with a single JSON object with exactly these fields ...
④ 边界         （写在规则里）不知道怎么办、前提错了怎么办
```

**① 角色**不是为了"入戏"，而是一次性交代背景：你是做什么的、依据什么回答（"answering from memory: you cannot look anything up"，让模型知道自己是闭卷）。

**② 规则**是这一课的重点。v2 的 9 条规则，每一条都对着 L01 实验里一类具体的失误：

| 规则 | 针对 L01 的哪个失误 |
|---|---|
| 1 财年按公司自己的定义 | NVIDIA 的财年到 1 月底、微软的到 6 月底 |
| 2 只答题目问的那一行，不换成相关指标 | Meta 答成含融资租赁的合计，Tesla 答成错误的拆分 |
| 3 金额统一用百万美元、百分比用数字 | billion / million 混用 |
| 4 计算题写出算式 | 计算过程不透明，没法检查 |
| 5 不确定就答 unknown，绝不猜 | 新财报上编数字 |
| 6 前提错误就答 invalid_premise | 陷阱题 |
| 7 一句话、不用 markdown 和 LaTeX | 回答里夹着 `\[ \frac{} \]` |
| 8 写明依据的文件 | 为 L05 的引用做准备 |
| 9 只有确定时才说 high 信心 | 为后面检验"自评信心准不准"做准备 |

**③ 格式**要写到能照着做的程度：字段名、顺序、类型、可选值都写清楚。

---

## 3. 什么是好规则

**好规则是具体、可执行、可检验的。** 对比一下：

| ❌ 空话 | ✅ 可执行的规则 |
|---|---|
| Be accurate. | Answer with exactly the line item that was asked for. Do not substitute a narrower sub-line or a broader aggregate figure. |
| Don't hallucinate. | If you do not reliably know the exact figure, set "status" to "unknown" and "value" to null. Never guess a number. |
| Use proper units. | Express monetary amounts in millions of US dollars (for example, $4.2 billion -> 4200). |
| Keep it short. | "answer_text": one plain sentence. No markdown, no LaTeX. |

区别在于：左边说的是"要好"，右边说的是"具体怎么做"。模型本来就想答对，它缺的不是态度，而是操作方法。

**常见的反模式：**
- **只说不许做什么，不说该怎么做**："不要编造"不如"不知道时输出 status: unknown"。给模型一条出路，它才能照做。
- **规则互相矛盾**：比如一边要求"尽量回答"，一边要求"不确定就不答"。
- **示例和规则不一致**：示例是模型最直接模仿的对象，规则说一套、示例做另一套，模型通常跟着示例走。
- **把评测题的答案写进 prompt**：分数会变好看，但那是作弊（数据泄漏），换一批题就露馅。
- **prompt 写在代码的字符串里**：没法单独查看、比较不同版本，也没法测试。

---

## 4. zero-shot 与 few-shot

- **zero-shot**：只给指令，不给示例（v1、v2）。
- **few-shot**：给几组"这样问 → 这样答"的示例（v3）。

**示例放在哪**：作为几轮真实的对话，放在 system 之后、真正的问题之前：

```
[0] system     规则和格式
[1] user       示例问题 1
[2] assistant  示例回答 1（一段合格的 JSON）
...
[9] user       示例问题 5
[10] assistant 示例回答 5
[11] user      真正的问题
```

模型会觉得"我前面已经这样回答过 5 次了"，于是延续同样的格式和风格。这比在 system prompt 里用文字描述格式更直接。

**示例怎么选：**
1. **覆盖不同的行为**：v3 的 5 个示例分别是计算、单位换算、前提错误、未来时期、不了解的公司。
2. **格式和真实输入一致**：示例的问题也带 `Today's date: ...`，和真正的问题长得一样。
3. **不能泄漏评测题**：v3 的示例刻意不包含评测集里公司的真实数字。`tests/test_prompts.py` 专门检查示例和评测题没有重合。
4. **示例本身必须合格**：示例的回答要能通过 schema 校验，示例错了，模型会跟着错（`test_example_answers_match_the_schema`）。
5. **注意示例的分布** ⚠️：本次实验里，v3 的 5 个示例中有 3 个是"不回答"（前提错误 1 个 + 不知道 2 个），结果 v3 比 v2 多拒答了一题（微软 q04）。**示例里什么行为占多数，模型就倾向于什么行为。** 这是一个合理的假设，但 10 道题不足以证实，作业 2 会让你动手验证。

---

## 5. 先推理，后回答

模型是一个 token 接一个 token 往后生成的，**后面的 token 能看到前面已经生成的所有内容**。所以：

- 先写数值、再写推理：数值是"凭直觉"蹦出来的，后面的推理只是在给它找理由；
- 先写推理、再写数值：推理过程成了模型的"草稿纸"，算出来的数值可以参考前面的计算。

这就是 chain of thought（CoT，思维链）有效的原因：**生成的顺序就是计算的顺序。**

我们在 JSON 里用**字段顺序**来实现它：prompt 的格式说明要求按顺序输出、`reasoning` 排第一，few-shot 示例也都是推理在前，模型于是先写推理，再写 `status` 和 `value`。`schemas.py` 里也是同样的顺序，但模型看不到 Python 代码，它的作用是让代码输出的 JSON（比如 L10 的训练数据）也保持推理在前。解析器不检查顺序，所以这是一个靠 prompt 实现的"软约束"：上次实验的 30 次输出全部遵守了。

**和 thinking 模式的区别：**

| | 在输出里写 reasoning（本课） | thinking 模式（L01） |
|---|---|---|
| 推理写在哪 | 答案的一个字段里，可以保存、检查 | 单独的 `reasoning_content` |
| 长度 | 我们限制在 3 句以内 | 可能几千 token |
| 成本 | 低 | 高（L01 里 flash+think 贵 11 倍） |
| 适合 | 简单的计算和判断 | 复杂的多步推理 |

---

## 6. 结构化输出：三道防线

```
第 1 道  prompt 里写清格式（字段、顺序、类型、可选值）      → 让模型"想"按格式来
第 2 道  JSON mode（response_format = json_object）           → 保证输出是合法的 JSON（语法）
第 3 道  pydantic 校验（schemas.Answer）                       → 保证内容合格（字段、类型、取值、一致性）
兜底     校验失败 → 把错误输出和错误原因一起发回去，让模型重试
```

### 6.1 DeepSeek JSON mode 的三条规矩

1. **prompt 里必须出现 "json" 这个词**，否则直接返回 400。演示脚本第 3 步实测：
   ```
   HTTP 400：Prompt must contain the word 'json' in some form to use 'response_format' of type 'json_object'.
   ```
2. **只保证语法，不保证内容**：`{"value": "36 dollars", "unit": "dollars"}` 是合法的 JSON，但数值是字符串、单位不在允许范围内。演示脚本第 4 步展示了 pydantic 怎么抓住它。
3. **偶尔会返回空内容**（官方文档说明）。所以代码要能处理空输出。

### 6.2 schema 设计要点（`schemas.py`）

- **用 `Literal` 限定取值**：`status`、`unit`、`confidence` 只能是几个固定的值，模型自创的写法（比如 `"USD billions"`）会被拒绝。
- **跨字段一致性**（`_check_consistency`）："说不知道却给了数值"、"给了数值却没给单位"，这类自相矛盾直接判为不合格。
- **`extra="forbid"`**：多出来的字段也算不合格，说明模型没按格式来。
- **长度限制**：`answer_text` 最多 500 字符，防止长篇大论。
- **把"不回答"也结构化**：`status` 有三种取值。"不知道"和"前提错误"是两种**正确的**行为，应当能被程序识别和统计，而不是埋在一段文字里。

### 6.3 带反馈的重试（`pipeline/answer.py`）

校验失败时，不是简单地再问一遍，而是：

```python
messages = [*messages,
            {"role": "assistant", "content": 它上次的错误输出},
            {"role": "user", "content": "Your previous reply was rejected: <具体错误>. Reply again with ONLY the JSON ..."}]
```

模型能看到"我上次输出了什么、错在哪"，改对的概率比盲目重试高得多。**空输出例外**：没有内容可反馈，就原样重试。

本次实验 30 次调用全部第一次就通过了校验，没有触发重试。重试流程由单元测试（`tests/test_answer.py`）保证正确，不能等线上出了问题才发现它有 bug。

---

## 7. prompt 是文件，不是字符串

`prompts/answer_v1.yaml`、`answer_v2.yaml`、`answer_v3.yaml`：每一版一个文件。好处：
- **能比较**：`git diff` 就能看出 v2 比 v1 多了什么；
- **能回滚**：新版本效果变差，换回旧文件名即可；
- **能测试**：`tests/test_prompts.py` 检查每个 prompt 文件都能加载、变量都填上了、包含 "json"、示例合格且不泄漏；
- **能记录**：配置文件里写的是 `prompt.name: answer_v2`，实验结果和 prompt 版本一一对应。

### 7.1 模板变量为什么用 `{{today}}`

| 写法 | 问题 |
|---|---|
| `str.format` 的 `{today}` | prompt 里的 JSON 示例全是花括号，format 会把 `{"value": 1}` 也当成变量，报 `KeyError: '"value"'`（演示脚本第 1 步） |
| `string.Template` 的 `$today` | 财报 prompt 里到处是 `$4.2 billion` 这样的金额，同样冲突 |
| 自己写的 `{{today}}` | 只认双花括号，和 JSON、美元符号都不冲突；缺变量就报错，不会留下没填的空位 |

### 7.2 日期为什么放在问题里，而不放在 system 里

L01 学过：前缀完全相同才能命中缓存。如果把每天都变的日期放进 system prompt，system 从第一个字就变了，后面的示例全都命中不了缓存。所以：

```
system（固定）→ 5 组示例（固定）→ 最后一条 user：Today's date + 问题（每次都变）
```

演示脚本第 5 步：两个不同问题的 12 条消息里，**前 11 条完全相同**。实验里的效果：

| prompt | 10 题输入 token | 其中缓存命中 | 花费 |
|---|---|---|---|
| v1 | 1,778 | 0 | $0.0013 |
| v3 | 11,808 | 10,240（87%） | **$0.0009** |

v3 的输入是 v1 的 6.6 倍，花费反而最低。**长 prompt 不一定贵，前提是它的前缀稳定。**

---

## 8. 代码导读（详细的阅读顺序和易错点见阅读指南）

| 顺序 | 文件 | 重点 |
|---|---|---|
| 1 | `prompts/answer_v1.yaml` → `v2` → `v3` | 用 diff 的眼光看：每一版多了什么 |
| 2 | `src/finagent/schemas.py` | 字段顺序、Literal、跨字段校验、`format_value` |
| 3 | `src/finagent/prompts.py` | `render`、`PromptTemplate.build_messages`、`load_prompt` |
| 4 | `src/finagent/pipeline/answer.py` ★ | `parse_answer` 的三种失败、`answer_question` 的重试循环 |
| 5 | `src/finagent/llm/openai_compat.py` | 新增的 `json_mode` → `response_format` |
| 6 | `src/finagent/config.py` | 新增的 `PromptConfig`；`prompt: None` 表示裸调模式 |
| 7 | `src/finagent/cli.py` | `_ask_raw` 和 `_ask_structured` 两条路径 |
| 8 | `tests/test_prompts.py`、`test_answer.py`、`test_cli.py` | 怎么测 prompt；怎么用"剧本客户端"测重试 |
| 9 | `scripts/l02_prompts.py` | 控制变量的实验设计、临时判分规则 |

---

## 9. 实验：三个 prompt 版本对比

```powershell
python scripts/l02_prompts.py
```

同样 10 道题，模型固定为 flash、关闭 thinking，**唯一的变量是 prompt**（控制变量）。报告在 `runs/l02_prompts/<时间>/report.md`。

输出变成结构化之后，**程序第一次可以自动判分了**，这是结构化输出最直接的价值。不过脚本里的 `quick_grade` 只是临时规则，L03 会正式建立评测。

### 我跑的一次结果（2026-09-26）

| 题 | 标准答案 | v1 | v2 | v3 |
|---|---|---|---|---|
| q01 NVDA 收入 | $130,497M | ✅ $130,500M | ✅ $130,497M | ✅ $130,497M |
| q02 NVDA 数据中心（新） | $193,737M | 🤷 | 🤷 | 🤷 |
| q03 AAPL 服务 | $96,169M | ✅ | ✅ $96,200M | ✅ |
| q04 MSFT 收入 | $281,724M | ✅ | ✅ | 🤷 |
| q05 GOOGL 收入（新） | $402,836M | 🤷 | 🤷 | 🤷 |
| q06 AWS 增长率 | 18.5% | 🤷 | ✅ 18.5% | ✅ 18.5% |
| q07 NVDA 毛利率 | 75.0% | ✅ | ✅ | ✅ |
| q08 META 购置固定资产 | $37,256M | ❌ $39,230M | ✅ | ✅ |
| q09 TSLA 汽车总收入 | $77,070M | ❌ $73,860M | ✅ $77,000M | ✅ $77,070M |
| q10 陷阱题 | 前提不成立 | ✅ | ✅ | ✅ |
| **合计 ✅ / ❌ / 🤷** | | **5 / 2 / 3** | **8 / 0 / 2** | **7 / 0 / 3** |

### 从结果里能学到什么

1. **规则的效果最明显**：v1 → v2 消灭了所有错误答案。q08、q09 靠规则 2（只答问的那一行），q06 靠规则 4（写出算式）。
2. **示例不是越多越好**：v3 比 v2 多拒答了一题，可能是因为示例里"不回答"占了 3/5（第 4 节）。
3. **"宁可不答，也不答错"**：q02、q05 三个版本都选择了"不知道"。L01 里没有规则时，flash 在 q02 编了一个错误的数。规则 5 把编造变成了拒答，代价是**覆盖率下降**：不答错了，但也答不出来。要真正答出来，需要 L05 的 RAG。
4. **自评信心只能部分参考**：报告最后一张表里，自评"高信心"的 21 个回答中有 2 个是错的（都来自 v1）；拒答时模型都自评"低信心"。自评信心有参考价值，但不能完全依赖。
5. **长 prompt 不一定贵**：v3 的输入最长，花费反而最低（第 7.2 节）。

### 最重要的一条警告

**v2 和 v3 只差一道题，这个差距不可靠。** 原因有两个：
- 样本只有 10 道，一道题就是 10 个百分点，随机波动就可能造成这么大的差异；
- 模型的输出本身有随机性，同样的 prompt 再跑一次，结果可能不同（作业 1）。

更危险的是：**如果在这 10 道题上反复修改 prompt，直到分数变高，你就是在对这 10 道题过拟合。** 换一批题，提升可能就消失了。这和训练模型时对训练集过拟合是一回事。L03 的第一件事，就是把评测集分成 dev（用来调）和 test（最后才看），并且学会判断一个差距是不是统计上显著。

v0.2 暂定用 v2（得分最高、更简单），L03 用更大的评测集再定。

---

## 10. 动手

```powershell
conda activate finagent
cd D:\claude\best\finagent
python -m pytest -q                                          # 53 个测试应全部通过
python walkthroughs/L02_json_mode.py                         # 5 个关键机制
finagent ask "What were Tesla's total automotive revenues in 2024?"   # 默认就是 v0.2
finagent ask "How much revenue did Apple's electric vehicle business generate in fiscal year 2024?" --json
finagent ask "What was NVIDIA's total revenue for fiscal year 2025?" --config configs/systems/v0.1.yaml   # 对比 v0.1
```

留意 `finagent ask` 底部信息里的"缓存命中"：连续问两个问题，第二次的缓存命中数会明显变多。

---

## 11. 作业

1. **体会随机性**：把实验再跑一次（`python scripts/l02_prompts.py --prompts answer_v2 answer_v3`），和第 9 节的结果逐题对比，看有哪些题的结果变了。然后思考：只有 10 道题时，"v2 比 v3 好"这个结论还站得住吗？
2. **验证示例分布的假设**：复制 `answer_v3.yaml` 为 `answer_v4.yaml`，把"不回答"类的示例减到 1 个，换成"回答"类的示例（注意：不能使用评测集里公司的真实数字，可以继续用"题目里给出数字"的写法）。跑 `--prompts answer_v3 answer_v4` 对比。
   跑完回答：如果你一直改到 v4 在这 10 道题上得分最高，就能说 v4 最好吗？为什么？
3. **thinking 的作用**：`python scripts/l02_prompts.py --prompts answer_v2 --thinking`，和关闭 thinking 时的 v2 比较准确率和花费。
4. （选做，代码练习）给 `Answer` 加一个 `fiscal_period` 字段（模型认为答案对应的财年，如 `"FY2025 (ended 2025-01-26)"`）。你会发现要改的地方不止 `schemas.py`：prompt 里的格式说明、few-shot 示例、测试里的 `GOOD` 字典……体会"改 schema 会牵动很多地方"。

---

## 12. 自测题（先自己答，再看第 13 节）

1. system prompt 和 user 消息分别放什么？为什么这个项目把日期放在 user 消息里？
2. "Be accurate." 为什么是一条坏规则？怎么改？
3. few-shot 示例怎么选？这次 v3 为什么可能变得更保守？
4. 为什么把 `reasoning` 放在 JSON 的第一个字段？
5. JSON mode 保证什么、不保证什么？DeepSeek 对它有什么特殊要求？
6. 解析失败时，为什么要把错误输出和错误原因一起发回去？空输出为什么不这样做？
7. 模板变量为什么用 `{{today}}`，而不是 `str.format` 或 `string.Template`？
8. 为什么把"不知道"做成 `status` 的一个取值，而不是让模型在 `answer_text` 里说？
9. 在同一批 10 道题上反复修改 prompt，有什么风险？
10. 为什么 v3 的输入 token 最多，花费却最低？

---

## 13. 参考答案

**1.** system 放对整个任务都成立的内容：角色、规则、格式、边界；user 放这一次的具体输入。日期每天都变，放进 system 会破坏前缀，导致 system 和后面的示例全都命中不了缓存；放在最后一条 user 消息里，前面的内容每次都一样，可以命中缓存。

**2.** 它只表达了愿望，没有给出做法，模型本来就"想"答对。应该改成具体可执行的规则，比如"只答题目问的那一行，不要换成更细的子项或更大的合计"，"不确定就输出 status: unknown，不要猜数字"。

**3.** 覆盖不同的行为；格式和真实输入一致；不泄漏评测题；示例本身合格；各类行为的比例要均衡。v3 的 5 个示例里有 3 个是"不回答"，示例里占多数的行为会被模型模仿，所以它可能更倾向于拒答。

**4.** 模型按顺序生成，后面的 token 能看到前面的内容。先写推理，推理就成了计算数值时的"草稿纸"；先写数值，数值就是凭直觉给的，后面的推理只是事后找理由。

**5.** 它保证输出是语法合法的 JSON，不保证字段齐全、类型正确、取值合法。DeepSeek 要求 prompt 里必须出现 "json" 这个词（否则返回 400），并建议给出格式示例；另外它偶尔会返回空内容。

**6.** 模型看到自己上一次的输出和具体错在哪，能有针对性地改正，比盲目重试成功率高。空输出没有内容可以反馈，只能原样再问一次。

**7.** `str.format` 会把 prompt 里 JSON 示例的花括号当成变量而报错；`string.Template` 会和金额里的 `$` 冲突。只认 `{{...}}` 的替换和两者都不冲突，缺变量时还能立刻报错。

**8.** 做成字段，程序才能识别和统计：拒答率多少、陷阱题有没有识破。它们都是**正确的**行为，判分时要和"答错"区别对待。如果埋在文字里，程序分不清"说不知道"和"答错了"。

**9.** 会对这 10 道题过拟合：改出来的 prompt 恰好适合这几道题，换一批题提升可能就消失了。而且 10 道题太少，随机波动就能造成一两道题的差异。正确做法是分 dev / test，并判断差异是否显著（L03）。

**10.** v3 的 system 和 5 组示例每次都完全一样，是一个很长的固定前缀，10 次调用里 87% 的输入 token 命中了缓存，而缓存价只有正常价格的约 1/50。

---

## 14. 下一课预告 · L03 Eval 驱动开发 ★

这一课结束时，我们手里有：一个能自动判分的结构化输出，一个只有 10 道题的评测集，和一个"v2 可能比 v3 好，但说不准"的结论。L03 要解决的就是"说不准"：
- 从 FinQA、FinanceBench 构建更大的评测集，分成 dev 和 test；
- 把 `quick_grade` 升级成正式的指标：数值容差、单位换算、陷阱题规则；
- 用 LLM 当裁判给文字类回答打分，并校准它的偏差；
- 用 bootstrap 计算置信区间，判断一个提升是真的还是运气；
- 每次实验生成一个不可变的记录目录，SCOREBOARD 的每一行都能复现。
