# 选修 S6 · LangChain 入门与对照

> ⏱ 约 3 小时（读讲义 1h + 跑演示 0.5h + 实验与作业 1.5h）
> 📍 前置：L01~L05。本课不改变 FinAgent 的核心代码：核心仍然不依赖任何框架（ADR-6），LangChain 只出现在 `walkthroughs/S6_*` 和 `scripts/s6_*` 里
> 📦 本课产出：LangChain 入门演示（6 步，每步对照我们自己的模块）、LangChain RAG 与 v0.5 的对照实验
> 🔢 版本：langchain 1.4.3、langchain-core 1.6.5、langgraph 1.2.12（2026-09 安装）。LangChain 的 API 变化很快，网上的旧教程常常跑不通，以你安装的版本为准

---

## 0. 这节课学完你能做到

1. 说清 LangChain 生态里各个包分别是干什么的。
2. 用 LangChain 写出模型调用、prompt 模板、结构化输出、工具、agent、RAG，并且知道每一样对应我们项目里的哪个模块。
3. 知道 LangChain 的默认值在哪些地方不适合我们的场景，以及怎么改。
4. 用我们自己的评测，量化"框架默认写法"和"工程化之后"的差距。
5. 面试时讲清楚：为什么先手写、什么时候用框架。

**阅读顺序**：先跑 `python walkthroughs/S6_langchain_tour.py`（约 8 次小调用，不到 $0.001），再读 §2（每一节对应演示的一步），然后看 §4 的实验。

---

## 1. LangChain 是什么

LangChain 是一个开源的 LLM 应用开发框架（有 Python 和 JavaScript 版本），把"调用模型、拼 prompt、解析输出、调用工具、检索文档"这些常见环节做成统一接口的组件，让你用很少的代码拼出一个应用。

它现在是一组包，不是一个包：

| 包 | 干什么 | 本课用到 |
|---|---|---|
| `langchain-core` | 基础抽象：消息、Runnable、prompt 模板、工具、Document、向量库接口 | ✅ |
| `langchain` | 上层功能，比如 `create_agent` | ✅ |
| `langchain-openai` 等 | 各家模型的集成包（OpenAI、Anthropic、DeepSeek……） | ✅ 用 `ChatOpenAI` 连 DeepSeek |
| `langchain-community` | 社区维护的大量集成：文档加载器、向量库…… | ✅ `PyMuPDFLoader` |
| `langchain-huggingface` | 本地的 HuggingFace 模型，包括 embedding | ✅ bge-small |
| `langchain-text-splitters` | 切块 | ✅ |
| **LangGraph** | 把 agent 写成"状态图"：节点、边、循环、状态保存、人工介入 | ✅ `create_agent` 底层就是它 |
| **LangSmith** | 商业平台：调用记录、调试、评测 | 只介绍 |

安装：`pip install -e ".[langchain]"`（`pyproject.toml` 里的可选依赖组）。

---

## 2. 六个核心概念（对照演示的 6 步）

### 2.1 模型与消息（↔ `llm/openai_compat.py`）

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-flash",
    base_url="https://api.deepseek.com",           # DeepSeek 兼容 OpenAI 的接口，换个地址就行
    api_key=os.environ["DEEPSEEK_API_KEY"],
    extra_body={"thinking": {"type": "disabled"}},  # DeepSeek 的扩展参数，和我们一样放 extra_body
)
reply = llm.invoke([SystemMessage("Reply in one short sentence."), HumanMessage("What is RAG?")])
```

- 消息是对象（`SystemMessage`、`HumanMessage`、`AIMessage`、`ToolMessage`），和我们用的 `{"role": ..., "content": ...}` 一一对应。
- `invoke` 返回 `AIMessage`：`content`、`response_metadata`（含 finish_reason）、`usage_metadata`（token 数）。
- **对照**：我们的 `LLMResponse` 还算好了花费、延迟、缓存命中，并写进 `runs/calls/`。LangChain 要自己加回调，或者接 LangSmith。

### 2.2 Runnable 与 LCEL：用 `|` 串起来

LangChain 里几乎所有东西（prompt、模型、解析器、检索器）都是 **Runnable**，有统一的三个方法：`invoke`（一次）、`batch`（并发多次）、`stream`（流式）。用 `|` 把它们串成管道，这叫 **LCEL**（LangChain Expression Language）：

```python
chain = prompt | llm | StrOutputParser()      # 渲染模板 → 调模型 → 取出文字
chain.invoke({"question": "What is EBITDA?"})
chain.batch([{"question": "What is ROA?"}, {"question": "What is free cash flow?"}])   # 并发
```

演示里 `type(chain)` 是 `RunnableSequence`。管道很方便，代价是出错时调用栈很深，不容易看出是哪一环出的问题。

### 2.3 prompt 模板（↔ `prompts.py`）

```python
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a financial analyst. Reply in one sentence."), ("human", "{question}")]
)
```

**一个 L02 就讲过的坑**：LangChain 的模板变量写成 `{变量}`（和 Python 的 `str.format` 一样），所以 prompt 里的 JSON 示例也会被当成变量。演示第 2 步的真实结果：

```
模板 'Reply like {"value": 1}' 被认出的变量：['"value"', 'question']
渲染时：KeyError  Input to ChatPromptTemplate is missing variables {'"value"'}
```

解决办法是 JSON 的花括号都写两遍：`{{"value": 1}}`。我们的 `render()` 只认 `{{变量}}`，正是为了避开这个坑。对照实验里，把 answer_rag_v1 的 system prompt 放进 LangChain 时，就得先把所有花括号双写。

### 2.4 结构化输出（↔ `llm/structured.py`）

```python
answer = llm.with_structured_output(Answer, method="function_calling").invoke(messages)
```

直接复用我们的 pydantic 模型 `Answer`，返回的就是 `Answer` 对象。演示第 3 步对三种 `method` 的真实结果：

| method | 原理 | 在 DeepSeek 上 |
|---|---|---|
| `json_schema`（**默认**） | OpenAI 的"结构化输出"：服务端按 schema 约束生成 | ❌ 400：`This response_format type is unavailable now` |
| `function_calling` | 把 `Answer` 变成一个工具的说明书，让模型"调用"它，参数就是答案（L04） | ✅ |
| `json_mode` | JSON mode + 解析 | ⚠️ 探测时解析失败，直接抛 `OutputParserException` |

两个教训：
1. **框架的默认值是按 OpenAI 设计的**。换一家模型，默认写法可能直接报错。
2. 我们的 `call_structured` 在解析失败时会把错误信息反馈给模型、让它重试（L02）；LangChain 默认直接抛异常，要自己包一层重试。

### 2.5 工具与 agent（↔ `tools/registry.py`、`agent/loop.py`）

```python
@tool(parse_docstring=True)
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly ...

    Args:
        expression: One arithmetic expression ...
    """
    return format_number(evaluate(expression))    # 干活的还是我们的 AST 白名单计算器
```

- LangChain 的 `@tool` 和我们 L04 写的 `@tool` **思路完全一样**：读函数签名和 docstring 生成说明书。`parse_docstring=True` 才会把 Args 段拆成参数描述（否则整段 docstring 都塞进工具描述）。
- 区别：我们的说明书有 `additionalProperties: false`（多传参数也算错），LangChain 的没有。

手写一轮工具调用（演示第 4 步），和 L04 的 `run_tool_loop` 一模一样：

```python
ai = llm.bind_tools([calculate]).invoke([question])       # bind_tools = 每次请求都带上 tools
# ai.tool_calls = [{'name': 'calculate', 'args': {'expression': '(5829 - 5735) / 5735 * 100'}, 'id': ...}]
tool_messages = [ToolMessage(calculate.invoke(tc["args"]), tool_call_id=tc["id"]) for tc in ai.tool_calls]
final = llm.bind_tools([calculate]).invoke([question, ai, *tool_messages])
```

注意 `args` 已经是 dict：LangChain 替你做了 `json.loads`（我们在 `Tool.run` 里自己做）。

一行建好 agent（演示第 5 步）：

```python
from langchain.agents import create_agent
agent = create_agent(llm, tools=[calculate], system_prompt="Use the calculate tool for every calculation.")
result = agent.invoke({"messages": [{"role": "user", "content": "Growth from 5735 to 5829, and from 5102 to 5735?"}]})
```

真实的消息序列：`HumanMessage` → `AIMessage`（一次请求了两个计算，并行调用）→ 两条 `ToolMessage` → `AIMessage`（最终回答）。和 L04 演示第 3 步看到的完全相同。

`create_agent` 返回的是 `CompiledStateGraph`：一张 LangGraph 状态图，节点是"模型"和"工具"，循环到模型不再调用工具为止。我们 30 行的 `run_tool_loop` 就是这张图最简单的版本。LangGraph 额外提供状态持久化、中途暂停等人工确认、多 agent 协作等能力，L07 做 agent 时会遇到类似需求。

### 2.6 RAG 组件（↔ `data/`、`rag/`）

| LangChain | 我们 | 默认值 / 注意 |
|---|---|---|
| `Document(page_content, metadata)` | `Page`、`Chunk`、`Source` | metadata 放出处（文档名、页码） |
| `PyMuPDFLoader` | `extract_pages` | 每页一个 Document，页码从 0 开始；每个单元格一行（不重建表格行） |
| `RecursiveCharacterTextSplitter` | `chunk_page` | **默认 4000 个字符、重叠 200**，按字符数不按 token 数；`from_huggingface_tokenizer` 可以按 token 切 |
| `HuggingFaceEmbeddings` | `Embedder` | **不加查询前缀**；用 `query_encode_kwargs={"prompt": 前缀}` 加上后，算出的问题向量和我们的完全一样（余弦 1.0，已验证） |
| `InMemoryVectorStore`、`FAISS`、`Chroma` | `NumpyIndex`、`FaissIndex` | 本质都是"存向量、找最近的 k 个" |
| `vectorstore.as_retriever()` | `Retriever.search` | **默认 k = 4** |
| `ParentDocumentRetriever` | `ContextBuilder` 的 `context: page` | small-to-big，要自己想到去配 |

演示第 6 步的 RAG 链（LCEL 写法）：

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}   # 两路并行准备输入
    | prompt
    | llm.with_structured_output(Answer, method="function_calling")
)
answer = rag_chain.invoke("How much cash did Block generate from operating activities in FY2020?")
```

**真实结果：它答不出来**。检索到的 Block 那一块（按 1000 个字符切）正好在 `Net cash provided by operating activities` 那一行之前断开，模型回答"材料被截断，无法确定"。这就是 L05 讲过的 small-to-big 问题，在 LangChain 里原样出现了一次。

---

## 3. 对照总表：同一件事，两种写法

| 环节 | FinAgent（我们手写） | LangChain |
|---|---|---|
| 调模型 | `OpenAICompatClient.chat`（带重试、计费、日志） | `ChatOpenAI.invoke` |
| prompt | YAML 文件 + `render()`（`{{变量}}`） | `ChatPromptTemplate`（`{变量}`，JSON 花括号要双写） |
| 结构化输出 | `call_structured`（JSON mode + pydantic + 带反馈重试） | `with_structured_output`（DeepSeek 上要用 function_calling） |
| 工具 | `@tool` + `ToolRegistry`（三关校验、错误回传） | `@tool(parse_docstring=True)` + `bind_tools` |
| 工具循环 | `run_tool_loop`（30 行） | `create_agent`（LangGraph） |
| 读 PDF | `extract_pages` + `rows_text` | `PyMuPDFLoader` |
| 切块 | `chunk_page`（按行、按 token、不跨页、加标题） | `RecursiveCharacterTextSplitter` |
| 向量化 | `Embedder`（查询前缀、归一化） | `HuggingFaceEmbeddings` |
| 索引与检索 | `NumpyIndex` / `FaissIndex` + `Retriever` | `InMemoryVectorStore` / `FAISS` + `as_retriever` |
| small-to-big | `ContextBuilder(context="page")` | `ParentDocumentRetriever` |
| 调用记录与评测 | `tracing.py`、`finagent eval`、`compare` | LangSmith（商业平台） |

LangChain 让每一行都更短；我们的版本让每一个决定都看得见、改得动、量得出。

---

## 4. 实验：LangChain 默认写法 vs L05 的做法

`python scripts/s6_langchain_rag.py --e2e`：三种 LangChain 写法，语料都是 84 份 PDF，向量模型都是 bge-small，用我们自己的指标在 dev 上评测（`runs/s6_langchain/20260929-063336`）。

| 写法 | 做法 | 块数 | 被截断的块 | R@4 | R@5 | 文档命中@5 |
|---|---|---|---|---|---|---|
| lc_default | 库的默认值：PyMuPDFLoader + 切块默认 4000 字符 + 不加前缀 | 16,282 | **8,778（54%）** | 28% | 34% | 80% |
| lc_tutorial | 常见教程：同上，但切成 1000 字符、重叠 200 | 54,539 | 104 | 28% | 32% | 74% |
| lc_tuned | 用 LangChain 的组件复现 L05：rows 页面 + 按 token 切 256 + 标题 + 前缀 | 41,739 | 0 | 42% | **48%** | **92%** |
| 对照：fb_bge_small（L05，手写） | | 41,766 | 0 | — | 48% | 92% |

（R@4 是因为 LangChain 的 `as_retriever()` 默认只取 4 个。）

**读法**：
1. **库的默认值有一半的块被截断**：4000 个字符约 1000 个 token，超过 bge 的 512 上限，后半截在向量化时被悄悄丢掉，不报错。只有像我们这样主动数一数才会发现。
2. **常见教程的写法也只有 32%**，比 L05 的做法低 16 个百分点。
3. **lc_tuned 和我们手写的版本结果完全一样（48% / 92%）**：用 LangChain 的组件，把 L05 的几个决定（表格行重建、按 token 切、块首标题、查询前缀）照做一遍，效果就回来了。**决定效果的是这些做法，不是"用不用框架"。**

**端到端**：用 lc_tutorial 的检索（k = 4），加上和 v0.5 相同的 system prompt（去掉计算器那段），`with_structured_output(method="function_calling")`，同一个裁判：

| | 准确率（95% 区间） | 答错 | 拒答 | 输出不合格 | 证据页命中 |
|---|---|---|---|---|---|
| lc_tutorial | **22%**（10% ~ 34%） | 12% | 58% | 4 题 | 28% |
| v0.5（L05） | **44%**（30% ~ 58%） | 14% | 42% | 0 题 | 48% |

配对比较：v0.5 比 lc_tutorial 高 **22 个百分点**（区间 +10 ~ +36，McNemar p = 0.003），**显著**。

那 4 道"输出不合格"的题，原因一模一样：模型把 `unit` 写成了**字符串** `"null"`（而不是 JSON 的 `null`），pydantic 校验不通过，LangChain 直接抛异常：

```
ValidationError: 1 validation error for Answer
unit
  Input should be 'USD_millions', 'percent', 'count' or 'other' [input_value='null', input_type=str]
```

我们的 `call_structured` 遇到这种情况，会把这条错误信息反馈给模型，让它改正后重试（L02 的"带反馈的重试"）。这 4 道题在 v0.5 里都得到了合格的输出。

**结论**：照教程几十行搭起来的 LangChain RAG，在我们的评测上只有 v0.5 一半的准确率；差距来自切块、标题、表格处理、给块还是给整页、失败重试这些细节。框架不会替你做这些决定，评测会告诉你它们有多重要。

---

## 5. 什么时候用框架，什么时候自己写

| 情况 | 建议 |
|---|---|
| 学习、面试准备 | 先手写：你得知道每一步在做什么（本课程的路线） |
| 快速做原型、验证想法 | 用框架：几十行就能跑通，先看方向对不对 |
| 团队已经在用 LangChain / LangGraph | 用框架：共同的语言比个人偏好重要 |
| 要上线、要调效果 | 框架可以用，但**默认值一个都别信**：用评测逐个验证（切块、前缀、k、上下文怎么给） |
| 需要复杂的 agent 流程（分支、暂停等人确认、多 agent） | LangGraph 很合适，它把状态和流程管理做好了 |
| 很简单的调用（一次请求、一个 prompt） | 直接用 SDK，不必引入框架 |

**一句话**：框架省的是"写代码的时间"，省不了"想清楚的时间"。本课实验里，决定效果的是切多大、加不加标题、给块还是给整页，这些框架都不会替你决定。

---

## 6. 面试怎么讲

被问"用过 LangChain 吗"，可以这样回答：

> 我的项目核心是手写的：工具循环、RAG 的切块、检索和评测都自己实现，所以清楚框架每个组件底层在做什么。我也用 LangChain 把同样的管线重写过一遍，用自己的评测集做了对照：照常见教程的写法，检索命中率 32%，端到端准确率 22%；库的默认切块甚至让一半的块超过向量模型的长度上限被截断。我在项目里加了表格行重建、块首标题、按 token 切块和 small-to-big，命中率 48%、准确率 44%，差距是显著的。用 LangChain 的组件照这些做法重写之后，结果和我手写的一样，说明关键在做法而不在框架。另外 LangChain 在 DeepSeek 上有兼容问题：结构化输出的默认模式直接报 400，要换成 function calling；解析失败时也不会自动带反馈重试。

这样回答同时体现了"懂原理"、"会用框架"和"会用数据说话"。

---

## 7. 作业

1. **small-to-big**：给 `scripts/s6_langchain_rag.py` 的 lc_tutorial 加上"检索块 → 返回整页"（可以用 `ParentDocumentRetriever`，也可以在检索后按 metadata 的页码自己换成整页），端到端再跑一次，和 §4 比较。
2. **用 LangGraph 重写 v0.5**：用 `create_agent`，把检索做成一个工具（`search_filings(question)` 返回整页材料），加上计算器，在 dev 上评测。这就是 L07 "agentic RAG" 的雏形：模型可以自己决定搜几次、搜什么。
3. **计费回调**：LangChain 默认不记录花费。查 `langchain_core.callbacks` 的文档，写一个回调，在每次调用结束时把 token 数和估算的花费记下来（对照我们的 `tracing.py` 和 `pricing.py`）。

---

## 8. 自测题

1. LangChain、langchain-core、LangGraph、LangSmith 分别是什么？
2. LCEL 的 `|` 做了什么？Runnable 有哪三个统一的方法？
3. 为什么把 answer_rag_v1 的 system prompt 放进 `ChatPromptTemplate` 之前，要把花括号双写？
4. `with_structured_output` 的默认 method 在 DeepSeek 上为什么报错？怎么解决？function_calling 模式的原理是什么？
5. LangChain 的 `@tool` 和我们的 `@tool` 有什么相同、什么不同？
6. `create_agent` 返回的是什么？它和我们的 `run_tool_loop` 是什么关系？
7. 用 LangChain 默认值搭 RAG，至少有哪 4 个地方和 L05 的做法不同？
8. 什么情况下你会选择用框架？什么情况下直接用 SDK？

**参考答案**

1. LangChain 是一组 LLM 应用开发框架的包；langchain-core 是基础抽象（消息、Runnable、模板、工具、Document）；LangGraph 用状态图编排 agent 和多步流程；LangSmith 是记录、调试、评测的商业平台。
2. `|` 把前一个组件的输出作为后一个的输入，串成一个新的 Runnable（`RunnableSequence`）。三个方法：`invoke`、`batch`、`stream`。
3. LangChain 的模板把 `{...}` 当作变量，system prompt 里的 JSON 示例会被误认为变量，渲染时报 KeyError。双写 `{{` `}}` 表示字面的花括号。
4. 默认是 `json_schema`，依赖 OpenAI 的服务端结构化输出，DeepSeek 不支持这种 response_format。改用 `method="function_calling"`：把 schema 变成一个工具的说明书，模型"调用"这个工具时的参数就是结构化的答案。
5. 相同：都从函数签名和 docstring 生成说明书、都用 pydantic 校验参数。不同：LangChain 要 `parse_docstring=True` 才解析 Args 段；我们的说明书禁止多余参数（`additionalProperties: false`）；我们把所有错误变成文字回传，LangChain 的工具单独调用时会抛异常（在 agent 里由工具节点处理）。
6. 返回一张编译好的 LangGraph 状态图（`CompiledStateGraph`），节点是模型和工具，循环到模型不再请求工具为止。它就是 `run_tool_loop` 的图版本，外加状态保存、中断等能力。
7. 抽取不重建表格行；按字符数切（默认 4000 字符，超过 bge 的 512 token 上限会被截断）；不加查询前缀；不加文档标题；默认 k = 4；只给块，不做 small-to-big。
8. 原型、团队统一、复杂 agent 流程时用框架；一次简单的调用直接用 SDK；无论哪种，默认值都要用评测验证。
