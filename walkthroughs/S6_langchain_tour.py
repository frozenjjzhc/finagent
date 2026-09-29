"""选修 S6 演示：LangChain 入门，每一步都对照 FinAgent 里我们亲手写的模块。

配合 lessons/S6_LangChain入门与对照.md 使用。
运行（先 conda activate finagent，在项目根目录；先 pip install -e ".[langchain]"）：
    python walkthroughs/S6_langchain_tour.py            # 全部 6 步，约 8 次很小的调用，不到 $0.001
    python walkthroughs/S6_langchain_tour.py --offline  # 只跑不调用模型的部分
第 6 步需要先 finagent build-corpus（用到 data/processed 里的 3 页财报）。

    第 1 步  模型：ChatOpenAI 连 DeepSeek，消息对象           ↔ llm/openai_compat.py
    第 2 步  prompt 模板 + LCEL 管道（prompt | model | parser）  ↔ prompts.py
    第 3 步  结构化输出：with_structured_output               ↔ llm/structured.py
    第 4 步  工具：@tool、bind_tools、手写一轮工具调用         ↔ tools/registry.py
    第 5 步  agent：create_agent 一行建好工具循环              ↔ agent/loop.py
    第 6 步  RAG：Document、切块、向量库、检索器、链           ↔ data/、rag/
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

from finagent.config import PROJECT_ROOT
from finagent.schemas import Answer
from finagent.tools.calculator import evaluate, format_number

OFFLINE = "--offline" in sys.argv


def banner(title: str) -> None:
    print("\n" + "=" * 76 + f"\n{title}\n" + "=" * 76)


def make_llm() -> ChatOpenAI:
    """DeepSeek 兼容 OpenAI 的接口，所以直接用 ChatOpenAI，改 base_url 即可（和我们的 ADR-1 一个道理）。"""
    load_dotenv(PROJECT_ROOT / ".env")
    return ChatOpenAI(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        extra_body={
            "thinking": {"type": "disabled"}
        },  # DeepSeek 的扩展参数，和 openai_compat.py 一样放 extra_body
        max_tokens=800,
    )


# ---------------------------------------------------------------- 第 1 步


def step1_model(llm: ChatOpenAI | None) -> None:
    banner("第 1 步 · 模型与消息（↔ llm/openai_compat.py）")
    messages = [SystemMessage("Reply in one short sentence."), HumanMessage("What is RAG?")]
    print("消息是对象，不是 dict：", [type(m).__name__ for m in messages])
    print("它们和我们用的 OpenAI 格式一一对应：SystemMessage ↔ {'role': 'system', ...}")
    if llm is None:
        return
    reply = llm.invoke(messages)  # 所有 LangChain 组件都有 invoke / batch / stream 三个方法
    print(f"\ninvoke 返回 {type(reply).__name__}：{reply.content[:90]!r}")
    print(f"  finish_reason：{reply.response_metadata.get('finish_reason')}")
    print(f"  usage_metadata：{reply.usage_metadata}")
    print(
        "  对照：我们的 LLMResponse 还记下了花费、延迟和缓存命中，并写进 runs/calls/；LangChain 要接 LangSmith 或回调才有"
    )


# ---------------------------------------------------------------- 第 2 步


def step2_prompt(llm: ChatOpenAI | None) -> None:
    banner("第 2 步 · prompt 模板 + LCEL 管道（↔ prompts.py）")
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a financial analyst. Reply in one sentence."), ("human", "{question}")]
    )
    print(f"模板变量：{prompt.input_variables}")
    print(f"渲染结果：{prompt.invoke({'question': 'What is EBITDA?'}).to_messages()}")

    print(
        "\n坑：LangChain 的模板用 {变量}，prompt 里的 JSON 示例会被当成变量（L02 我们专门躲开的坑）："
    )
    try:
        bad = ChatPromptTemplate.from_messages(
            [("system", 'Reply like {"value": 1}'), ("human", "{question}")]
        )
        print(f"  它认出的变量：{bad.input_variables}")
        bad.invoke({"question": "Q"})
    except KeyError as e:
        print(f"  渲染报错：KeyError {str(e)[:70]}…")
    print(
        '  解决：JSON 的花括号写两遍 {{"value": 1}}。我们的 render() 只认 {{变量}}，所以没有这个问题'
    )

    if llm is None:
        return
    chain = prompt | llm | StrOutputParser()  # LCEL：用 | 把组件串成一条管道
    print(f"\nprompt | llm | StrOutputParser() 的类型：{type(chain).__name__}")
    print(f"chain.invoke：{chain.invoke({'question': 'What is EBITDA?'})[:100]!r}")
    answers = chain.batch([{"question": "What is ROA?"}, {"question": "What is free cash flow?"}])
    print(f"chain.batch（并发执行两个）：{[a[:50] for a in answers]}")


# ---------------------------------------------------------------- 第 3 步


def step3_structured(llm: ChatOpenAI | None) -> None:
    banner("第 3 步 · 结构化输出（↔ llm/structured.py）")
    print("直接复用我们的 pydantic 模型 finagent.schemas.Answer")
    if llm is None:
        return
    messages = [
        SystemMessage("You are a financial analyst. Answer with the Answer schema."),
        HumanMessage(
            "Revenue was 5829 in 2019 and 5735 in 2018. What is the growth rate in percent?"
        ),
    ]
    try:
        llm.with_structured_output(Answer).invoke(messages)  # 默认 method="json_schema"
    except Exception as e:  # noqa: BLE001
        print(f"\n默认方式（json_schema）：{type(e).__name__}：{str(e)[:110]}")
        print("  → OpenAI 支持的 json_schema 模式，DeepSeek 不支持。框架的默认值是按 OpenAI 设计的")
    answer = llm.with_structured_output(Answer, method="function_calling").invoke(messages)
    print(
        f"\nmethod='function_calling'：得到 {type(answer).__name__}，value={answer.value}，unit={answer.unit}"
    )
    print(
        '  原理：把 Answer 变成一个"工具"的说明书，让模型"调用"它，参数就是答案（L04 学的 function calling）'
    )
    print(
        "  对照：我们的 call_structured 解析失败时会把错误反馈给模型重试；LangChain 默认直接抛异常"
    )


# ---------------------------------------------------------------- 第 4 步


@tool(parse_docstring=True)
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly and return the numeric result. Use it for every calculation.

    Args:
        expression: One arithmetic expression using plain numbers, for example "(5829 - 5735) / 5735 * 100".
    """
    return format_number(evaluate(expression))  # 干活的还是我们的 AST 白名单计算器


def step4_tools(llm: ChatOpenAI | None) -> None:
    banner("第 4 步 · 工具（↔ tools/registry.py、agent/loop.py）")
    print(
        "@tool(parse_docstring=True) 生成的说明书（和我们的注册表一样，也是读签名和 docstring）："
    )
    print(json.dumps(convert_to_openai_tool(calculate), ensure_ascii=False, indent=1)[:520])
    print("  对照：我们的说明书多了 additionalProperties: false（多传参数也算错）")
    if llm is None:
        return
    llm_with_tools = llm.bind_tools([calculate])  # 相当于每次请求都带上 tools 参数
    question = HumanMessage("Use the calculate tool: what is (5829 - 5735) / 5735 * 100?")
    ai = llm_with_tools.invoke([question])
    print(f"\n模型回复的 tool_calls：{ai.tool_calls}")
    print("  注意 args 已经是 dict：LangChain 替你做了 json.loads（我们在 Tool.run 里自己做）")
    # 手写一轮：执行 → ToolMessage 回传 → 再请求。和 L04 的 run_tool_loop 一模一样
    tool_messages = [
        ToolMessage(calculate.invoke(tc["args"]), tool_call_id=tc["id"]) for tc in ai.tool_calls
    ]
    final = llm_with_tools.invoke([question, ai, *tool_messages])
    print(f"回传 ToolMessage 之后的最终回答：{final.content[:100]!r}")


# ---------------------------------------------------------------- 第 5 步


def step5_agent(llm: ChatOpenAI | None) -> None:
    banner("第 5 步 · agent：create_agent（↔ agent/loop.py）")
    if llm is None:
        print("（离线模式跳过）")
        return
    from langchain.agents import create_agent

    agent = create_agent(
        llm,
        tools=[calculate],
        system_prompt="Use the calculate tool for every calculation. Answer in one sentence.",
    )
    print(
        f"create_agent 返回的是 {type(agent).__name__}：一张 LangGraph 状态图（节点：模型、工具；循环到不再调用工具为止）"
    )
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Growth from 5735 to 5829, and from 5102 to 5735, in percent?",
                }
            ]
        }
    )
    for m in result["messages"]:
        extra = (
            f"  tool_calls={[tc['args'] for tc in m.tool_calls]}"
            if getattr(m, "tool_calls", None)
            else ""
        )
        print(f"  {type(m).__name__:<13} {' '.join(str(m.content).split())[:70]}{extra}")
    print(
        "  对照：我们 30 行的 run_tool_loop 做的就是这件事；LangGraph 额外提供状态保存、中断、人工确认等能力（L07 会用到类似的需求）"
    )


# ---------------------------------------------------------------- 第 6 步


def step6_rag(llm: ChatOpenAI | None) -> None:
    banner("第 6 步 · RAG：Document → 切块 → 向量库 → 检索器 → 链（↔ data/、rag/）")
    from langchain_core.documents import Document
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from finagent.data.ingest import load_docs, load_pages

    wanted = {("NETFLIX_2017_10K", 44), ("BLOCK_2020_10K", 89), ("3M_2018_10K", 59)}
    titles = {name: d["title"] for name, d in load_docs().items()}
    # 用我们 rows 版本的页面构造 Document：page_content 是文字，metadata 放出处
    docs = [
        Document(
            page_content=p.text,
            metadata={"doc_name": p.doc_name, "page": p.page, "title": titles[p.doc_name]},
        )
        for p in load_pages("rows")
        if (p.doc_name, p.page) in wanted
    ]
    print(f"3 页 → {len(docs)} 个 Document；metadata 例子：{docs[0].metadata}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )  # 按字符数切（常见教程的写法）
    chunks = splitter.split_documents(docs)
    print(
        f"RecursiveCharacterTextSplitter(1000 字符, 重叠 200) → {len(chunks)} 块（metadata 自动继承）"
    )
    print(
        f"  默认参数是 chunk_size={RecursiveCharacterTextSplitter()._chunk_size} 个字符：约 1000 个 token，超过 bge 的 512 上限，后半截会被截掉"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        query_encode_kwargs={
            "prompt": "Represent this sentence for searching relevant passages: "
        },  # 默认不加
    )
    store = InMemoryVectorStore.from_documents(
        chunks, embeddings
    )  # 算向量 + 存起来（相当于 build_index）
    retriever = store.as_retriever(search_kwargs={"k": 3})  # 默认 k=4
    question = (
        "How much cash did Block generate from operating activities in FY2020 (in USD millions)?"
    )
    hits = retriever.invoke(question)
    print(
        f"\n问题：{question}\n检索到：{[(d.metadata['doc_name'], d.metadata['page']) for d in hits]}"
    )
    print(
        "  对照：我们的 Chunk / Hit / Source 就是 Document + metadata；as_retriever 就是 Retriever.search"
    )

    if llm is None:
        return

    def format_docs(documents: list) -> str:  # 和我们的 format_context 一样：编号 + 出处
        return "\n\n".join(
            f"[{i}] {d.metadata['title']}, page {d.metadata['page'] + 1}\n{d.page_content}"
            for i, d in enumerate(documents, 1)
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ("Answer using only the excerpts. Cite excerpt numbers in sources. "
                        "Express money in USD millions (a table in thousands: divide by 1000).")),
            ("human", "Excerpts:\n\n{context}\n\nQuestion: {question}"),
        ]
    )  # fmt: skip
    # LCEL：先并行准备两个输入（检索并格式化、原样传递问题），再进 prompt 和结构化模型
    from langchain_core.runnables import RunnablePassthrough

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm.with_structured_output(Answer, method="function_calling")
    )
    answer = rag_chain.invoke(question)
    print(f"\nRAG 链的回答：value={answer.value} {answer.unit}，sources={answer.sources}")
    print(f"  {answer.answer_text}")
    print("  标准答案（FinanceBench fb07661）：$382.00（381,603 千美元）")
    block_chunks = [d for d in hits if d.metadata["doc_name"] == "BLOCK_2020_10K"]
    key = "Net cash provided by operating activities"
    if block_chunks and not any(key in d.page_content for d in block_chunks):
        print(
            f"\n这就是 L05 讲的 small-to-big 问题：检索到的 Block 那一块里没有 {key!r} 这一行\n"
            "（按 1000 个字符切，这一行被切到了另一块里）。模型只看到半张表，只能拒答。\n"
            "LangChain 里的对策是 ParentDocumentRetriever（用小块检索、返回整页），和我们 v0.5 的 context: page 一样；\n"
            "但这需要你先知道有这个问题：默认写法不会替你想到。"
        )


def main() -> None:
    llm = None if OFFLINE else make_llm()
    step1_model(llm)
    step2_prompt(llm)
    step3_structured(llm)
    step4_tools(llm)
    step5_agent(llm)
    step6_rag(llm)


if __name__ == "__main__":
    main()
