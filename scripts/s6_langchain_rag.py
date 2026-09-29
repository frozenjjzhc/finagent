"""选修 S6 实验：用 LangChain 搭 RAG，和 v0.5 比较。评测用的是我们自己的指标和判分。

三种 LangChain 写法（语料都是 FinanceBench 的 84 份 PDF，向量模型都是 bge-small）：
    lc_default   库的默认值：PyMuPDFLoader + RecursiveCharacterTextSplitter()（4000 字符 / 重叠 200）
                 + 不加查询前缀 + as_retriever() 默认取 4 个
    lc_tutorial  常见教程的写法：同上，但切成 1000 字符 / 重叠 200
    lc_tuned     用 LangChain 的组件复现 L05 的做法：rows 版页面 + 按 bge 的 token 切（256 / 32）
                 + 块首加文档标题 + 查询前缀，取 5 个

用法（先 conda activate finagent，在项目根目录；需要先 build-corpus）：
    python scripts/s6_langchain_rag.py            # 只评检索（免费，建三个向量库约 6~8 分钟）
    python scripts/s6_langchain_rag.py --e2e      # 再用 lc_tutorial 端到端跑一遍 dev（调用模型，约几美分）

只在 dev 上做。输出：runs/s6_langchain/<时间戳>/report.md 和 results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from finagent.config import PROJECT_ROOT, load_eval_config
from finagent.data.ingest import PDF_DIR, corpus_doc_names, load_docs, load_pages
from finagent.eval.datasets import load_evalset
from finagent.eval.judge import Judge
from finagent.eval.metrics import (
    bootstrap_ci,
    grade_item,
    mcnemar_exact,
    paired_bootstrap_diff,
    summarize,
)
from finagent.eval.retrieval import evidence_pages, first_doc_rank, first_hit_rank, recall_at
from finagent.eval.run import load_results
from finagent.llm.openai_compat import OpenAICompatClient
from finagent.prompts import load_prompt
from finagent.schemas import Answer

OUT_ROOT = PROJECT_ROOT / "runs" / "s6_langchain"
MODEL = "BAAI/bge-small-en-v1.5"
PREFIX = "Represent this sentence for searching relevant passages: "
V05_RUN = PROJECT_ROOT / "runs" / "eval" / "20260928-094320_v0.5_answer_rag_v1_financebench-dev"


def load_pdf_documents() -> list[Document]:
    """LangChain 的标准做法：PyMuPDFLoader 每页一个 Document，metadata 里有 page（从 0 开始）。"""
    docs = []
    for name in corpus_doc_names():
        for d in PyMuPDFLoader(str(PDF_DIR / f"{name}.pdf")).load():
            d.metadata = {"doc_name": name, "page": d.metadata["page"]}  # 只保留要用的
            docs.append(d)
    return docs


def build_store(name: str) -> tuple[InMemoryVectorStore, int, int, int]:
    """按配置建一个 LangChain 向量库，返回（库，块数，超过 512 token 的块数，用时秒）。"""
    t0 = time.perf_counter()
    if name == "lc_tuned":
        embeddings = HuggingFaceEmbeddings(model_name=MODEL, query_encode_kwargs={"prompt": PREFIX})
        tokenizer = embeddings._client.tokenizer  # bge 自己的分词器
        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer, chunk_size=256, chunk_overlap=32
        )
        titles = {n: d["title"] for n, d in load_docs().items()}
        pages = [
            Document(p.text, metadata={"doc_name": p.doc_name, "page": p.page})
            for p in load_pages("rows")
            if p.text
        ]
        chunks = splitter.split_documents(pages)
        for c in chunks:  # 块首加文档标题（L05 §3.5）
            c.page_content = f"{titles[c.metadata['doc_name']]}\n{c.page_content}"
    else:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL)  # 默认：不加查询前缀
        tokenizer = embeddings._client.tokenizer
        splitter = (
            RecursiveCharacterTextSplitter()  # 默认 4000 字符 / 重叠 200
            if name == "lc_default"
            else RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        )
        chunks = splitter.split_documents(load_pdf_documents())
    lengths = [
        len(x) for x in tokenizer([c.page_content for c in chunks], verbose=False)["input_ids"]
    ]
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    return store, len(chunks), sum(n > 512 for n in lengths), round(time.perf_counter() - t0)


def retrieval_table(stores: dict, items: list) -> tuple[list[str], dict]:
    lines = [
        "| 写法 | 块数 | 超过 512 token（被截断）的块 | R@4 | R@5 | 文档命中@5 | R@5 的 95% 区间 | 建库用时 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    per_item = {}
    for name, (store, n_chunks, n_long, seconds) in stores.items():
        ranks, doc_ranks = [], []
        for it in items:
            hits = store.similarity_search(it.question, k=5)
            keys = [(d.metadata["doc_name"], d.metadata["page"]) for d in hits]
            ranks.append(first_hit_rank(keys, evidence_pages(it)))
            doc_ranks.append(first_doc_rank(keys, evidence_pages(it)))
        per_item[name] = ranks
        _, lo, hi = bootstrap_ci([1.0 if r and r <= 5 else 0.0 for r in ranks])
        lines.append(
            f"| {name} | {n_chunks:,} | {n_long:,}（{n_long / n_chunks:.0%}） | {recall_at(ranks, 4):.0%} "
            f"| {recall_at(ranks, 5):.0%} | {recall_at(doc_ranks, 5):.0%} | {lo:.0%} ~ {hi:.0%} | {seconds}s |"
        )
    return lines, per_item


def end_to_end(store: InMemoryVectorStore, items: list, out_dir: Path) -> list[str]:
    """lc_tutorial 的端到端：检索 4 块 → 教程式拼接 → 和 v0.5 同样的 system prompt（去掉计算器那段）→ 结构化输出。"""
    load_dotenv(PROJECT_ROOT / ".env")
    llm = ChatOpenAI(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        extra_body={"thinking": {"type": "disabled"}},
        max_tokens=2048,
    )
    # 去掉计算器那一段（这里不带工具）。YAML 的 | 块会去掉公共缩进，所以行首没有空格
    system = load_prompt("answer_rag_v1").system
    system = system[: system.index("Calculator:")] + system[system.index("Respond with") :]
    # 坑：system prompt 里有 JSON 示例，LangChain 会把 {...} 当成变量，花括号要写两遍
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system.replace("{", "{{").replace("}", "}}")),
            ("human", "Excerpts:\n\n{context}\n\nQuestion: {question}"),
        ]
    )
    chain = prompt | llm.with_structured_output(Answer, method="function_calling")
    retriever = store.as_retriever()  # 默认 k=4，和教程一样

    def answer(it):
        docs = retriever.invoke(it.question)
        context = "\n\n".join(
            f"[{i}] {d.metadata['doc_name']}, page {d.metadata['page'] + 1}\n{d.page_content}"
            for i, d in enumerate(docs, 1)
        )
        # 解析失败时 LangChain 直接抛异常，不会像 call_structured 那样带着错误反馈重试
        try:
            return chain.invoke({"context": context, "question": it.question}), docs
        except Exception as e:  # noqa: BLE001
            return e, docs

    with ThreadPoolExecutor(max_workers=5) as pool:
        outputs = list(pool.map(answer, items))

    eval_cfg = load_eval_config()
    judge = Judge(
        OpenAICompatClient(eval_cfg.judge.llm, "judge:s6_langchain"),
        load_prompt(eval_cfg.judge.prompt),
        eval_cfg.judge.max_attempts,
    )
    rows, labels = [], []
    for it, (out, docs) in zip(items, outputs, strict=True):
        ans = out if isinstance(out, Answer) else None
        grade = grade_item(it, ans, judge, eval_cfg.relative_tolerance)
        labels.append(grade.label)
        keys = [(d.metadata["doc_name"], d.metadata["page"]) for d in docs]
        rows.append({
            "id": it.id, "grade": grade.label, "tag": grade.tag, "detail": grade.detail,
            "retrieval_hit": first_hit_rank(keys, evidence_pages(it)) is not None,
            "answer": ans.model_dump() if ans else None,
            "error": None if ans else f"{type(out).__name__}: {str(out)[:200]}",
        })  # fmt: skip
    with (out_dir / "results_e2e.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    m = summarize(labels)
    correct = [1.0 if lab == "correct" else 0.0 for lab in labels]
    _, lo, hi = bootstrap_ci(correct)
    lines = [
        "",
        "## 端到端：lc_tutorial vs v0.5（dev 50 题，同一个裁判）",
        "",
        (
            f"lc_tutorial：准确率 {m['accuracy']:.0%}（95% 区间 {lo:.0%} ~ {hi:.0%}），"
            f"答错 {m['hallucination_rate']:.0%}，拒答 {m['abstain_rate']:.0%}，输出不合格 {m['invalid']} 题；"
            f"检索命中 {sum(r['retrieval_hit'] for r in rows) / len(rows):.0%}；裁判花费 ${judge.cost_usd:.4f}"
        ),
    ]
    if V05_RUN.exists():
        v05 = load_results(V05_RUN)
        a = [1.0 if v05[it.id]["grade"] == "correct" else 0.0 for it in items]
        diff, dlo, dhi = paired_bootstrap_diff(a, correct)
        p = mcnemar_exact(
            sum(x > y for x, y in zip(a, correct)), sum(y > x for x, y in zip(a, correct))
        )
        lines.append(
            f"v0.5（L05 第 1 次运行）：准确率 {sum(a) / len(a):.0%}；v0.5 − lc_tutorial = "
            f"{diff * 100:+.0f} 个百分点（区间 {dlo * 100:+.0f} ~ {dhi * 100:+.0f}，McNemar p = {p:.3f}）"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--e2e", action="store_true", help="再用 lc_tutorial 端到端跑一遍 dev（调用模型）"
    )
    args = parser.parse_args()

    items = load_evalset("financebench", "dev")
    out_dir = OUT_ROOT / datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    stores = {}
    for name in ("lc_default", "lc_tutorial", "lc_tuned"):
        print(f"建向量库 {name} …", flush=True)
        stores[name] = build_store(name)
    lines = [
        f"# S6 · LangChain RAG 对照（FinanceBench dev，{len(items)} 题）",
        "",
        "## 检索（不调用大模型）",
        "",
        "对照：我们的 fb_bge_small（L05）R@5 = 48%，文档命中@5 = 92%。",
        "",
    ]
    table, _ = retrieval_table(stores, items)
    lines += table
    # 检索结果先写盘：后面的端到端要调用模型，万一出错，建库花的十几分钟不会白费
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if args.e2e:
        e2e = end_to_end(stores["lc_tutorial"][0], items, out_dir)
        lines += e2e
        (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(e2e))
    print(f"记录目录：{out_dir}")


if __name__ == "__main__":
    main()
