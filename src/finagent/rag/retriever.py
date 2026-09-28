"""检索器（L05）：建索引、查索引，以及把检索结果整理成交给模型的上下文。

整条 RAG 管线：
    ingest   PDF → 每页的文字                          data/ingest.py
    chunk    每页 → 若干文本块（带文档标题）              rag/chunker.py
    embed    每个文本块 → 一个向量                       rag/embedder.py
    index    所有向量存起来，支持"找最相似的 k 个"        rag/index.py
    retrieve 问题 → 向量 → 最相似的 k 个文本块           本文件 Retriever
    generate 文本块（或它们所在的整页）→ prompt → 模型回答  本文件 ContextBuilder + pipeline/answer.py

前四步是离线的（build_index，几分钟做一次）；后两步是每个问题都要做的（在线）。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path

import numpy as np

from finagent.config import PROJECT_ROOT, IndexConfig, RetrievalConfig, load_index_config
from finagent.data.ingest import load_docs, load_pages, pages_file
from finagent.rag.chunker import Chunk, chunk_pages
from finagent.rag.embedder import Embedder
from finagent.rag.index import VectorIndex, make_index

INDEX_ROOT = PROJECT_ROOT / "data" / "index"


def make_embedder(cfg: IndexConfig) -> Embedder:
    e = cfg.embedder
    return Embedder(e.model, e.query_prefix, e.device, e.batch_size, e.revision)


# ---------------------------------------------------------------- 建索引（离线）


def build_index(cfg: IndexConfig, show_progress: bool = True) -> Path:
    """切块 + 向量化 + 存盘。索引是可以随时重建的派生数据，所以直接覆盖旧的。"""
    t0 = time.perf_counter()
    pages = load_pages(cfg.corpus_layout)
    titles = {name: d["title"] for name, d in load_docs().items()}
    embedder = make_embedder(cfg)
    chunks = chunk_pages(
        pages, titles, cfg.chunk.max_tokens, cfg.chunk.overlap_tokens, embedder.count_tokens
    )
    texts = [c.embed_text(cfg.chunk.title) for c in chunks]
    vectors = embedder.embed_passages(texts, show_progress)

    # 检查有没有块超过模型的长度上限（会被悄悄截断）。+2 是 [CLS] 和 [SEP]
    lengths = np.array(embedder.count_tokens(texts)) + 2
    out = INDEX_ROOT / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vectors)
    with (out / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    meta = {
        "config": cfg.model_dump(),
        "n_chunks": len(chunks),
        "dim": int(vectors.shape[1]),
        "tokens": {
            "mean": round(float(lengths.mean()), 1),
            "max": int(lengths.max()),
            "truncated": int((lengths > embedder.max_tokens).sum()),
        },
        "corpus_sha256": hashlib.sha256(pages_file(cfg.corpus_layout).read_bytes()).hexdigest(),
        "build_seconds": round(time.perf_counter() - t0, 1),
        "packages": {p: metadata.version(p) for p in ("sentence-transformers", "torch", "numpy")},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------- 查索引（在线）


@dataclass
class Hit:
    chunk: Chunk
    score: float  # 余弦相似度
    rank: int  # 从 1 开始


class Retriever:
    def __init__(
        self, cfg: IndexConfig, chunks: list[Chunk], index: VectorIndex, embedder: Embedder
    ) -> None:
        self.cfg, self.chunks, self.index, self.embedder = cfg, chunks, index, embedder

    @classmethod
    def load(cls, name: str) -> Retriever:
        cfg = load_index_config(name)
        folder = INDEX_ROOT / name
        if not (folder / "meta.json").exists():
            raise FileNotFoundError(
                f"索引 {name} 还没建：先运行 finagent build-index --index {name}"
            )
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        if meta["config"] != cfg.model_dump():
            # 配置改了、索引却是旧的：检索结果和配置对不上，而且没人会发现。宁可报错
            raise ValueError(f"configs/index/{name}.yaml 在建索引之后改过：请重新 build-index")
        with (folder / "chunks.jsonl").open(encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        vectors = np.load(folder / "vectors.npy")
        index = make_index(vectors, cfg.backend, cfg.faiss_factory)
        return cls(cfg, chunks, index, make_embedder(cfg))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: list[str], k: int = 5) -> list[list[Hit]]:
        """一次查多个问题：问题一起算向量（批量），比一个一个算快得多。"""
        if not queries:
            return []
        scores, ids = self.index.search(self.embedder.embed_queries(queries), k)
        return [
            [
                Hit(self.chunks[int(i)], float(s), rank)
                for rank, (s, i) in enumerate(zip(row_s, row_i, strict=True), 1)
                if i >= 0  # FAISS 结果不足 k 个时用 -1 占位
            ]
            for row_s, row_i in zip(scores, ids, strict=True)
        ]


# ---------------------------------------------------------------- 整理成上下文


@dataclass
class Source:
    """放进 prompt 的一段材料，编号 [ref]。模型在 sources 字段里用这个编号引用它。"""

    ref: int
    doc_name: str
    page: int
    title: str
    text: str
    score: float | None = None  # 检索的相似度；oracle 时为 None
    chunk_ids: list[str] = field(default_factory=list)  # 是哪些检索到的块把它带进来的

    def header(self) -> str:
        return f"[{self.ref}] {self.title}, page {self.page + 1}"  # 给人和模型看的页码从 1 开始


def format_context(sources: list[Source]) -> str:
    """[1] 3M 10-K 2018, page 60
    <正文>

    [2] ...
    """
    return "\n\n".join(f"{s.header()}\n{s.text}" for s in sources)


class ContextBuilder:
    """按系统配置的 retrieval 段，为问题准备上下文。三种方式：
    - 检索 + chunk：交给模型的就是检索到的 k 个文本块
    - 检索 + page：交给模型的是这些块所在的整页（去重，保持排名顺序）。
      用小块检索（准），用整页回答（全）：一张表被切成两块时，表头年份和数字都还在同一页上。
      这叫 small-to-big（也叫 parent document retrieval）
    - oracle：不检索，直接用题目标注的证据页（只在评测时可用，用来测"阅读器"的上限）
    """

    def __init__(
        self,
        cfg: RetrievalConfig,
        retriever: Retriever | None = None,
        pages: dict[tuple[str, int], str] | None = None,
        titles: dict[str, str] | None = None,
    ) -> None:
        """retriever、pages、titles 默认从磁盘加载；测试时可以直接传入，不需要真实的语料和索引。"""
        self.cfg = cfg
        self.retriever = retriever
        if retriever is None and not cfg.oracle:
            assert cfg.index is not None
            self.retriever = Retriever.load(cfg.index)
        self._pages = pages
        self._titles = titles

    @property
    def pages(self) -> dict[tuple[str, int], str]:
        if self._pages is None:
            layout = self.retriever.cfg.corpus_layout if self.retriever else "rows"
            self._pages = {(p.doc_name, p.page): p.text for p in load_pages(layout)}
        return self._pages

    @property
    def titles(self) -> dict[str, str]:
        if self._titles is None:
            self._titles = {name: d["title"] for name, d in load_docs().items()}
        return self._titles

    def retrieve(self, questions: list[str]) -> list[list[Source]]:
        assert self.retriever is not None, "oracle 模式没有检索器"
        results = self.retriever.search_many(questions, self.cfg.top_k)
        if self.cfg.context == "chunk":
            return [
                [
                    Source(h.rank, h.chunk.doc_name, h.chunk.page, h.chunk.title, h.chunk.text,
                           h.score, [h.chunk.id])
                    for h in hits
                ]
                for hits in results
            ]  # fmt: skip
        return [self._expand_to_pages(hits) for hits in results]

    def _expand_to_pages(self, hits: list[Hit]) -> list[Source]:
        sources: dict[tuple[str, int], Source] = {}
        for h in hits:
            key = (h.chunk.doc_name, h.chunk.page)
            if key in sources:  # 同一页上的第二个块：页已经在了，只记下是哪个块
                sources[key].chunk_ids.append(h.chunk.id)
                continue
            sources[key] = Source(
                len(sources) + 1, key[0], key[1], h.chunk.title, self.pages[key], h.score,
                [h.chunk.id],
            )  # fmt: skip
        return list(sources.values())

    def oracle(self, evidence: list[list[dict]]) -> list[list[Source]]:
        out = []
        for ev in evidence:
            keys = list(dict.fromkeys((e["doc_name"], e["page"]) for e in ev))  # 去重、保持顺序
            out.append(
                [
                    Source(i, doc, page, self.titles[doc], self.pages[(doc, page)])
                    for i, (doc, page) in enumerate(keys, 1)
                ]
            )
        return out
