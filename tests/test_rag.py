"""RAG 的测试（L05）：切块、索引、检索、上下文、检索指标。

不需要真实的语料、索引和向量模型：用几页手写的"财报"和一个假的向量模型（词袋），
一秒跑完，而且结果完全确定。
"""

import hashlib

import numpy as np
import pytest

from finagent.config import IndexConfig, RetrievalConfig
from finagent.data.ingest import Page, clean_text, doc_title, rows_text
from finagent.eval.retrieval import first_doc_rank, first_hit_rank, mrr, recall_at
from finagent.rag.chunker import Chunk, chunk_page, chunk_pages, count_words
from finagent.rag.index import FaissIndex, NumpyIndex
from finagent.rag.retriever import ContextBuilder, Retriever, format_context

# ---------------------------------------------------------------- 语料


@pytest.mark.parametrize(
    ("doc_name", "company", "doc_type", "title"),
    [
        ("3M_2018_10K", "3M", "10k", "3M 10-K 2018"),
        ("3M_2023Q2_10Q", "3M", "10q", "3M 10-Q 2023Q2"),
        ("AMCOR_2022_8K_dated-2022-07-01", "Amcor", "8k", "Amcor 8-K 2022 (dated 2022-07-01)"),
        ("FOOTLOCKER_2022_8K_dated_2022-08-19", "Foot Locker", "8k", "Foot Locker 8-K 2022 (dated 2022-08-19)"),
        ("PEPSICO_2023Q1_EARNINGS", "PepsiCo", "Earnings", "PepsiCo earnings release 2023Q1"),
    ],
)  # fmt: skip
def test_doc_title(doc_name, company, doc_type, title) -> None:
    assert doc_title(doc_name, company, doc_type) == title


def test_clean_text_keeps_lines() -> None:
    assert clean_text("  Net  sales \n\n   5,829  \n") == "Net sales\n5,829"


def test_rows_text_rebuilds_table_rows() -> None:
    # 单元格是分开的文字块，纵坐标相同（允许 1~2 磅的错位）→ 拼回同一行，按横坐标排序
    words = [
        (300, 100.5, 330, 110, "1,544", 0, 0, 0),
        (10, 100, 60, 110, "Depreciation", 1, 0, 0),
        (240, 99.8, 270, 110, "1,488", 2, 0, 0),
        (360, 101, 390, 111, "1,474", 3, 0, 0),
        (10, 120, 60, 130, "Inventories", 4, 0, 0),
    ]
    assert rows_text(words) == "Depreciation 1,488 1,544 1,474\nInventories"


# ---------------------------------------------------------------- 切块


def fake_counter(lines: list[str]) -> list[int]:
    """每行的 token 数就写在行尾：'A 3' → 3。方便精确地构造切块的例子。"""
    return [int(line.split()[-1]) for line in lines]


def test_chunk_packing_and_overlap() -> None:
    # 和 chunk_page 文档里的例子一样：A(3) B(2) C(2) D(4)，max 6，overlap 2
    page = Page("DOC", 7, "A 3\nB 2\nC 2\nD 4")
    chunks = chunk_page(page, "T", max_tokens=6, overlap_tokens=2, count=fake_counter)
    assert [c.text.replace("\n", " ") for c in chunks] == ["A 3 B 2", "B 2 C 2", "C 2 D 4"]
    assert [c.id for c in chunks] == ["DOC:p7:c0", "DOC:p7:c1", "DOC:p7:c2"]
    assert all(c.page == 7 and c.title == "T" for c in chunks)


def test_chunk_never_crosses_pages_and_skips_empty() -> None:
    pages = [Page("DOC", 0, "a b c"), Page("DOC", 1, ""), Page("DOC", 2, "d e")]
    chunks = chunk_pages(pages, {"DOC": "Doc"}, max_tokens=100, overlap_tokens=0)
    assert [(c.page, c.text) for c in chunks] == [(0, "a b c"), (2, "d e")]


def test_long_line_is_split_until_it_fits() -> None:
    page = Page("DOC", 0, " ".join(f"w{i}" for i in range(50)))
    chunks = chunk_page(page, "T", max_tokens=8, overlap_tokens=0, count=count_words)
    assert all(len(c.text.split()) <= 8 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == [f"w{i}" for i in range(50)]  # 一个词都没丢


def test_embed_text_title_switch() -> None:
    c = Chunk("id", "DOC", 0, "3M 10-K 2018", "Net sales")
    assert c.embed_text() == "3M 10-K 2018\nNet sales"
    assert c.embed_text(with_title=False) == "Net sales"


# ---------------------------------------------------------------- 索引


def random_unit_vectors(n: int, d: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_numpy_index_matches_full_sort() -> None:
    vectors, queries = random_unit_vectors(500, 16, 0), random_unit_vectors(3, 16, 1)
    scores, ids = NumpyIndex(vectors).search(queries, 10)
    expected = np.argsort(-(queries @ vectors.T), axis=1)[:, :10]
    assert (ids == expected).all()
    assert (np.diff(scores, axis=1) <= 0).all()  # 从高到低排好


def test_faiss_flat_equals_numpy() -> None:
    pytest.importorskip("faiss")
    vectors, queries = random_unit_vectors(500, 16, 0), random_unit_vectors(3, 16, 1)
    _, ids_np = NumpyIndex(vectors).search(queries, 10)
    _, ids_faiss = FaissIndex(vectors, "Flat").search(queries, 10)
    assert (ids_np == ids_faiss).all()  # 精确检索：结果逐一相同


# ---------------------------------------------------------------- 检索器（假的向量模型）


class BagOfWordsEmbedder:
    """假的向量模型：每个词哈希到一个维度上计数，再归一化。只认字面上相同的词，但结果完全确定。"""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1.0)

    def embed_passages(self, texts, show_progress=False):
        return np.stack([self._vec(t) for t in texts])

    def embed_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])


CHUNKS = [
    Chunk("A:p0:c0", "A", 0, "Acme 10-K 2018", "capital expenditures 1577 purchases of equipment"),
    Chunk("A:p0:c1", "A", 0, "Acme 10-K 2018", "dividends paid to shareholders 3193"),
    Chunk("A:p5:c0", "A", 5, "Acme 10-K 2018", "revenue by segment industrial safety"),
    Chunk("B:p2:c0", "B", 2, "Beta 10-K 2022", "capital expenditures 900 purchases of equipment"),
]
PAGES = {("A", 0): "FULL PAGE A0", ("A", 5): "FULL PAGE A5", ("B", 2): "FULL PAGE B2"}


def make_retriever() -> Retriever:
    embedder = BagOfWordsEmbedder()
    vectors = embedder.embed_passages([c.embed_text() for c in CHUNKS])
    return Retriever(IndexConfig(name="t"), CHUNKS, NumpyIndex(vectors), embedder)


def test_retriever_finds_the_matching_chunk() -> None:
    hits = make_retriever().search("dividends paid", k=2)
    assert hits[0].chunk.id == "A:p0:c1" and hits[0].rank == 1
    assert hits[0].score >= hits[1].score


def test_title_helps_tell_companies_apart() -> None:
    # 两块正文几乎一样（都是 capex），只有标题能区分公司：问题里提到 Beta，就该排 Beta 在前
    hits = make_retriever().search("Beta capital expenditures", k=2)
    assert hits[0].chunk.doc_name == "B"


def test_context_page_mode_dedupes_pages() -> None:
    cfg = RetrievalConfig(index="t", top_k=3, context="page")
    builder = ContextBuilder(cfg, make_retriever(), pages=PAGES, titles={})
    # "capital expenditures purchases equipment dividends" 会命中 A:p0:c0、A:p0:c1（同一页）、B:p2:c0
    sources = builder.retrieve(["capital expenditures purchases equipment dividends"])[0]
    keys = [(s.doc_name, s.page) for s in sources]
    assert len(keys) == len(set(keys))  # 同一页只出现一次
    first = sources[0]
    assert first.text.startswith("FULL PAGE")  # 给模型的是整页
    assert [s.ref for s in sources] == list(range(1, len(sources) + 1))  # 编号连续


def test_context_chunk_mode_keeps_chunks() -> None:
    cfg = RetrievalConfig(index="t", top_k=2, context="chunk")
    sources = ContextBuilder(cfg, make_retriever(), pages=PAGES).retrieve(["dividends paid"])[0]
    assert sources[0].text == "dividends paid to shareholders 3193"
    assert sources[0].chunk_ids == ["A:p0:c1"]


def test_oracle_uses_evidence_pages() -> None:
    cfg = RetrievalConfig(oracle=True)
    builder = ContextBuilder(cfg, pages=PAGES, titles={"A": "Acme 10-K 2018"})
    evidence = [
        [{"doc_name": "A", "page": 5}, {"doc_name": "A", "page": 5}]
    ]  # 重复的证据页只给一次
    sources = builder.oracle(evidence)[0]
    assert [(s.doc_name, s.page, s.text) for s in sources] == [("A", 5, "FULL PAGE A5")]
    assert (
        format_context(sources) == "[1] Acme 10-K 2018, page 6\nFULL PAGE A5"
    )  # 显示的页码从 1 开始


def test_retrieval_config_requires_index_unless_oracle() -> None:
    with pytest.raises(ValueError, match="index"):
        RetrievalConfig()


# ---------------------------------------------------------------- 检索指标


def test_retrieval_metrics_example() -> None:
    # eval/retrieval.py 文档里的例子：证据在 (3M_2018_10K, 59)
    gold = {("3M_2018_10K", 59)}
    retrieved = [("3M_2022_10K", 40), ("3M_2018_10K", 58), ("3M_2018_10K", 59)]
    assert first_hit_rank(retrieved, gold) == 3
    assert first_doc_rank(retrieved, gold) == 2
    ranks = [3, 1, None]
    assert recall_at(ranks, 1) == pytest.approx(1 / 3)
    assert recall_at(ranks, 3) == pytest.approx(2 / 3)
    assert mrr(ranks) == pytest.approx((1 / 3 + 1) / 3)
