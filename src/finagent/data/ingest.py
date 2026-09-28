"""语料（L05）：下载 FinanceBench 的财报 PDF 原文，逐页抽取文字。

为什么用 FinanceBench 自带的 PDF，而不是去 SEC EDGAR 下载 HTML：
    FinanceBench 每道题都标注了证据所在的**文档名和页码**（evidence_page_num）。
    用同一批 PDF、同样的页码，就能自动判断"检索有没有找到正确的那一页"（Recall@k）。
    换成 EDGAR 的 HTML，页码就对不上了。

语料范围：评测集（evalsets/financebench.jsonl）里所有题目用到的 84 份文档，dev 和 test 都包括。
    语料是"图书馆"，不是"答案"：test 题目用到的文档也要在库里，v0.7 跑 test 时才找得到；
    而且它们对 dev 题目来说是干扰项（同一家公司不同年份的报告），让检索更接近真实情况。

页码约定：page 从 0 开始，和 FinanceBench 的 evidence_page_num 一致（已用证据原文核对过，见 L05 讲义 §2.2）。
    给人看时（引用、报告）显示 page + 1，也就是 PDF 阅读器里看到的页码。

输出（都在 data/ 下，已被 .gitignore 忽略；任何人都能用 finagent build-corpus 重新生成）：
    data/raw/financebench_pdfs/<doc_name>.pdf       原始 PDF
    data/processed/financebench_pages.rows.jsonl    每页一行：doc_name, page, text（按行重建表格，默认）
    data/processed/financebench_pages.text.jsonl    同上，PyMuPDF 的默认输出（对比实验用）
    data/processed/financebench_docs.jsonl          每份文档一行：公司、类型、年份、标题、页数、sha256
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from finagent.config import PROJECT_ROOT

FINANCEBENCH_COMMIT = "6e138b2f29670bc416456cd16258f9a48b93b76e"  # 和评测集用同一个版本（L03）
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PDF_DIR = RAW_DIR / "financebench_pdfs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_FILE = PROCESSED_DIR / "financebench_docs.jsonl"

_DOC_TYPE_LABEL = {"10k": "10-K", "10q": "10-Q", "8k": "8-K", "earnings": "earnings release"}


@dataclass(frozen=True)
class Page:
    doc_name: str
    page: int  # 从 0 开始，和 FinanceBench 的 evidence_page_num 一致
    text: str


# ---------------------------------------------------------------- 文档信息


def doc_title(doc_name: str, company: str, doc_type: str) -> str:
    """给人（和模型）看的文档标题，例如：
    3M_2018_10K                     → "3M 10-K 2018"
    3M_2023Q2_10Q                   → "3M 10-Q 2023Q2"
    AMCOR_2022_8K_dated-2022-07-01  → "Amcor 8-K 2022 (dated 2022-07-01)"
    这个标题会加在每个文本块的开头（L05 讲义 §3.3）：否则一段"Net sales increased 5%"
    根本看不出是哪家公司、哪一年的。
    """
    period = re.search(r"_(\d{4}(?:Q\d)?)_", doc_name)
    dated = re.search(r"dated[-_](\d{4}-\d{2}-\d{2})", doc_name)
    parts = [company, _DOC_TYPE_LABEL.get(doc_type.lower(), doc_type)]
    if period:
        parts.append(period.group(1))
    title = " ".join(parts)
    return f"{title} (dated {dated.group(1)})" if dated else title


def corpus_doc_names() -> list[str]:
    """评测集用到的全部文档（dev + test）。"""
    from finagent.eval.datasets import load_evalset

    names = {e["doc_name"] for it in load_evalset("financebench") for e in it.evidence}
    return sorted(names)


# ---------------------------------------------------------------- 下载与抽取


def download_pdf(doc_name: str) -> Path:
    """从固定 commit 下载一份 PDF（已有就跳过）。固定 commit 的地址，内容永远不会变。"""
    path = PDF_DIR / f"{doc_name}.pdf"
    if not path.exists():
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://raw.githubusercontent.com/patronus-ai/financebench/"
            f"{FINANCEBENCH_COMMIT}/pdfs/{doc_name}.pdf"
        )
        tmp = path.with_suffix(".part")  # 先写临时文件，下载完整了再改名：中断时不会留下半个 PDF
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def clean_text(text: str) -> str:
    """整理一页的文字：每行去掉首尾空白、行内的连续空白合成一个，去掉空行。

    **保留换行**：PDF 里的表格，一行文字往往就是表格的一行（或一个单元格），
    切块时按行切（rag/chunker.py），才不会把一个数字切成两半。
    """
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def rows_text(words: list[tuple], tolerance: float = 3.0) -> str:
    """按坐标把同一行的词拼回一行（L05 的表格处理）。

    PDF 里没有"表格"这个概念，只有一个个带坐标的文字。PyMuPDF 默认按文本块输出，
    财务报表里每个单元格都是一个块，结果变成"一个单元格一行"：
        Depreciation and amortization
        1,488
        1,544
        1,474
    模型很难看出 1,544 属于哪一年。这里改为：纵坐标（行的中线）相差不超过 tolerance 磅的词
    算作同一行，再按横坐标从左到右排好：
        Depreciation and amortization 1,488 1,544 1,474
    表头 "(Millions) 2018 2017 2016" 也会变成一行，数字和年份的对应关系就清楚了。

    words 是 page.get_text("words") 的结果：每项是 (x0, y0, x1, y1, 词, 块号, 行号, 词号)。
    局限：双栏排版会把左右两栏的同一行拼在一起；一个单元格折成两行时也会被拆开。
    """
    ordered = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[tuple]] = []
    center = 0.0
    for w in ordered:
        y = (w[1] + w[3]) / 2
        if rows and abs(y - center) <= tolerance:
            rows[-1].append(w)
            center += (y - center) / len(rows[-1])  # 这一行的中线取平均，容忍轻微的上下错位
        else:
            rows.append([w])
            center = y
    return "\n".join(" ".join(w[4] for w in sorted(row, key=lambda w: w[0])) for row in rows)


Layout = Literal["rows", "text"]


def extract_pages(pdf_path: Path, layout: Layout = "rows") -> list[Page]:
    """用 PyMuPDF 逐页抽取文字。扫描版 PDF（图片）抽不出文字，那一页的 text 为空。

    layout="rows"：按坐标重建表格行（默认，见 rows_text）；
    layout="text"：PyMuPDF 的默认输出（一个文本块一行），留着做对比实验。
    """
    import pymupdf  # 放在函数里导入：只有建语料时才需要它

    doc_name = pdf_path.stem
    with pymupdf.open(pdf_path) as pdf:
        return [
            Page(
                doc_name,
                i,
                clean_text(
                    rows_text(page.get_text("words")) if layout == "rows" else page.get_text("text")
                ),
            )
            for i, page in enumerate(pdf)
        ]


def pages_file(layout: Layout = "rows") -> Path:
    return PROCESSED_DIR / f"financebench_pages.{layout}.jsonl"


def build_corpus(doc_names: Iterable[str] | None = None, layout: Layout = "rows") -> dict:
    """下载 + 抽取 + 写文件，返回统计信息。"""
    names = list(doc_names) if doc_names is not None else corpus_doc_names()
    info = {
        d["doc_name"]: d
        for d in (
            json.loads(line)
            for line in (RAW_DIR / "financebench_document_information.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    n_pages = n_empty = 0
    with (
        pages_file(layout).open("w", encoding="utf-8", newline="\n") as pages_out,
        DOCS_FILE.open("w", encoding="utf-8", newline="\n") as docs_out,
    ):
        for name in names:
            pdf = download_pdf(name)
            pages = extract_pages(pdf, layout)
            for p in pages:
                pages_out.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
            empty = sum(1 for p in pages if not p.text)
            n_pages, n_empty = n_pages + len(pages), n_empty + empty
            meta = info[name]
            docs_out.write(
                json.dumps(
                    {
                        "doc_name": name,
                        "company": meta["company"],
                        "doc_type": meta["doc_type"],
                        "doc_period": meta["doc_period"],
                        "title": doc_title(name, meta["company"], meta["doc_type"]),
                        "pages": len(pages),
                        "empty_pages": empty,
                        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {"docs": len(names), "pages": n_pages, "empty_pages": n_empty}


# ---------------------------------------------------------------- 读取


def load_pages(layout: Layout = "rows") -> list[Page]:
    path = pages_file(layout)
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}：先运行 finagent build-corpus --layout {layout}")
    with path.open(encoding="utf-8") as f:
        return [Page(**json.loads(line)) for line in f if line.strip()]


def load_docs(path: Path = DOCS_FILE) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}：先运行 finagent build-corpus")
    with path.open(encoding="utf-8") as f:
        return {d["doc_name"]: d for d in (json.loads(line) for line in f if line.strip())}
