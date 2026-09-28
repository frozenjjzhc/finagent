"""切块（L05）：把一页一页的财报切成适合检索的小块。

为什么要切：
  - 一份 10-K 有 100~300 页、5~10 万个 token，整份塞进 prompt 又贵又慢，模型还容易漏掉中间的内容
  - 向量模型有输入长度上限（bge-small 是 512 个 token），超过的部分会被**悄悄截掉**
  - 一整页往往讲好几件事，把它压成一个向量，每件事都被"平均"掉了，检索时反而不容易命中

切多大：本课默认每块最多 256 个 token、相邻两块重叠 32 个 token（L06 会扫一遍不同的大小）。
    **token 数按向量模型自己的分词器来数**，而不是按单词或字符估：同样 100 个词，
    满是数字的表格会被切成多得多的 token。

怎么切（三个设计选择）：
  1. **按行累加**，不在行中间切：一行往往是表格的一行，切在中间，数字就和它的行名分家了
  2. **不跨页**：每块只来自一页，这样每块都有确定的页码，才能和 FinanceBench 的证据页比较（Recall@k），
     引用时也能说清"第几页"
  3. **每块开头加上文档标题**（如 "3M 10-K 2018"）：一块文字"Net sales increased 5%"本身
     看不出是哪家公司、哪一年；84 份文档里同一家公司常有好几年的报告，没有标题就会张冠李戴
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from finagent.data.ingest import Page

# 数 token 的函数：输入一批文本行，返回每行的 token 数（批量处理，比一行一行调用快得多）
TokenCounter = Callable[[list[str]], list[int]]


def count_words(lines: list[str]) -> list[int]:
    """默认的计数方式：按空格数单词。测试用；建索引时换成向量模型的分词器。"""
    return [len(line.split()) for line in lines]


@dataclass(frozen=True)
class Chunk:
    id: str  # 例如 "3M_2018_10K:p59:c1"：文档名 + 页码 + 这一页的第几块
    doc_name: str
    page: int  # 从 0 开始，和 FinanceBench 的 evidence_page_num 一致
    title: str  # 文档标题，例如 "3M 10-K 2018"
    text: str

    def embed_text(self, with_title: bool = True) -> str:
        """送进向量模型的文字。with_title=False 用来做"不加标题"的对比实验。"""
        return f"{self.title}\n{self.text}" if with_title else self.text


def chunk_page(
    page: Page,
    title: str,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
    count: TokenCounter = count_words,
) -> list[Chunk]:
    """把一页切成若干块：一行一行往里装，装不下了就开一块新的，新块开头重复上一块末尾的几行。

    例子（max_tokens=6、overlap_tokens=2，每行的 token 数写在括号里）：
        A(3) B(2) C(2) D(4)
        → 块 1：A B    再装 C 就是 3+2+2=7 > 6，开新块
        → 块 2：B C    新块先带上块 1 末尾不超过 2 个 token 的行（B），再装 C
        → 块 3：C D    B C 再装 D 是 8 > 6；新块带上 C（2），再装 D（4），正好 6
    重叠让落在两块交界处的内容，至少完整地出现在其中一块里。
    """
    lines = [line for line in page.text.splitlines() if line.strip()]
    units: list[tuple[str, int]] = []
    for line, n in zip(lines, count(lines), strict=True):
        # 极少数超长的行（比如一大段没有换行的文字）要先拆开
        units += [(line, n)] if n <= max_tokens else _split_long(line, max_tokens, count)

    groups: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    size = 0
    for line, n in units:
        if current and size + n > max_tokens:
            groups.append(current)
            # 新块从上一块末尾的几行开始（总共不超过 overlap_tokens）
            keep: list[tuple[str, int]] = []
            kept = 0
            for prev_line, prev_n in reversed(current):
                if kept + prev_n > overlap_tokens:
                    break
                keep.insert(0, (prev_line, prev_n))
                kept += prev_n
            if kept + n > max_tokens:  # 带上重叠部分就装不下这一行了：不要重叠
                keep, kept = [], 0
            current, size = keep, kept
        current.append((line, n))
        size += n
    if current:
        groups.append(current)

    return [
        Chunk(
            id=f"{page.doc_name}:p{page.page}:c{i}",
            doc_name=page.doc_name,
            page=page.page,
            title=title,
            text="\n".join(line for line, _ in group),
        )
        for i, group in enumerate(groups)
    ]


def _split_long(line: str, max_tokens: int, count: TokenCounter) -> list[tuple[str, int]]:
    """把超长的一行按词对半拆，拆到每段都不超过 max_tokens 为止（递归）。"""
    words = line.split()
    if len(words) <= 1:  # 一个"词"本身就超长（极少见），没法再拆，只能让模型截断
        return [(line, count([line])[0])]
    halves = [" ".join(words[: len(words) // 2]), " ".join(words[len(words) // 2 :])]
    out: list[tuple[str, int]] = []
    for piece, n in zip(halves, count(halves), strict=True):
        out += [(piece, n)] if n <= max_tokens else _split_long(piece, max_tokens, count)
    return out


def chunk_pages(
    pages: Iterable[Page],
    titles: dict[str, str],
    max_tokens: int = 256,
    overlap_tokens: int = 32,
    count: TokenCounter = count_words,
) -> list[Chunk]:
    """切整个语料。titles：文档名 → 标题。空白页（扫描版 PDF 抽不出文字）没有块。"""
    chunks: list[Chunk] = []
    for page in pages:
        if page.text:
            chunks += chunk_page(page, titles[page.doc_name], max_tokens, overlap_tokens, count)
    return chunks
