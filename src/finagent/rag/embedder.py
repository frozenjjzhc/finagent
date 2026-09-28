"""向量化（L05）：把一段文字变成一个向量，意思相近的文字，向量也相近。

用的是 bi-encoder（双塔）：问题和文本块**分别**过同一个模型，各自得到一个向量，再算相似度。
好处是文本块的向量可以提前算好存起来（建索引），查询时只需要算问题这一个向量。
（L06 的 reranker 是 cross-encoder：把问题和文本块拼在一起过模型，更准，但不能提前算。）

模型：BAAI/bge-small-en-v1.5（3300 万参数，输出 384 维向量，最多读 512 个 token），
    在大量"问题-相关段落"对上用对比学习训练过：让相关的一对向量靠近、不相关的远离。
    本地 GPU 上几十秒就能把整个语料（约 5 万块）算完，不花 API 的钱。

两个细节：
  1. **归一化**：向量都缩放到长度为 1，这样两个向量的点积就等于余弦相似度（-1 ~ 1），
     检索时只需要做矩阵乘法
  2. **查询前缀**：bge 系列训练时，"问题"前面都加了一句
     "Represent this sentence for searching relevant passages: "，文本块前面不加。
     使用时也要照做，否则问题和文本块的向量不在同一个"语境"里，效果会变差
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        device: str = "auto",
        batch_size: int = 64,
        revision: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.device = device
        self.batch_size = batch_size
        self.revision = revision
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """第一次用到时才加载模型：import torch 和加载权重要好几秒，不用向量的命令不该为此等待。"""
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(
                self.model_name, device=device, revision=self.revision
            )
        return self._model

    @property
    def max_tokens(self) -> int:
        """模型最多读多少个 token，超过的部分会被截掉。"""
        return int(self.model.max_seq_length)

    def count_tokens(self, lines: list[str]) -> list[int]:
        """用模型自己的分词器数 token（不含 [CLS]、[SEP] 这类特殊 token）。给切块用。"""
        if not lines:
            return []
        # verbose=False：只是数长度，超过 512 的行不必警告（切块时会把它拆开）
        ids = self.model.tokenizer(lines, add_special_tokens=False, verbose=False)["input_ids"]
        return [len(x) for x in ids]

    def embed_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """文本块 → 向量，形状 (块数, 维度)，已归一化，float32。"""
        return self._encode(texts, show_progress)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """问题 → 向量。问题前面要加上查询前缀（见文件开头的说明）。"""
        return self._encode([self.query_prefix + t for t in texts], show_progress=False)

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)
