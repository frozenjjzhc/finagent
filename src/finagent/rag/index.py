"""向量索引（L05）：给一个问题向量，找出最相似的 k 个文本块。

两种实现，接口相同（search(queries, k) → (scores, ids)）：

  NumpyIndex   手写的暴力检索：问题向量和**每一个**块的向量都算一次点积，取最大的 k 个。
               5 万块 × 384 维，一次查询就是一次矩阵乘法，几毫秒。先手写，才知道 FAISS 在做什么（ADR-7）。
  FaissIndex   FAISS 库。"Flat" 和暴力检索完全一样（结果逐一相同，有测试保证）；
               "HNSW32"、"IVF256,Flat" 是近似检索（ANN）：不和每个向量都比，用少量精度换速度。
               语料到了几百万、上亿块时，暴力检索就太慢了，才需要 ANN（直觉见 L05 讲义 §4.3）。

向量都已归一化，所以点积 = 余弦相似度，越大越相似。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """queries 形状 (m, d)；返回 scores、ids，形状都是 (m, k)，每行按相似度从高到低排好。"""
        ...


class NumpyIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.vectors))
        scores = queries @ self.vectors.T  # (m, n)：每个问题和每个块的相似度
        # argpartition 只保证"前 k 个都比其余的大"，不排序，比完整排序 n 个数快得多；
        # 然后只对这 k 个排序
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        return np.take_along_axis(top_scores, order, axis=1), np.take_along_axis(top, order, axis=1)


class FaissIndex:
    def __init__(self, vectors: np.ndarray, factory: str = "Flat") -> None:
        import faiss

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        # METRIC_INNER_PRODUCT：按点积排序（向量已归一化，也就是余弦相似度）
        self.index = faiss.index_factory(vectors.shape[1], factory, faiss.METRIC_INNER_PRODUCT)
        if not self.index.is_trained:  # IVF 这类索引要先"训练"：用 k-means 把向量分成若干簇
            self.index.train(vectors)
        self.index.add(vectors)

    def __len__(self) -> int:
        return int(self.index.ntotal)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, ids = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return scores, ids


def make_index(vectors: np.ndarray, backend: str = "numpy", factory: str = "Flat") -> VectorIndex:
    return FaissIndex(vectors, factory) if backend == "faiss" else NumpyIndex(vectors)
