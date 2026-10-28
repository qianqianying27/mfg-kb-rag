"""混合检索：向量召回 + BM25 关键词召回，用 RRF 融合排序。

为什么必须加 BM25：
制造业问答里大量查询带型号/编号（"XYZ-320 轴承更换周期""E-102 报警"）。
向量模型把型号当成普通词，容易泛化成"某个轴承"，导致召回失败；
BM25 对稀有词（型号、错误码）极其敏感，两者互补。

RRF（Reciprocal Rank Fusion）只用排名不用分数，避免余弦相似度与 BM25 分数量纲
不一致的问题：score = Σ 1 / (k + rank)，k 取 60。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from app.config.settings import get_settings
from app.retrieval import embedder
from app.retrieval.store import SearchHit, VectorStore

try:  # jieba 只用于中文分词，缺失时退化为字符二元组，保证可运行
    import jieba
except ImportError:  # pragma: no cover
    jieba = None


@dataclass(slots=True)
class Retrieved(SearchHit):
    vector_sim: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    score: float = 0.0


class _BM25Cache:
    """BM25 索引缓存：语料变化（条数变化）时自动重建。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25 = None
        self._ids: list[str] = []
        self._hits: dict[str, SearchHit] = {}
        self._count = -1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if jieba is not None:
            return [t for t in jieba.lcut(text) if t.strip()]
        return [text[i : i + 2] for i in range(len(text) - 1)]

    def get(self, store: VectorStore):
        count = store.count()
        if self._bm25 is not None and count == self._count:
            return self._bm25, self._ids, self._hits
        with self._lock:
            if self._bm25 is not None and count == self._count:
                return self._bm25, self._ids, self._hits
            docs = store.all_documents()
            if not docs:
                return None, [], {}
            from rank_bm25 import BM25Okapi

            corpus = [self._tokenize(d.text) for d in docs]
            self._bm25 = BM25Okapi(corpus)
            self._ids = [d.chunk_id for d in docs]
            self._hits = {d.chunk_id: d for d in docs}
            self._count = count
            return self._bm25, self._ids, self._hits

    def search(self, store: VectorStore, query: str, top_k: int) -> list[SearchHit]:
        bm25, ids, hits = self.get(store)
        if bm25 is None:
            return []
        scores = bm25.get_scores(self._tokenize(query))
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [hits[ids[i]] for i in order if scores[i] > 0]


def _rrf(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


class HybridRetriever:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore()
        self._bm25 = _BM25Cache()

    def retrieve(self, query: str, top_k: int | None = None) -> list[Retrieved]:
        settings = get_settings()
        top_k = top_k or settings.top_k

        if self.store.count() == 0:
            return []

        q_vec = embedder.encode_query(query)
        vector_hits = self.store.query(q_vec, settings.vector_top_k)

        if not settings.enable_hybrid:
            merged = [
                Retrieved(
                    chunk_id=h.chunk_id,
                    text=h.text,
                    metadata=h.metadata,
                    similarity=h.similarity,
                    vector_sim=h.similarity,
                    vector_rank=i + 1,
                    score=_rrf(i + 1, settings.rrf_k),
                )
                for i, h in enumerate(vector_hits)
            ]
            return merged[:top_k]

        bm25_hits = self._bm25.search(self.store, query, settings.bm25_top_k)

        pool: dict[str, Retrieved] = {}
        for rank, hit in enumerate(vector_hits, start=1):
            item = pool.setdefault(
                hit.chunk_id,
                Retrieved(chunk_id=hit.chunk_id, text=hit.text, metadata=hit.metadata),
            )
            item.vector_sim = hit.similarity
            item.vector_rank = rank
            item.score += _rrf(rank, settings.rrf_k)

        for rank, hit in enumerate(bm25_hits, start=1):
            item = pool.setdefault(
                hit.chunk_id,
                Retrieved(chunk_id=hit.chunk_id, text=hit.text, metadata=hit.metadata),
            )
            item.bm25_rank = rank
            item.score += _rrf(rank, settings.rrf_k)

        ranked = sorted(
            pool.values(),
            key=lambda r: (r.score, r.vector_sim or 0.0),
            reverse=True,
        )
        return ranked[:top_k]

    def warmup(self) -> None:
        """预热：加载 embedding 模型并构建 BM25 索引。

        两个动作都是首问的主要耗时来源（bge 权重加载约 2 s、BM25 建索引与 jieba
        首次分词约 1 s），放在服务启动阶段做，避免首个真实用户请求等 30 s。
        """
        embedder.encode_query("预热")
        if self.store.count() > 0:
            self._bm25.get(self.store)

    def best_similarity(self, hits: list[Retrieved]) -> float:
        """以向量相似度作为"检索置信度"（BM25 分数无量纲，不适合做阈值）。"""
        sims = [h.vector_sim for h in hits if h.vector_sim is not None]
        return max(sims) if sims else 0.0
