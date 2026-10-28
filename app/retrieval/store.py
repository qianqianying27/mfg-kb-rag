"""向量库层：ChromaDB 持久化（零部署成本，本地单文件目录）。

为什么选 Chroma 而不是 FAISS：
- Chroma 自带持久化 + metadata 过滤 + 增删改，适合"文档会不断新增"的知识库；
- FAISS 只是索引库，metadata 与持久化要自己维护，MVP 阶段成本更高。
"""
from __future__ import annotations

from dataclasses import dataclass

import chromadb

from app.config.settings import get_settings
from app.ingestion.splitter import Chunk


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    text: str
    metadata: dict
    similarity: float | None = None   # 1 - cosine_distance

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def page(self) -> int | None:
        page = self.metadata.get("page")
        return int(page) if page else None

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", ""))

    def cite(self) -> str:
        """生成引用标注：文件名 + 页码/小节。"""
        where = f"p.{self.page}" if self.page else f"§{self.section}"
        return f"{self.source} {where}"


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self._name = settings.collection_name
        self._collection = self._create_collection()
        self._version = 0        # 每次写入自增，供 BM25 索引判断是否需要重建

    def _create_collection(self):
        # Chroma 1.x 用 configuration 声明 cosine；旧版本用 metadata，这里兼容两种写法
        try:
            return self._client.get_or_create_collection(
                name=self._name,
                configuration={"hnsw": {"space": "cosine"}},
            )
        except Exception:
            return self._client.get_or_create_collection(
                name=self._name,
                metadata={"hnsw:space": "cosine"},
            )

    # ---------- 写入 ----------
    def add(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 64) -> int:
        if not chunks:
            return 0
        total = 0
        for start in range(0, len(chunks), batch_size):
            part = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[c.chunk_id for c in part],
                documents=[c.text for c in part],
                embeddings=embeddings[start : start + batch_size],
                metadatas=[c.to_metadata() for c in part],
            )
            total += len(part)
        self._version += 1
        return total

    # ---------- 查询 ----------
    def query(self, embedding: list[float], top_k: int) -> list[SearchHit]:
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append(
                SearchHit(
                    chunk_id=cid,
                    text=doc,
                    metadata=dict(meta or {}),
                    similarity=1.0 - float(dist),   # cosine distance → 相似度
                )
            )
        return hits

    # ---------- 统计与维护 ----------
    def count(self) -> int:
        return self._collection.count()

    @property
    def version(self) -> int:
        return self._version

    def all_documents(self) -> list[SearchHit]:
        """取出全部分块，供 BM25 构建倒排。"""
        if self.count() == 0:
            return []
        res = self._collection.get(include=["documents", "metadatas"])
        return [
            SearchHit(chunk_id=cid, text=doc, metadata=dict(meta or {}))
            for cid, doc, meta in zip(
                res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
            )
        ]

    def has_source(self, filename: str) -> bool:
        return bool(self._collection.get(where={"source": filename}, limit=1)["ids"])

    def delete_source(self, filename: str) -> int:
        res = self._collection.get(where={"source": filename})
        ids = res.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            self._version += 1
        return len(ids)

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._create_collection()
        self._version += 1
