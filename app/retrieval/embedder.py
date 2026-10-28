"""向量化层：本地 bge-small-zh-v1.5（sentence-transformers）。

选型理由：
- bge-small-zh-v1.5 只有 93MB、512 维，中文语义检索效果好，CPU 单条约 7ms，
  不占用显卡（本机 vLLM 已占约 78% 显存），离线可用、零 API 成本；
- 按 bge 官方用法，查询侧加指令前缀、文档侧不加，检索精度明显更高；
- 全部输出做 L2 归一化，配合 Chroma 的 cosine 空间，相似度即点积。
"""
from __future__ import annotations

import threading

from app.config.settings import get_settings

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                settings = get_settings()
                _model = SentenceTransformer(
                    settings.embedding_model,
                    device=settings.embedding_device,
                    local_files_only=True,   # 模型已在本机 HF 缓存，强制离线避免联网卡住
                )
    return _model


def encode_documents(texts: list[str]) -> list[list[float]]:
    """文档侧向量化（不加指令前缀）。"""
    if not texts:
        return []
    settings = get_settings()
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def encode_query(text: str) -> list[float]:
    """查询侧向量化（加 bge 官方检索指令）。"""
    model = _get_model()
    vector = model.encode(
        [BGE_QUERY_INSTRUCTION + text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector[0].tolist()
