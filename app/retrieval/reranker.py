"""重排序层：bge-reranker-base 交叉编码器。

为什么需要它：
向量检索是"问题与文档各自编码再算距离"，精度有限；交叉编码器把
(问题, 文档) 拼成一对一起过模型，能捕捉词级交互，对"数值型/多条款"问题
提升明显（q03 这类要列全 4 个步骤的题最受益）。

为什么默认跑 CPU：
本机 8 GB 显存，vLLM 按 0.72 已占 6.4 GB，reranker fp16 再要 0.6 GB 必然 OOM。
278M 参数在 CPU 上给 8 条候选打分约 150~400 ms，换来的是答案完整度提升。
显存充足的机器（24 GB 卡）把 RERANK_DEVICE 改成 cuda 即可。
"""
from __future__ import annotations

import logging
import threading

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import CrossEncoder

                settings = get_settings()
                _model = CrossEncoder(
                    settings.rerank_model,
                    device=settings.rerank_device,
                    local_files_only=True,
                )
                logger.info("reranker 加载完成：%s @ %s", settings.rerank_model, settings.rerank_device)
    return _model


def rerank(question: str, hits: list, top_n: int | None = None) -> list:
    """按 (问题, 分块) 相关性重新排序，返回得分最高的 top_n 条。

    失败不致命：reranker 加载或推理异常时退回原顺序，保证问答可用。
    """
    if not hits:
        return hits
    settings = get_settings()
    if not settings.enable_rerank:
        return hits

    top_n = top_n or settings.rerank_top_n
    try:
        model = _get_model()
        pairs = [(question, h.text) for h in hits]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)
        kept = [h for h, _ in ranked[:top_n]]
        logger.info(
            "rerank 重排 %d → %d 条，最高分 %.3f",
            len(hits), len(kept), float(ranked[0][1]),
        )
        return kept
    except Exception as exc:  # noqa: BLE001 - 重排失败不应让问答 500
        logger.warning("rerank 失败，退回原始顺序：%s", exc)
        return hits