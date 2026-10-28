"""向量化层：本地 bge-small-zh-v1.5（sentence-transformers）。

选型理由：
- bge-small-zh-v1.5 只有 93MB、512 维，中文语义检索效果好，离线可用、零 API 成本；
- 按 bge 官方用法，查询侧加指令前缀、文档侧不加，检索精度明显更高；
- 全部输出做 L2 归一化，配合 Chroma 的 cosine 空间，相似度即点积。

设备策略：默认用本机已装好的 **CUDA 版 torch** 跑在 GPU 上（EMBEDDING_DEVICE=cuda）。
- 只“借用”现有的 CUDA 版 torch，不安装、不替换任何 torch 版本；
- CUDA 不可用时自动回退 CPU 并告警，保证无显卡的机器上也能启动；
- 推理时若显存被 vLLM 抢占导致 OOM，自动回退 CPU 重试，不让整个问答失败。
"""
from __future__ import annotations

import logging
import threading

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

_model = None
_resolved_device: str | None = None
_lock = threading.Lock()


def _resolve_device(requested: str) -> str:
    """把配置里的 device 解析成真实可用设备。

    只做“检测 + 回退”，绝不安装或替换 torch：配置写 cuda 但 torch 是 CPU 版
    （或驱动缺失）时回退 cpu，保证服务能起来而不是启动即崩。
    """
    req = (requested or "cpu").strip().lower()
    if not req.startswith("cuda"):
        return req
    try:
        import torch
    except ImportError:
        logger.warning("配置要求 %s，但当前环境没有 torch，回退 CPU", req)
        return "cpu"
    if not torch.cuda.is_available():
        logger.warning(
            "配置要求 %s，但 CUDA 不可用（当前 torch=%s，可能是 CPU 版或驱动缺失），已回退 CPU；"
            "要用显卡请装 CUDA 版 torch：pip install torch --index-url "
            "https://download.pytorch.org/whl/cu126",
            req,
            getattr(torch, "__version__", "?"),
        )
        return "cpu"
    logger.info(
        "embedding 使用 GPU：%s（torch=%s, cuda=%s）",
        torch.cuda.get_device_name(0),
        torch.__version__,
        torch.version.cuda,
    )
    return req


def _get_model():
    global _model, _resolved_device
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                settings = get_settings()
                _resolved_device = _resolve_device(settings.embedding_device)
                _model = SentenceTransformer(
                    settings.embedding_model,
                    device=_resolved_device,
                    local_files_only=True,   # 模型已在本机 HF 缓存，强制离线避免联网卡住
                )
                logger.info(
                    "embedding 模型加载完成：%s @ %s",
                    settings.embedding_model,
                    _resolved_device,
                )
    return _model


def _fallback_to_cpu() -> None:
    """显存不足时丢掉 GPU 上的模型，下次重新加载到 CPU。"""
    global _model, _resolved_device
    with _lock:
        _model = None
        _resolved_device = "cpu"
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - 清理失败不影响回退
            pass
    logger.warning("embedding 已回退 CPU（GPU 显存不足）")


def _encode(texts: list[str], **kwargs):
    """统一的 encode 入口：GPU 显存不足时自动回退 CPU 重试一次。"""
    model = _get_model()
    try:
        return model.encode(texts, **kwargs)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        logger.warning("向量化时显存不足，本次请求回退 CPU 重试：%s", exc)
        _fallback_to_cpu()
        return _get_model().encode(texts, **kwargs)


def encode_documents(texts: list[str]) -> list[list[float]]:
    """文档侧向量化（不加指令前缀）。"""
    if not texts:
        return []
    settings = get_settings()
    vectors = _encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def encode_query(text: str) -> list[float]:
    """查询侧向量化（加 bge 官方检索指令）。"""
    vector = _encode(
        [BGE_QUERY_INSTRUCTION + text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector[0].tolist()
