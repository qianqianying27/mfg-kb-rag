"""全局配置。

设计理由：所有可变参数（模型、base_url、top_k、chunk_size）统一走环境变量，
这样同一份代码既能连本地 vLLM（http://127.0.0.1:8000/v1），
也能一键切到 DeepSeek / Qwen 等云端 OpenAI 兼容服务，符合"大模型接入可替换"的工程要求。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- LLM（OpenAI 兼容）----------
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "qwen-scorer"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 900
    llm_timeout: float = 120.0
    llm_max_retries: int = 2

    # ---------- Embedding ----------
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cuda"         # 用本机已装的 CUDA 版 torch；CUDA 不可用时自动回退 CPU
    embedding_batch_size: int = 32

    # ---------- 重排序（默认关闭；开启后召回 8 条精排取 3 条） ----------
    enable_rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_device: str = "cpu"             # 8GB 显存放不下 vLLM+reranker，显存充足时改 cuda
    rerank_top_n: int = 3

    # ---------- 分块 ----------
    chunk_size: int = 480                  # 中文约等于 480 字，接近一段完整工艺描述
    chunk_overlap: int = 80                # 重叠 80 字，避免"操作步骤被切断"
    min_chunk_chars: int = 40              # 过短的碎片（页眉页脚）直接丢弃

    # ---------- 检索 ----------
    top_k: int = 5
    vector_top_k: int = 8
    bm25_top_k: int = 8
    rrf_k: int = 60                        # RRF 平滑常数，工业界常用 60
    min_similarity: float = 0.40           # 低于该值视为"检索置信度低"
    enable_hybrid: bool = True             # 向量 + BM25 融合（型号/编号类查询靠它）
    enable_rewrite: bool = True            # 低置信时先改写查询再检索一次

    # ---------- 存储 ----------
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    collection_name: str = "mfg_docs"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    return s
