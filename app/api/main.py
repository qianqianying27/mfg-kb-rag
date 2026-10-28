"""FastAPI 服务层。

三个接口：
  POST /upload  上传文档并入知识库（PDF / TXT / MD）
  POST /chat    自然语言问答，返回答案 + 引用
  GET  /health  探活（含向量库规模、模型连通性）
Swagger 文档：http://127.0.0.1:8080/docs
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationOut,
    DocumentOut,
    HealthResponse,
    UploadResponse,
)
from app.config.settings import get_settings
from app.generation import llm
from app.generation.rag import RagService
from app.ingestion import loader
from app.ingestion.pipeline import ingest_file
from app.retrieval.store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时在后台线程预热检索层，不阻塞端口监听（/health 可立即可用）。"""
    threading.Thread(target=_warmup_bg, name="warmup", daemon=True).start()
    yield


app = FastAPI(
    title="制造业设备知识库 RAG 问答系统",
    description="上传设备手册/工艺文件，提问即得带原文引用的答案。",
    version="0.1.0",
    lifespan=lifespan,
)

store = VectorStore()
rag = RagService()


def _warmup_bg() -> None:
    started = time.perf_counter()
    try:
        rag.warmup()
        logger.info("后台预热耗时 %.0f ms", (time.perf_counter() - started) * 1000)
    except Exception:  # 预热失败不影响服务可用，首个请求会自然加载
        logger.exception("后台预热失败，将在首个请求时惰性加载")


@app.get("/health", response_model=HealthResponse, summary="健康检查")
def health() -> HealthResponse:
    settings = get_settings()
    llm_state = llm.health()
    return HealthResponse(
        status="ok" if llm_state.get("ok") else "degraded",
        documents=len({d.source for d in store.all_documents()}),
        chunks=store.count(),
        embedding_model=settings.embedding_model,
        llm=llm_state,
    )


@app.post("/upload", response_model=UploadResponse, summary="上传文档并入库")
def upload(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in loader.SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix or '(无后缀)'}，仅支持 {sorted(loader.SUPPORTED_SUFFIXES)}",
        )

    settings = get_settings()
    target = settings.raw_dir / (file.filename or "unnamed")
    try:
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"文件写入失败：{exc}") from exc
    finally:
        file.file.close()

    result = ingest_file(target, store)
    if result.error:
        raise HTTPException(status_code=400, detail=f"解析失败：{result.error}")

    return UploadResponse(
        file=result.file,
        pages=result.pages,
        chunks=result.chunks,
        added=result.added,
        total_chunks=store.count(),
    )


@app.post("/chat", response_model=ChatResponse, summary="知识库问答")
def chat(req: ChatRequest) -> ChatResponse:
    if store.count() == 0:
        raise HTTPException(status_code=409, detail="知识库为空，请先调用 /upload 入库文档")

    try:
        result = rag.answer(req.question, top_k=req.top_k)
    except llm.LLMUnavailable as exc:
        # 模型不可用属于外部依赖故障，返回 503 而不是 500，便于前端做降级提示
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.answer,
        refused=result.refused,
        confidence=result.confidence,
        used_rewrite=result.used_rewrite,
        retrieved=result.retrieved,
        latency_ms=result.latency_ms,
        citations=[
            CitationOut(
                index=c.index,
                source=c.source,
                page=c.page,
                section=c.section,
                snippet=c.snippet,
            )
            for c in result.citations
        ],
    )


@app.get("/documents", response_model=list[DocumentOut], summary="列出已入库文档")
def documents() -> list[DocumentOut]:
    counter = Counter(d.source for d in store.all_documents())
    return [DocumentOut(source=s, chunks=c) for s, c in sorted(counter.items())]


@app.delete("/documents/{source}", summary="按文件名删除文档")
def delete_document(source: str) -> dict:
    removed = store.delete_source(source)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"未找到文档：{source}")

    # 同步删除源文件：否则下次跑 ingest 会把这份文档重新入库，向量库与磁盘不一致
    settings = get_settings()
    raw_file = settings.raw_dir / Path(source).name
    file_removed = False
    try:
        if raw_file.is_file():
            raw_file.unlink()
            file_removed = True
    except OSError as exc:
        logger.warning("向量已删除但源文件删除失败 %s: %s", raw_file, exc)

    return {"deleted_chunks": removed, "deleted_file": file_removed}
