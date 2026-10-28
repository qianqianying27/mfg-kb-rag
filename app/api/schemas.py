"""接口层的数据模型（Pydantic v2），Swagger 文档直接由它们生成。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户问题")
    top_k: int | None = Field(default=None, ge=1, le=10, description="检索条数，默认取配置值")


class CitationOut(BaseModel):
    index: int
    source: str
    page: int | None = None
    section: str = ""
    snippet: str = ""


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    confidence: float
    used_rewrite: bool
    retrieved: int
    latency_ms: int
    citations: list[CitationOut] = []


class UploadResponse(BaseModel):
    file: str
    pages: int
    chunks: int
    added: int
    total_chunks: int
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    embedding_model: str
    llm: dict


class DocumentOut(BaseModel):
    source: str
    chunks: int
