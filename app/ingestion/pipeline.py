"""入库流水线：解析 → 分块 → 向量化 → 写入 Chroma。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.config.settings import get_settings
from app.ingestion import loader, splitter
from app.retrieval import embedder
from app.retrieval.store import VectorStore


@dataclass(slots=True)
class IngestResult:
    file: str
    pages: int
    chunks: int
    added: int
    skipped: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def ingest_file(path: Path, store: VectorStore, replace: bool = True) -> IngestResult:
    """解析并入库单个文件。已存在同名来源时先删后写，保证幂等。"""
    settings = get_settings()
    try:
        pages = loader.load_file(path)
    except (loader.DocParseError, loader.UnsupportedFileType, FileNotFoundError) as exc:
        return IngestResult(file=path.name, pages=0, chunks=0, added=0, error=str(exc))

    chunks = splitter.split_pages(
        pages,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        min_chars=settings.min_chunk_chars,
    )
    if not chunks:
        return IngestResult(file=path.name, pages=len(pages), chunks=0, added=0,
                            error="未切出有效分块（内容过短）")

    if replace and store.has_source(path.name):
        store.delete_source(path.name)

    vectors = embedder.encode_documents([c.text for c in chunks])
    added = store.add(chunks, vectors)
    return IngestResult(file=path.name, pages=len(pages), chunks=len(chunks), added=added)


def ingest_directory(directory: Path, store: VectorStore, replace: bool = True) -> list[IngestResult]:
    results: list[IngestResult] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in loader.SUPPORTED_SUFFIXES:
            results.append(ingest_file(p, store, replace=replace))
    return results
