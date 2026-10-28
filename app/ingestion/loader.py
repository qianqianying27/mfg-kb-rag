"""文档解析：把 PDF / TXT / Markdown 统一解析成 (source, page, text) 的段落列表。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


class UnsupportedFileType(ValueError):
    """文件类型不支持。"""


class DocParseError(RuntimeError):
    """文档解析失败（损坏、加密、空文件）。"""


@dataclass(slots=True)
class RawPage:
    source: str          # 文件名，用于引用溯源
    page: int | None     # PDF 页码从 1 开始；md/txt 为 None
    text: str


def _load_pdf(path: Path) -> list[RawPage]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # 加密 / 损坏 / 非 PDF
        raise DocParseError(f"PDF 打开失败：{path.name}（{exc}）") from exc

    pages: list[RawPage] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""   # 单页失败不影响整份文档，记为空页
        if text.strip():
            pages.append(RawPage(source=path.name, page=idx, text=text))
    if not pages:
        raise DocParseError(f"PDF 未解析出任何文本（可能是扫描件，需 OCR）：{path.name}")
    return pages


def _load_text(path: Path) -> list[RawPage]:
    for encoding in ("utf-8", "utf-8-sig", "gbk"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocParseError(f"文本编码无法识别：{path.name}")

    if not text.strip():
        raise DocParseError(f"文件内容为空：{path.name}")
    return [RawPage(source=path.name, page=None, text=text)]


def load_file(path: str | Path) -> list[RawPage]:
    """解析单个文件，返回按页/整体切开的原文。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在：{p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileType(f"不支持的文件类型：{suffix}（支持 {sorted(SUPPORTED_SUFFIXES)}）")
    if suffix == ".pdf":
        return _load_pdf(p)
    return _load_text(p)


def load_directory(directory: str | Path) -> tuple[list[RawPage], list[str]]:
    """批量解析目录，返回 (成功页面, 失败原因列表)。失败不中断，便于定位坏文件。"""
    d = Path(directory)
    pages: list[RawPage] = []
    errors: list[str] = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                pages.extend(load_file(p))
            except (DocParseError, UnsupportedFileType) as exc:
                errors.append(str(exc))
    return pages, errors
