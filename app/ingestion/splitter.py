"""分块策略：标题优先 + 定长窗口 + 重叠。

为什么这样切：
1. 制造业文档（手册、规程）几乎都是"标题 → 步骤/参数"结构，按标题切能保证
   一个块内语义完整（例如"E-102 报警处理"不会被拆到两处）；
2. 单个小节超过 chunk_size 时再按段落累积，超长段落硬切并保留 overlap，
   避免答案被切断在边界上；
3. overlap 只在上一步硬切时产生，段落级合并不重复，减少冗余检索噪声。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import md5

from app.ingestion.loader import RawPage

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 中文标点断句，用于超长段落的硬切（优先在句号/分号处断开）
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；\n])")


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int | None
    section: str
    index: int
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        md = {
            "source": self.source,
            "page": self.page if self.page is not None else 0,
            "section": self.section,
            "index": self.index,
        }
        md.update(self.metadata)
        return md


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按句子边界累积切块，块间保留 overlap 个字符。"""
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    pieces: list[str] = []
    buf = ""
    for sent in sentences:
        if len(sent) > chunk_size:                      # 单句仍超长，退化为定长切
            for i in range(0, len(sent), chunk_size - overlap):
                piece = sent[i : i + chunk_size]
                if piece.strip():
                    pieces.append(piece)
            continue
        if len(buf) + len(sent) <= chunk_size:
            buf += sent
        else:
            if buf.strip():
                pieces.append(buf)
            buf = (buf[-overlap:] if overlap and buf else "") + sent
    if buf.strip():
        pieces.append(buf)
    return pieces


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题切成 (小节标题, 正文)。无标题时整篇算一节。"""
    sections: list[tuple[str, str]] = []
    current_title = "正文"
    buf: list[str] = []

    for line in text.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            if "".join(buf).strip():
                sections.append((current_title, "\n".join(buf)))
            current_title = m.group(2).strip()
            buf = []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((current_title, "\n".join(buf)))
    return sections


def split_pages(
    pages: list[RawPage],
    chunk_size: int = 480,
    overlap: int = 80,
    min_chars: int = 40,
) -> list[Chunk]:
    """把解析后的页面切成分块列表。"""
    chunks: list[Chunk] = []
    seen: set[str] = set()

    for page in pages:
        for title, body in _split_sections(page.text):
            body = body.strip()
            if not body:
                continue
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

            # 段落合并到 chunk_size 以内
            buf = ""
            merged: list[str] = []
            for para in paragraphs:
                para = re.sub(r"\s+", " ", para)
                if len(para) > chunk_size:
                    if buf:
                        merged.append(buf)
                        buf = ""
                    merged.extend(_hard_split(para, chunk_size, overlap))
                elif len(buf) + len(para) + 1 <= chunk_size:
                    buf = f"{buf}\n{para}" if buf else para
                else:
                    merged.append(buf)
                    buf = para
            if buf:
                merged.append(buf)

            for piece in merged:
                piece = piece.strip()
                if len(piece) < min_chars:
                    continue
                # 内容去重：同一段文字在文档里重复出现只保留一次
                digest = md5(piece.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)

                idx = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{page.source}#{idx}",
                        text=piece,
                        source=page.source,
                        page=page.page,
                        section=title,
                        index=idx,
                    )
                )
    return chunks
