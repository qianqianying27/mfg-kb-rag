"""分块逻辑的单测：标题优先、按句切分、去重、短碎片丢弃。"""
from __future__ import annotations

from app.ingestion.loader import RawPage
from app.ingestion.splitter import _hard_split, _split_sections, split_pages

LONG_BODY = "主控柜、上料位、取件位、包装位、两侧安全通道都需要确认按钮已复位。" * 3


def test_split_sections_uses_nearest_heading():
    text = "# 注塑机 IMM-260\n\n一级正文\n\n## 3.2 E-206 急停按钮被按下\n\n二级正文"
    titles = [title for title, _ in _split_sections(text)]
    assert titles == ["注塑机 IMM-260", "3.2 E-206 急停按钮被按下"]


def test_split_sections_without_heading_falls_back():
    titles = [title for title, _ in _split_sections("没有标题的一段正文内容")]
    assert titles == ["正文"]


def test_chunk_carries_section_for_citation():
    """引用要能定位到小节，section 必须是最近的标题。"""
    page = RawPage(source="PLC故障代码对照表.md", page=None, text=f"## 3.2 E-206 急停按钮被按下\n\n{LONG_BODY}")
    chunks = split_pages([page])
    assert chunks
    assert all(c.section == "3.2 E-206 急停按钮被按下" for c in chunks)
    assert all(c.source == "PLC故障代码对照表.md" for c in chunks)


def test_hard_split_breaks_long_text():
    text = "这是一句用于测试分块逻辑的中文句子。" * 30
    pieces = _hard_split(text, chunk_size=100, overlap=20)
    assert len(pieces) >= 4
    assert all(piece.strip() for piece in pieces)


def test_split_pages_dedups_identical_content():
    """同一段文字在两份文件里重复出现时只保留一份（md5 去重）。"""
    body = f"## 记录\n\n{LONG_BODY}"
    pages = [
        RawPage(source="a.md", page=None, text=body),
        RawPage(source="b.md", page=None, text=body),
    ]
    assert len(split_pages(pages)) == 1


def test_split_pages_drops_short_fragments():
    """页眉页脚这类碎片（短于 min_chars）直接丢弃。"""
    page = RawPage(source="x.md", page=None, text="## 短\n\n太短了")
    assert split_pages([page], min_chars=40) == []


def test_chunk_index_is_sequential():
    page = RawPage(source="a.md", page=None, text="## 一\n\n" + LONG_BODY + "\n\n## 二\n\n" + LONG_BODY)
    chunks = split_pages([page])
    assert [c.index for c in chunks] == list(range(len(chunks)))
