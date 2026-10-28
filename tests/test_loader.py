"""文档解析容错：不支持类型、空文件、坏文件不中断批量导入。"""
from __future__ import annotations

import pytest

from app.ingestion.loader import (
    SUPPORTED_SUFFIXES,
    DocParseError,
    UnsupportedFileType,
    load_directory,
    load_file,
)


def test_load_markdown_returns_single_page(tmp_path):
    p = tmp_path / "轴承保养.md"
    p.write_text("# 轴承\n\n更换周期 2000 小时。", encoding="utf-8")
    pages = load_file(p)
    assert len(pages) == 1
    assert pages[0].source == "轴承保养.md"
    assert pages[0].page is None          # md 没有页码概念
    assert "2000 小时" in pages[0].text


def test_load_text_falls_back_to_gbk(tmp_path):
    """设备手册常见 GBK 编码，解析要能自动兜底。"""
    p = tmp_path / "gbk.txt"
    p.write_bytes("料筒温度 240 度。".encode("gbk"))
    pages = load_file(p)
    assert "240" in pages[0].text


def test_unsupported_suffix_raises(tmp_path):
    p = tmp_path / "手册.docx"
    p.write_text("x", encoding="utf-8")
    assert ".docx" not in SUPPORTED_SUFFIXES
    with pytest.raises(UnsupportedFileType):
        load_file(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "不存在.md")


def test_empty_file_raises_parse_error(tmp_path):
    """空文件必须抛错，不能返回空字符串 —— 上层靠它决定是否回 400。"""
    p = tmp_path / "empty.txt"
    p.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(DocParseError):
        load_file(p)


def test_load_directory_collects_errors_without_stopping(tmp_path):
    (tmp_path / "good.md").write_text("# 正常\n\n这里有足够长的正文内容用于切块。", encoding="utf-8")
    (tmp_path / "bad.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "skip.docx").write_text("x", encoding="utf-8")

    pages, errors = load_directory(tmp_path)
    assert len(pages) == 1                      # 好文件正常解析
    assert len(errors) == 1                     # 坏文件只记一条错误
    assert "bad.txt" in errors[0]
