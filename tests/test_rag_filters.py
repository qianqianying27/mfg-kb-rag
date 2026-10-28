"""检索后处理的单测：全是纯函数，不加载任何模型。

这里覆盖的是项目里最容易回归出问题的几处：型号实体门控、跨文档噪声过滤、
相似度落差裁剪，以及答案里引用编号的解析与清洗。
"""
from __future__ import annotations

from app.generation.rag import (
    MAX_SOURCES,
    _citation_marks,
    _entity_terms,
    _filter_hits,
    _gate_by_entity,
    _sanitize_citation_marks,
)
from app.retrieval.hybrid import Retrieved


def _hit(text: str, source: str, sim: float) -> Retrieved:
    return Retrieved(
        chunk_id=f"{source}#0",
        text=text,
        metadata={"source": source, "page": 0, "section": "s", "index": 0},
        similarity=sim,
        vector_sim=sim,
    )


def test_entity_terms_matches_single_letter_prefix():
    """E-102 是单字母前缀；早期正则用 {2,} 时匹配不到，导致答案被噪声污染。"""
    assert _entity_terms("注塑机出现 E-102 报警怎么办") == ["E-102"]
    assert _entity_terms("XYZ-320 轴承更换周期是多少") == ["XYZ-320"]
    assert _entity_terms("AC-75 空压机排气温度") == ["AC-75"]


def test_entity_terms_returns_empty_without_entity():
    assert _entity_terms("轴承振动值超过多少必须停机") == []


def test_gate_by_entity_keeps_only_matching_chunks():
    hits = [
        _hit("E-102：料筒温度超限，立即停止射胶", "PLC故障代码对照表.md", 0.70),
        _hit("XYZ-320 轴承更换周期为 2000 小时", "轴承保养与更换规程.md", 0.68),
    ]
    kept = _gate_by_entity(hits, ["E-102"])
    assert [h.source for h in kept] == ["PLC故障代码对照表.md"]


def test_gate_by_entity_falls_back_when_nothing_matches():
    """一个都不命中时不能过滤，否则会把可回答的问题误判成库外。"""
    hits = [_hit("料筒温度超限处理步骤", "PLC故障代码对照表.md", 0.70)]
    assert _gate_by_entity(hits, ["E-999"]) == hits


def test_filter_hits_drops_low_similarity_noise():
    hits = [
        _hit("轴承更换周期 2000 小时", "轴承保养与更换规程.md", 0.72),
        _hit("料筒温度 240 度", "注塑机IMM-260操作与维护手册.md", 0.70),
        _hit("空压机压力 0.7 MPa", "螺杆空压机AC-75点检与安全规程.md", 0.45),
    ]
    kept = _filter_hits(hits, 0.40)
    assert "螺杆空压机AC-75点检与安全规程.md" not in [h.source for h in kept]


def test_filter_hits_caps_document_count():
    """单次问答最多引用 MAX_SOURCES 份文档，避免“什么都答一点”。"""
    hits = [_hit("a", "d1.md", 0.70), _hit("b", "d2.md", 0.69), _hit("c", "d3.md", 0.68)]
    kept = _filter_hits(hits, 0.40)
    assert len({h.source for h in kept}) == MAX_SOURCES


def test_filter_hits_never_returns_empty():
    """全部低于阈值时也要留一条，交给上层置信度闸门判断，而不是在这里静默清空。"""
    hits = [_hit("无关内容", "d1.md", 0.10)]
    assert len(_filter_hits(hits, 0.40)) == 1


def test_citation_marks_expands_grouped_form():
    """模型会写 [1, 5]；早期只用 r"\\[(\\d+)\\]" 时只能抽出 1，把 [1，2] 的第 2 条也丢了。"""
    assert _citation_marks("见 [1]") == [1]
    assert _citation_marks("见 [1, 5]") == [1, 5]
    assert _citation_marks("见 [1，2] 与 [3、4]") == [1, 2, 3, 4]
    assert _citation_marks("没有任何编号") == []


def test_sanitize_drops_dangling_mark():
    """只召回 1 条却写 [1, 5] 时，必须抹掉 5，否则看起来像有第 5 条依据。"""
    assert _sanitize_citation_marks("检查固态继电器 [1, 5]。", {1}) == "检查固态继电器 [1]。"
    assert _sanitize_citation_marks("只有 [5] 存在", {1}) == "只有 存在"


def test_sanitize_keeps_leading_space():
    """编号前的空格要保留，否则中文与 [n] 会挤在一起。"""
    assert _sanitize_citation_marks("温度 240 度 [1]。", {1}) == "温度 240 度 [1]。"
    assert _sanitize_citation_marks("温度 240 度[1]。", {1}) == "温度 240 度[1]。"
