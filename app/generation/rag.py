"""RAG 编排层：检索 → 置信度判断 → (必要时改写查询) → 拼 Prompt → 生成 → 附引用。

关键设计：
1. 先用向量相似度做"检索置信度"判据，低置信时让模型改写查询再检索一次
   （简单的 Agentic 检索，避免答非所问时还硬编答案）；
2. 两次都低置信直接拒答，不把问题抛给模型，减少幻觉；
3. 上下文按字符上限截断，保证 8192 上下文窗口下不溢出。
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from app.config.settings import get_settings
from app.generation import llm, prompts
from app.retrieval.hybrid import HybridRetriever, Retrieved
from app.retrieval import reranker


logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 6000   # 约 3500 token，为 8192 窗口留出答案空间
REL_GAP = 0.12             # 与最高相似度的相对落差，用于剔除跨文档噪声
MAX_SOURCES = 2            # 单次问答最多引用 2 份文档，避免“什么都答一点”
# 型号/错误码/编号识别（XYZ-320、E-102、IMM-260、AC-75 这类）
ENTITY_RE = re.compile(r"[A-Za-z]{1,}[-_]?\d+[A-Za-z0-9-]*")


def _entity_terms(question: str) -> list[str]:
    return [t.upper() for t in ENTITY_RE.findall(question)]


def _gate_by_entity(hits: list[Retrieved], terms: list[str]) -> list[Retrieved]:
    """型号强匹配过滤。

    问句里出现 XYZ-320 / E-102 这类实体时，只有真正包含该实体的分块才可能是答案。
    实测：不加这层过滤时，“E-102 怎么处理”会把另一份文档里“轴承更换周期 2000 小时”
    的分块召回进上下文，模型随即把这句话复述到答案开头。
    """
    if not terms:
        return hits
    matched = [h for h in hits if any(t in h.text.upper() for t in terms)]
    return matched or hits      # 一个都没命中时不过滤，避免误杀


def _filter_hits(hits: list[Retrieved], floor: float) -> list[Retrieved]:
    """噪声过滤：按相对相似度 + 文档数上限裁剪候选。

    制造业文档词汇高度重叠（都有“更换周期”“温度”“压力”），向量检索很容易把
    其他设备的分块召回进 top-k。实测不加过滤时，模型会把轴承的“2000 小时”
    复述进注塑机 E-102 的回答里，所以必须在这里裁掉。
    """
    if not hits:
        return hits
    best = max((h.vector_sim or 0.0) for h in hits)
    kept = [h for h in hits if (h.vector_sim or 0.0) >= max(floor, best - REL_GAP)]
    order: list[str] = []
    for h in kept:
        if h.source not in order:
            order.append(h.source)
    allowed = set(order[:MAX_SOURCES])
    filtered = [h for h in kept if h.source in allowed]
    return filtered or hits[:1]


@dataclass(slots=True)
class Citation:
    index: int
    source: str
    page: int | None
    section: str
    snippet: str


@dataclass(slots=True)
class RagAnswer:
    answer: str
    citations: list[Citation]
    refused: bool
    confidence: float
    used_rewrite: bool
    retrieved: int
    latency_ms: int
    debug: dict = field(default_factory=dict)


def _build_context(hits: list[Retrieved]) -> tuple[str, list[Citation]]:
    """把检索结果编号后拼成资料区，同时生成引用列表。"""
    blocks: list[str] = []
    citations: list[Citation] = []
    used = 0

    for i, hit in enumerate(hits, start=1):
        cite = hit.cite()
        block = f"[{i}] 来源：{cite}\n{hit.text}"
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        used += len(block)
        citations.append(
            Citation(
                index=i,
                source=hit.source,
                page=hit.page,
                section=hit.section,
                snippet=hit.text[:80],
            )
        )
    return "\n\n".join(blocks), citations


class RagService:
    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    # ---------- 内部工具 ----------
    def warmup(self) -> None:
        """服务启动时调用，把 embedding 与 BM25 的加载成本移出首个请求。"""
        self.retriever.warmup()
        logger.info("检索层预热完成（embedding 模型 + BM25 索引就绪）")

    def _rewrite(self, question: str) -> str:
        try:
            new_q = llm.chat(
                [
                    {"role": "system", "content": "你是检索查询改写助手，只输出改写后的查询。"},
                    {"role": "user", "content": prompts.REWRITE_PROMPT.format(question=question)},
                ],
                temperature=0.0,
                max_tokens=60,
            )
            return new_q.strip().splitlines()[0][:60] or question
        except llm.LLMUnavailable:
            return question        # 改写失败不影响主流程，退回原问题

    # ---------- 主流程 ----------
    def answer(self, question: str, top_k: int | None = None) -> RagAnswer:
        settings = get_settings()
        started = time.perf_counter()

        if self.retriever.store.count() == 0:
            return RagAnswer(
                answer=prompts.REFUSE_TEXT,
                citations=[],
                refused=True,
                confidence=0.0,
                used_rewrite=False,
                retrieved=0,
                latency_ms=int((time.perf_counter() - started) * 1000),
                debug={"reason": "vector store is empty"},
            )

        hits = self.retriever.retrieve(question, top_k=top_k)
        confidence = self.retriever.best_similarity(hits)
        used_rewrite = False

        if confidence < settings.min_similarity and settings.enable_rewrite:
            rewritten = self._rewrite(question)
            if rewritten and rewritten != question:
                logger.info("检索置信度 %.3f 偏低，改写查询：%s → %s", confidence, question, rewritten)
                alt_hits = self.retriever.retrieve(rewritten, top_k=top_k)
                alt_conf = self.retriever.best_similarity(alt_hits)
                if alt_conf > confidence:
                    hits, confidence, used_rewrite = alt_hits, alt_conf, True

        if confidence < settings.min_similarity or not hits:
            return RagAnswer(
                answer=prompts.REFUSE_TEXT,
                citations=[],
                refused=True,
                confidence=round(confidence, 4),
                used_rewrite=used_rewrite,
                retrieved=len(hits),
                latency_ms=int((time.perf_counter() - started) * 1000),
                debug={"reason": "low retrieval confidence"},
            )

        hits = _gate_by_entity(hits, _entity_terms(question))
        hits = reranker.rerank(question, hits)  # 精排：交叉编码器重排
        hits = _filter_hits(hits, settings.min_similarity)
        context, citations = _build_context(hits)
        answer = llm.chat(
            [
                {"role": "system", "content": prompts.SYSTEM_PROMPT},
                {"role": "user", "content": prompts.USER_TEMPLATE.format(
                    context=context, question=question
                )},
            ]
        )

        refused = prompts.REFUSE_TEXT.rstrip("。") in answer and len(answer) < 40
        if refused:
            return RagAnswer(
                answer=prompts.REFUSE_TEXT,
                citations=[],
                refused=True,
                confidence=round(confidence, 4),
                used_rewrite=used_rewrite,
                retrieved=len(hits),
                latency_ms=int((time.perf_counter() - started) * 1000),
                debug={"reason": "model refused"},
            )

        # 只保留模型真正引用过的条目，避免“引了一堆但一个没用”的假溯源
        used_marks = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        citations = [c for c in citations if c.index in used_marks] or citations

        # 引用溯源：固定附在答案末尾，前端与评测都按"依据："识别
        cite_lines = "；".join(
            f"[{c.index}] {c.source}" + (f" p.{c.page}" if c.page else f" §{c.section}")
            for c in citations
        )
        final = f"{answer}\n\n{prompts.CITATION_HEADER}{cite_lines}"

        return RagAnswer(
            answer=final,
            citations=citations,
            refused=False,
            confidence=round(confidence, 4),
            used_rewrite=used_rewrite,
            retrieved=len(hits),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
