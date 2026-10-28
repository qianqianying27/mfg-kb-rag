"""离线评测脚本：命中率 / 引用准确率 / 拒答正确率 / 延迟。

为什么要有它：
RAG 项目最常见的坑是"改了 Prompt 或分块，某些问题变好、另一些变差"。
拿 10 条固定用例跑基线，改完再跑一次，用数据说话（面试时这张表就是成果）。

用法：
    python -m eval.run_eval            # 跑全部用例并生成 eval/report.md
    python -m eval.run_eval --limit 3  # 只跑前 3 条，调参时省时间
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.rag import RagService   # noqa: E402

CASES_PATH = Path(__file__).with_name("cases.jsonl")
REPORT_PATH = Path(__file__).with_name("report.md")


def load_cases(limit: int | None = None) -> list[dict]:
    cases = []
    for line in CASES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 评测")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases(args.limit)
    if not cases:
        print("没有找到评测用例")
        return 1

    rag = RagService()
    # 先预热再计时：否则第一条用例会把 embedding 模型加载（约 30 s）算进延迟，
    # 把平均值拉到几千毫秒，线上真实体验被严重低估。
    warm_started = time.perf_counter()
    rag.warmup()
    warmup_ms = int((time.perf_counter() - warm_started) * 1000)
    print(f"预热完成：{warmup_ms} ms（不计入延迟统计）")

    rows = []
    t_start = time.perf_counter()

    for case in cases:
        started = time.perf_counter()
        answer = rag.answer(case["question"])
        elapsed = (time.perf_counter() - started) * 1000

        text = answer.answer
        keywords_hit = (
            all(kw in text for kw in case["expected_keywords"])
            if case["expected_keywords"]
            else None
        )
        cite_sources = {c.source for c in answer.citations}
        cite_hit = (
            case["expected_source"] in cite_sources if case["expected_source"] else None
        )
        refuse_ok = answer.refused == case["expect_refuse"]

        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "expect_refuse": case["expect_refuse"],
                "refused": answer.refused,
                "refuse_ok": refuse_ok,
                "keywords_hit": keywords_hit,
                "cite_hit": cite_hit,
                "confidence": answer.confidence,
                "rewrite": answer.used_rewrite,
                "latency_ms": round(elapsed),
                "answer": text.replace("\n", " ")[:110],
            }
        )

    # ---------- 指标 ----------
    in_kb = [r for r in rows if not r["expect_refuse"]]
    out_kb = [r for r in rows if r["expect_refuse"]]
    hit_rate = sum(1 for r in in_kb if r["keywords_hit"]) / len(in_kb) if in_kb else 0.0
    cite_rate = (
        sum(1 for r in in_kb if r["cite_hit"]) / len(in_kb) if in_kb else 0.0
    )
    refuse_rate = (
        sum(1 for r in out_kb if r["refused"]) / len(out_kb) if out_kb else 0.0
    )
    avg_latency = sum(r["latency_ms"] for r in rows) / len(rows)

    # ---------- 控制台输出 ----------
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="RAG 评测明细")
        for col in ("id", "问题", "拒答?", "关键词", "引用", "置信度", "耗时ms"):
            table.add_column(col)
        for r in rows:
            table.add_row(
                r["id"],
                r["question"][:26],
                "Y" if r["refuse_ok"] else "N",
                "-" if r["keywords_hit"] is None else ("Y" if r["keywords_hit"] else "N"),
                "-" if r["cite_hit"] is None else ("Y" if r["cite_hit"] else "N"),
                f"{r['confidence']:.3f}",
                str(r["latency_ms"]),
            )
        # Windows 控制台默认 GBK，遇到特殊字符会抛 UnicodeEncodeError，这里降级为纯文本输出
        try:
            Console().print(table)
        except (UnicodeEncodeError, UnicodeError):
            Console(no_color=True, legacy_windows=False).print(table, soft_wrap=True)
    except ImportError:
        for r in rows:
            print(r)

    summary = (
        f"命中率 {hit_rate:.0%} | 引用准确率 {cite_rate:.0%} | "
        f"拒答正确率 {refuse_rate:.0%} | 平均延迟 {avg_latency:.0f} ms"
    )
    print("\n" + summary)

    # ---------- 写报告 ----------
    lines = [
        "# RAG 评测报告",
        "",
        f"- 用例数：{len(rows)}（库内 {len(in_kb)} 条 / 库外 {len(out_kb)} 条）",
        f"- 命中率（答案包含标准关键词）：**{hit_rate:.0%}**",
        f"- 引用准确率（引用包含期望文档）：**{cite_rate:.0%}**",
        f"- 拒答正确率（库外问题正确拒答）：**{refuse_rate:.0%}**",
        f"- 平均延迟：{avg_latency:.0f} ms（总耗时 {time.perf_counter() - t_start:.1f} s，已预热，不含模型加载 {warmup_ms} ms）",
        "",
        "| ID | 问题 | 期望拒答 | 实际拒答 | 关键词 | 引用 | 置信度 | 延迟(ms) | 答案摘要 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        kw = "-" if r["keywords_hit"] is None else ("Y" if r["keywords_hit"] else "N")
        ct = "-" if r["cite_hit"] is None else ("Y" if r["cite_hit"] else "N")
        lines.append(
            f"| {r['id']} | {r['question']} | {r['expect_refuse']} | {r['refused']} | "
            f"{kw} | {ct} | {r['confidence']:.3f} | {r['latency_ms']} | {r['answer']} |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已写入：{REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
