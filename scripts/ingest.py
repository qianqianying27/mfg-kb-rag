"""命令行入库脚本（无需启动 Web 服务）。

用法：
    python -m scripts.ingest --dir data/raw          # 批量入库
    python -m scripts.ingest --file data/raw/a.md    # 单文件
    python -m scripts.ingest --dir data/raw --reset  # 先清空向量库
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings            # noqa: E402
from app.ingestion.pipeline import ingest_directory, ingest_file  # noqa: E402
from app.retrieval.store import VectorStore              # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="文档入库")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path, help="批量入库目录")
    group.add_argument("--file", type=Path, help="入库单个文件")
    parser.add_argument("--reset", action="store_true", help="入库前清空向量库")
    args = parser.parse_args()

    store = VectorStore()
    if args.reset:
        store.reset()
        print("已清空向量库")

    settings = get_settings()
    if args.file:
        results = [ingest_file(args.file, store)]
    else:
        results = ingest_directory(args.dir or settings.raw_dir, store)

    failed = 0
    for r in results:
        if r.error:
            failed += 1
            print(f"  [FAIL] {r.file}: {r.error}")
        else:
            print(f"  [ OK ] {r.file}: pages={r.pages} chunks={r.chunks} added={r.added}")

    print(f"\n总计 {len(results)} 个文件，失败 {failed} 个；向量库当前 {store.count()} 个分块")
    print(f"向量库路径：{settings.chroma_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
