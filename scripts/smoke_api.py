"""HTTP 接口冒烟测试：不依赖浏览器，一条命令验证三个接口是否可用。

用法：python -m scripts.smoke_api [--base http://127.0.0.1:8080]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests  # noqa: E402

SAMPLE_DOC = """# 激光焊接机 LW-500 安全须知

## 1. 激光防护

1.1 LW-500 焊接机工作时激光等级为 4 类，进入工作区必须佩戴 1064 nm 专用护目镜。

1.2 焊接舱门未关闭时联锁会切断出光，禁止短接联锁开关。

## 2. 冷却要求

2.1 冷却水温控制在 22～25 ℃，流量不低于 12 L/min。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    base = args.base.rstrip("/")

    print("1) GET /health")
    health = requests.get(f"{base}/health", timeout=30).json()
    print(f"   status={health['status']} documents={health['documents']} chunks={health['chunks']} "
          f"llm_ok={health['llm'].get('ok')}")

    print("2) POST /chat（库内问题）")
    r = requests.post(f"{base}/chat", json={"question": "XYZ-320 轴承的更换周期是多少？"}, timeout=120)
    r.raise_for_status()
    data = r.json()
    print(f"   耗时 {data['latency_ms']} ms 置信度 {data['confidence']} 引用 {len(data['citations'])} 条")
    print("   " + data["answer"].replace("\n", " ")[:150])

    print("3) POST /chat（库外问题，应拒答）")
    r = requests.post(f"{base}/chat", json={"question": "车间里有几台叉车？"}, timeout=120)
    data = r.json()
    print(f"   refused={data['refused']} answer={data['answer'][:40]}")

    print("4) POST /upload（上传样例文档）")
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "激光焊接机LW-500安全须知.md"
        p.write_text(SAMPLE_DOC, encoding="utf-8")
        with p.open("rb") as fh:
            r = requests.post(f"{base}/upload", files={"file": (p.name, fh, "text/markdown")}, timeout=120)
        r.raise_for_status()
        up = r.json()
    print(f"   {up['file']}: pages={up['pages']} chunks={up['chunks']} total={up['total_chunks']}")

    print("5) GET /documents")
    docs = requests.get(f"{base}/documents", timeout=30).json()
    for d in docs:
        print(f"   - {d['source']}: {d['chunks']} 块")

    # 收尾：删掉本次上传的测试文档，保证 eval/report.md 的基线（4 文档/38 块）不被污染
    print("6) DELETE /documents/{source}（清理测试文档）")
    r = requests.delete(f"{base}/documents/{up['file']}", timeout=60)
    r.raise_for_status()
    print(f"   已删除 {r.json()['deleted_chunks']} 块")
    after = requests.get(f"{base}/health", timeout=30).json()
    print(f"   documents={after['documents']} chunks={after['chunks']}")

    print("\n全部接口可用。Swagger: %s/docs" % base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
