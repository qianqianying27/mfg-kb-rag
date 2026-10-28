# 贡献指南

## 环境

- Python 3.11+（本项目在 3.12.7 验证）
- 需要 CUDA 版 torch（embedding 跑 GPU）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

装了 CUDA 版之后**不要再加 `--upgrade` / `--force-reinstall`**，否则会被 PyPI 的 CPU 版覆盖，
`EMBEDDING_DEVICE=cuda` 会在启动时回退到 CPU（日志有 warning）。

## 本地开发

```bash
pip install -r requirements.txt
cp .env.example .env
python -m scripts.ingest --dir data/raw     # 首次入库
python -m pytest                            # 单测，秒级，不需要 GPU/vLLM
python -m eval.run_eval                     # 端到端评测，需要 vLLM 已启动
uvicorn app.api.main:app --port 8080        # 打开 http://127.0.0.1:8080/ 为对话页
```

## 提交规范

commit message 用英文祈使句，一条提交只做一件事：

```
feat: add rerank with bge-reranker-base
fix: filter cross-document noise in retrieved chunks
docs: explain why refusal happens before generation
test: cover entity gating and noise filtering
```

## 合并前必须满足

1. `python -m pytest` 全绿；
2. `python -m eval.run_eval` 三项指标不下降（命中率 / 引用准确率 / 拒答正确率）；
3. 若改动了 Prompt、分块参数或检索策略，在 `eval/report.md` 里附改动前后对比；
4. 不提交 `.env`、`data/chroma/`、任何密钥。

## 新增文档类型

在 `app/ingestion/loader.py` 的 `SUPPORTED_SUFFIXES` 注册后缀，并在 `load_file()` 加解析分支。
解析失败必须抛 `DocParseError` 而不是返回空字符串 —— 上层靠它决定是否返回 400。

## 新增 Agent / 检索能力

检索后处理集中在 `app/generation/rag.py`，顺序是：

```
retrieve → 置信度闸门 → (改写重试) → 型号实体门控 → rerank → 噪声过滤 → 拼上下文
```

往中间插环节时注意：便宜的过滤（字面匹配）放前面，贵的模型推理（rerank）放后面。
