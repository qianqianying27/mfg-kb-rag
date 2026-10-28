"""大模型调用层：OpenAI 兼容接口封装。

可切换性：只依赖 base_url + model + api_key 三个环境变量，
本地 vLLM（http://127.0.0.1:8000/v1）、DeepSeek、Qwen、OpenAI 都能直接跑。
"""
from __future__ import annotations

import logging

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """模型服务不可用（超时 / 连接失败 / 限流），由 API 层转成 503。"""


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        s = get_settings()
        _client = OpenAI(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
            timeout=s.llm_timeout,
            max_retries=s.llm_max_retries,
        )
    return _client


def chat(
    messages: list[dict],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """调用对话模型，返回纯文本。失败统一抛 LLMUnavailable，避免把 SDK 异常泄漏到接口层。"""
    s = get_settings()
    try:
        resp = _get_client().chat.completions.create(
            model=s.llm_model,
            messages=messages,
            temperature=s.llm_temperature if temperature is None else temperature,
            max_tokens=s.llm_max_tokens if max_tokens is None else max_tokens,
        )
    except (APITimeoutError, APIConnectionError) as exc:
        logger.warning("模型服务连接失败：%s", exc)
        raise LLMUnavailable(f"模型服务不可用：{s.llm_base_url}") from exc
    except RateLimitError as exc:
        logger.warning("模型服务限流：%s", exc)
        raise LLMUnavailable("模型服务繁忙（限流），请稍后重试") from exc
    except APIStatusError as exc:
        logger.error("模型服务返回错误 %s：%s", exc.status_code, exc.message)
        raise LLMUnavailable(f"模型服务返回错误（HTTP {exc.status_code}）") from exc

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise LLMUnavailable("模型返回空内容")
    return content


def health() -> dict:
    """探活：供 /health 与启动自检使用。"""
    s = get_settings()
    try:
        models = _get_client().models.list()
        ids = [m.id for m in models.data]
        return {"ok": s.llm_model in ids, "base_url": s.llm_base_url, "models": ids}
    except Exception as exc:  # 探活失败不应让进程崩溃
        return {"ok": False, "base_url": s.llm_base_url, "error": str(exc)[:200]}
