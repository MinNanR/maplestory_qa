"""LLM 调用封装。

统一模式：**调用方先建 span 并挂到 TurnTrace 上，client 负责把实际发生的事填进去。**
- 谁构造 messages，谁把它写进 span（只有 client 知道最终发出去的 payload）；
- 先挂后填 —— 请求失败/超时/被取消时 span 仍在列表里，才能看到"失败前烧了多少"。

关于流式 usage 的两个坑（都实测复现过）：
1. OpenAI 兼容接口的流式响应**默认不带 usage**，必须显式传
   `stream_options={"include_usage": True}`；不传的话 `chunk.usage` 恒为 None，
   token 采集是空转。
2. 带 usage 的那个尾块**`choices` 是空列表** —— 若先写 `chunk.choices[0]` 会
   IndexError。所以必须先判 usage 并 `continue`，再对 choices 做空判断。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from backend.config import settings
from backend.models import ChatMessage
from backend.observability.trace import LLMSpan, apply_usage


def _as_payload(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


class LLMClient:

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    # ------------------------------------------------------------------
    # span 记账
    # ------------------------------------------------------------------

    def _prepare(self, span: LLMSpan | None, messages: list[ChatMessage], stream: bool) -> list[dict]:
        payload = _as_payload(messages)
        if span is not None:
            span.model = settings.llm_model
            span.messages = payload
            span.stream = stream
        return payload

    @staticmethod
    def _finish(span: LLMSpan | None, t0: float) -> None:
        if span is not None:
            span.duration_ms = (time.perf_counter() - t0) * 1000

    @classmethod
    def _fail(cls, span: LLMSpan | None, exc: Exception, t0: float) -> None:
        if span is not None:
            span.error = str(exc)
        cls._finish(span, t0)

    # ------------------------------------------------------------------
    # 三种调用
    # ------------------------------------------------------------------

    async def chat(self, messages: list[ChatMessage], span: LLMSpan | None = None) -> str:
        payload = self._prepare(span, messages, stream=False)
        t0 = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=settings.llm_model,
                messages=payload,
            )
        except Exception as e:
            self._fail(span, e, t0)
            raise

        content = response.choices[0].message.content or ""
        if span is not None:
            span.response = content
            apply_usage(span, response.usage)
        self._finish(span, t0)
        return content

    async def chat_json(self, messages: list[ChatMessage], span: LLMSpan | None = None) -> dict:
        payload = self._prepare(span, messages, stream=False)
        t0 = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=settings.llm_model,
                messages=payload,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            self._fail(span, e, t0)
            raise

        content = response.choices[0].message.content
        if span is not None:
            span.response = content or ""
            apply_usage(span, response.usage)
        self._finish(span, t0)

        content = (content or "").strip()
        if not content:
            # 原代码先把空内容兜成 "{}"，那一句 raise 成了死代码 —— 空响应会被
            # 当成"分析成功但没有结论"，静默降级。这里让它明确失败。
            raise ValueError("LLM returned empty content")
        return json.loads(content)

    async def stream_chat(
        self, messages: list[ChatMessage], span: LLMSpan | None = None
    ) -> AsyncGenerator[str, None]:
        payload = self._prepare(span, messages, stream=True)
        t0 = time.perf_counter()
        parts: list[str] = []

        try:
            stream = await self._open_stream(payload)
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    apply_usage(span, usage)
                    continue                      # usage 尾块没有正文，且 choices 为空
                if not chunk.choices:             # 空 choices 保护（别照抄 choices[0]）
                    continue
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                parts.append(content)
                if span is not None and span.ttft_ms is None:
                    span.ttft_ms = (time.perf_counter() - t0) * 1000
                yield content
        except Exception as e:
            if span is not None:
                span.error = str(e)
            raise
        finally:
            # 客户端断流（GeneratorExit）时也会走到这里：保留已生成的部分。
            # 注意 finally 里不要 yield。
            if span is not None:
                span.response = "".join(parts)
                span.duration_ms = (time.perf_counter() - t0) * 1000

    async def _open_stream(self, payload: list[dict]):
        """开流。带 include_usage；provider 不认时降级重试一次。"""
        kwargs = {"model": settings.llm_model, "messages": payload, "stream": True}

        if settings.llm_stream_include_usage:
            try:
                return await self.client.chat.completions.create(
                    **kwargs, stream_options={"include_usage": True}
                )
            except Exception as e:
                # 有些 OpenAI 兼容服务不认 stream_options。降级后这次调用没有 usage
                # （span 里 token 会是 None，而不是 0 —— 两者要能区分）。
                if "stream_options" not in str(e):
                    raise
                print("[WARN] provider 不支持 stream_options.include_usage，已降级：本次调用无 usage")

        return await self.client.chat.completions.create(**kwargs)
