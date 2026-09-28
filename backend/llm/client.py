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
from backend.models import (
    ChatMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)
from backend.observability.trace import LLMSpan, apply_usage


def _as_message(message: ChatMessage) -> dict:
    """ChatMessage → OpenAI 兼容的 dict。

    这里踩过一个很隐蔽的坑（后果是**每一次** LLM 调用都失败，报错却只有一句
    `'SystemMessage' object has no attribute 'tool_calls'`）：
    `match` 里的**裸类名不是类模式，是捕获模式**。于是
      - `case SystemMessage, UserMessage, ToolMessage:` 被解析成"长度为 3 的
        序列模式"（逗号 = 序列，不是 `|` 或模式），BaseModel 不是序列，永不匹配；
      - `case AssistantMessage:` 变成"捕获一切"，把**所有**消息都按 assistant 处理，
        非 assistant 的消息一访问 `.tool_calls` 就 AttributeError。
    类模式必须带括号：`case AssistantMessage():`。
    """
    match message:
        case SystemMessage() | UserMessage() | ToolMessage():
            return message.model_dump()
        case AssistantMessage():
            m = {"role": message.role, "content": message.content}
            if message.tool_calls:
                if message.reasoning_content:
                    m["reasoning_content"] = message.reasoning_content
                # 这里必须转成 **OpenAI 线格式**，不能直接 model_dump：
                # 我们的 ToolCall 是扁平的 {id, name, arguments}，而接口要的是
                # {id, type:"function", function:{name, arguments}}。扁平形状发出去
                # 第一轮没事（那时还没有 tool_calls），第二轮带工具历史时会被
                # provider 以 400 拒绝 —— 表现为"工具一调就崩"。
                # 本地 ToolCall 形状保持不变（累加器、span.detail 都还在用它）。
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in message.tool_calls
                ]
            return m
        case _:
            # 不写这个分支的话，未识别的类型会静默返回 None，
            # 变成 payload 里混进一个 None —— 又是一次难查的失败。
            raise TypeError(f"未知的消息类型：{type(message).__name__}")


def _as_payload(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [_as_message(m) for m in messages]


def _capture_finish_reason(span: LLMSpan | None, choice) -> None:
    """把 choice.finish_reason 记进 span（只记非空值）。

    流式下它可能出现在**任何**一个带 choices 的块上（常与空 delta 同块），
    所以必须在 `if not content: continue` **之前**调用 —— 否则只捕获到
    "文本非空的那一块"，而结束原因恰恰常在不带正文的尾块上。
    usage 尾块的 choices 是空列表，不携带它（已被上游的 choices 判空挡掉）。
    """
    if span is None or choice is None:
        return
    reason = getattr(choice, "finish_reason", None)
    if reason:
        span.finish_reason = reason


class LLMClient:

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            max_retries=3,
            timeout=180
        )

    # ------------------------------------------------------------------
    # span 记账
    # ------------------------------------------------------------------

    def _prepare(
        self, span: LLMSpan | None, messages: list[ChatMessage], stream: bool
    ) -> list[dict]:
        try:
            payload = _as_payload(messages)
        except Exception as e:
            # 构造 payload 就失败时也要落到 span 上。三个调用方法里的 try 都从
            # "发请求"才开始，不在这里兜一下，trace 里会留下一个 duration/error
            # 全空的空白 span，控制台只剩一句裸异常 —— 最需要线索的时候没线索。
            if span is not None:
                span.error = f"构造 messages payload 失败：{e!r}"
            raise
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

    async def chat(
        self, messages: list[ChatMessage], span: LLMSpan | None = None
    ) -> str:
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
        _capture_finish_reason(span, response.choices[0])
        self._finish(span, t0)
        return content

    async def chat_json(
        self, messages: list[ChatMessage], span: LLMSpan | None = None
    ) -> dict:
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
        _capture_finish_reason(span, response.choices[0])
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
                    continue  # usage 尾块没有正文，且 choices 为空
                if not chunk.choices:  # 空 choices 保护（别照抄 choices[0]）
                    continue
                choice = chunk.choices[0]
                _capture_finish_reason(span, choice)
                content = choice.delta.content
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

    async def _open_stream(self, payload: list[dict], tools: list[dict] | None = None):
        """开流。带 include_usage；provider 不认时降级重试一次。"""
        kwargs = {"model": settings.llm_model, "messages": payload, "stream": True}
        if tools:
            kwargs["tools"] = tools

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
                print(
                    "[WARN] provider 不支持 stream_options.include_usage，已降级：本次调用无 usage"
                )

        return await self.client.chat.completions.create(**kwargs)

    async def stream_chat_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict],
        span: LLMSpan | None = None,
        tool_acc: dict[int, ToolCall] | None = None,
    ) -> AsyncGenerator[str, None]:
        payload = self._prepare(span, messages, True)
        t0 = time.perf_counter()
        content_parts: list[str] = []
        reasoning_content_parts: list[str] = []
        if tool_acc is None:
            # 不能写成 `if not tool_acc`：调用方传进来的是一个**空 dict**，
            # "not 空 dict" 为真，于是这里新建一个 dict 顶掉调用方那个 ——
            # 工具调用参数被填进了没人读的副本里，编排器看到 tool_acc 仍为空，
            # 就把"模型要求调工具"当成"模型直接作答"。失败形态是静默的：
            # 不报错、工具链路全线失效（正是 orchestrator 里那两条注释警告的坑）。
            tool_acc = {}

        try:
            stream = await self._open_stream(payload, tools)
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    apply_usage(span, usage)
                if not chunk.choices:
                    continue
                # 层次别写错：chunk.choices[0] 是 Choice，正文/工具增量在它下面的
                # `.delta` 上。把 Choice 当 delta 用，`choice.tool_calls` 会直接
                # AttributeError（pydantic 模型没有的字段就抛），整条工具链路必炸。
                choice = chunk.choices[0]
                delta = choice.delta
                _capture_finish_reason(span, choice)

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        tc_idx = tc.index
                        if tc_idx not in tool_acc:
                            tool_acc[tc_idx] = ToolCall(id="", name="", arguments="")

                        tool_call = tool_acc[tc_idx]
                        if tc.id:
                            tool_call.id = tc.id
                        if tc.function and tc.function.name:
                            tool_call.name = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_call.arguments += tc.function.arguments

                reasoning_content = getattr(delta, "reasoning_content", None)
                if reasoning_content:
                    reasoning_content_parts.append(reasoning_content)

                content = delta.content
                if not content:
                    continue
                content_parts.append(content)
                if span is not None and span.ttft_ms is None:
                    span.ttft_ms = (time.perf_counter() - t0) * 1000
                yield content
        except Exception as e:
            if span:
                span.error = str(e)
            raise
        finally:
            if span:
                span.response = "".join(content_parts)
                span.duration_ms = (time.perf_counter() - t0) * 1000
                if reasoning_content_parts:
                    span.reasoning_content = "".join(reasoning_content_parts)
                if tool_acc:
                    span.detail["tool_calls"] = [
                        tc.model_dump() for tc in tool_acc.values()
                    ]
