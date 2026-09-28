"""观测契约：一次调用一个 span，一轮一个 TurnTrace。

为什么用 span 列表而不是"往一个对象上累加"
------------------------------------------
1. 累加会丢掉"哪一步花了多少"——成本与延迟都无从归因；
2. 台阶 1/2 的工具循环会让调用次数变成动态的（1~6 次），累加只知道总数，
   数不出循环了几轮；
3. 每个 span 的 messages/response 是"某一次请求的输入输出"，累加后这些字段
   语义不成立（response 拼接？messages 取谁的？）。

**累加只发生在读的时候**：TurnTrace 提供派生属性，落盘时顺手把 totals 也写进去
（spans 是真相、totals 是便利，读端不必重复实现聚合）。

span ≠ LLM 调用
---------------
检索、工具执行同样有耗时，也要 span，只是 kind 不同。否则"这一轮 8 秒花在哪"
永远答不出来——而检索/工具很可能才是大头。
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# span 的种类：决定哪些字段有意义（token 只对 llm 有意义）
KIND_LLM = "llm"
KIND_RETRIEVAL = "retrieval"
KIND_TOOL = "tool"


def create_request_id() -> str:
    return str(uuid.uuid4())


@dataclass
class LLMSpan:
    """一次可观测的执行步骤：一次 LLM 调用 / 一次检索 / 一次工具执行。"""

    # —— 通用（所有 kind 都有）——
    name: str                                  # "analysis" / "generation" / "retrieval" / "tool:xxx"
    kind: str = KIND_LLM
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started_at: datetime = field(default_factory=datetime.now)
    t0: float | None = None
    duration_ms: float | None = None
    error: str | None = None

    # —— llm 专属 ——
    model: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)   # 实际发出去的 payload
    response: str = ""
    reasoning_content: str | None = None   # 推理内容
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None       # 预留：prompt cache 命中
    usage_estimated: bool = False          # 预留：provider 不给 usage 时是估算值
    ttft_ms: float | None = None           # 仅流式：首 token 延迟
    stream: bool = False
    # 最后一次 choice 的结束原因："stop" / "tool_calls" / "length" / "content_filter"。
    # 这是判断"模型是想调工具还是想回答"的唯一可靠依据：
    #   - 拿到 tool_calls 但 finish_reason 不是 "tool_calls" → 流被截断，工具参数可能不完整；
    #   - finish_reason 是 "length" → 上下文或 max_tokens 不够，答案/参数被切掉。
    # None = provider 没给（流式下由带 choices 的尾块携带；usage 尾块 choices 为空，不带它）。
    finish_reason: str | None = None
    attempt: int = 1                       # 预留：重试
    parent_span_id: str | None = None      # 预留：多 agent / 工具由某次调用触发

    # —— retrieval / tool 专属 ——
    detail: dict[str, Any] = field(default_factory=dict)
    
    # tool 专属
    tool_output: str | None = None
    

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    @property
    def usage_known(self) -> bool:
        return self.kind != KIND_LLM or self.input_tokens is not None


def apply_usage(span: LLMSpan | None, usage: Any) -> None:
    """把 provider 的 usage 映射到 span 字段。

    OpenAI 兼容接口用的是 prompt_tokens / completion_tokens，而我们统一叫
    input_tokens / output_tokens —— 映射只写这一处，换 provider 只改这里。
    """
    if span is None or usage is None:
        return
    span.input_tokens = getattr(usage, "prompt_tokens", None) or span.input_tokens
    span.output_tokens = getattr(usage, "completion_tokens", None) or span.output_tokens
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", None)
        if cached is not None:
            span.cached_tokens = cached


@dataclass
class TurnTrace:
    """一轮对话的全部 span。"""

    request_id: str
    conversation_id: str
    spans: list[LLMSpan] = field(default_factory=list)

    def start_span(self, name: str, kind: str = KIND_LLM, **kw) -> LLMSpan:
        """创建 span 并**立即挂上**（先挂后填）。

        这样请求失败/超时/被取消时，span 仍然留在列表里 —— 你才能看到
        "生成阶段失败，但已经烧了 400 token"。
        """
        span = LLMSpan(name=name, kind=kind, **kw)
        self.spans.append(span)
        return span

    # —— 累加（读的时候算）——
    @property
    def llm_spans(self) -> list[LLMSpan]:
        return [s for s in self.spans if s.kind == KIND_LLM]

    @property
    def llm_calls(self) -> int:
        return len(self.llm_spans)

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens or 0 for s in self.llm_spans)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens or 0 for s in self.llm_spans)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def wall_ms(self) -> float:
        return sum(s.duration_ms or 0 for s in self.spans)

    @property
    def usage_estimated(self) -> bool:
        return any(s.usage_estimated for s in self.llm_spans)

    def summary(self) -> dict[str, Any]:
        """给事件流/录制的摘要：**不含 messages/response**（那是 trace 文件的职责）。"""
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "wall_ms": self.wall_ms,
            "estimated": self.usage_estimated,
            "usage_missing": any(not s.usage_known for s in self.llm_spans),
            "spans": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "model": s.model,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    # provider 报的 prompt cache 命中量。**input_tokens 已经包含它**，
                    # 所以成本必须按 (input - cached) 计全价、cached 计缓存价；
                    # 少了这个字段，评测算成本时只能把命中部分按全价算（上界）。
                    "cached_tokens": s.cached_tokens,
                    "duration_ms": s.duration_ms,
                    "ttft_ms": s.ttft_ms,
                    "finish_reason": s.finish_reason,
                    "usage_estimated": s.usage_estimated,
                    "error": s.error,
                    "detail": s.detail,
                }
                for s in self.spans
            ],
        }


@contextmanager
def span_timer(span: LLMSpan):
    """给同步步骤（检索、工具执行）计时，并保证异常时也留下 error + duration。

    用法：
        span = turn.trace.start_span("retrieval", kind=KIND_RETRIEVAL)
        with span_timer(span):
            chunk_ids = manager.refresh_knowledge(...)
    """
    t0 = time.perf_counter()
    try:
        yield span
    except Exception as e:  # noqa: BLE001 - 记下来再往上抛
        span.error = str(e)
        raise
    finally:
        span.duration_ms = (time.perf_counter() - t0) * 1000


def print_turn_trace(trace: TurnTrace | None) -> None:
    """控制台摘要。只用 ASCII 标记：Windows 控制台是 GBK，打不出 ✓/⚠ 会直接崩。"""
    if trace is None:
        return
    print()
    print("====== Turn Trace ======")
    print(f"request_id={trace.request_id}")
    print(f"conversation_id={trace.conversation_id}")
    print(f"spans={len(trace.spans)} llm_calls={trace.llm_calls} "
          f"tokens in/out={trace.input_tokens}/{trace.output_tokens}")
    for s in trace.spans:
        bits = [f"{s.name}({s.kind})"]
        if s.duration_ms is not None:
            bits.append(f"{s.duration_ms:.0f}ms")
        if s.kind == KIND_LLM:
            bits.append(f"in={s.input_tokens} out={s.output_tokens}")
            if s.ttft_ms is not None:
                bits.append(f"ttft={s.ttft_ms:.0f}ms")
            if s.finish_reason:
                bits.append(f"finish={s.finish_reason}")
        if s.detail:
            bits.append(str(s.detail))
        if s.error:
            bits.append(f"ERROR={s.error[:60]}")
        print("  - " + " | ".join(bits))
