import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal

from backend.agent.events import (
    StagePayload,
    TokenPayload,
    ToolCallPayload,
    ToolResultPayload,
)
from backend.agent.execution import ToolExecutor
from backend.conversation.manager import ConversationManager
from backend.llm.client import LLMClient
from backend.models import AssistantMessage, ChatMessage, ToolCall, ToolMessage
from backend.observability.trace import KIND_LLM, KIND_TOOL, TurnContext, TurnTrace
from backend.config import settings
from backend.prompts.system import get_wrap_up_note
from backend.tool.dispatcher import ToolCallResult, build_tool_call_request


@dataclass
class Budget:
    max_steps: int  # = max_tool_rounds
    max_tool_chars: int  # 累计回灌字符（新增）
    max_wall_s: float  # 建议 90（现状 p95 23.5s）
    steps_remaining: int


# 收口保留量：预算跌破它就不再开新工具轮，把剩下的额度留给"不带 tools 的收口请求"。
# 必须是显式常量，而不是散在各处条件里的魔法数 —— 否则调预算的人不会知道
# "把 max_tool_chars 配成 12000 会让机器人从第一轮起就完全不检索"。
WRAP_UP_RESERVE_CHARS = 15_000
WRAP_UP_RESERVE_S = 15.0


def _finish_reason(
    steps_remaining: int,
    tool_chars_remaining: int,
    remaining_wall_s: float,
    stall_rounds: int,
) -> Literal[
    "no_tool_calls", "steps_exhausted", "chars_exhausted", "wall_clock", "no_progress"
]:
    """预算触顶的原因。

    成因要在**触发的那一刻**从剩余量判定，而不是事后从被改过的 Budget 反推 ——
    多个预算可能同时触顶，反推出来的原因会随判定顺序变化。
    """
    if stall_rounds >= 2:
        return "no_progress"
    if steps_remaining <= 0:
        return "steps_exhausted"
    if tool_chars_remaining <= WRAP_UP_RESERVE_CHARS:
        return "chars_exhausted"
    if remaining_wall_s <= WRAP_UP_RESERVE_S:
        return "wall_clock"
    return "no_tool_calls"


def _finalize(
    messages: list[ChatMessage], reason: str, llm: LLMClient, llm_span
) -> None:
    """在预算用尽时，给模型发一条 wrap-up note，让它直接给出最终答案。"""
    stream = llm.stream_chat(
        messages=[*messages, get_wrap_up_note(reason=reason)], span=llm_span
    )
    return stream


EVIDENCE_KEYS = ("chunk_ids",)


def _collect_evidence(structured: dict) -> set[str]:
    out: set[str] = set()
    for k in EVIDENCE_KEYS:
        for v in structured.get(k) or ():
            out.add(f"{k}:{v}")
    return out


async def run_loop(
    *,
    cm: ConversationManager,
    llm: LLMClient,
    executor: ToolExecutor,
    turn: TurnContext,
    budget: Budget,
    now: Callable[[], float] = time.monotonic,
) -> AsyncIterator[tuple[str, object]]:
    conversation_id = turn.conversation_id
    tools_schema = executor.dispatcher.build_schema()
    step = 1

    # 先挂后填：span 在发起请求前就进了列表，失败/超时也留得下记录
    def start_span(name: str, kind: str = KIND_LLM, **kw):
        nonlocal step
        span = turn.trace.start_span(name, kind=kind, **kw)
        span.step = step
        return span

    seen_evidence: set[str] = set()
    stall_rounds = 0

    start = now()
    steps_remaining, tool_chars_remaining = budget.max_steps, budget.max_tool_chars
    try:
        while True:
            messages = cm.build_messages(conversation_id)
            turn.answer = ""
            turn.stage = "generation"
            llm_span = start_span(
                "token", kind=KIND_LLM, model=settings.llm_model, stream=True
            )

            full_content = ""
            wrap_up = False

            tool_acc: dict[int, ToolCall] = {}

            # 剩余墙钟 = 总预算 - 已耗时。**不要**把已耗时写回 budget.max_wall_s：
            # 那会让"剩余"的语义反转，`<= 保留量` 在第一次工具轮之后恒为真 ——
            # 表现是"永远只检索一次就收口"，多步能力被静默关掉。
            remaining_wall_s = budget.max_wall_s - (now() - start)
            if (
                steps_remaining <= 0
                or tool_chars_remaining <= WRAP_UP_RESERVE_CHARS
                or remaining_wall_s <= WRAP_UP_RESERVE_S
                or stall_rounds >= 2
            ):
                reason = _finish_reason(steps_remaining, tool_chars_remaining, remaining_wall_s, stall_rounds)
                stream = _finalize(messages, reason, llm, llm_span)
                turn.finished = True
                turn.finish_reason = reason
                turn.trace.finish_reason = reason
                wrap_up = True
            else:
                stream = llm.stream_chat_with_tools(
                    messages=messages,
                    tools=tools_schema,
                    span=llm_span,
                    tool_acc=tool_acc,
                )

            yield "stage", StagePayload(name="generation", state="start")
            async for content in stream:
                full_content += content
                yield "token", TokenPayload(text=content)
            yield "stage", StagePayload(name="generation", state="end")

            tool_calls: list[ToolCall] = list(tool_acc[i] for i in sorted(tool_acc))

            turn.answer = full_content
            assistant_message = AssistantMessage(
                content=full_content,
                reasoning_content=llm_span.reasoning_content or "",
                tool_calls=tool_calls,
            )

            if tool_calls:
                assistant_message.tool_calls = tool_calls
                for tc in tool_calls:
                    yield "tool_call", ToolCallPayload(
                        tool=tc.name, args=tc.arguments, call_id=tc.id, step=step
                    )
            cm.add_message(conversation_id, assistant_message)

            if not tool_calls:
                turn.finished = True
                if not wrap_up:
                    turn.finish_reason = "no_tool_calls"
                    turn.trace.finish_reason = "no_tool_calls"
                break

            turn.stage = "tool_call"
            yield "stage", StagePayload(
                name="tool_call", state="start", message="正在执行工具调用"
            )
            # 构建工具调用请求
            tools_executions = [
                (
                    start_span(
                        name=f"tool_call:{tc.name}",
                        kind=KIND_TOOL,
                        t0=time.perf_counter(),
                        parent_span_id=llm_span.span_id,
                    ),
                    build_tool_call_request(tc),
                )
                for tc in tool_calls
            ]

            executor_results = await executor.run_batch(tools_executions)
            round_evidence: set[str] = set()

            # 与 tools_executions 并行取，这样即使 run_tool 抛异常逃出来，
            # 也能拿回这条调用的 tool_call_id —— 缺了它，下一轮请求会因为
            # "assistant 的 tool_calls 没有对应的 tool 消息" 被 provider 拒绝。
            for (_, request), outcome in zip(tools_executions, executor_results):
                # return_exceptions=True 会把"逃出 run_tool 的异常"混进结果列表。
                # 不在这里收窄类型，下面访问 .content 就会炸成一个
                # 把原始错误盖掉的 AttributeError。
                if isinstance(outcome, BaseException):
                    result = ToolCallResult(
                        tool_call_id=request.tool_call_id,
                        name=request.name,
                        succeeded=False,
                        error_msg=f"工具调度失败：{outcome!r}",
                    )
                else:
                    result = outcome

                if isinstance(outcome, ToolCallResult):
                    round_evidence |= _collect_evidence(outcome.structured)
                result_content = (
                    result.content
                    if result.succeeded
                    else f"调用失败，错误原因：{result.error_msg}"
                )
                tm = ToolMessage(
                    tool_call_id=result.tool_call_id,
                    content=result_content,
                )
                cm.add_message(conversation_id, tm)
                yield "tool_result", ToolResultPayload(
                    tool=result.name,
                    result=result_content[:200],
                    structured=dict(result.structured),
                    chars=len(result_content),
                    step=step,
                    succeeded=result.succeeded,
                )

            yield "stage", StagePayload(
                name=turn.stage, state="end", message="工具调用完毕"
            )

            if not round_evidence:
                if seen_evidence:
                    stall_rounds += 1
            elif round_evidence - seen_evidence:
                seen_evidence |= round_evidence
                stall_rounds = 0
            else:
                stall_rounds += 1

            steps_remaining -= 1
            tool_chars_remaining -= sum(
                len(r.content)
                for r in executor_results
                if isinstance(r, ToolCallResult) and r.succeeded
            )
            step += 1
            # 墙钟按"总预算 - 已耗时"在每轮开头现算（见上），这里不写回 budget。
            # 单次工具输出上限是**每次调用**的约束，不由累计量递减；更不该拿
            # 助手回答的字数去扣 —— 两者语义无关（那会让上限随回答长度漂移）。
        budget.steps_remaining = steps_remaining
    except Exception:
        # 这里**只抛不报**：error 事件由外壳（turn 生命周期）统一发。
        # 两边都发会得到两个 error，而且第一个 error 后面还跟着 usage ——
        # 直接破坏"error 必须是最后一个事件"这条契约（前端据此判断是否失败）。
        raise
