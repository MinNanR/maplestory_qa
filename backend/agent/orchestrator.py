import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from dataclasses import dataclass, asdict, field

from backend.analysis.query_analyzer import QueryAnalyzer
from backend.conversation.manager import ConversationManager
from backend.llm.client import LLMClient
from backend.prompts.system import get_wrap_up_note
from backend.models import (
    ChatMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
    AssistantMessage,
    ToolMessage,
)
from backend.observability.trace import (
    KIND_LLM,
    KIND_RETRIEVAL,
    KIND_TOOL,
    LLMSpan,
    TurnTrace,
    create_request_id,
    print_turn_trace,
    span_timer,
)
import backend.tool.tools_impl
from backend.tool.dispatcher import (
    dispatcher,
    ToolCallRequest,
    ToolCallResult,
    build_tool_call_request,
)
from backend.config import settings

import json
import time
from typing import Literal, Union


@dataclass(frozen=True)
class StagePayload:
    name: str
    state: Literal["start", "end"]
    message: str = ""
    stage_context: dict[str, any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenPayload:
    text: str


@dataclass(frozen=True)
class ToolCallPayload:
    tool: str
    args: dict
    call_id: str


@dataclass(frozen=True)
class ToolResultPayload:
    tool: str
    result: str
    # 工具自报的结构化结果（原样透传，编排层不解释里面的键）。
    # 只带摘要级原始类型、不带正文 —— 正文仍然只进 trace 文件，不进事件流。
    structured: dict = field(default_factory=dict)
    # 回灌给模型的**完整**字符数。result 只是 200 字符预览，数不出真实规模，
    # 所以单独给一个数：前端用它显示"已返回 N 字符"。
    chars: int = 0


@dataclass(frozen=True)
class FinalPayload:
    text: str
    request_id: str


@dataclass(frozen=True)
class ErrorPayload:
    code: str
    message: str
    # 失败发生在哪个阶段（turn.stage：generation / tool_call ...）。
    # 前端据此提示"（调用工具 阶段）"—— 只报"处理失败"没法定位。
    stage: str = ""


@dataclass(frozen=True)
class UsagePayload:
    """本轮的成本/耗时摘要。

    spans 只带**摘要**（不含 messages/response —— 那是 trace 文件的职责），
    所以这个事件不会把上下文正文泄进事件流。
    """

    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    wall_ms: float
    estimated: bool  # 有 span 的 token 是估算值
    usage_missing: bool  # 有 llm span 完全没拿到 usage
    spans: list[dict]


@dataclass(frozen=True)
class AgentEvent:
    seq: int
    type: Literal[
        "stage", "token", "tool_call", "tool_result", "usage", "final", "error"
    ]
    payload: Union[
        StagePayload,
        TokenPayload,
        ToolCallPayload,
        ToolResultPayload,
        UsagePayload,
        FinalPayload,
        ErrorPayload,
    ]


@dataclass
class TurnContext:
    conversation_id: str
    request_id: str
    answer: str = ""
    stage: str = "init"
    finished: bool = False
    trace: TurnTrace | None = None


class Orchestrator:
    def __init__(
        self,
        llm_client: LLMClient,
        conversation_manager: ConversationManager,
        trace_dir: str | None,
    ):
        self.llm_client = llm_client
        self.conversation_manager = conversation_manager
        self.analyzer = QueryAnalyzer(llm_client)
        self.trace_dir = trace_dir

    def _persist(self, turn: TurnContext):
        from pathlib import Path

        trace_dir = self.trace_dir or "./turn"
        now = datetime.now()
        time_str = now.strftime("%Y%m%d_%H%M%S")
        p = Path(f"{trace_dir}/{turn.request_id}_{time_str}.json")
        p.parent.mkdir(parents=True, exist_ok=True)

        def json_default(o):
            if isinstance(o, datetime):
                return o.isoformat()  # "2026-09-11T10:30:00"
            raise TypeError(f"Type {type(o)} not serializable")

        with open(p.resolve(), "w", encoding="utf-8") as f:
            json.dump(
                asdict(turn), f, default=json_default, ensure_ascii=False, indent=2
            )
        if turn.trace:
            print_turn_trace(turn.trace)

    async def run(
        self, conversation_id: str, message: str
    ) -> AsyncIterator[AgentEvent]:
        turn = TurnContext(
            conversation_id=conversation_id, request_id=create_request_id()
        )
        # 在 try 之外建：TurnTrace 构造不会失败，这样任何失败路径都能取到它
        turn.trace = TurnTrace(
            request_id=turn.request_id, conversation_id=conversation_id
        )

        seq = 0

        def emit(type_: str, payload) -> AgentEvent:
            nonlocal seq
            seq += 1
            return AgentEvent(seq=seq, type=type_, payload=payload)

        def usage_payload() -> UsagePayload:
            return UsagePayload(**turn.trace.summary())

        # 先挂后填：span 在发起请求前就进了列表，失败/超时也留得下记录
        def start_span(name: str, kind: str = KIND_LLM, **kw):
            return turn.trace.start_span(name, kind=kind, **kw)

        tool_seen: set[str] = set()

        async def run_tool(request: ToolCallRequest, span: LLMSpan) -> ToolCallResult:
            try:
                paras_json = json.loads(request.params)
                canon = json.dumps(
                    paras_json,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                call_key = f"{request.name}|{canon}"
            except Exception:
                call_key = ""
            if call_key in tool_seen:
                result = ToolCallResult(
                    tool_call_id=request.tool_call_id,
                    name=request.name,
                    succeeded=True,
                    content=(
                        f"你已用完全相同的参数调用过 {request.name}，"
                        f"结果就在先前工具消息里，可以直接使用。"
                        f"如果需要更多信息，请更换参数（更具体的关键词，或另一个 doc_id），"
                        f"或者直接基于已有信息作答。"
                    ),
                    deduped=True,
                )
            else:
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(dispatcher.dispatch, request),
                        timeout=settings.tool_timeout_s,
                    )
                    if call_key:
                        tool_seen.add(call_key)
                except Exception as e:
                    result = ToolCallResult(
                        tool_call_id=request.tool_call_id,
                        name=request.name,
                        succeeded=False,
                        error_msg=f"工具执行失败：{e}",
                    )
            span.detail = {
                "tool_call_request": asdict(request),
                "tool_call_result": {
                    "succeeded": result.succeeded,
                    "deduped": result.deduped,
                    "chars_count": len(result.content),
                    # 工具自报的结构化结果（知识检索是 {"chunk_ids": [...]}）：
                    # 评测据此算 doc_recall / ctx_recall，编排层不解释它
                    "structured": result.structured,
                },
            }
            if not result.succeeded:
                span.detail["tool_call_result"]["error_message"] = result.error_msg
            span.tool_output = result.content
            span.duration_ms = (time.perf_counter() - span.t0) * 1000
            return result

        tool_rounds = 0
        max_tool_rounds = settings.max_tool_rounds

        user_message = UserMessage(content=message)
        self.conversation_manager.add_message(conversation_id, user_message)

        try:
            while True:
                messages: list[ChatMessage] = self.conversation_manager.build_messages(
                    conversation_id
                )
                # 每轮清空最终答案
                turn.answer = ""
                turn.stage = "generation"
                llm_span = start_span(
                    "token", KIND_LLM, model=settings.llm_model, stream=True
                )

                full_content = ""
                tool_acc: dict[int, ToolCall] = {}
                if tool_rounds < max_tool_rounds:
                    # 还有工具调用轮次
                    tools_schema = dispatcher.build_schema()
                    # 注意两个坑，都踩过：
                    # 1. stream_chat_with_tools 是 async generator 函数，调用它只创建
                    #    生成器对象，不能 await（会 TypeError）；
                    # 2. 生成器的函数体要等第一次 __anext__ 才执行，所以 tool_acc
                    #    必须在下面**消费完流之后**才读 —— 在这里读永远是空的，
                    #    后果是"模型要求调工具、编排器却当成直接作答"，整条工具链路
                    #    静默失效（不报错、只是永远不调工具）。
                    stream = self.llm_client.stream_chat_with_tools(
                        messages=messages,
                        tools=tools_schema,
                        span=llm_span,
                        tool_acc=tool_acc,
                    )
                else:
                    # 不带 tools：模型无法再要求调工具，循环必然在这里收口
                    stream = self.llm_client.stream_chat(
                        messages=[*messages, get_wrap_up_note()], span=llm_span
                    )

                async for content in stream:
                    full_content += content
                    yield emit(type_="token", payload=TokenPayload(text=content))

                # 流已消费完，累加器这时才是准的。按 index 排序：分片到达顺序不保证。
                tool_calls: list[ToolCall] = [tool_acc[i] for i in sorted(tool_acc)]

                turn.answer = full_content
                assistant_message = AssistantMessage(
                    content=full_content,
                    # LLMSpan.reasoning_content 的 None 表示"provider 没给这个字段"，
                    # 而 AssistantMessage 要的是 str —— 不兜这一下，每轮都会
                    # pydantic 校验失败（Input should be a valid string）。
                    reasoning_content=llm_span.reasoning_content or "",
                )
                if tool_calls:
                    assistant_message.tool_calls = tool_calls
                    for tc in tool_calls:
                        yield emit(
                            type_="tool_call",
                            payload=ToolCallPayload(
                                tool=tc.name, args=tc.arguments, call_id=tc.id
                            ),
                        )

                # 回灌模型消息
                self.conversation_manager.add_message(
                    conversation_id, assistant_message
                )

                if not tool_calls:
                    turn.finished = True
                    break

                turn.stage = "tool_call"
                yield emit(
                    type_="stage",
                    payload=StagePayload(
                        name=turn.stage, state="start", message="正在执行工具调用"
                    ),
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

                # 执行工具调用请求
                tool_call_results: list[ToolCallResult] = await asyncio.gather(
                    *(
                        run_tool(request=request, span=span)
                        for span, request in tools_executions
                    ),
                    return_exceptions=True,
                )

                # 与 tools_executions 并行取，这样即使 run_tool 抛异常逃出来，
                # 也能拿回这条调用的 tool_call_id —— 缺了它，下一轮请求会因为
                # "assistant 的 tool_calls 没有对应的 tool 消息" 被 provider 拒绝。
                for (_, request), outcome in zip(tools_executions, tool_call_results):
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
                    result_content = (
                        result.content
                        if result.succeeded
                        else f"调用失败，错误原因：{result.error_msg}"
                    )
                    tm = ToolMessage(
                        tool_call_id=result.tool_call_id,
                        content=result_content,
                    )
                    self.conversation_manager.add_message(conversation_id, tm)
                    yield emit(
                        "tool_result",
                        ToolResultPayload(
                            tool=result.name,
                            result=result_content[:200],
                            structured=dict(result.structured),
                            chars=len(result_content),
                        ),
                    )
                yield emit(
                    type_="stage",
                    payload=StagePayload(
                        name=turn.stage, state="end", message="工具调用完成"
                    ),
                )

                tool_rounds += 1

        except Exception as e:
            # 失败也要报已花掉的成本：放在 error 之前，保持 error 是最后一个事件
            yield emit(type_="usage", payload=usage_payload())
            yield emit(
                type_="error",
                payload=ErrorPayload(
                    code="internal", message=str(e), stage=turn.stage
                ),
            )
            raise e
        finally:
            try:
                self._persist(turn)
            except Exception as e:
                print(f"[persist failed] {e}")

        # # usage 放在 final 之前：final 仍是"本轮最后一个事件"
        yield emit(type_="usage", payload=usage_payload())
        if turn.finished:
            yield emit(
                type_="final",
                payload=FinalPayload(text=turn.answer, request_id=turn.request_id),
            )
