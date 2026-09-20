from collections.abc import AsyncIterator
from datetime import datetime
from dataclasses import dataclass, asdict, field

from backend.analysis.query_analyzer import QueryAnalyzer
from backend.conversation.manager import ConversationManager
from backend.llm.client import LLMClient
from backend.models import ChatMessage
from backend.observability.trace import (
    KIND_LLM,
    KIND_RETRIEVAL,
    TurnTrace,
    create_request_id,
    print_turn_trace,
    span_timer,
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


@dataclass(frozen=True)
class ToolResultPayload:
    tool: str
    result: dict


@dataclass(frozen=True)
class FinalPayload:
    text: str
    request_id: str


@dataclass(frozen=True)
class ErrorPayload:
    code: str
    message: str


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
    estimated: bool          # 有 span 的 token 是估算值
    usage_missing: bool      # 有 llm span 完全没拿到 usage
    spans: list[dict]


@dataclass(frozen=True)
class AgentEvent:
    seq: int
    type: Literal["stage", "token", "tool_call", "tool_result", "usage", "final", "error"]
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

        try:
            previous_messages = self.conversation_manager.get_message(conversation_id)
            previous_user_texts = [
                m.content for m in previous_messages if m.role == "user"
            ][-2:]
            turn.stage = "analysis"
            yield emit(
                type_="stage",
                payload=StagePayload(turn.stage, "start", "正在分析用户意图"),
            )
            analysis_span = start_span("analysis", KIND_LLM, model=settings.llm_model)
            analysis = await self.analyzer.analyze(
                query=message, history=previous_user_texts, span=analysis_span
            )
            yield emit(type_="stage", payload=StagePayload(turn.stage, "end", ""))

            if len(previous_user_texts) > 0:
                dialog_query = (
                    message + "\n前文问题：\n" + ";".join(previous_user_texts)
                )
            else:
                dialog_query = message

            if analysis.needs_knowledge:
                # 命中文档并入会话知识池，并按 dialog_query 截取片段做本轮注入；
                # 池中历史文档（如塞伦机制）在预算内仍会保留在场。
                turn.stage = "retrieval"
                yield emit(
                    type_="stage",
                    payload=StagePayload(turn.stage, "start", "正在检索知识库"),
                )
                # 检索没有 token，但有耗时 —— 也要 span，否则"这一轮几秒花在哪"答不出来
                retrieval_span = start_span("retrieval", KIND_RETRIEVAL)
                with span_timer(retrieval_span):
                    retrieval_chunk_ids: list[str] = (
                        self.conversation_manager.refresh_knowledge(
                            conversation_id, analysis.knowledge_ids, dialog_query
                        )
                    )
                retrieval_span.detail = {
                    "chunk_count": len(retrieval_chunk_ids),
                    "chunk_ids": retrieval_chunk_ids,
                }
                retrieval_result = ";".join(retrieval_chunk_ids)
                yield emit(
                    type_="stage",
                    payload=StagePayload(
                        turn.stage,
                        "end",
                        f"检索数据库完成, 注入知识id：{retrieval_result}",
                        stage_context={"retrieval_result": retrieval_chunk_ids},
                    ),
                )
            else:
                self.conversation_manager.clear_knowledge(conversation_id)

            user_message = ChatMessage(role="user", content=message)

            self.conversation_manager.add_message(conversation_id, user_message)

            messages = self.conversation_manager.build_messages(conversation_id)

            turn.stage = "generation"
            generation_span = start_span(
                "generation", KIND_LLM, model=settings.llm_model, stream=True
            )
            full_content = ""
            async for content in self.llm_client.stream_chat(
                messages, span=generation_span
            ):
                full_content += content
                yield emit(type_="token", payload=TokenPayload(content))

            turn.answer = full_content
            assistant_message = ChatMessage(role="assistant", content=full_content)
            self.conversation_manager.add_message(conversation_id, assistant_message)
            turn.finished = True
        except Exception as e:
            # 失败也要报已花掉的成本：放在 error 之前，保持 error 是最后一个事件
            yield emit(type_="usage", payload=usage_payload())
            yield emit(
                type_="error", payload=ErrorPayload(code="internal", message=str(e))
            )
            raise e
        finally:
            try:
                self._persist(turn)
            except Exception as e:
                print(f"[persist failed] {e}")

        # usage 放在 final 之前：final 仍是"本轮最后一个事件"
        yield emit(type_="usage", payload=usage_payload())
        if turn.finished:
            yield emit(type_="final", payload=FinalPayload(text=turn.answer, request_id=turn.request_id))
