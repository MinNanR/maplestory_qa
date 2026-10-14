import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from dataclasses import dataclass, asdict, field

from backend.agent.events import (
    AbortedPayload,
    AgentEvent,
    ErrorPayload,
    FinalPayload,
    StagePayload,
    TokenPayload,
    ToolCallPayload,
    ToolResultPayload,
    UsagePayload,
)
from backend.agent.execution import ToolExecutor
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
    TurnContext,
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
import backend.agent.loop as loop

import json
import time
from typing import Literal, Union


class Orchestrator:
    def __init__(
        self,
        llm_client: LLMClient,
        conversation_manager: ConversationManager,
        trace_dir: str | None,
    ):
        self.llm_client = llm_client
        self.conversation_manager = conversation_manager
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
        tool_executor = ToolExecutor(dispatcher=dispatcher)

        seq = 0

        def emit(type_: str, payload) -> AgentEvent:
            nonlocal seq
            seq += 1
            return AgentEvent(seq=seq, type=type_, payload=payload)

        def usage_payload() -> UsagePayload:
            return UsagePayload(**turn.trace.summary())
        
        def mark_interrupted() -> None:
            turn.interrupted = True
            turn.finished = False
            if not turn.finish_reason:
                turn.finish_reason = "user_cancel"

        user_message = UserMessage(content=message)
        self.conversation_manager.add_message(conversation_id, user_message)

        budget = loop.Budget(
            max_steps=settings.max_tool_rounds,
            max_tool_chars=settings.max_tool_chars,
            max_wall_s=settings.max_wall_s,
            steps_remaining=settings.max_tool_rounds
        )
        try:
            async for event_type, payload in loop.run_loop(
                cm=self.conversation_manager,
                llm=self.llm_client,
                executor=tool_executor,
                turn=turn,
                budget=budget,
            ):
                yield emit(type_=event_type, payload=payload)
        except asyncio.CancelledError:
            mark_interrupted()
            yield emit(type_="aborted", payload=AbortedPayload(reason=turn.finish_reason))
            raise
        except GeneratorExit:
            mark_interrupted()
            raise
        except Exception as e:
            # 失败也要报已花掉的成本：放在 error 之前，保持 error 是最后一个事件
            yield emit(type_="usage", payload=usage_payload())
            yield emit(
                type_="error",
                payload=ErrorPayload(code="internal", message=str(e), stage=turn.stage),
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
                payload=FinalPayload(
                    text=turn.answer,
                    request_id=turn.request_id,
                    finish_reason=turn.finish_reason,
                    steps=budget.max_steps - budget.steps_remaining,
                ),
            )
