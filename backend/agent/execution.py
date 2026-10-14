import asyncio
from dataclasses import asdict
import json
import time

from backend.observability.trace import LLMSpan
from backend.tool.dispatcher import (
    dispatcher,
    ToolDispatcher,
    ToolCallRequest,
    ToolOutput,
    ToolCallResult,
)
from backend.config import settings


class ToolExecutor:
    def __init__(
        self,
        dispatcher: ToolDispatcher = dispatcher,
        *,
        per_call_timeout_s: float | None = 20,
        max_parallel: int = 4,
    ):
        self.dispatcher = dispatcher
        self.tool_seen: set[str] = set()  # 用于去重，记录已经执行过的工具调用ID
        self.per_call_timeout_s = per_call_timeout_s
        self.max_parallel = max_parallel

    @staticmethod
    def _dedup_key(request: ToolCallRequest) -> str:
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
        return call_key

    async def run_tool(self, request: ToolCallRequest, span: LLMSpan) -> ToolCallResult:
        call_key = self._dedup_key(request)

        if call_key in self.tool_seen:
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
                execute_result = await asyncio.wait_for(
                    asyncio.to_thread(self.dispatcher.dispatch, request),
                    timeout=self.per_call_timeout_s or settings.tool_timeout_s,
                )
                if isinstance(execute_result, BaseException):
                    result = ToolCallResult(
                        tool_call_id=request.tool_call_id,
                        name=request.name,
                        succeeded=False,
                        error_msg=f"工具调度失败：{execute_result!r}",
                    )
                else:
                    result = execute_result
            
                if call_key:
                    self.tool_seen.add(call_key)
            except Exception as e:
                result = ToolCallResult(
                    tool_call_id=request.tool_call_id,
                    name=request.name,
                    succeeded=False,
                    error_msg=f"工具执行失败：{e}",
                )

        # 统一记账出口：成功 / 失败 / 参数校验失败 / 去重**都**走这里。
        # 这段以前写在 else 分支里，于是"去重拦下"和"dispatch 抛异常"两条路径
        # 既不写 span.detail 也不写 duration —— trace 里那是一行空白：
        # 评测既算不出失败率与去重数，也看不到它花了多久（先挂后填原则的反面）。
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
        if span.t0 is not None:
            span.duration_ms = (time.perf_counter() - span.t0) * 1000
        return result

    async def run_batch(
        self, calls: list[tuple[LLMSpan, ToolCallRequest]]
    ) -> list[ToolCallResult]:
        sem = asyncio.Semaphore(self.max_parallel)

        async def execute_with_semaphore(
            request: ToolCallRequest, span: LLMSpan
        ) -> ToolCallResult:
            async with sem:
                return await self.run_tool(request, span)

        tasks = [
            asyncio.create_task(execute_with_semaphore(request, span))
            for span, request in calls
        ]

        results = await asyncio.gather(*tasks)
        return results