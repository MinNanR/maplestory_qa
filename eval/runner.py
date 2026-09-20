"""评测入口（采集侧）。

职责边界：**只负责采集**。
    逐用例驱动 Orchestrator → 收集事件 → 压成 TurnRecord → 落盘 RunRecord。
打分不在这里（见 eval/metrics.py），这样改断言/加指标不需要重跑 LLM。

两种运行方式都支持：
    python -m eval.runner     # 从仓库根目录
    python eval/runner.py     # 任意目录
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# 以脚本方式运行（python eval/runner.py）时，sys.path[0] 是 eval/ 而不是仓库根目录，
# 于是 `import backend` / `import eval.case` 都会失败。这里显式补上仓库根目录。
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agent.orchestrator import (  # noqa: E402
    AgentEvent,
    ErrorPayload,
    FinalPayload,
    Orchestrator,
    StagePayload,
    UsagePayload,
)
from backend.conversation.manager import ConversationManager  # noqa: E402
from backend.knowledge.retrieval import get_chunk_text  # noqa: E402
from backend.llm.client import LLMClient  # noqa: E402

from eval.case import Case, load_cases, validate_cases  # noqa: E402
from eval.record import (  # noqa: E402
    CaseRecord,
    RunRecord,
    TurnRecord,
    build_meta,
    record_run,
)


def safe_name(name: str) -> str:
    """中和路径分隔符与 Windows 保留字符。

    case.id 会被当作目录名使用。案例 id `multi-blackmage-archmage(f/p)-01` 里的 `/`
    会让 `mkdir` 把目录拆成两层（`...(f/` + `p)-01/`），导致每个 case 的落盘结构不一致。
    """
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_ .")
    return cleaned or "case"


def collect_turn(
    query: str, events: list[AgentEvent], turn_idx: int, t0: float, ttft_ms: float | None
) -> TurnRecord:
    """把一轮的事件流压成结构化事实。"""
    request_id, answer, error = "", "", None
    injected_chunk_ids: list[str] = []
    injected_doc_ids: list[str] = []
    did_retrieve = False
    usage: dict = {}

    for event in events:
        if event.type == "final":
            payload: FinalPayload = event.payload
            answer = payload.text
            request_id = payload.request_id
        elif event.type == "stage":
            payload: StagePayload = event.payload
            if payload.name == "retrieval":
                did_retrieve = True
                if payload.state == "end":
                    injected_chunk_ids = list(payload.stage_context.get("retrieval_result") or [])
                    injected_doc_ids = sorted({cid.split("#")[0] for cid in injected_chunk_ids})
        elif event.type == "usage":
            payload: UsagePayload = event.payload
            usage = {
                "llm_calls": payload.llm_calls,
                "input_tokens": payload.input_tokens,
                "output_tokens": payload.output_tokens,
                "usage_estimated": payload.estimated,
                "spans": payload.spans,
            }
        elif event.type == "error":
            payload: ErrorPayload = event.payload
            error = payload.message

    # 注入片段的文本：录制自带证据，之后改锚点/加指标不必重跑 LLM。
    chunk_texts = [t for cid in injected_chunk_ids if (t := get_chunk_text(cid))]

    return TurnRecord(
        turn_idx=turn_idx,
        request_id=request_id,
        query=query,
        answer=answer,
        error=error,
        injected_chunk_ids=injected_chunk_ids,
        injected_doc_ids=injected_doc_ids,
        injected_chunk_texts=chunk_texts,
        did_retrieve=did_retrieve,
        latency_ms=(time.perf_counter() - t0) * 1000,
        ttft_ms=ttft_ms,
        llm_calls=usage.get("llm_calls"),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        usage_estimated=bool(usage.get("usage_estimated")),
        spans=list(usage.get("spans") or []),
    )


async def run_case(
    orch: Orchestrator, case: Case, case_id: str
) -> tuple[list[TurnRecord], Exception | None]:
    """跑完一个用例的全部轮次。

    返回 (已收集的轮记录, 失败原因)，**不抛异常**。
    原因：`turns = await run_case(...)` 一旦抛异常，赋值就不会发生，调用方手里
    仍是一个空列表 —— 失败轮的 error 文本、部分答案、注入片段会全部丢失，
    报告里那条轮只能显示成 not_run，最该看的诊断信息反而没了。
    """
    turns: list[TurnRecord] = []
    for turn_idx, turn_expect in enumerate(case.turns):
        t0 = time.perf_counter()
        events: list[AgentEvent] = []
        ttft_ms: float | None = None
        failure: Exception | None = None

        try:
            async for event in orch.run(case_id, turn_expect.query):
                if ttft_ms is None and event.type == "token":
                    ttft_ms = (time.perf_counter() - t0) * 1000     # 用户感知首字延迟（含分析+检索）
                events.append(event)
        except Exception as e:  # noqa: BLE001 - 先收集部分事件，再交给调用方
            failure = e
        finally:
            turns.append(collect_turn(turn_expect.query, events, turn_idx, t0, ttft_ms))

        if failure is not None:
            return turns, failure
    return turns, None


async def run(run_result_folder: str = "./run_dir") -> RunRecord:
    cases = load_cases()

    problems = validate_cases(cases)
    if problems:
        print(f"评测集非法，共 {len(problems)} 个问题：")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    llm_client = LLMClient()
    run_id = uuid.uuid4().hex[:8]
    run_dir = f"{run_result_folder.rstrip('/')}/{run_id}"
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    case_records: list[CaseRecord] = []
    failed_cases = 0

    for case_idx, case in enumerate(cases):
        case_id = f"eval-{case.id}-{uuid.uuid4().hex[:8]}"     # 每题一个全新会话
        manager = ConversationManager()                        # 每题一个干净的管理器
        turns: list[TurnRecord] = []
        case_t0 = time.perf_counter()
        failure: Exception | None = None

        try:
            orch = Orchestrator(
                llm_client, manager, trace_dir=f"{run_dir}/{safe_name(case.id)}"
            )
            turns, failure = await run_case(orch, case, case_id)
        except Exception as e:  # noqa: BLE001 - 单题失败不能毁掉整轮评测
            failure = e
        finally:
            case_records.append(
                CaseRecord(
                    case=asdict(case),                         # 用例快照：录制自包含
                    turns=turns,
                    status="error" if failure else "ok",
                )
            )
            manager.clear(case_id)
            elapsed = (time.perf_counter() - case_t0) * 1000
            if failure:
                failed_cases += 1
                print(f"测试 {case_idx + 1}/{len(cases)} 失败（{elapsed:.0f} ms）：{case.id} -> {failure}")
            else:
                print(f"测试 {case_idx + 1}/{len(cases)} 完成（{elapsed:.0f} ms）：{case.id}")

    run_record = RunRecord(
        run_id=run_id,
        started_at=started_at,
        meta=build_meta(cases),
        cases=case_records,
    )
    path = record_run(run_record, save_dir=run_dir)

    print()
    print(f"录制已保存：{path}")
    print(f"用例 {len(cases)} 个，失败 {failed_cases} 个；run_id={run_id}")
    print("打分请执行：python -m eval.metrics " + str(path))
    return run_record


async def test_usage() -> None:
    """联调小工具：验证流式 usage 真能采到（走真实 API，会花钱）。

    运行：python -m eval.runner --usage
    """
    from backend.models import ChatMessage
    from backend.observability.trace import TurnTrace, print_turn_trace

    llm = LLMClient()
    trace = TurnTrace(request_id="usage-probe", conversation_id="usage-probe")
    span = trace.start_span("generation", model="usage-probe", stream=True)

    async for piece in llm.stream_chat(
        [ChatMessage(role="user", content="你是谁？")], span=span
    ):
        print(piece, end="", flush=True)

    print()
    print_turn_trace(trace)
    if span.input_tokens is None:
        print("[WARN] 没拿到 usage：确认 provider 是否支持 stream_options.include_usage "
              "（或 settings.llm_stream_include_usage 被关掉了）")


async def main() -> None:
    await run(run_result_folder="./run_dir")


if __name__ == "__main__":
    if "--usage" in sys.argv:
        asyncio.run(test_usage())
    else:
        asyncio.run(main())
