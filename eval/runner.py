"""评测入口（采集侧）。

职责边界：**只负责采集**。
    逐用例驱动 Orchestrator → 收集事件 → 压成 TurnRecord → 落盘 RunRecord。
打分不在这里（见 eval/metrics.py），这样改断言/加指标不需要重跑 LLM。

两种运行方式都支持：
    python -m eval.runner     # 从仓库根目录
    python eval/runner.py     # 任意目录

用例选择（调试期的关键能力，全量一轮要十几分钟，单条只要几秒）：
    python -m eval.runner                                   # 全量
    python -m eval.runner --case job-adele-hexa              # 单条（可重复传）
    python -m eval.runner --pattern "eval/case/negative.jsonl"   # 整类
    python -m eval.runner --limit 3                         # 只录前 N 条
    python -m eval.runner --usage                           # 联调：验证流式 usage 能采到
"""

from __future__ import annotations

import argparse
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
    ToolCallPayload,
    ToolResultPayload,
    UsagePayload,
)
from backend.conversation.manager import ConversationManager  # noqa: E402
from backend.knowledge.retrieval import get_chunk_text  # noqa: E402
from backend.llm.client import LLMClient  # noqa: E402
from backend.tool.tools_impl.knowledge import RETRIEVAL_TOOL_NAMES  # noqa: E402

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
    """把一轮的事件流压成结构化事实。

    证据口径（工具模式下唯一正确的来源）：
      - did_retrieve   ← 本轮是否发起过检索类工具调用（tool_call 事件）
      - injected_*_ids ← 工具**实际返回**的片段 id
                         （tool_result 事件的 structured["chunk_ids"]）

    注意分层：dispatcher 只透传 structured、不认识任何键；"chunk_ids 是知识检索的
    证据键、且能从它派生出文档级召回"这条领域知识**属于评测层**（它本来就是知识问答
    评测器）。所以将来接外部搜索时，dispatcher 和编排器一行都不用改，
    只需要在下面加一条对应键的采集规则。

    刻意不再从 stage 事件里读 retrieval_result：那条链路属于已删除的管道模式，
    它现在永远不会触发，而失败形态是"指标全线归零、报告却一字不差"——
    是这份评测里最容易被误读的一种错。
    """
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
        elif event.type == "tool_call":
            payload: ToolCallPayload = event.payload
            if payload.tool in RETRIEVAL_TOOL_NAMES:
                did_retrieve = True
        elif event.type == "tool_result":
            payload: ToolResultPayload = event.payload
            # structured 由工具自报，可能根本没有这个键（目录类工具、失败路径）
            chunk_ids = (payload.structured or {}).get("chunk_ids") or []
            # 保序去重：多次调用/多关键词可能返回同一片段
            for cid in chunk_ids:
                if cid not in injected_chunk_ids:
                    injected_chunk_ids.append(cid)
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
    injected_doc_ids = sorted({cid.split("#")[0] for cid in injected_chunk_ids})

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


def select_cases(
    all_cases: list[Case],
    *,
    case_ids: list[str] | None = None,
    limit: int | None = None,
) -> tuple[list[Case], list[str]]:
    """按 --case / --limit 从已加载的用例里筛选。

    返回 (选中的用例, 没匹配上的 case id)。没匹配上必须回报：`--case` 拼错一个字母
    就会静默录一条空 run —— 那是"没有结果"被当成"结果为空"，最难发现的一种错。

    --pattern 不在这里处理：它是"从哪些文件加载"，由 load_cases 的 glob 参数负责。
    同一件事只留一处实现，否则两处过滤条件迟早会分歧。
    """
    cases = all_cases
    unmatched: list[str] = []

    if case_ids:
        wanted = set(case_ids)
        by_id = {c.id: c for c in cases}
        unmatched = [cid for cid in case_ids if cid not in by_id]
        cases = [c for c in cases if c.id in wanted]

    if limit is not None:
        cases = cases[: max(0, limit)]

    return cases, unmatched


def run_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.runner",
        description="评测录制：驱动 Orchestrator 跑用例并落盘（打分见 eval.metrics）。",
    )
    parser.add_argument(
        "--case", action="append", default=None, metavar="CASE_ID",
        help="只录指定 case id（可重复传，与 --pattern/--limit 叠加）",
    )
    parser.add_argument(
        "--pattern", default=None, metavar="GLOB",
        help="用例文件 glob，默认 eval/case/*.jsonl",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="最多录前 N 条（按加载顺序）",
    )
    parser.add_argument(
        "--run-dir", default="./run_dir", metavar="DIR",
        help="录制根目录（默认 ./run_dir）",
    )
    parser.add_argument(
        "--usage", action="store_true",
        help="联调小工具：验证流式 usage 能不能采到（走真实 API，会花钱）",
    )
    return parser


async def run(
    run_result_folder: str = "./run_dir",
    *,
    pattern: str | None = None,
    case_ids: list[str] | None = None,
    limit: int | None = None,
) -> RunRecord:
    loaded = load_cases(pattern) if pattern else load_cases()

    cases, unmatched = select_cases(
        loaded, case_ids=case_ids, limit=limit
    )
    if unmatched:
        print(f"[WARN] --case 里有 {len(unmatched)} 个 id 没匹配到：{unmatched}")
    if not cases:
        # 关键守卫：静默录一条空 run 比报错难查得多
        sample = ", ".join(sorted(c.id for c in loaded)[:8])
        print(
            f"选中的用例数为 0（可用 case 共 {len(loaded)} 条）。"
            f"检查 --case 的拼写 / --pattern / --limit；例如：{sample} ..."
        )
        sys.exit(2)
    if len(cases) < len(loaded):
        print(f"用例筛选：{len(cases)}/{len(loaded)} 条")
        print(
            "[WARN] 这是子集录制：总体均值不能与全量 run 直接对比"
            "（逐轮明细里那几行仍然可比）"
        )

    # 校验在筛选**之后**做：--case 的意义是"聚焦一条快速迭代"，
    # 不该被另一条无关用例的锚点问题挡住。全量 run 仍然是完整的发布闸门。
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


async def main(argv: list[str] | None = None) -> int:
    args = run_arg_parser().parse_args(argv)
    if args.usage:
        await test_usage()
        return 0
    await run(
        run_result_folder=args.run_dir,
        pattern=args.pattern,
        case_ids=args.case,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
