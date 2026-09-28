# smoke_orchestrator.py  （仓库根目录，运行：.\.venv\Scripts\python.exe smoke_orchestrator.py）
"""冒烟测试：不联网、不要 API key，跑通工具化之后的 Orchestrator.run()。

这是这个项目唯一的**零成本回归网**：改编排器 / 改 dispatcher / 改事件契约之后，
先跑它，再决定值不值得烧钱跑全量评测。

为什么从"fail-fast 的 assert"改成"逐用例收集失败、最后汇总"：
    改动密集期一次可能同时弄坏好几条契约，fail-fast 会在第一条就停下 ——
    修一个跑一次，太慢。现在每个用例互相独立（各自建假件、各自落自己的临时 trace 目录），
    一次跑完能把所有坏掉的契约列出来。

覆盖的契约（每条都对应一次真实踩过的坑）：
    1  无工具回合：事件序列 / 只调一次 LLM / 第一轮必须带 tools / 落历史 / trace 落盘
    2  工具回合：tool_call + tool_result / 收口那轮不带 tools / 消息配对 / structured 透传
    3  连续两轮工具：轮次上限到点后不再给 tools（这是"必然终止"的保证）
    4  工具执行抛异常：不产生 error 事件，模型拿到观察后仍能作答
    5  参数校验失败：同上，且错误文本指向缺失的参数
    6  相同参数重复调用：被拦下、工具只真跑一次、但仍回灌一条 tool 消息
    7  LLM 失败：usage + error + raise（错误必须走事件，异常必须同时抛给调用方）
    8  会话历史：user/assistant/tool 的落库顺序，以及"配对不变式"在真实 payload 上成立
"""

import asyncio
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.agent.orchestrator import Orchestrator
from backend.config import settings
from backend.models import ToolCall
from backend.prompts.system import get_wrap_up_note
from backend.tool.dispatcher import ToolOutput, dispatcher

# trace 落在这里而不是系统临时目录：一是可事后翻看，二是不依赖系统 temp 的写权限
# （受限环境下 %TEMP% 可能不可写，那会让"落盘失败"这条断言变成假失败）。
# 建议把它加进 .gitignore。
TRACE_ROOT = Path(__file__).resolve().parent / ".smoke_traces"


# ---------------------------------------------------------------------------
# 假工具：注册到全局 dispatcher（编排器用的就是这一个实例）
# ---------------------------------------------------------------------------

FAKE_LOOKUP_CALLS: list[str] = []      # 记录真实执行过的关键词，用来验证"去重没有真跑"


@dispatcher.register
def fake_lookup(keyword: str) -> ToolOutput:
    """假检索工具（冒烟测试专用，返回固定内容）。

    何时用：任何时候都能调，它只用来验证编排链路。
    何时不用：无。

    Args:
        keyword (string): 任意关键词
    """
    FAKE_LOOKUP_CALLS.append(keyword)
    return ToolOutput(
        text=f"[fake] 命中 {keyword} 的原文片段",
        structured={"chunk_ids": [f"fake#{keyword}"]},
    )


@dispatcher.register
def fake_boom(keyword: str) -> str:
    """假工具：一定抛异常，用来验证"工具异常 = 观察，而不是整轮失败"。

    Args:
        keyword (string): 任意关键词
    """
    raise RuntimeError("boom in tool")


@dispatcher.register
def fake_needs_arg(required_arg: str) -> str:
    """假工具：漏传必填参数时走参数校验失败路径。

    Args:
        required_arg (string): 必传参数，冒烟测试会故意不传
    """
    return "never reached"


# ---------------------------------------------------------------------------
# 假 LLM：行为由脚本驱动
# ---------------------------------------------------------------------------


def tc(call_id: str, name: str, **args) -> ToolCall:
    """构造一个工具调用（arguments 必须是 JSON 字符串，与真件一致）。"""
    return ToolCall(id=call_id, name=name, arguments=json.dumps(args, ensure_ascii=False))


@dataclass
class Step:
    """假 LLM 一次调用的行为。

    stream     → 这次调用流式吐出的文本分片（工具轮里就是"前导文本"）
    tool_calls → 非空表示这次调用要求调工具；空表示直接作答
    """

    stream: tuple[str, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()


class FakeLLM:
    """签名对齐真实 LLMClient 的两个流式方法。

    注意：真件是 async generator 函数，调用它得到的是**异步生成器对象**，
    所以编排器必须 `async for ... in client.stream_chat(...)`，
    **不能** `await client.stream_chat(...)`。假件保持同样的形状，
    这样这个错误在冒烟测试里就会立刻暴露。
    """

    def __init__(self, steps, fail_on: int | None = None, tokens=(100, 20)):
        self.steps = list(steps)
        self.fail_on = fail_on          # 第 N 次调用抛异常（从 1 起）
        self.tokens = tokens
        self.calls: list[dict] = []     # 每次调用的记账，供断言

    def _begin(self, messages, tools, span) -> Step:
        self.calls.append({
            "n": len(self.calls) + 1,
            "messages": list(messages),
            "tools": list(tools) if tools else [],
        })
        if self.fail_on == len(self.calls):
            raise RuntimeError(f"boom in llm call #{len(self.calls)}")
        if not self.steps:
            raise AssertionError(
                f"FakeLLM 的脚本用尽了（这是第 {len(self.calls)} 次调用）——"
                f"说明编排器多调了一轮，或终止条件没生效"
            )
        return self.steps.pop(0)

    def _fill(self, span, text: str, finish_reason: str) -> None:
        if span is None:                # 假件也要填 span，否则 usage 契约测不到
            return
        span.response = text
        span.finish_reason = finish_reason
        span.input_tokens, span.output_tokens = self.tokens

    async def stream_chat(self, messages, span=None):
        step = self._begin(messages, None, span)
        for chunk in step.stream:
            if span is not None and span.ttft_ms is None:
                span.ttft_ms = 1.0
            yield chunk
        self._fill(span, "".join(step.stream), "stop")

    async def stream_chat_with_tools(self, messages, tools, span=None, tool_acc=None):
        step = self._begin(messages, tools, span)
        for chunk in step.stream:
            if span is not None and span.ttft_ms is None:
                span.ttft_ms = 1.0
            yield chunk
        if step.tool_calls:
            # 真件按 index 累积；假件直接把整条塞进去，编排器只读 values()
            for i, call in enumerate(step.tool_calls):
                tool_acc[i] = call
        self._fill(span, "".join(step.stream), "tool_calls" if step.tool_calls else "stop")


class FakeManager:
    """假会话管理器：只记录调用事实，不碰真实知识库。

    必须与真件契约一致的两点：
      - build_messages() 每次返回**新列表**（真件就是新建，绝不能让工作副本污染历史）；
      - 只存 user / assistant / tool 三种消息，不掺别的东西。
    """

    def __init__(self):
        self.added: list = []
        self._history: list = []

    def get_message(self, cid):
        return list(self._history)

    def add_message(self, cid, message):
        self.added.append(message)
        self._history.append(message)

    def build_messages(self, cid):
        # 真件返回 [SystemMessage, *history]；这里用 SystemMessage 而不是
        # ChatMessage(...) —— 后者是 Annotated 类型别名，调用它会 TypeError
        from backend.models import SystemMessage

        return [SystemMessage(content="SYS"), *self._history]


# ---------------------------------------------------------------------------
# 断言助手
# ---------------------------------------------------------------------------


def types_of(events) -> list[str]:
    return [e.type for e in events]


def of_type(events, type_: str) -> list:
    return [e for e in events if e.type == type_]


def event_size(event) -> int:
    return len(json.dumps(asdict(event.payload), ensure_ascii=False))


def assert_seq(events) -> None:
    seqs = [e.seq for e in events]
    assert seqs == list(range(1, len(events) + 1)), f"seq 必须从 1 连续递增：{seqs}"


def assert_events_carry_no_body(events, limit: int = 1500) -> None:
    """守卫：上下文正文（知识块几千字符）不许回到事件流。

    阈值不是精确指标，而是量级守卫：usage 事件天然带每 span 摘要，几百字符属正常，
    但工具返回的正文（fake 里是短文本，真件里可达 6000 字符）一旦漏进事件流就会击穿它。
    """
    biggest = max(events, key=event_size)
    size = event_size(biggest)
    assert size < limit, f"事件 {biggest.type} 有 {size} 字符，疑似把正文塞进了事件流"


def assert_no_error_event(events) -> None:
    errs = of_type(events, "error")
    assert not errs, f"不应出现 error 事件：{[e.payload.message for e in errs]}"


def assert_tool_pairs(messages) -> None:
    """配对不变式：每个带 tool_calls 的 assistant 消息，其后必须紧跟覆盖全部 id 的 tool 消息。

    这是 provider 侧的真实约束（少一条就 400），也是"工具消息要不要落历史"那个
    设计讨论的硬前提 —— 所以它必须在真实 payload 上被验证，而不是靠人肉 review。
    """
    pending: list[str] = []
    for m in messages:
        role = getattr(m, "role", "?")
        if role == "assistant":
            assert not pending, f"上一条 assistant 的 tool_calls 没被回应就来了下一条：{pending}"
            pending = [c.id for c in (getattr(m, "tool_calls", None) or [])]
        elif role == "tool":
            cid = getattr(m, "tool_call_id", "")
            assert cid in pending, f"tool 消息的 tool_call_id={cid!r} 不在待回应的 {pending} 里"
            pending.remove(cid)
        else:
            assert not pending, f"tool_calls 还没回应就出现了 {role} 消息：{pending}"
    assert not pending, f"结尾仍有未被回应的 tool_calls：{pending}"


def span_names(spans, kind: str) -> list[str]:
    return [s["name"] for s in spans if s.get("kind") == kind]


# ---------------------------------------------------------------------------
# 跑一轮
# ---------------------------------------------------------------------------


async def run_turn(llm, mgr, message: str = "你好", *, rounds: int = 1):
    """跑一轮，返回 (events, exc, new_trace_files)。

    两者都要：`[e async for e in ...]` 在生成器抛异常时会把已收到的事件整段丢弃，
    所以这里手工收集 —— 失败路径的断言同样需要看到已产出的事件。

    trace 落到 .smoke_traces/（扁平目录，所有轮共用）：冒烟测试不该往 ./turn/ 里
    塞测试记录。返回的是**本轮新增**的文件（前后 diff），所以断言"落了一份 trace"
    不会受其他用例影响，也不依赖给每一轮单独建目录。
    """
    settings.max_tool_rounds = rounds
    TRACE_ROOT.mkdir(exist_ok=True)
    before = set(TRACE_ROOT.glob("*.json"))
    orch = Orchestrator(llm_client=llm, conversation_manager=mgr, trace_dir=str(TRACE_ROOT))
    events: list = []
    exc: Exception | None = None
    try:
        async for event in orch.run("conv-1", message):
            events.append(event)
    except Exception as e:  # noqa: BLE001
        exc = e
    new_files = sorted(set(TRACE_ROOT.glob("*.json")) - before)
    return events, exc, new_files


CASES: list[tuple[str, object]] = []


def case(desc: str):
    def deco(fn):
        CASES.append((desc, fn))
        return fn

    return deco


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------


@case("1 无工具回合：事件序列 / 单次调用 / 第一轮带 tools / 落历史 / trace 落盘")
async def case_plain():
    llm = FakeLLM([Step(stream=("你好", "，", "世界"))])
    mgr = FakeManager()
    events, exc, traces = await run_turn(llm, mgr)
    assert exc is None, f"不应抛异常：{exc!r}"

    ts = types_of(events)
    assert "error" not in ts, f"不应有 error：{ts}"
    assert ts[-1] == "final", f"末事件必须是 final：{ts}"
    assert ts[:3] == ["token"] * 3, f"应逐个吐 token：{ts}"
    assert_seq(events)
    assert events[-1].payload.text == "你好，世界", "final 必须携带完整答案"

    # 无工具回合只调一次 LLM，而且**这一次必须带 tools** —— 否则模型永远没机会调工具
    assert len(llm.calls) == 1, f"无工具回合应只调一次 LLM：{len(llm.calls)}"
    assert llm.calls[0]["tools"], "第一轮必须把 tools 传给模型"
    assert [m.role for m in llm.calls[0]["messages"]] == ["system", "user"], \
        f"第一次请求应是 system+user：{[m.role for m in llm.calls[0]['messages']]}"

    assert [m.role for m in mgr.added] == ["user", "assistant"], "user / assistant 各落一次"

    usage = of_type(events, "usage")
    assert len(usage) == 1, f"应有且仅有一个 usage 事件：{ts}"
    assert ts[-2] == "usage", f"usage 应在 final 之前：{ts}"
    u = usage[0].payload
    assert u.llm_calls == 1, f"无工具回合 llm_calls 应为 1：{u.llm_calls}"
    assert (u.input_tokens, u.output_tokens) == (100, 20), f"token 应等于各 span 之和：{u}"
    assert u.usage_missing is False, "假件已填 token，不应报 usage_missing"
    assert span_names(u.spans, "llm") == ["token"], f"llm span 名单：{u.spans}"
    assert all("messages" not in s and "response" not in s for s in u.spans), \
        "事件流里的 span 摘要不许带 messages/response"

    assert_events_carry_no_body(events)

    # trace 落盘：正文只进 trace 文件（这条以前被注释掉过，所以单独断言）
    assert len(traces) == 1, f"应恰好落一份 trace：{traces}"
    data = json.loads(traces[0].read_text(encoding="utf-8"))
    spans = data["trace"]["spans"]
    assert len(spans) == 1 and spans[0]["response"] == "你好，世界", \
        f"trace 里必须留得下完整 response：{spans}"


@case("2 工具回合：tool_call/tool_result / 收口轮不带 tools / 配对 / structured 透传")
async def case_tool_round():
    llm = FakeLLM([
        Step(stream=("我查一下。",), tool_calls=(tc("call_1", "fake_lookup", keyword="adele"),)),
        Step(stream=("答", "案")),
    ])
    mgr = FakeManager()
    FAKE_LOOKUP_CALLS.clear()
    events, exc, traces = await run_turn(llm, mgr, "阿黛尔的 HEXA 技能有哪些？", rounds=1)
    assert exc is None, f"不应抛异常：{exc!r}"
    assert_no_error_event(events)

    ts = types_of(events)
    assert ts[0] == "token", f"前导文本应先流出去（保住 TTFT）：{ts}"
    assert ts.count("tool_call") == 1 and ts.count("tool_result") == 1, ts
    assert ts.index("tool_call") < ts.index("tool_result"), f"顺序必须是 call 先于 result：{ts}"
    assert ts[-1] == "final", ts
    assert_seq(events)

    call_ev = of_type(events, "tool_call")[0].payload
    assert call_ev.tool == "fake_lookup" and call_ev.call_id == "call_1", call_ev
    assert json.loads(call_ev.args) == {"keyword": "adele"}, call_ev.args

    result_ev = of_type(events, "tool_result")[0].payload
    assert result_ev.structured == {"chunk_ids": ["fake#adele"]}, \
        f"structured 必须原样透传（评测靠它取证据）：{result_ev.structured}"
    assert len(result_ev.result) <= 200, "事件里只放预览，不放正文"

    # 收口那一轮**不能**带 tools —— 这是"必然终止"的唯一保证
    assert len(llm.calls) == 2, f"工具回合应是两次调用（决策 + 收口）：{len(llm.calls)}"
    assert llm.calls[0]["tools"], "决策轮必须带 tools"
    assert not llm.calls[1]["tools"], "收口轮必须不带 tools，否则可能再次要求调工具"

    # 第二次请求的 payload：工具配对必须成立，且末尾挂着"收口提示"
    wrap_messages = llm.calls[1]["messages"]
    wrap_note = get_wrap_up_note().content
    assert wrap_messages[-1].content == wrap_note, \
        "收口轮的最后一条必须是收口提示（告诉模型别再要工具了）"
    assert wrap_messages[-1].role == "user", \
        "收口提示要用 UserMessage：非首条 system 消息在部分 provider 上不被接受"
    # 提示本身只活在这一轮工作副本里，绝不能落进会话历史 ——
    # 否则下一轮模型会以为"用户"真的说过这句话
    assert wrap_note not in [getattr(m, "content", "") for m in mgr.added], \
        "收口提示不许落进会话历史"
    assert_tool_pairs(wrap_messages)
    roles = [m.role for m in wrap_messages]
    assert roles == ["system", "user", "assistant", "tool", "user"], \
        f"回灌后的消息顺序：{roles}"

    # 落库顺序：user → assistant(带 tool_calls) → tool → assistant(最终答案)
    assert [m.role for m in mgr.added] == ["user", "assistant", "tool", "assistant"], \
        f"落库顺序：{[m.role for m in mgr.added]}"
    assert mgr.added[1].tool_calls[0].id == mgr.added[2].tool_call_id == "call_1", \
        "assistant 的 tool_calls.id 必须与 tool 消息的 tool_call_id 一致"
    assert events[-1].payload.text == "答案", "final 应是收口轮的答案，不含前导文本"

    assert_events_carry_no_body(events)
    assert FAKE_LOOKUP_CALLS == ["adele"], f"工具应被真实执行一次：{FAKE_LOOKUP_CALLS}"

    # span 契约：llm 与 tool 分开，tool span 带 name / parent / detail / tool_output
    u = of_type(events, "usage")[0].payload
    assert span_names(u.spans, "llm") == ["token", "token"], u.spans
    assert span_names(u.spans, "tool") == ["tool_call:fake_lookup"], u.spans
    tool_span = [s for s in u.spans if s["kind"] == "tool"][0]
    assert tool_span["detail"]["tool_call_result"]["structured"] == {"chunk_ids": ["fake#adele"]}
    assert tool_span["detail"]["tool_call_result"]["chars_count"] > 0
    assert tool_span["duration_ms"] > 0, "工具 span 必须有耗时（且单位是 ms）"
    assert u.llm_calls == 2, u.llm_calls

    # 工具正文只进 trace 文件，不进事件流
    data = json.loads(traces[0].read_text(encoding="utf-8"))
    tspan = [s for s in data["trace"]["spans"] if s["kind"] == "tool"][0]
    assert tspan["tool_output"] == "[fake] 命中 adele 的原文片段", tspan["tool_output"]
    assert "tool_output" not in json.dumps(asdict(u)), "tool_output 不许出现在事件流摘要里"
    # 因果链只在 trace 文件里（summary 不带 parent_span_id）：
    # 工具 span 应挂在产生它的那次 LLM span 上
    llm_span_ids = [s["span_id"] for s in data["trace"]["spans"] if s["kind"] == "llm"]
    assert tspan["parent_span_id"] in llm_span_ids, \
        f"工具 span 的 parent 应指向产生它的 LLM span：{tspan['parent_span_id']} 不在 {llm_span_ids}"


@case("3 连续两轮工具：轮次上限到点后不再给 tools")
async def case_rounds_cap():
    llm = FakeLLM([
        Step(tool_calls=(tc("c1", "fake_lookup", keyword="one"),)),
        Step(tool_calls=(tc("c2", "fake_lookup", keyword="two"),)),
        Step(stream=("最终答案",)),
    ])
    mgr = FakeManager()
    events, exc, traces = await run_turn(llm, mgr, "连查两次", rounds=2)
    assert exc is None, f"不应抛异常：{exc!r}"
    assert_no_error_event(events)
    assert types_of(events)[-1] == "final"
    assert [bool(c["tools"]) for c in llm.calls] == [True, True, False], \
        f"轮次上限到了就必须停发 tools：{[bool(c['tools']) for c in llm.calls]}"
    assert len(of_type(events, "tool_result")) == 2, "两次工具都应有结果事件"
    assert_tool_pairs(llm.calls[-1]["messages"])
    assert llm.calls[-1]["messages"][-1].content == get_wrap_up_note().content, \
        "用满轮次后的收口调用必须带上收口提示"
    assert events[-1].payload.text == "最终答案"


@case("4 工具执行抛异常：不产生 error 事件，模型拿到观察后仍能作答")
async def case_tool_raises():
    llm = FakeLLM([
        Step(tool_calls=(tc("c1", "fake_boom", keyword="x"),)),
        Step(stream=("工具挂了，我直接说。",)),
    ])
    mgr = FakeManager()
    events, exc, traces = await run_turn(llm, mgr, rounds=1)
    assert exc is None, f"工具异常绝不该冒泡成整轮失败：{exc!r}"
    assert_no_error_event(events)
    assert types_of(events)[-1] == "final", "工具失败也必须产出 final"

    # 失败是"观察"：错误文本回灌给模型，而不是抛异常
    tool_msgs = [m for m in mgr.added if m.role == "tool"]
    assert len(tool_msgs) == 1, f"即使失败也必须回灌一条 tool 消息：{[m.role for m in mgr.added]}"
    assert "boom in tool" in tool_msgs[0].content, tool_msgs[0].content
    assert tool_msgs[0].tool_call_id == "c1", "失败路径也必须带上正确的 tool_call_id"

    u = of_type(events, "usage")[0].payload
    tool_span = [s for s in u.spans if s["kind"] == "tool"][0]
    assert tool_span["detail"]["tool_call_result"]["succeeded"] is False, tool_span["detail"]


@case("5 参数校验失败：错误文本指向缺失参数，且不炸整轮")
async def case_bad_params():
    llm = FakeLLM([
        Step(tool_calls=(tc("c1", "fake_needs_arg"),)),
        Step(stream=("参数不对，我换个说法。",)),
    ])
    mgr = FakeManager()
    events, exc, traces = await run_turn(llm, mgr, rounds=1)
    assert exc is None, f"参数错误不该抛异常：{exc!r}"
    assert_no_error_event(events)
    tool_msgs = [m for m in mgr.added if m.role == "tool"]
    assert len(tool_msgs) == 1, "参数失败也必须回灌 tool 消息（否则下一次请求 400）"
    assert "required_arg" in tool_msgs[0].content, \
        f"错误信息必须点名缺失的参数，模型才可能自纠：{tool_msgs[0].content}"


@case("6 相同参数重复调用：被拦下 / 工具只真跑一次 / 仍回灌 tool 消息")
async def case_dedup():
    same = tc("c1", "fake_lookup", keyword="same")
    same_again = tc("c2", "fake_lookup", keyword="same")   # 参数完全相同，只是 id 不同
    llm = FakeLLM([Step(tool_calls=(same,)), Step(tool_calls=(same_again,)), Step(stream=("好了",))])
    mgr = FakeManager()
    FAKE_LOOKUP_CALLS.clear()
    events, exc, traces = await run_turn(llm, mgr, rounds=2)
    assert exc is None, f"去重路径不该抛异常：{exc!r}"
    assert_no_error_event(events)
    assert FAKE_LOOKUP_CALLS == ["same"], f"第二次不该真的执行工具：{FAKE_LOOKUP_CALLS}"

    tool_msgs = [m for m in mgr.added if m.role == "tool"]
    assert len(tool_msgs) == 2, f"去重也必须回灌 tool 消息，否则配对断裂：{len(tool_msgs)}"
    # 去重的返回是"引导复用已有结果"，不是"调用失败" —— 后者会诱导模型反复重试
    assert "完全相同的参数" in tool_msgs[1].content, \
        f"第二次应是引导文本：{tool_msgs[1].content}"
    assert "调用失败" not in tool_msgs[1].content, \
        f"去重不能报成失败：{tool_msgs[1].content}"
    assert tool_msgs[1].tool_call_id == "c2", "去重回灌也必须用本次的 tool_call_id"
    assert_tool_pairs(llm.calls[-1]["messages"])

    u = of_type(events, "usage")[0].payload
    deduped = [s for s in u.spans if s["kind"] == "tool"
               and s["detail"]["tool_call_result"]["deduped"]]
    assert len(deduped) == 1, f"trace 里必须能看出有一次被去重：{u.spans}"


@case("7 LLM 失败：usage + error + raise（错误走事件，异常同时抛给调用方）")
async def case_llm_failure():
    llm = FakeLLM([Step(stream=("不会走到这里",))], fail_on=1)
    mgr = FakeManager()
    events, exc, traces = await run_turn(llm, mgr)

    ts = types_of(events)
    assert ts.count("error") == 1, f"应恰好一个 error 事件：{ts}"
    assert ts[-1] == "error", f"error 必须是最后一个事件：{ts}"
    assert "final" not in ts, f"失败时不应有 final：{ts}"
    assert "token" not in ts, f"失败发生在生成前，不该有 token：{ts}"
    assert ts[-2] == "usage", f"失败也要报已花掉的成本（在 error 之前）：{ts}"
    assert "boom in llm call #1" in events[-1].payload.message, "error 事件必须带上原因"
    assert exc is not None, "契约要求异常同时抛给调用方"
    assert "boom in llm call #1" in str(exc), "抛出的异常必须保留原始原因"

    # 失败路径仍然要落 trace（finally 里落盘），否则"失败前烧了多少"就查不到了
    assert len(traces) == 1, f"失败也要落 trace：{traces}"


@case("8 会话历史：落库顺序 / build_messages 返回新列表 / 真实 payload 满足配对不变式")
async def case_history():
    llm = FakeLLM([
        Step(tool_calls=(tc("c1", "fake_lookup", keyword="hist"),)),
        Step(stream=("第一轮答案",)),
    ])
    mgr = FakeManager()
    events, exc, traces = await run_turn(llm, mgr, "第一个问题", rounds=1)
    assert exc is None, f"不应抛异常：{exc!r}"

    # build_messages 必须是新列表：否则编排器往工作副本里 append 工具消息，
    # 会顺手改写会话历史（真件里就是新建列表，这里把这条不变量钉住）
    a = mgr.build_messages("conv-1")
    b = mgr.build_messages("conv-1")
    assert a is not b, "build_messages 必须每次返回新列表"
    a.append("脏数据")
    assert "脏数据" not in mgr.build_messages("conv-1"), "工作副本不许污染历史"

    hist = mgr.get_message("conv-1")
    assert_tool_pairs(hist)
    assert [m.role for m in hist] == ["user", "assistant", "tool", "assistant"], \
        f"历史形状：{[m.role for m in hist]}"

    # 第二轮对话：历史应被完整带上（含上一轮的工具消息）
    llm2 = FakeLLM([Step(stream=("第二轮答案",))])
    events2, exc2, _ = await run_turn(llm2, mgr, "第二个问题", rounds=1)
    assert exc2 is None, f"第二轮不应抛异常：{exc2!r}"
    sent = llm2.calls[0]["messages"]
    assert [m.role for m in sent] == ["system", "user", "assistant", "tool", "assistant", "user"], \
        f"第二轮请求应带完整历史：{[m.role for m in sent]}"
    assert_tool_pairs(sent)
    assert types_of(events2)[-1] == "final"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def main() -> int:
    saved_rounds, saved_timeout = settings.max_tool_rounds, settings.tool_timeout_s
    settings.tool_timeout_s = 5          # 冒烟测试不该等 10 分钟
    shutil.rmtree(TRACE_ROOT, ignore_errors=True)
    failures: list[tuple[str, BaseException]] = []
    try:
        for desc, fn in CASES:
            try:
                await fn()
            except BaseException as e:  # noqa: BLE001 - 逐用例收集，跑完再汇总
                failures.append((desc, e))
                print(f"[FAIL] {desc}\n       {type(e).__name__}: {e}")
            else:
                print(f"[ ok ] {desc}")
    finally:
        settings.max_tool_rounds, settings.tool_timeout_s = saved_rounds, saved_timeout

    print()
    print(f"trace 落在：{TRACE_ROOT}")
    if failures:
        print(f"smoke FAILED — {len(failures)}/{len(CASES)} 个用例没过：")
        for desc, e in failures:
            print(f"  - {desc}  ({type(e).__name__})")
        return 1
    print(f"smoke ok — {len(CASES)}/{len(CASES)} 个用例全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
