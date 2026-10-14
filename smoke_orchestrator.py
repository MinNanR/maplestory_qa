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
    9  步数预算到点：必须收口（不带 tools）**并产出 final** —— 不能只 break 不吐字
   10  字符预算：不足以再开一轮时直接收口；一轮之后跌破保留量也必须收口
   11  墙钟预算：到点后不再开工具轮（用 Step.delay 造真实耗时，不用长 sleep）
   12  预算参数不进 schema：模型看不到 max_chars，硬塞也改不了它
   13  单次输出上限：不接受该参数的工具不被注入打断 / 接受的上限真生效 / executor 会传值

关于 9~11：这三条守的是同一件事 —— **预算触顶是"降级收口"，不是"异常"**。
用户已经等了很久，此时回一句"超时"是最差结果；正确行为是用已有信息给出答案。

台阶 2 的 W0（14~18）—— **先写测试，再动生产代码**：
    14 工具执行中取消：取消必须传播 / 末尾事件是 aborted / trace 标记 interrupted   （W3）
    15 不完整 tool 组：投影后必须满足配对不变式（整组丢弃或补齐）                     （W1）
    16 窗口裁剪：丢最老的整轮组、system 在首位、最新 user 在末尾、不切碎消息组         （W1）
    17 跨轮证据：第 2 轮能带回第 1 轮取到的证据，且注入块不在 system 位置             （W2）
    18 前缀稳定：后一轮的 payload 以前一轮为逐字节前缀（缓存命中的前提）               （一直要绿）

关于 [PEND]：能力尚未落地时用 `pending("缺什么")` 报告，**不算失败**（否则天天红，
人会学会忽略它）；前置条件一旦成立，同一条用例里的断言全部真跑，该 FAIL 就 FAIL。
"""

import asyncio
import json
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import get_args

# Windows 控制台默认 GBK：用例描述里有中文与 emoji 级的符号，直接 print 会
# UnicodeEncodeError（backend/main.py 顶部踩过同一个坑）。统一按 UTF-8 输出。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from backend.config import settings
from backend.models import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from backend.prompts.system import get_wrap_up_note
from backend.tool.dispatcher import (
    ToolCallRequest,
    ToolOutput,
    build_tool_call_request,
    dispatcher,
)

import backend.conversation.projection as projection

# 编排器 import 失败（改契约、删旧模块、循环 import）是一个**必须立刻看到**的失败，
# 但不该让整个文件都跑不了：dispatcher / executor 层的用例不依赖它，仍然要给结论。
try:
    from backend.agent.orchestrator import Orchestrator

    ORCHESTRATOR_IMPORT_ERROR: BaseException | None = None
except BaseException as _e:  # noqa: BLE001
    Orchestrator = None
    ORCHESTRATOR_IMPORT_ERROR = _e

try:
    from backend.agent.execution import ToolExecutor
    from backend.observability.trace import KIND_TOOL, LLMSpan

    import backend.tool.tools_impl  # noqa: F401  注册真实知识工具（13 号用例要用）
    from backend.tool.tools_impl.knowledge import read_knowledge_sub_catalog

    EXECUTOR_IMPORT_ERROR: BaseException | None = None
except BaseException as _e:  # noqa: BLE001
    ToolExecutor = None
    LLMSpan = None
    KIND_TOOL = "tool"
    read_knowledge_sub_catalog = None
    EXECUTOR_IMPORT_ERROR = _e

# trace 落在这里而不是系统临时目录：一是可事后翻看，二是不依赖系统 temp 的写权限
# （受限环境下 %TEMP% 可能不可写，那会让"落盘失败"这条断言变成假失败）。
# 建议把它加进 .gitignore。
TRACE_ROOT = Path(__file__).resolve().parent / ".smoke_traces"

# 配置里的预算默认值，在 import 时抓一份快照。
# 用例之间会改 settings，若某条用例不显式传预算就"继承上一条用例改过的值"，
# 会得到随执行顺序变化的假失败 —— 所以每条用例都按这份快照复位。
DEFAULT_BUDGET = {
    "tool_chars": settings.max_tool_chars,
    "wall_s": settings.max_wall_s,
}


class Pending(BaseException):
    """用例的"前置能力还没实现"。刻意继承 BaseException：

    用例里到处是 `except Exception`（生产代码也是），继承 Exception 会被中途吞掉，
    表现成"用例莫名通过"——那比失败更糟。
    """


def pending(reason: str) -> None:
    raise Pending(reason)


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


BUDGET_ECHO: list[int | None] = []      # 记录工具实际收到的 max_chars


@dispatcher.register
def fake_budget_echo(max_chars: int | None = None) -> str:
    """假工具：记录本次收到的 max_chars，用来验证单次输出预算真的传到了工具层。

    它的 docstring 里**没有** max_chars 参数 —— 和真件里那些"由 dispatcher 注入
    预算"的工具同一种形状：模型在 schema 里看不到它，但函数签名收得到。
    """
    BUDGET_ECHO.append(max_chars)
    return "ok"


@dispatcher.register
def fake_big(size: int) -> str:
    """假工具：返回指定字符数，用来把累计字符预算打下去。

    Args:
        size (integer): 返回的字符数
    """
    return "字" * int(size)


@dispatcher.register
def fake_repeat(keyword: str) -> ToolOutput:
    """假工具：不管关键词是什么都返回**同一批**片段。

    用来模拟 agent 循环里最典型的空转：模型换了关键词、也确实拿到了返回，
    但证据和上一轮完全一样 —— 去重拦不住它（它并不是"完全相同的调用"）。

    Args:
        keyword (string): 任意关键词（返回结果与它无关）
    """
    return ToolOutput(
        text="[fake] 同一批片段",
        structured={"chunk_ids": ["fake#fixed"]},
    )


@dispatcher.register
def fake_slow(seconds: float) -> str:
    """假工具：睡指定秒数 —— 用来在"工具执行中"取消这一轮（用例 14）。

    Args:
        seconds (number): 睡眠秒数
    """
    time.sleep(float(seconds))
    return "睡完了"


@dispatcher.register
def fake_evidence(keyword: str) -> ToolOutput:
    """假检索工具：返回**真实存在**的 chunk_id（用例 17 测跨轮证据）。

    为什么必须是真 id：证据池的重投影要拿它回知识库取原文 —— 假 id 取不到任何
    东西，用例会永远停在 PEND，而看不出"到底哪一步没接上"。

    Args:
        keyword (string): 任意关键词（不影响返回的 id）
    """
    return ToolOutput(
        text=f"[fake] 假装检索了 {keyword}",
        structured={"chunk_ids": ["starforce#1", "starforce#4"]},
    )


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
    delay      → 这次调用"花掉"的秒数（墙钟用例专用；其余用例保持 0）
    """

    stream: tuple[str, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    delay: float = 0.0


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
        if step.delay:
            await asyncio.sleep(step.delay)     # 墙钟用例造耗时；其余用例 delay=0
        for chunk in step.stream:
            if span is not None and span.ttft_ms is None:
                span.ttft_ms = 1.0
            yield chunk
        self._fill(span, "".join(step.stream), "stop")

    async def stream_chat_with_tools(self, messages, tools, span=None, tool_acc=None):
        step = self._begin(messages, tools, span)
        if step.delay:
            await asyncio.sleep(step.delay)     # 墙钟用例造耗时；其余用例 delay=0
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

    preset：预置历史（用例 15/16 要造"被中断的半截轮""7 轮长历史"这类现场）。
    没有它就只能靠真跑几轮来产生历史，而"不完整的消息组"根本跑不出来。
    """

    def __init__(self, preset: list | None = None):
        self.added: list = []
        self._history: list = list(preset or [])
        self.policy = projection.HistoryPolicy(
            answer_turns=settings.answer_turns_projection,
            max_prompt_chars=settings.max_prompt_chars,
            max_total_tool_chars=settings.max_tool_chars_projection,
        )

    def get_message(self, cid):
        return list(self._history)

    def add_message(self, cid, message):
        self.added.append(message)
        self._history.append(message)

    def build_messages(self, cid):
        # 真件返回 [SystemMessage, *history]；这里用 SystemMessage 而不是
        # ChatMessage(...) —— 后者是 Annotated 类型别名，调用它会 TypeError
        working_messages = self.get_message(cid)
        projected_messages = projection.build_message(
            messages=working_messages, policy=self.policy
        )
        return [SystemMessage(content="SYS"), *projected_messages]


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


def assert_is_wrap_up(msg) -> None:
    """收口提示的不变量（刻意不绑定具体 reason 文案）。

    它是"必然终止"的最后一道保险，三个属性缺一不可：
      - 挂在**消息末尾**（不是 system：改 system 会击穿前缀缓存，见 001 §4.2）；
      - 是 UserMessage（部分 provider 不收非首条 system）；
      - 同时禁止"再要工具"和"输出工具调用格式"（少后半句会出现协议泄漏，001 §4.1）。
    """
    assert getattr(msg, "role", "") == "user", f"收口提示必须是 user 消息：{msg}"
    content = getattr(msg, "content", "") or ""
    assert "不要再请求调用工具" in content, f"收口提示必须明确禁止再调工具：{content}"
    assert "预算" in content, f"收口提示必须说明收口原因（reason 会拼进文案）：{content}"


def assert_prefix_stable(earlier, later) -> None:
    """后一次请求必须以前一次请求为**逐字节前缀**（用例 18）。

    这是 prompt cache 命中的前提（001 §4.2：工具模式的成本优势全靠它）。
    历史只会往后追加，所以 `later` 的开头必须与 `earlier` 完全一致。

    为什么值得单独守：有人把动态内容（跨轮证据块、临时提示）插到**前部**（尤其
    system）时，功能全都正常，只有成本和延迟变差 —— 指标上很难看出来，
    只能靠这条断言。
    """
    assert earlier and later, "两边都不该是空 payload"
    for i, msg in enumerate(earlier):
        assert i < len(later), (
            f"后一次请求反而更短：{len(earlier)} → {len(later)}"
            f"（第 {i} 条 role={getattr(msg, 'role', '?')} 不见了）"
        )
        assert earlier[i] == later[i], (
            f"第 {i} 条消息被改写（role={getattr(earlier[i], 'role', '?')}）"
            f"—— 动态内容插到了前部，前缀缓存会失效"
        )


def span_names(spans, kind: str) -> list[str]:
    return [s["name"] for s in spans if s.get("kind") == kind]


def tb(exc: BaseException | None, *, skip_frames: int = 2) -> str:
    """把异常现场格式化成可以塞进断言消息的字符串（含**生产代码的行号**）。

    run_turn() 把异常收进返回值，断言原本只剩 `f"...：{exc!r}"` —— 只有异常类型和
    message，traceback 是丢掉的。于是冒烟测试红的时候只能看到
    `TypeError("'NoneType' object is not iterable")` 这种"没有位置的报错"，
    还得另写脚本复现才能定位。用 tb(exc) 替代 `{exc!r}`，失败信息里直接带文件行号。

    skip_frames=2 跳过 traceback 自己的帧（tb 与调用它的用例函数）：
    最上面一帧就落在真正出错的代码上（通常是 backend/… 里的生产代码）。
    异常没有 traceback（比如我们自己 assert 造出来的）时退化成 `TypeError: msg`。
    """
    if exc is None:
        return "<none>"
    frames = list(traceback.extract_tb(exc.__traceback__))[skip_frames:]
    if not frames:
        return f"{type(exc).__name__}: {exc}"
    lines = traceback.format_list(frames)
    lines.append(f"{type(exc).__name__}: {exc}")
    body = "".join(lines).rstrip().replace("\n", "\n       ")
    return f"\n       {body}"


# ---------------------------------------------------------------------------
# 跑一轮
# ---------------------------------------------------------------------------


async def run_turn(
    llm,
    mgr,
    message: str = "你好",
    *,
    rounds: int = 1,
    tool_chars: int | None = None,
    wall_s: int | None = None,
):
    """跑一轮，返回 (events, exc, new_trace_files)。

    两者都要：`[e async for e in ...]` 在生成器抛异常时会把已收到的事件整段丢弃，
    所以这里手工收集 —— 失败路径的断言同样需要看到已产出的事件。

    trace 落到 .smoke_traces/（扁平目录，所有轮共用）：冒烟测试不该往 ./turn/ 里
    塞测试记录。返回的是**本轮新增**的文件（前后 diff），所以断言"落了一份 trace"
    不会受其他用例影响，也不依赖给每一轮单独建目录。

    预算通过 settings 注入（Orchestrator 就是从 settings 建 Budget 的），
    传 None 表示"用配置里的默认值"。main() 会在跑完后统一还原。
    """
    if Orchestrator is None:
        raise AssertionError(
            f"生产代码 import 失败，编排器用例跑不了：{ORCHESTRATOR_IMPORT_ERROR!r}"
        )
    settings.max_tool_rounds = rounds
    settings.max_tool_chars = (
        tool_chars if tool_chars is not None else DEFAULT_BUDGET["tool_chars"]
    )
    settings.max_wall_s = wall_s if wall_s is not None else DEFAULT_BUDGET["wall_s"]
    TRACE_ROOT.mkdir(exist_ok=True)
    before = set(TRACE_ROOT.glob("*.json"))
    orch = Orchestrator(llm_client=llm, conversation_manager=mgr, trace_dir=str(TRACE_ROOT))
    events: list = []
    exc: Exception | None = None
    try:
        async for event in orch.run("conv-1", message):
            events.append(event)
    except Exception as e:  # noqa: BLE001
        # 异常对象自带 __traceback__，原样带回去即可：断言里用 tb(exc) 打出生产代码行号，
        # 不必再另写复现脚本（见 tb() 的说明）。
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
    assert exc is None, f"不应抛异常：{(tb(exc))}"

    ts = types_of(events)
    assert "error" not in ts, f"不应有 error：{ts}"
    assert ts[-1] == "final", f"末事件必须是 final：{ts}"
    assert ts[1:4] == ["token"] * 3, f"应逐个吐 token：{ts}"
    assert_seq(events)
    assert events[-1].payload.text == "你好，世界", "final 必须携带完整答案"
    # 收口原因要能从事件里读到（评测的预算归因格靠它；None 会被当成"旧录制"）
    assert events[-1].payload.finish_reason == "no_tool_calls", \
        f"自然收尾必须标注 no_tool_calls：{events[-1].payload}"

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
    assert exc is None, f"不应抛异常：{(tb(exc))}"
    assert_no_error_event(events)

    ts = types_of(events)
    assert ts[1] == "token", f"前导文本应先流出去（保住 TTFT）：{ts}"
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
    assert_is_wrap_up(wrap_messages[-1])
    wrap_note = wrap_messages[-1].content
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
    assert exc is None, f"不应抛异常：{(tb(exc))}"
    assert_no_error_event(events)
    assert types_of(events)[-1] == "final"
    assert [bool(c["tools"]) for c in llm.calls] == [True, True, False], \
        f"轮次上限到了就必须停发 tools：{[bool(c['tools']) for c in llm.calls]}"
    assert len(of_type(events, "tool_result")) == 2, "两次工具都应有结果事件"
    assert_tool_pairs(llm.calls[-1]["messages"])
    assert_is_wrap_up(llm.calls[-1]["messages"][-1])
    assert events[-1].payload.text == "最终答案"


@case("4 工具执行抛异常：不产生 error 事件，模型拿到观察后仍能作答")
async def case_tool_raises():
    llm = FakeLLM([
        Step(tool_calls=(tc("c1", "fake_boom", keyword="x"),)),
        Step(stream=("工具挂了，我直接说。",)),
    ])
    mgr = FakeManager()
    events, exc, traces = await run_turn(llm, mgr, rounds=1)
    assert exc is None, f"工具异常绝不该冒泡成整轮失败：{(tb(exc))}"
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
    assert exc is None, f"参数错误不该抛异常：{(tb(exc))}"
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
    assert exc is None, f"去重路径不该抛异常：{(tb(exc))}"
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
    assert exc is None, f"不应抛异常：{(tb(exc))}"

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
    assert exc2 is None, f"第二轮不应抛异常：{(tb(exc2))}"
    sent = llm2.calls[0]["messages"]
    assert [m.role for m in sent] == ["system", "user", "assistant", "tool", "assistant", "user"], \
        f"第二轮请求应带完整历史：{[m.role for m in sent]}"
    assert_tool_pairs(sent)
    assert types_of(events2)[-1] == "final"


@case("9 步数预算到点：收口轮不带 tools，且必须产出 final")
async def case_steps_budget_wrap_up():
    llm = FakeLLM([
        Step(tool_calls=(tc("c1", "fake_lookup", keyword="budget"),)),
        Step(stream=("收口答案",)),
    ])
    mgr = FakeManager()
    events, exc, traces = await run_turn(llm, mgr, "预算到点", rounds=1)
    assert exc is None, f"不应抛异常：{(tb(exc))}"
    assert_no_error_event(events)

    ts = types_of(events)
    # 这条是收口改造最容易翻车的地方：预算触顶时若只 break、没真的发出收口请求，
    # 客户端连 final 都收不到 —— 表现是"转了半天，什么都没有"。
    assert ts[-1] == "final", f"收口路径必须产出 final：{ts}"
    assert events[-1].payload.text == "收口答案", "final 必须是收口轮给出的答案"
    assert events[-1].payload.finish_reason == "steps_exhausted", \
        f"步数触顶必须标注 steps_exhausted：{events[-1].payload}"
    assert "usage" in ts, "收口路径同样要报成本"
    assert_seq(events)

    assert [bool(c["tools"]) for c in llm.calls] == [True, False], \
        f"收口轮必须不带 tools：{[bool(c['tools']) for c in llm.calls]}"
    note = llm.calls[-1]["messages"][-1].content
    assert_is_wrap_up(llm.calls[-1]["messages"][-1])
    # reason 必须来自一个有限集合：评测要靠它把"预算不足"单独归因成一格
    assert any(
        note == get_wrap_up_note(r).content
        for r in ("steps_exhausted", "chars_exhausted", "wall_clock", "no_tool_calls")
    ), f"收口提示必须来自 get_wrap_up_note(reason)：{note}"
    assert note not in [getattr(m, "content", "") for m in mgr.added], \
        "收口提示不许落进会话历史"


@case("10 字符预算：不足以开新轮时直接收口，且必须给答案")
async def case_chars_budget_wrap_up():
    # (a) 预算小到连一轮都不该开：直接收口，而不是空转到步数上限
    llm = FakeLLM([Step(stream=("预算不足，直接答",))])
    events, exc, _ = await run_turn(llm, FakeManager(), "字符预算", rounds=8, tool_chars=100)
    assert exc is None, f"不应抛异常：{(tb(exc))}"
    assert_no_error_event(events)
    assert types_of(events)[-1] == "final", types_of(events)
    assert events[-1].payload.finish_reason == "chars_exhausted", \
        f"字符预算触顶必须标注 chars_exhausted：{events[-1].payload}"
    assert [bool(c["tools"]) for c in llm.calls] == [False], \
        f"预算不足以开新轮时，收口请求不能带 tools：{[bool(c['tools']) for c in llm.calls]}"

    # (b) 一轮工具之后跌破保留量：第二轮必须是收口，而不是继续检索
    llm2 = FakeLLM([
        Step(tool_calls=(tc("c1", "fake_big", size=2000),)),
        Step(stream=("基于已有信息作答",)),
    ])
    events2, exc2, _ = await run_turn(
        llm2, FakeManager(), "字符预算2", rounds=8, tool_chars=16000
    )
    assert exc2 is None, f"不应抛异常：{(tb(exc2))}"
    assert_no_error_event(events2)
    assert [bool(c["tools"]) for c in llm2.calls] == [True, False], \
        f"累计字符跌破保留量后应停止开新轮：{[bool(c['tools']) for c in llm2.calls]}"
    assert types_of(events2)[-1] == "final", types_of(events2)
    assert len(of_type(events2, "tool_result")) == 1, "预算已耗尽的那一轮不该再执行工具"


@case("11 墙钟预算：到点后不再开工具轮，且仍然给出答案")
async def case_wall_budget_wrap_up():
    # (a) 剩余时间不够再走一轮：一轮工具都不该开
    llm = FakeLLM([Step(stream=("时间不够，直接答",))])
    events, exc, _ = await run_turn(llm, FakeManager(), "墙钟", rounds=8, wall_s=1)
    assert exc is None, f"墙钟路径绝不能抛异常（超时不等于失败）：{(tb(exc))}"
    assert_no_error_event(events)
    assert [bool(c["tools"]) for c in llm.calls] == [False], \
        f"时间预算不足时不该开工具轮：{[bool(c['tools']) for c in llm.calls]}"
    assert types_of(events)[-1] == "final", types_of(events)
    assert events[-1].payload.finish_reason == "wall_clock", \
        f"墙钟触顶必须标注 wall_clock：{events[-1].payload}"

    # (b) 一轮工具真的花掉 4 秒，剩余时间跌破保留量 → 收口
    llm2 = FakeLLM([
        Step(
            stream=("先查一下",),
            tool_calls=(tc("c1", "fake_lookup", keyword="wall"),),
            delay=4.0,
        ),
        Step(stream=("收口答案",)),
    ])
    events2, exc2, _ = await run_turn(llm2, FakeManager(), "墙钟2", rounds=8, wall_s=18)
    assert exc2 is None, f"不应抛异常：{(tb(exc2))}"
    assert_no_error_event(events2)
    assert [bool(c["tools"]) for c in llm2.calls] == [True, False], \
        f"耗时吃掉预算后应停止开新轮：{[bool(c['tools']) for c in llm2.calls]}"
    assert types_of(events2)[-1] == "final", types_of(events2)


@case("12 预算参数不进 schema：模型看不到 max_chars，硬塞也到不了工具")
async def case_budget_param_hidden():
    schema = dispatcher.build_schema()
    for tool in schema:
        props = ((tool.get("function") or {}).get("parameters") or {}).get("properties") or {}
        assert "max_chars" not in props, \
            f"{tool['function']['name']} 把 max_chars 暴露给模型了 —— 预算不该由模型决定：{props}"

    # 模型硬塞一个 max_chars：必须被忽略。
    # 它不在 docstring 里 → 不在 properties 里 → dispatch 只按 properties 取参数，
    # 于是它连工具函数都到不了（工具拿到的仍是自己的默认 None）。
    BUDGET_ECHO.clear()
    result = dispatcher.dispatch(
        ToolCallRequest(
            tool_call_id="t1",
            name="fake_budget_echo",
            params=json.dumps({"max_chars": 9_999_999}),
        )
    )
    assert result.succeeded, result.error_msg
    assert BUDGET_ECHO == [None], \
        f"模型给的 max_chars 不该传到工具（工具应拿到默认值）：{BUDGET_ECHO}"


@case("13 输出预算归工具自己管：调度层不注入 / 失败仍是观察 / 工具自身预算仍生效")
async def case_output_budget_is_tool_local():
    if EXECUTOR_IMPORT_ERROR is not None:
        raise AssertionError(f"executor / 工具层 import 失败：{EXECUTOR_IMPORT_ERROR!r}")

    # (0) 调度层不注入任何预算参数：没有预算概念的工具拿到的是它的默认值。
    #     曾经有过"dispatcher 给工具塞 max_chars"的隐式约定，代价是每个工具都必须
    #     记得声明它，忘一个就在运行期变成 TypeError（报错还像参数写错）—— 已整条移除。
    BUDGET_ECHO.clear()
    r0 = dispatcher.dispatch(ToolCallRequest(tool_call_id="t0", name="fake_budget_echo"))
    assert r0.succeeded, r0.error_msg
    assert BUDGET_ECHO == [None], f"调度层不该注入 max_chars：{BUDGET_ECHO}"

    # (1) 工具自己抛异常要变成"观察"，而不是整轮失败；失败原因不得被调度层污染
    span = LLMSpan(name="tool_call:fake_boom", kind=KIND_TOOL, t0=time.perf_counter())
    r1 = await ToolExecutor().run_tool(
        build_tool_call_request(tc("c1", "fake_boom", keyword="x")), span
    )
    assert r1.succeeded is False, "fake_boom 本来就该失败（它验证的是失败原因）"
    assert "boom in tool" in (r1.error_msg or ""), \
        f"失败原因必须是工具自己的，不能被别的层覆盖：{r1.error_msg}"

    # (1b) 真实目录工具照常可调（它不接受任何预算参数）
    result = dispatcher.dispatch(
        ToolCallRequest(tool_call_id="t1b", name="read_knowledge_catalog")
    )
    assert result.succeeded, f"目录工具调用失败：{result.error_msg}"

    # (2) 工具**自身的**预算仍然生效（输出上限归工具负责，但预算不能消失）：
    #     直接调工具函数，小预算 < 大预算。
    small = read_knowledge_sub_catalog(folder="job_skill", max_chars=300)
    big = read_knowledge_sub_catalog(folder="job_skill", max_chars=30_000)
    assert len(small) <= 300 + 200, f"工具自身的小预算没有生效：{len(small)} 字符"
    assert len(big) > len(small), \
        "大预算应能取回更多内容 —— 否则说明工具级预算根本没生效"




# ---------------------------------------------------------------------------
# W0（台阶 2）：先写测试，再动生产代码
# ---------------------------------------------------------------------------
# 这五条守的是"接下来要做的三件事"的对外契约：
#   14          取消/中断（W3）
#   15/16/17    投影层与证据池（W1 / W2）
#   18          缓存前缀（一直要绿）
# 约定：能力没落地时用 pending(...) 说明缺什么（[PEND]，不算失败）；
#       前置条件一旦成立，同一条用例里的断言全部真跑（可能 [FAIL]）。
# ---------------------------------------------------------------------------


def _agent_event_types() -> set[str]:
    """AgentEvent.type 的取值集合（读 Literal，不硬编码）。"""
    from backend.agent.events import AgentEvent

    return set(get_args(AgentEvent.__annotations__["type"]))


def _projection():
    """投影层模块（W1 的产物，候选路径 backend/conversation/projection.py）。"""
    try:
        from backend.conversation import projection

        return projection, ""
    except BaseException as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _preset_rounds(n: int) -> list:
    """造 n 个"完整轮组"，最后一轮带一次工具交互（验证裁剪不切碎消息组）。"""
    msgs: list = []
    for i in range(n - 1):
        msgs.append(UserMessage(content=f"第{i}问"))
        msgs.append(AssistantMessage(content=f"第{i}答"))
    last = n - 1
    msgs.append(UserMessage(content=f"第{last}问"))
    msgs.append(
        AssistantMessage(
            content="我查一下",
            tool_calls=[tc("ct", "fake_lookup", keyword="win")],
        )
    )
    msgs.append(ToolMessage(tool_call_id="ct", content="[fake] 命中 win 的原文片段"))
    msgs.append(AssistantMessage(content=f"第{last}答"))
    return msgs


@case("14 工具执行中取消：取消必须传播 / 末尾事件是 aborted / trace 标记 interrupted")
async def case_cancel_during_tool():
    from backend.observability.trace import TurnContext

    missing = []
    if "aborted" not in _agent_event_types():
        missing.append("AgentEvent.type 里没有 'aborted'")
    if not hasattr(TurnContext, "interrupted"):
        missing.append("TurnContext 没有 interrupted 字段")
    if missing:
        pending("W3（事务性/中断）未实现：" + "；".join(missing))

    llm = FakeLLM([
        Step(tool_calls=(tc("c1", "fake_slow", seconds=1.5),)),
        Step(stream=("不该走到这里",)),
    ])
    mgr = FakeManager()
    orch = Orchestrator(
        llm_client=llm, conversation_manager=mgr, trace_dir=str(TRACE_ROOT)
    )
    events: list = []

    async def consume():
        async for e in orch.run("conv-cancel", "取消测试"):
            events.append(e)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.4)          # 让工具睡在中间（fake_slow 睡 1.5 秒）
    task.cancel()
    cancelled = False
    try:
        await task
    except asyncio.CancelledError:
        cancelled = True

    assert cancelled, "取消必须传播到调用方（被吞掉就变成「看起来正常结束」）"
    assert events, "取消之前至少应该已经产出过事件"
    assert events[-1].type == "aborted", (
        f"取消路径的最后一个事件必须是 aborted：{[e.type for e in events]}"
    )
    assert not of_type(events, "final"), "被取消的轮不该再产出 final"

    traces = sorted(TRACE_ROOT.glob("*.json"), key=lambda p: p.stat().st_mtime)
    assert traces, "取消也要落 trace，否则事后查不出「半截轮」"
    data = json.loads(traces[-1].read_text(encoding="utf-8"))
    assert data.get("interrupted") is True, (
        f"trace 必须标记 interrupted：{data.get('interrupted')!r}"
    )
    assert data.get("finished") is False, (
        f"被取消的轮 finished 必须是 False：{data.get('finished')!r}"
    )


@case("15 不完整 tool 组：投影后必须满足配对不变式（整组丢弃或补齐）")
async def case_orphan_tool_group():
    proj, err = _projection()
    if proj is None:
        pending(f"W1（投影层）未实现：backend/conversation/projection 不可用（{err}）")

    # 现场：assistant 声明了两条 tool_calls，但只回来一条 —— 中断/崩溃后的真实形态
    broken = [
        UserMessage(content="第一个问题"),
        AssistantMessage(
            content="我查一下",
            tool_calls=[
                tc("c1", "fake_lookup", keyword="a"),
                tc("c2", "fake_lookup", keyword="b"),
            ],
        ),
        ToolMessage(tool_call_id="c1", content="只有这一条回来了"),
    ]
    llm = FakeLLM([Step(stream=("第二个答案",))])
    mgr = FakeManager(preset=broken)
    events, exc, _ = await run_turn(llm, mgr, "第二个问题", rounds=1)
    assert exc is None, f"不完整的旧历史不该让整轮失败：{(tb(exc))}"
    assert_no_error_event(events)
    # 硬断言：不满足就是 FAIL（不是 PEND）—— provider 侧少一条直接 400
    assert_tool_pairs(llm.calls[0]["messages"])


@case("16 窗口裁剪：丢最老的整轮组 / system 在首位 / 最新 user 在末尾 / 不切碎消息组")
async def case_history_window():
    proj, err = _projection()
    if proj is None:
        pending(f"W1（投影层）未实现：backend/conversation/projection 不可用（{err}）")

    llm = FakeLLM([Step(stream=("答",))])
    mgr = FakeManager(preset=_preset_rounds(30))
    events, exc, _ = await run_turn(llm, mgr, "第30问", rounds=1)
    assert exc is None, f"长历史不该让整轮失败：{(tb(exc))}"
    assert_no_error_event(events)

    sent = llm.calls[0]["messages"]
    roles = [m.role for m in sent]
    contents = [getattr(m, "content", "") or "" for m in sent]

    assert roles[0] == "system", f"system 必须在第 0 位：{roles[:3]}"
    assert roles[-1] == "user" and contents[-1] == "第30问", (
        f"最新的用户消息必须在末尾：{roles[-2:]} / {contents[-2:]}"
    )
    assert not any("第0问" in c for c in contents), (
        "30 轮历史里最老的一轮必须被裁掉（否则等于没有窗口）"
    )
    assert any("第29问" in c for c in contents), "最近的轮次必须保留"
    assert_tool_pairs(sent)          # 裁剪不能切碎最后一个工具组



@case("17 前缀稳定：后一轮的 payload 以前一轮为逐字节前缀（缓存命中的前提）")
async def case_prefix_stability():
    first = FakeLLM([
        Step(tool_calls=(tc("c1", "fake_lookup", keyword="prefix"),)),
        Step(stream=("答一",)),
    ])
    mgr = FakeManager()
    events, exc, _ = await run_turn(first, mgr, "第一问", rounds=1)
    assert exc is None, f"第一轮不该失败：{(tb(exc))}"
    assert_no_error_event(events)
    first_payload = first.calls[0]["messages"]

    second = FakeLLM([Step(stream=("答二",))])
    events2, exc2, _ = await run_turn(second, mgr, "第二问", rounds=1)
    assert exc2 is None, f"第二轮不该失败：{(tb(exc2))}"
    assert_no_error_event(events2)

    assert_prefix_stable(first_payload, second.calls[0]["messages"])


@case("18 无进展收口：换了关键词但拿不到新证据 → 必须判 no_progress")
async def case_no_progress_wrap_up():
    """L1 的回归网：连续两轮"有返回、没新证据"就必须收口。

    现场刻意构造成最阴的形态（去重**拦不住**的那种）：
      - 第 1 轮拿到 fake#fixed；
      - 第 2/3 轮各发两条：一条与上一轮完全相同（会被去重），另一条换了关键词
        但返回的还是同一批 chunk_id。
    于是一轮里同时满足"round_evidence 非空"和"saw_duplicate_call 为真" ——
    这正是"重复调用"与"没有新证据"的交叉点，也是停滞判定最容易漏掉的地方。
    """
    llm = FakeLLM([
        Step(tool_calls=(tc("c1", "fake_repeat", keyword="a"),)),
        Step(
            tool_calls=(
                tc("c2", "fake_repeat", keyword="a"),   # 与 c1 完全相同 → 被去重
                tc("c3", "fake_repeat", keyword="b"),   # 换了关键词，证据不变
            )
        ),
        Step(
            tool_calls=(
                tc("c4", "fake_repeat", keyword="a"),
                tc("c5", "fake_repeat", keyword="b"),
            )
        ),
        Step(stream=("基于已有信息作答",)),
    ])
    mgr = FakeManager()
    events, exc, _ = await run_turn(llm, mgr, "无进展", rounds=3)
    assert exc is None, f"不应抛异常：{tb(exc)}"
    assert_no_error_event(events)
    assert types_of(events)[-1] == "final", types_of(events)

    # 关键断言：收口原因必须是 no_progress。
    # 若这里是 steps_exhausted / no_tool_calls，说明循环是"把预算烧完才停"，
    # 而不是"识别出原地打转" —— 归因矩阵会永远看不到 no_progress 这一格。
    assert events[-1].payload.finish_reason == "no_progress", (
        "连续两轮没有新证据时应判 no_progress，实际："
        f"{events[-1].payload.finish_reason!r}"
    )


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def main() -> int:
    saved = (
        settings.max_tool_rounds,
        settings.tool_timeout_s,
        settings.max_tool_chars,
        settings.max_wall_s,
    )
    settings.tool_timeout_s = 5          # 冒烟测试不该等 10 分钟
    shutil.rmtree(TRACE_ROOT, ignore_errors=True)

    if ORCHESTRATOR_IMPORT_ERROR is not None:
        print(
            "[BLOCKED] 生产代码 import 失败，编排器用例会全部报错："
            f"{type(ORCHESTRATOR_IMPORT_ERROR).__name__}: {ORCHESTRATOR_IMPORT_ERROR}"
        )
        print("          先修 import 链再跑；dispatcher / executor 层用例不受影响。")
    if EXECUTOR_IMPORT_ERROR is not None:
        print(
            "[BLOCKED] executor 层 import 失败："
            f"{type(EXECUTOR_IMPORT_ERROR).__name__}: {EXECUTOR_IMPORT_ERROR}"
        )

    failures: list[tuple[str, BaseException]] = []
    pendings: list[tuple[str, str]] = []
    try:
        for desc, fn in CASES:
            try:
                await fn()
            except Pending as p:
                # 能力还没落地：不算失败，但必须点名缺什么（见文件头对 [PEND] 的说明）
                pendings.append((desc, str(p)))
                print(f"[PEND] {desc}\n       {p}")
            except BaseException as e:  # noqa: BLE001 - 逐用例收集，跑完再汇总
                failures.append((desc, e))
                print(f"[FAIL] {desc}\n       {type(e).__name__}: {e}")
            else:
                print(f"[ ok ] {desc}")
    finally:
        (
            settings.max_tool_rounds,
            settings.tool_timeout_s,
            settings.max_tool_chars,
            settings.max_wall_s,
        ) = saved

    print()
    print(f"trace 落在：{TRACE_ROOT}")
    if failures:
        print(f"smoke FAILED — {len(failures)}/{len(CASES)} 个用例没过：")
        for desc, e in failures:
            print(f"  - {desc}  ({type(e).__name__})")
        return 1
    if pendings:
        print(
            f"smoke ok — {len(CASES) - len(pendings)}/{len(CASES)} 个用例通过，"
            f"{len(pendings)} 条待实现（W0 已就位，能力落地后自动转绿）："
        )
        for desc, reason in pendings:
            print(f"  [PEND] {desc}\n         {reason}")
    else:
        print(f"smoke ok — {len(CASES)}/{len(CASES)} 个用例全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
