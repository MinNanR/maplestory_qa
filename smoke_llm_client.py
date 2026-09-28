# smoke_llm_client.py  （仓库根目录，运行：.\.venv\Scripts\python.exe smoke_llm_client.py）
"""冒烟测试：真实 LLMClient 全链路，**不联网、不要 API key**。

为什么还要有这一份 —— smoke_orchestrator.py 里的 FakeLLM 把 LLMClient **整个替换掉**了，
于是 backend/llm/client.py 从来没被执行过。后果是一串 bug 同时潜伏在"每次真实调用
都必经"的路上，而单用例录制的全部症状只有一句
`'SystemMessage' object has no attribute 'tool_calls'`（2026-09-27），无从定位：

    1. _as_message：`match` 里写了裸类名 —— 那是**捕获模式**不是类模式，
       于是任何非 assistant 的消息一取 .tool_calls 就 AttributeError；
       而 system 消息在每次请求的第一条，等于**每一次调用都失败**。
    2. stream_chat_with_tools：把 `chunk.choices[0]`（Choice）当成 delta 用，
       正文/工具增量其实在 `.delta` 下 —— 一有工具调用就 AttributeError。
    3. tool_acc：`if not tool_acc: tool_acc = {}` 把调用方传进来的**空 dict 顶掉**，
       工具调用被填进没人读的副本 —— 静默失效，工具链路全线不工作。
    4. 线格式：assistant.tool_calls 发的是扁平 {id,name,arguments}，而接口要
       {id,type,function:{name,arguments}} —— 第二轮带工具历史时被 provider 400。

做法：盖掉 transport（httpx.MockTransport），其余全走生产代码路径。
断言直接打在**最终 HTTP 请求体**上 —— 那才是 provider 真正看到的东西。
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import traceback
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from backend.agent.orchestrator import Orchestrator
from backend.conversation.manager import ConversationManager
from backend.llm.client import LLMClient, _as_payload
from backend.models import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from backend.observability.trace import KIND_LLM, TurnTrace
from backend.tool.dispatcher import ToolOutput, dispatcher

# trace 落在这里（同 smoke_orchestrator）：一是可事后翻看，二是不依赖 %TEMP% 写权限
TRACE_ROOT = Path(__file__).resolve().parent / ".smoke_traces_llm"

SSE_HEADERS = {"content-type": "text/event-stream"}


# ---------------------------------------------------------------------------
# 假工具：注册到全局 dispatcher（编排器用的就是这一个实例）
# ---------------------------------------------------------------------------


@dispatcher.register
def smoke_probe(keyword: str) -> ToolOutput:
    """冒烟测试专用的假检索工具（固定返回，不读知识库）。

    何时用：验证工具链路契约时。
    何时不用：无。

    Args:
        keyword (string): 任意关键词
    """
    return ToolOutput(
        text=f"[smoke] 命中 {keyword} 的原文片段",
        structured={"chunk_ids": [f"smoke#{keyword}"]},
    )


# ---------------------------------------------------------------------------
# 假 provider：把 SSE 流按真实形状拼出来
# ---------------------------------------------------------------------------


def chunk(delta: dict, finish: str | None = None) -> dict:
    return {
        "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "fake",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def usage_chunk(input_tokens: int = 11, output_tokens: int = 7) -> dict:
    """usage 尾块的 choices **是空列表** —— 照抄 choices[0] 会 IndexError。"""
    return {
        "id": "c", "object": "chat.completion.chunk", "created": 0, "model": "fake",
        "choices": [],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def sse(chunks: list[dict]) -> str:
    body = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


class FakeProvider:
    """收请求、回 SSE。保留**真实收到的请求体**，断言就打在这上面。"""

    def __init__(self, responder):
        self.responder = responder
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        return httpx.Response(200, headers=SSE_HEADERS, text=self.responder(body))


def make_llm(provider: FakeProvider) -> LLMClient:
    """真 LLMClient，只把 HTTP transport 换成假的（其余都是生产代码）。"""
    llm = LLMClient()
    llm.client = AsyncOpenAI(
        api_key="fake-key",
        base_url="http://fake.local/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
        max_retries=0,
    )
    return llm


def roles(body: dict) -> list[str]:
    return [m["role"] for m in body["messages"]]


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

CASES: list[tuple[str, object]] = []


def case(desc: str):
    def deco(fn):
        CASES.append((desc, fn))
        return fn

    return deco


@case("1 payload：四种角色各按 role 分流，未知类型显式报错，整体可 json 序列化")
async def case_payload_shape():
    messages = [
        SystemMessage(content="sys"),
        UserMessage(content="u"),
        AssistantMessage(
            content="a",
            reasoning_content="想一下",
            tool_calls=[ToolCall(id="call_1", name="smoke_probe", arguments='{"keyword":"x"}')],
        ),
        ToolMessage(content="r", tool_call_id="call_1"),
    ]
    payload = _as_payload(messages)

    assert payload[0] == {"role": "system", "content": "sys"}, payload[0]
    assert payload[1] == {"role": "user", "content": "u"}, payload[1]
    assert payload[2]["role"] == "assistant", payload[2]
    assert payload[3] == {"role": "tool", "content": "r", "tool_call_id": "call_1"}, payload[3]

    # 落 trace 时整段 payload 要能直接 json.dump：pydantic 模型混进来会 persist failed
    json.dumps(payload, ensure_ascii=False)

    try:
        _as_payload([object()])
    except TypeError:
        pass
    else:
        raise AssertionError("未知消息类型必须抛 TypeError，不能静默返回 None")


@case("2 线格式：tool_calls 必须是 {id,type,function:{name,arguments}}")
async def case_wire_format():
    payload = _as_payload([
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="call_1", name="smoke_probe", arguments='{"keyword":"x"}')],
        )
    ])
    sent = payload[0]["tool_calls"][0]

    assert sent == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "smoke_probe", "arguments": '{"keyword":"x"}'},
    }, f"发出去的 tool_calls 形状不对（provider 会 400）：{sent}"


@case("3 流式解析：分片的 tool_calls 累加进【调用方传入的那个】dict")
async def case_tool_acc():
    def responder(body):
        return sse([
            chunk({"role": "assistant", "content": "我查一下。"}),
            # 真实 provider 会把 arguments 切成多块：id/name 只在第一块
            chunk({"tool_calls": [{
                "index": 0, "id": "call_9", "type": "function",
                "function": {"name": "smoke_probe", "arguments": '{"key'},
            }]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'word": "x"}'}}]}),
            chunk({}, "tool_calls"),
            usage_chunk(),
        ])

    provider = FakeProvider(responder)
    llm = make_llm(provider)
    acc: dict[int, ToolCall] = {}                     # 调用方的累加器
    span = TurnTrace(request_id="r", conversation_id="c").start_span("token", KIND_LLM)

    pieces = [
        p async for p in llm.stream_chat_with_tools(
            messages=[SystemMessage(content="sys"), UserMessage(content="u")],
            tools=dispatcher.build_schema(),
            span=span,
            tool_acc=acc,
        )
    ]

    assert pieces == ["我查一下。"], f"正文应逐块吐出：{pieces}"
    assert acc, "工具调用没有被累加进调用方传入的 dict（acc 被顶掉了？）"
    tc = acc[0]
    assert (tc.id, tc.name) == ("call_9", "smoke_probe"), tc
    assert tc.arguments == '{"keyword": "x"}', f"分片参数没拼全：{tc.arguments!r}"
    assert span.finish_reason == "tool_calls", span.finish_reason
    assert span.detail["tool_calls"][0]["name"] == "smoke_probe", span.detail


@case("4 端到端：工具回合走通，第二轮请求体里的配对与形状正确，trace 落盘无 error")
async def case_end_to_end():
    def responder(body):
        if "tool" in roles(body):
            return sse([
                chunk({"role": "assistant", "content": "最终答案"}),
                chunk({}, "stop"),
                usage_chunk(),
            ])
        return sse([
            chunk({"role": "assistant", "content": "我先查一下。"}),
            chunk({"tool_calls": [{
                "index": 0, "id": "call_1", "type": "function",
                "function": {"name": "smoke_probe", "arguments": '{"keyword": "adele"}'},
            }]}),
            chunk({}, "tool_calls"),
            usage_chunk(),
        ])

    provider = FakeProvider(responder)
    llm = make_llm(provider)
    TRACE_ROOT.mkdir(exist_ok=True)
    before = set(TRACE_ROOT.glob("*.json"))
    orch = Orchestrator(
        llm_client=llm,
        conversation_manager=ConversationManager(),
        trace_dir=str(TRACE_ROOT),
    )

    events = [e async for e in orch.run("conv-smoke", "阿黛尔的 HEXA 技能是什么？")]

    kinds = [e.type for e in events]
    assert "error" not in kinds, f"不应有 error 事件：{kinds}"
    assert kinds[-1] == "final", f"末事件必须是 final：{kinds}"
    assert "tool_call" in kinds and "tool_result" in kinds, f"工具链路没走通：{kinds}"
    final = [e for e in events if e.type == "final"][0]
    assert final.payload.text == "最终答案", final.payload.text

    assert len(provider.requests) == 2, f"应恰好两轮请求：{len(provider.requests)}"
    first, second = provider.requests
    assert roles(first) == ["system", "user"], roles(first)
    assert first["tools"], "第一轮必须把 tools 传给模型，否则模型永远没机会调工具"
    assert roles(second) == ["system", "user", "assistant", "tool"], roles(second)

    assistant = [m for m in second["messages"] if m["role"] == "assistant"][-1]
    tool_msg = [m for m in second["messages"] if m["role"] == "tool"][-1]
    sent_call = assistant["tool_calls"][0]
    assert sent_call["type"] == "function" and "function" in sent_call, sent_call
    assert sent_call["id"] == tool_msg["tool_call_id"] == "call_1", (sent_call, tool_msg)
    assert "[smoke] 命中 adele" in tool_msg["content"], tool_msg["content"]

    new_files = sorted(set(TRACE_ROOT.glob("*.json")) - before)
    assert new_files, "没有落盘 trace"
    trace = json.loads(new_files[-1].read_text(encoding="utf-8"))["trace"]
    spans = trace["spans"]
    assert [s["error"] for s in spans] == [None] * len(spans), \
        f"span 不该有 error：{[(s['name'], s['error']) for s in spans]}"
    llm_spans = [s for s in spans if s["kind"] == "llm"]
    assert len(llm_spans) == 2, f"应有两个 llm span：{len(llm_spans)}"
    assert all(s["messages"] for s in llm_spans), "span.messages 没记到实际发出去的 payload"
    # 落盘必须能反序列化出工具消息的配对（ToolCall 是 pydantic 模型时这里会炸）
    assert llm_spans[1]["messages"][2]["tool_calls"][0]["id"] == "call_1", llm_spans[1]["messages"]


@case("5 失败可见性：payload 构造失败要落到 span.error（不能只剩一个空白 span）")
async def case_failure_is_visible():
    llm = make_llm(FakeProvider(lambda body: sse([chunk({}, "stop")])))
    span = TurnTrace(request_id="r", conversation_id="c").start_span("token", KIND_LLM)

    try:
        llm._prepare(span, [object()], True)
    except TypeError:
        pass
    else:
        raise AssertionError("未知消息类型必须抛 TypeError")

    assert span.error, "构造 payload 失败没有记进 span.error —— trace 上会是个空白 span"
    assert "payload" in span.error, span.error


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def main() -> int:
    shutil.rmtree(TRACE_ROOT, ignore_errors=True)
    failures: list[tuple[str, BaseException]] = []
    for desc, fn in CASES:
        try:
            await fn()
        except BaseException as e:  # noqa: BLE001 - 逐用例收集，跑完再汇总
            failures.append((desc, e))
            print(f"[FAIL] {desc}\n       {type(e).__name__}: {e}")
            if "--trace" in sys.argv:
                traceback.print_exc()
        else:
            print(f"[ ok ] {desc}")

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
