# smoke_orchestrator.py  （仓库根目录，运行：.\.venv\Scripts\python.exe smoke_orchestrator.py）
"""冒烟测试：不联网、不要 API key，跑通 Orchestrator.run() 的最短可信路径。"""
import asyncio
import json
from dataclasses import asdict

from backend.agent.orchestrator import Orchestrator
from backend.models import ChatMessage


class FakeLLM:
    """假 LLM：签名对齐真实 LLMClient，但只实现 run() 真正会调的两个方法。"""

    def __init__(self, analysis=None, chunks=("你好", "，", "世界"), fail=None):
        self.analysis = analysis or {
            "intent": "问候", "entities": [],
            "needs_knowledge": False, "knowledge_ids": [],
            "needs_external_search": False, "external_search_queries": [],
        }
        self.chunks = chunks
        self.fail = fail            # "analysis" | "generation" | None
        self.seen_messages = []     # 记录被喂进来的 messages，供断言

    async def chat_json(self, messages, span=None):
        if span is not None:            # 假件也要填 span：否则 usage 契约测不到
            span.input_tokens, span.output_tokens = 100, 20
        if self.fail == "analysis":
            raise RuntimeError("boom in analysis")
        return dict(self.analysis)

    async def stream_chat(self, messages, span=None):   # ← 必须是 async generator
        self.seen_messages = list(messages)
        if self.fail == "generation":
            raise RuntimeError("boom in generation")
        for i, chunk in enumerate(self.chunks):
            if span is not None and i == 0:
                span.ttft_ms = 1.0
            yield chunk
        if span is not None:
            span.response = "".join(self.chunks)
            span.input_tokens, span.output_tokens = 200, 40


class FakeManager:
    """假会话管理器：只记录调用事实，不碰真实知识库。"""

    def __init__(self, block="K" * 6000):    # 模拟真实的知识块大小（knowledge_max_chars）
        self.block = block
        self.added = []
        self.refreshed = []
        self.cleared = 0
        self._history = []

    def get_message(self, cid):
        return list(self._history)

    def add_message(self, cid, message):
        self.added.append(message)
        self._history.append(message)

    def refresh_knowledge(self, cid, ids, query):
        self.refreshed.append((ids, query))
        # 必须与真件契约一致：真件返回「片段 id 列表」，不是 ";" 拼接的字符串。
        # 返回字符串会让 stage_context 里变成 str，消费方 list() 会把字符逐个拆开。
        return ["skill_adele#1", "skill_adele#2"]

    def clear_knowledge(self, cid):
        self.cleared += 1

    def build_messages(self, cid):
        return [ChatMessage(role="system", content="SYS"), *self._history]


async def collect(llm, mgr, message="你好"):
    # trace_dir=None：落盘到默认 ./turn/，冒烟测试不需要独立目录
    orch = Orchestrator(llm_client=llm, conversation_manager=mgr, trace_dir=None)
    return [event async for event in orch.run("conv-1", message)]


def event_size(event) -> int:
    return len(json.dumps(asdict(event.payload), ensure_ascii=False))


async def collect_raising(llm, mgr, message="你好"):
    """失败路径专用：既要拿到已产出的事件，也要拿到逃出的异常。

    不能复用 collect()：`[event async for event in ...]` 在生成器抛异常时整段丢弃，
    已经收集到的事件就看不到了。
    """
    orch = Orchestrator(llm_client=llm, conversation_manager=mgr, trace_dir=None)
    events, exc = [], None
    try:
        async for event in orch.run("conv-1", message):
            events.append(event)
    except Exception as e:  # noqa: BLE001
        exc = e
    return events, exc


def assert_error_contract(events, exc, *, expect_tokens=False):
    """当前契约：orchestrator 在 yield 完 error 事件后会 `raise e`。

    1. 恰好一个 error 事件，且它是最后一个 —— 流式消费方（SSE / 前端）据此报错；
    2. 没有 final、失败前没有 token；
    3. 异常同时抛给程序化调用方（评测 / 脚本），不必自己去翻事件流。
    """
    types = [e.type for e in events]
    assert types.count("error") == 1, f"应恰好一个 error 事件：{types}"
    assert types[-1] == "error", f"error 必须是最后一个事件：{types}"
    assert "final" not in types, f"失败时不应有 final：{types}"
    if not expect_tokens:
        assert "token" not in types, f"生成前的失败不应产出 token：{types}"
    assert exc is not None, "契约要求异常同时抛给调用方（orchestrator 里的 raise e）"


async def main():
    # 用例 1：正常路径（不需要知识）
    llm, mgr = FakeLLM(), FakeManager()
    events = await collect(llm, mgr)
    types = [e.type for e in events]

    assert types[0] == "stage", f"首个事件应为 stage：{types}"
    assert types[-1] == "final", f"末事件应为 final：{types}"
    assert [e.seq for e in events] == list(range(1, len(events) + 1)), \
        f"seq 必须从 1 连续递增：{[e.seq for e in events]}"
    assert events[-1].payload.text == "你好，世界", "final 必须携带完整答案"
    assert [m.role for m in llm.seen_messages][-1] == "user", "生成时最后一条必须是当前用户消息"
    assert [m.role for m in mgr.added] == ["user", "assistant"], "user / assistant 各落一次"
    # 阈值不是精确指标，而是守卫："上下文正文（几千字符）不许回到事件流"。
    # usage 事件天然带每 span 摘要，几百字符属正常。
    assert max(event_size(e) for e in events) < 1500, "事件里不能塞正文（知识块应进 trace）"
    assert mgr.cleared == 1, "needs_knowledge=False 应清空知识块"

    # usage 事件契约：恰好一个、紧邻 final 之前、合计 = 各 span 之和
    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 1, f"应有且仅有一个 usage 事件：{types}"
    assert types[-2] == "usage" and types[-1] == "final", f"usage 应在 final 之前：{types}"
    u = usage_events[0].payload
    assert u.llm_calls == 2, f"一轮两次 LLM 调用（分析 + 生成）：{u.llm_calls}"
    assert (u.input_tokens, u.output_tokens) == (300, 60), \
        f"token 应是各 span 之和（100+200 / 20+40）：{u.input_tokens}/{u.output_tokens}"
    assert u.usage_missing is False, "假件已填 token，不应报 usage_missing"
    assert [s["name"] for s in u.spans] == ["analysis", "generation"], u.spans
    # 事件流里不许出现 span 的 messages/response（正文只进 trace 文件）
    assert all("messages" not in s and "response" not in s for s in u.spans), u.spans

    # 用例 2：知识命中路径
    # 用例 2a：知识命中、无历史 —— dialog_query 不应夹带多余内容
    hit = FakeLLM(analysis={"intent": "x", "entities": [], "needs_knowledge": True,
                            "knowledge_ids": ["skill_adele"],
                            "needs_external_search": False, "external_search_queries": []})
    mgr2 = FakeManager()
    events2 = await collect(hit, mgr2, message="阿黛尔的 HEXA 技能有哪些？")
    assert [e.type for e in events2][-1] == "final"
    assert mgr2.refreshed, "命中知识时应调用 refresh_knowledge"
    _, query_2a = mgr2.refreshed[0]
    assert query_2a == "阿黛尔的 HEXA 技能有哪些？", f"无历史时不应有后缀：{query_2a!r}"
    assert max(event_size(e) for e in events2) < 1500, "检索事件不能带正文"
    assert [s["name"] for s in [e for e in events2 if e.type == "usage"][0].payload.spans] \
        == ["analysis", "retrieval", "generation"], "命中知识时应有 retrieval span"

    # 用例 2b：多轮指代 —— 验证 dialog_query 带上了前文问题
    mgr2b = FakeManager()
    mgr2b.add_message("conv-1", ChatMessage(role="user", content="塞伦有哪些阶段机制？"))
    events2b = await collect(hit, mgr2b, message="那阿黛尔怎么应对？")
    assert [e.type for e in events2b][-1] == "final", f"2b 应正常收尾：{[e.type for e in events2b]}"
    _, query_2b = mgr2b.refreshed[0]
    assert "塞伦" in query_2b, f"dialog_query 必须带上下文（指代消解）：{query_2b!r}"

    # 用例 3：分析阶段失败
    llm3, mgr3 = FakeLLM(fail="analysis"), FakeManager()
    events3, exc3 = await collect_raising(llm3, mgr3)
    assert_error_contract(events3, exc3)
    assert "boom in analysis" in events3[-1].payload.message, "error 事件必须带上原因"
    assert "boom in analysis" in str(exc3), "抛出的异常必须保留原始原因"
    assert [e.type for e in events3][-2] == "usage", "失败也要报已花掉的成本（在 error 之前）"

    # 用例 4：生成阶段失败
    llm4, mgr4 = FakeLLM(fail="generation"), FakeManager()
    events4, exc4 = await collect_raising(llm4, mgr4)
    assert_error_contract(events4, exc4)
    assert "boom in generation" in events4[-1].payload.message
    assert [m.role for m in mgr4.added] == ["user"], \
        "生成失败时不落 assistant —— 这是当前策略，明确断言它而不是默认它"
        
    print("smoke ok — event types:", types)


if __name__ == "__main__":
    asyncio.run(main())