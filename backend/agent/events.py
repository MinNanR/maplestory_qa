"""编排层的事件契约（事件类型 + payload 数据类）。

为什么单独一个模块
------------------
orchestrator（外壳：turn 生命周期、seq 分配、usage/final/error 顺序）与
loop（循环控制：预算、终止、消息组原子性）**都要 yield 这些类型**。
把它们留在 orchestrator 里，loop 就只能 `from backend.agent.orchestrator import ...`
—— 而 orchestrator 又要 import loop，于是形成 orchestrator → loop → orchestrator
的循环导入：模块半初始化，报 "cannot import name 'X' from partially initialized
module"，而且报错位置会指向一个和原因无关的名字。

两条契约
--------
1. **事件里只放"UI 需要渐进渲染的"和"评测需要断言的"**：正文（工具输出、知识块）
   只进 trace 文件，不进事件流 —— smoke 有 `<1500 字符` 的守卫盯着这条。
2. **seq 由外壳分配**：loop yield `(type, payload)`，不带 seq。这样 seq 在整个 turn
   内单调递增，与内部有几个生成器、是否并行执行工具都无关。
"""

from dataclasses import dataclass, field
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
    step: int


@dataclass(frozen=True)
class ToolResultPayload:
    tool: str
    result: str
    # 产生的轮次
    step: int
    # 调用是否成功。失败也要回灌一条 tool 消息 —— 前端据此标红，别去嗅探 result 文案。
    succeeded: bool
    # ⚠️ 必填字段必须写在**所有默认字段之前**：dataclass 会按顺序生成 __init__，
    # 一个没有默认值的字段跟在有默认值的字段后面，会在**导入期**直接
    # TypeError: non-default argument follows default argument —— 整个模块起不来。
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
    finish_reason: str | None = None  # 终止原因（budget / no_tool_calls / over_budget / user_cancel）
    steps: int = 0


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
    

@dataclass(frozen= True)
class AbortedPayload:
    reason: str = ""
    

@dataclass(frozen=True)
class AgentEvent:
    seq: int
    type: Literal[
        "stage", "token", "tool_call", "tool_result", "usage", "final", "error", "aborted"
    ]
    payload: Union[
        StagePayload,
        TokenPayload,
        ToolCallPayload,
        ToolResultPayload,
        UsagePayload,
        FinalPayload,
        ErrorPayload,
        AbortedPayload
    ]
