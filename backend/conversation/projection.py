from dataclasses import dataclass

from backend.models import AssistantMessage, ChatMessage, ToolMessage, UserMessage


@dataclass(frozen=True)
class HistoryPolicy:
    answer_turns: int = 6  # 完整保留的最近轮数
    max_prompt_chars: int = 1024_000  # 投影后正文总预算（硬上限）
    max_total_tool_chars: int = 60_000


class MessageGroup:
    def __init__(self, user_message: UserMessage):
        self.user_message = user_message
        self.assistant_messages: list[AssistantMessage] = []
        self.tool_messages: list[ToolMessage] = []

    def add_assistant_message(self, assistant_message: AssistantMessage):
        self.assistant_messages.append(assistant_message)

    def _get_request_tool_call_ids(self) -> list[str]:
        return [
            tool_call.id
            for am in self.assistant_messages
            for tool_call in (am.tool_calls or [])
        ]

    def collect_tool_message(self, tool_messages: list[ToolMessage]) -> None:
        """把本组 assistant 声明过的 tool_calls 对应的 tool 消息收进来。

        只收**被声明过**的 id：历史里可能有别的组的工具消息，按 id 配对而不是按位置。
        """
        request_tool_call_ids = self._get_request_tool_call_ids()
        for tool_call_id in request_tool_call_ids:
            response_message = next(
                (tm for tm in tool_messages if tm.tool_call_id == tool_call_id), None
            )
            if response_message is not None:
                self.tool_messages.append(response_message)

    def missing_tool_call_ids(self) -> list[str]:
        """还没拿到 tool 消息的调用 id。非空 = 本组**不完整**，不能进 payload。"""
        response_tool_call_ids = {tm.tool_call_id for tm in self.tool_messages}
        return [
            request_id
            for request_id in self._get_request_tool_call_ids()
            if request_id not in response_tool_call_ids
        ]

    def is_complete(self) -> bool:
        if not self.assistant_messages:
            return False
        last_msg = self.assistant_messages[-1]
        return not getattr(last_msg, "tool_calls", None)

    def build_message(self) -> list[ChatMessage]:
        """把一组展开成 payload 片段：user → (assistant → tool)*。

        调用方（build_message）负责先做两件事：把 tool 消息收进来、把不完整的组
        整组丢掉。这里**只做拼装，不做校验**，所以 tool 消息必然找得到。

        刻意不用 `next(...)` 裸取：取不到时 StopIteration 在生成器帧里会被 PEP 479
        改写成 "async generator raised StopIteration" —— 报错点离真正原因（投影层
        放行了不完整的组）十万八千里。这里显式抛 KeyError，让报错指向本条调用。
        """
        result: list[ChatMessage] = [self.user_message]
        for am in self.assistant_messages:
            result.append(am)
            for tc in am.tool_calls or []:
                matched = [tm for tm in self.tool_messages if tm.tool_call_id == tc.id]
                if not matched:
                    raise KeyError(
                        f"投影层放行了一个不完整的消息组：assistant 声明了 tool_call "
                        f"{tc.id}（{tc.name}）但没有对应的 tool 消息"
                    )
                result.append(matched[0])
        return result


def build_message(
    messages: list[ChatMessage], policy: HistoryPolicy
) -> list[ChatMessage]:
    """把原始 transcript 投影成一次请求要发的消息列表（**扁平**列表）。

    窗口规则：
      - 最新一组（本轮问题）永远保留；
      - 从新到旧回溯，最多带 policy.answer_turns 组；
      - 遇到不完整组（assistant 声明了 tool_calls 却没有对应 tool 消息，比如上轮
        被中断）就**整组丢掉并停止回溯** —— 残缺的组会让 provider 直接 400，
        而且它的 user 消息也不能单独留下（否则与新问题组成"双 user"的假现场）；
      - 字符预算放不下时同样停止回溯。

    返回的永远是 list[ChatMessage]，不含 None、不含嵌套列表：
    拼 payload 的地方是 `[system, *projected]`，中间混进 None 会炸成一个
    跟真正原因完全无关的 TypeError。
    """
    message_group: list[MessageGroup] = []

    for message in messages:
        if message.role == "tool":
            continue  # 工具消息不单独成组，归到它前面的用户消息里
        if message.role == "user":
            group = MessageGroup(message)
            message_group.append(group)
        if message.role == "assistant":
            if not message_group:
                # 没有归属的 assistant（截断的历史）：丢掉，别让 message_group[-1] 炸
                continue
            message_group[-1].add_assistant_message(message)

    if not message_group:
        return []

    tool_messages = [m for m in messages if m.role == "tool"]

    prompt_chars_used = 0
    tool_chars_used = 0

    def fits(group: MessageGroup) -> bool:
        nonlocal prompt_chars_used, tool_chars_used
        user_chars = len(group.user_message.content or "")
        assistant_chars = sum(len(m.content or "") for m in group.assistant_messages)
        tool_chars = sum(len(m.content or "") for m in group.tool_messages)
        if (
            prompt_chars_used + user_chars + assistant_chars + tool_chars
            > policy.max_prompt_chars
        ):
            return False
        if tool_chars_used + tool_chars > policy.max_total_tool_chars:
            return False
        prompt_chars_used += user_chars + assistant_chars + tool_chars
        tool_chars_used += tool_chars
        return True

    group_len = len(message_group)
    message_group = [
        m
        for idx, m in enumerate(message_group)
        if idx == group_len - 1 or m.is_complete()
    ]
    # 倒序累积（最新在前），循环里用 reversed() —— list.reverse() 返回 None，
    # 直接对它迭代会得到 "'NoneType' object is not iterable"。
    # 最新的组也要 collect_tool_message()：本轮工具轮的 assistant + tool 就在这一组里，
    # 漏掉它就会拼出一个"声明了 tool_calls 却没有 tool 消息"的残缺组（provider 直接 400）。

    projected_group: list[MessageGroup] = []
    for group in reversed(message_group[-policy.answer_turns - 1 :]):
        group.collect_tool_message(tool_messages)
        if projected_group and group.missing_tool_call_ids():
            # 残缺的旧组不进 payload，它之前的更旧组也就不必再看了
            break
        if projected_group and not fits(group):
            break
        projected_group.append(group)

    projected_messages: list[ChatMessage] = []
    for group in reversed(projected_group):
        projected_messages.extend(group.build_message())

    return projected_messages
