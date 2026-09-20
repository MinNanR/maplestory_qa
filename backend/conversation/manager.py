from dataclasses import dataclass
import sqlite3
from zipfile import Path

from backend.conversation.context import ConversationContext
from backend.models import ChatMessage



class ConversationManager:

    def __init__(self):
        self.conversations: dict[str, ConversationContext] = {}

    def _get_or_create(self, conversation_id: str) -> ConversationContext:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(conversation_id)
        return self.conversations[conversation_id]

    def get_message(self, conversation_id: str) -> list[ChatMessage]:
        context = self.conversations.get(conversation_id)
        return context.messages if context else []

    def add_message(self, conversation_id: str, message: ChatMessage) -> None:
        self._get_or_create(conversation_id).add_message(message)

    def refresh_knowledge(
        self, conversation_id: str, new_ids: list[str], query: str
    ) -> list[str]:
        """当前用户问题处理后调用：把命中文档并入会话知识池并做本轮片段投影。

        返回本轮实际注入的片段 id 列表（供事件流/评测采集使用）。
        """
        return self._get_or_create(conversation_id).refresh_knowledge(new_ids, query)

    def clear_knowledge(self, conversation_id: str) -> None:
        """当前问题不需要知识（闲聊等）：本轮不注入，知识池保留供后续使用。"""
        context = self.conversations.get(conversation_id)
        if context:
            context.clear_knowledge_block()

    def clear(self, conversation_id: str) -> None:
        self.conversations.pop(conversation_id, None)

    def build_messages(self, conversation_id: str) -> list[ChatMessage]:
        context = self.conversations.get(conversation_id)
        if context:
            return context.build_messages()
        else:
            return []
