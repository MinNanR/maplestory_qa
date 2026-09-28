from dataclasses import dataclass
from zipfile import Path
from backend.prompts.system import get_system_message

from backend.conversation.context import ConversationContext
from backend.models import ChatMessage



class ConversationManager:

    def __init__(self):
        self.conversations: dict[str, list[ChatMessage]] = {}

    def _get_or_create(self, conversation_id: str) -> list[ChatMessage]:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationContext(conversation_id)
        return self.conversations[conversation_id]

    def get_message(self, conversation_id: str) -> list[ChatMessage]:
        return self.conversations.get(conversation_id, [])

    def add_message(self, conversation_id: str, message: ChatMessage) -> None:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = [message]
        else:
            self.conversations[conversation_id].append(message)

    def clear(self, conversation_id: str) -> None:
        self.conversations.pop(conversation_id, None)

    def build_messages(self, conversation_id: str) -> list[ChatMessage]:
        system_message = get_system_message()
        messages = [system_message]
        working_messages = self.get_message(conversation_id)
        if working_messages:
            messages.extend(working_messages)
        return messages
