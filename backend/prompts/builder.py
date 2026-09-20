from backend.models import ChatMessage
from backend.prompts.system import get_system_message

def build_message(conversation_messages: list[ChatMessage]) -> list[ChatMessage]:
    system_message = get_system_message()
    return [
        system_message,
        *conversation_messages
    ]