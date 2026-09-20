from backend.models import ChatMessage
from backend.prompts.system import get_system_message
from backend.config import settings
from backend.knowledge import retrieval


class ConversationContext:

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages: list[ChatMessage] = []
        # 会话级知识池：doc_id -> 添加顺序（OrderedDict 首部最旧、尾部最新）。
        # 多轮对话中历史轮次解析出的知识会持续累积在这里（按文档去重），
        # 避免后续追问（如「那阿黛尔怎么应对塞伦」）丢失塞伦机制知识。
        self._knowledge_pool: dict[str, None] = {}
        # 本轮投影好的知识注入文本（build_messages 时拼进 system）
        self._knowledge_block: str | None = None

    def add_message(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def _add_to_pool(self, doc_ids: list[str]) -> None:
        """把新命中的文档并入知识池（去重 + 刷新新鲜度到尾部）。"""
        for doc_id in retrieval.filter_valid_ids(doc_ids):
            self._knowledge_pool.pop(doc_id, None)
            self._knowledge_pool[doc_id] = None

    def _evict_pool(self) -> None:
        """超出池内文档数上限时淘汰最旧文档。

        注：池只存文档 id，字符预算由每轮投影的 knowledge_max_chars 控制，
        因此这里按文档数淘汰（文档再大也不挤占跨轮保真的名额）。
        """
        while len(self._knowledge_pool) > settings.knowledge_pool_max_docs:
            oldest = next(iter(self._knowledge_pool))
            self._knowledge_pool.pop(oldest, None)

    def refresh_knowledge(self, new_ids: list[str], query: str) -> list[str]:
        """当前用户问题处理后调用：并入新知识并做本轮片段投影。"""
        self._add_to_pool(new_ids)
        self._evict_pool()

        chunks = retrieval.retrieve_chunks(
            query,
            list(self._knowledge_pool),
            max_chars=settings.knowledge_max_chars,
        )
        block = retrieval.build_knowledge_block(chunks)
        # 预算针对最终注入文本（含片段头/分隔符开销）；超出时按优先级从尾部裁剪
        while block and len(block) > settings.knowledge_max_chars and len(chunks) > 1:
            chunks = chunks[:-1]
            block = retrieval.build_knowledge_block(chunks)
        self._knowledge_block = block
        
        retrieval_chunk_ids = [c.chunk_id for c in chunks]
        return retrieval_chunk_ids

    def clear_knowledge_block(self) -> None:
        """当前问题不需要知识（如闲聊）时：本轮不注入，但知识池保留供后续使用。"""
        self._knowledge_block = None

    def build_messages(self) -> list[ChatMessage]:
        system_message = get_system_message()
        messages = [system_message]
        if self._knowledge_block:
            messages.append(ChatMessage(
                role="system",
                content=self._knowledge_block
            ))
        messages.extend(self.messages)
        return messages
