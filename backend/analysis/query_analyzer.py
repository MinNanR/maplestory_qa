from pathlib import Path

from backend.llm.client import LLMClient
from backend.models import ChatMessage, QueryAnalysis
from backend.observability.trace import LLMSpan
from backend.knowledge.retrieval import retrieve_documents, format_candidates, filter_valid_ids

SYSTEM_PROMPT = """
你是一个 MapleStory 用户问题分析器。
你的任务不是回答用户问题，而是分析用户的问题。

请返回 JSON。

JSON 必须包含以下字段：

{
    "intent": "用户问题的意图",
    "entities": ["用户提到的冒险岛实体"],
    "needs_knowledge": true,
    "knowledge_ids": ["需要查询的知识 ID 列表"],
    "needs_external_search": false,
    "external_search_queries": ["需要进行外部搜索的查询参数列表"]
}

字段要求：

- intent：描述用户问题的主要意图
- entities：问题中涉及的游戏实体
- needs_knowledge：是否需要查询冒险岛知识
- knowledge_ids：需要查询的知识 ID 列表，只能从系统消息给出的「候选知识条目」中选择其 id，
  可多选；若未提供候选清单或候选均与问题不匹配，返回空数组 []
- needs_external_search：是否需要进行外部搜索
- external_search_queries：需要进行外部搜索的查询参数列表

注意事项：
- 若系统消息包含「历史问题」，仅用于理解指代（如“那它呢”“然后呢”），
  intent / entities / knowledge_ids 一律针对最后一条用户问题。
- 不要回答用户的问题。
- 只输出 JSON。
"""


class QueryAnalyzer:

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def analyze(
        self,
        query: str,
        history: list[str] | None = None,
        span: LLMSpan | None = None,
    ) -> QueryAnalysis:
        system_message = ChatMessage(
            role="system",
            content=SYSTEM_PROMPT
        )

        messages = [system_message]

        # 本地检索先筛出少量候选条目，替代把整份知识库目录塞进提示词；
        # 提示词大小与知识库总量解耦（只与候选数有关）。
        candidates = retrieve_documents(query)
        if candidates:
            candidate_ids = [doc_id for doc_id, _ in candidates]
            candidate_text = (
                "候选知识条目如下（knowledge_ids 只能从中选择其 id，可多选；"
                "若均不匹配请返回 []）：\n"
                + format_candidates(candidate_ids)
            )
            messages.append(ChatMessage(role="system", content=candidate_text))

        if history:
            messages.append(ChatMessage(
                role="system",
                content=f"历史问题（仅用于理解指代，不改变当前问题）：{'；'.join(history)}"
            ))

        user_message = ChatMessage(
            role="user",
            content=query
        )

        messages.append(user_message)
        print(messages)
        data = await self.llm_client.chat_json(messages, span=span)
        print(f"Query analysis result: {data}")
        # 过滤 LLM 可能输出的幻觉 id（目录中不存在者剔除），避免知识读取/注入异常
        data["knowledge_ids"] = filter_valid_ids(data.get("knowledge_ids") or [])
        return QueryAnalysis.model_validate(data)
