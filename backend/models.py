from pydantic import BaseModel, Field
from typing import Literal

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant" ]
    content: str


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ChatResponse(BaseModel):
    conversation_id: str
    content: str
    
    
class QueryAnalysis(BaseModel):
    intent: str = Field(default="", description="用户问题的意图")
    entities: list[str] = Field(default=[], description="用户提到的冒险岛实体")

    needs_knowledge: bool = Field(default=True, description="是否需要查询冒险岛知识")
    knowledge_ids: list[str] = Field(default=[], description="需要查询的知识 ID 列表")

    needs_external_search: bool = Field(default=False, description="是否需要进行外部搜索")
    external_search_queries: list[str] = Field(default=[], description="需要进行外部搜索的查询参数列表")
    