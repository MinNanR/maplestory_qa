from dataclasses import field

from pydantic import BaseModel, Field
from typing import Literal, Any, Union, Annotated, TypeAlias


"""
与大模型交互的消息封装
"""
#模型请求调用tool的信息
class ToolCall(BaseModel):
    id: str #调用id，由模型提供
    name: str  # 调用的工具名
    arguments: str # 调用工具的参数，是一个json字符串




class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str = "" 

class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str = ""
    
    
class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    
class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    content: str = ""
    tool_call_id: str = ""
    

ChatMessage: TypeAlias = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="role")
]


class FunctionParameter(BaseModel):
    type: str = "object"  
    properties: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
    
class ToolFunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: FunctionParameter

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunctionDefinition
    


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
    