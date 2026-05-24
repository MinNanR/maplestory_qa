from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import DB_PATH, KNOWLEDGE_DIR, STATIC_DIR, settings
from app.knowledge import KnowledgeBase
from app.llm import LLMClient


app = FastAPI(title=settings.app_name)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

kb = KnowledgeBase(KNOWLEDGE_DIR, DB_PATH)
kb.load()
llm = LLMClient(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.llm_model,
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = Field(default_factory=list)


def build_search_query(question: str, history: list[ChatMessage]) -> str:
    recent_user_turns = [item.content.strip() for item in history if item.role == "user" and item.content.strip()]
    recent_user_turns = recent_user_turns[-2:]
    parts = recent_user_turns + [question.strip()]
    return "\n".join(part for part in parts if part)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/entries")
def entries() -> dict:
    return {"items": kb.list_entries()}


@app.post("/api/reload")
def reload_knowledge() -> dict:
    kb.load()
    return {"ok": True, "count": len(kb.chunks)}


@app.post("/api/search")
def search(payload: AskRequest) -> dict:
    results = kb.search(payload.question, top_k=settings.top_k)
    return {"items": results}


@app.post("/api/ask")
def ask(payload: ChatRequest) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    search_query = build_search_query(question, payload.history)
    contexts = kb.search(search_query, top_k=settings.top_k)
    if not contexts:
        return {
            "answer": "当前知识库里没有找到相关资料。",
            "sources": [],
        }

    answer = llm.answer(
        question=question,
        contexts=contexts,
        history=[item.model_dump() for item in payload.history],
    )
    return {"answer": answer, "sources": contexts}


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    search_query = build_search_query(question, payload.history)
    contexts = kb.search(search_query, top_k=settings.top_k)
    history = [item.model_dump() for item in payload.history]

    def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': contexts}, ensure_ascii=False)}\n\n"
        if not contexts:
            yield f"data: {json.dumps({'type': 'chunk', 'content': '当前知识库里没有找到相关资料。'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
            return

        try:
            for chunk in llm.stream_answer(question=question, contexts=contexts, history=history):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
