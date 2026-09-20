import sys

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json

# Windows 控制台默认 GBK，知识文本含 GBK 无法编码的字符（如技能箭头 ⤢），
# 直接 print 会抛 UnicodeEncodeError 导致请求 500；统一按 UTF-8 输出。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from backend.models import ChatRequest
from backend.agent.orchestrator import Orchestrator
from backend.llm.client import LLMClient

from backend.conversation.manager import ConversationManager
from backend.agent.orchestrator import AgentEvent
from dataclasses import asdict


app = FastAPI(
    title="Maplestory Knowledge Agent",
    version="0.1.0",
)

conversation_manager = ConversationManager()

llm_client = LLMClient()

# trace_dir=None：落到默认 ./turn/；传目录可以让每次运行各写一处
orchestrator = Orchestrator(
    llm_client=llm_client, conversation_manager=conversation_manager, trace_dir=None
)
    
def encode_sse(event: AgentEvent) -> str:
    data = {"seq": event.seq, "type": event.type, "payload": asdict(event.payload)}
    return f"event: {event.type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

class ClearRequest(BaseModel):
    conversation_id: str

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "0.1.0",
    }

@app.post("/api/chat/clear")
async def clear_conversation(request: ClearRequest):
    conversation_manager.clear(request.conversation_id)
    return {"ok": True, "conversation_id": request.conversation_id}


@app.post("/api/chat/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        async for event in orchestrator.run(conversation_id=request.conversation_id, message=request.message):
            yield encode_sse(event)
        
    return StreamingResponse(generate(), media_type="text/event-stream",
                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")