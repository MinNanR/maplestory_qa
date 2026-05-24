from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
STATIC_DIR = BASE_DIR / "app" / "static"
DB_PATH = BASE_DIR / "db" / "robot.db"


class Settings:
    app_name = "Game Knowledge QA"
    llm_base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_api_key = os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", "sk-d66e7442da94489f985d3dfc983cd8b6"))
    llm_model = os.getenv("LLM_MODEL", "qwen-plus")
    top_k = int(os.getenv("TOP_K", "5"))


settings = Settings()
