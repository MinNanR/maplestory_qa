from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "Maplestory Knowledge Agent"
    llm_provider: str = "deepseek"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str
    llm_model: str = "deepseek-flash"
    top_k: int = 10
    max_tool_rounds: int = 8
    tool_timeout_s: int = 20

    # 知识库检索与注入相关配置
    knowledge_dir: str = "knowledge"  # 知识库根目录（相对仓库根）
    retrieval_top_docs: int = 15  # 意图解析阶段给 LLM 的候选知识条目数
    chunk_top_k: int = 6  # 每个命中文档单次检索最多取的片段数
    chunk_min_chars: int = 300  # 分块最小字符数（过小片段会合并）
    chunk_max_chars: int = 2500  # 分块最大字符数（超大块按段落硬切）
    knowledge_max_chars: int = 6000  # 单轮注入知识片段的总预算（字符）
    knowledge_pool_max_docs: int = 6  # 会话知识池最多同时保留的文档数（超出淘汰最旧）

    # token 采集与成本核算
    # 流式响应默认不带,True时为每个块都携带用量统计，False时只在最后一个块提供用量统计
    # 届时 client 会自动降级重试一次（那一次调用就没有 usage）。
    llm_stream_include_usage: bool = True
    # 每百万 token 单价。None = 未知：成本指标显示为不可用，而不是 0。
    # 只存 token、不把金额写进 span —— 价格变了可以按新价重算历史数据。
    llm_price_input_per_mtok: float | None = None
    llm_price_output_per_mtok: float | None = None
    llm_price_cached_per_mtok: float | None = None


settings = Settings()
