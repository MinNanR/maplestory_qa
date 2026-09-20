# Game Knowledge QA MVP

一个本地运行的游戏知识问答原型，包含：

- FastAPI 后端
- 本地 Markdown 知识库
- 本地关键词检索
- 通义千问云模型接入
- 聊天式 Web 页面
- 流式输出

## 1. 安装依赖

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. 配置通义

设置 DashScope API Key：

```powershell
$env:DASHSCOPE_API_KEY="your_api_key"
```

可选覆盖项：

```powershell
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_API_KEY=$env:DASHSCOPE_API_KEY
$env:LLM_MODEL="qwen-plus"
```

## 3. 启动

最直接的启动方式：

```powershell
python run.py
```

如果你想一键创建虚拟环境、安装依赖并启动：

```powershell
.\start.ps1
```

底层等价命令是：

```powershell
uvicorn app.main:app --reload
```

打开 `http://127.0.0.1:8000`

## 4. 知识文件格式

知识文件放在 `knowledge/` 目录下，使用 Markdown。

```md
---
id: flame-slash
title: 怒焰斩
type: skill
source: 手工整理自游戏 Wiki
tags:
  - 狂战士
  - 技能
  - 伤害
---

# 技能效果

怒焰斩基础倍率为 320%。

# 怒气加成

- 怒气 0 到 49：无额外加成。
- 怒气 50 到 79：最终倍率提高到 380%。
- 怒气 80 到 100：最终倍率提高到 450%。
```

字段说明：

- `id`：唯一标识
- `title`：条目名
- `type`：如 `class`、`skill`、`boss`、`item`、`formula`
- `source`：来源说明
- `tags`：检索标签
- 正文：按 Markdown 标题切分成检索片段

## 5. 对话历史

前端会保存最近几轮 `user/assistant` 消息，并在提问时一并发送给后端。  
后端会：

- 用最近几轮用户问题补全检索 query
- 把最近几轮对话历史一起发给模型

这样像“那满怒的时候呢”这类省略主语的问题也能接住。

## 6. backend 知识检索与上下文控制

`backend/` 下 `/api/chat/stream` 链路用两处本地检索控制上下文长度（零新增依赖）：

- **意图解析（`backend/analysis/query_analyzer.py`）**：不再把整份知识库目录（`build_catalog()`，会随知识库线性增长）塞进提示词；改为先用本地检索（`backend/knowledge/retrieval.py` 的 BM25-风格打分，词元 = 英文单词 + 中文单字）从目录条目（title/description/keyword）筛出少量候选，只把压缩后的候选清单（默认 15 条）交给 LLM 选 `knowledge_ids`，提示词大小与知识库总量解耦。
- **知识注入（`backend/conversation/context.py`）**：不再把命中文档全文拼入 system 消息；文档先按标题结构分块（`backend/knowledge/chunker.py`，过小块合并、超大块按段落硬切），回答时按当前问题（含最近两轮用户问题用于指代消解）从**会话知识池**内截取相关片段注入，总量受 `knowledge_max_chars`（默认 6000 字符）预算约束。
  - 会话知识池跨轮累积、按文档去重：历史轮次解析出的知识（如塞伦机制）在后续追问（如「那阿黛尔怎么应对」）中仍会被注入，每个池内文档至少保留其最高分片段；
  - 池内文档数超过 `knowledge_pool_max_docs`（默认 6）时淘汰最旧文档；
  - 单轮不需要知识（闲聊）时不注入，但池保留供后续使用。

相关配置项见 `backend/config.py`（`.env` 可覆盖），如 `retrieval_top_docs` / `chunk_top_k` / `chunk_min_chars` / `chunk_max_chars` / `knowledge_max_chars` / `knowledge_pool_max_docs`。检索演示：`python -m backend.knowledge.retrieval --query "虎影 Bravado 技能效果" --top 5`。知识库文件变更（新增/修改文档）后，索引在下一次调用时按 mtime 指纹自动重建，无需重启服务。

注意：上面的第 1~5 节描述的是旧版 `app/` 原型；当前生效的实现为 `backend/` 包（`uvicorn backend.main:app` 启动，静态页在 `frontend/`）。
