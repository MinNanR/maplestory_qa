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
