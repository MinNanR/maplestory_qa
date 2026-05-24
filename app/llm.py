from __future__ import annotations

from typing import Iterable

from openai import OpenAI


SYSTEM_PROMPT = """你是一个游戏知识问答助手。

回答规则：
1. 只能根据给定资料回答，不要编造游戏数据。
2. 如果资料不足，明确说“当前知识库里没有足够信息回答这个问题”。
3. 优先给出结论，再给出必要解释。
4. 涉及公式、数值、条件时，保留原始条件，不要擅自补全。
5. 如果用户使用代词或省略主语，要结合对话历史理解。
6. 输出使用简洁中文。
"""


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def _build_context_text(self, contexts: list[dict]) -> str:
        return "\n\n".join(
            [
                (
                    f"[资料{index}] 标题: {item['title']}\n"
                    f"类型: {item['type']}\n"
                    f"章节: {item['section']}\n"
                    f"来源: {item['source']}\n"
                    f"内容:\n{item['text']}"
                )
                for index, item in enumerate(contexts, start=1)
            ]
        )

    def _build_messages(
        self,
        question: str,
        contexts: list[dict],
        history: list[dict] | None = None,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for item in (history or [])[-8:]:
            role = item.get("role", "")
            content = item.get("content", "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            messages.append({"role": role, "content": content})

        context_text = self._build_context_text(contexts)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"用户当前问题:\n{question}\n\n"
                    f"可用资料:\n{context_text}\n\n"
                    "请根据资料回答。先直接给结论，再补充必要说明，最后列出你使用到的资料标题。"
                ),
            }
        )
        return messages

    def answer(
        self,
        question: str,
        contexts: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=self._build_messages(question, contexts, history),
            timeout=60.0,
        )
        return response.choices[0].message.content or ""

    def stream_answer(
        self,
        question: str,
        contexts: list[dict],
        history: list[dict] | None = None,
    ) -> Iterable[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=self._build_messages(question, contexts, history),
            timeout=60.0,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
