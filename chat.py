import os

from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

stream = client.chat.completions.create(
    model="qwen-plus",
    temperature=0.2,
    stream=True,
    messages=[
        {"role": "system", "content": "你是一个中文游戏知识助手。"},
        {"role": "user", "content": "解释一下游戏里的伤害倍率和最终伤害的区别。"},
    ],
)

for chunk in stream:
    delta = chunk.choices[0].delta.content or ""
    if delta:
        print(delta, end="", flush=True)
