from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage


llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0.5,
    reasoning=False
    # other params...
)

system_template = """
你是一个{role}
请用通俗易懂的语言回答以下问题
"""
# prompt = PromptTemplate.from_template(template)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_template,
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm
res = chain.invoke(
    {
        "role": "数学专家",
        "input": "解释洛必达法则",
    }
)
print(res.content)