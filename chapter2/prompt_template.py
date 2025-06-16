from langchain.prompts import PromptTemplate

template = """
你是一位{role}。
请用{num}句话解释：{topic}
"""

prompt = PromptTemplate(
    input_variables=["role", "num", "topic"],
    template=template,
)

filled_prompt = prompt.format(role="AI Agent开发专家", num="三", topic="RAG架构")
print(filled_prompt)