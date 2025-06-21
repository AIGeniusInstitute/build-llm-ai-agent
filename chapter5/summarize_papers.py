import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# LLM Setup
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("Doubao_Seed_16_Flash")
llm = ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url)


def summarize_papers(paper_chunks):
    prompt = "请根据以下多篇论文的内容，生成一段简明的综述，突出主要创新点和研究趋势：\n"
    for chunk in paper_chunks:
        prompt += f"- {chunk}\n"

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content.strip()


# 示例
chunks = [
    "本文提出了基于注意力机制的Transformer模型...",
    "我们改进了Transformer结构，提升了效率..."
]
summary = summarize_papers(chunks)
print(summary)
