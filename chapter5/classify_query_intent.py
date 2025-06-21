import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# LLM and Graph setup
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("Doubao_Seed_16_Flash")
llm = ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url)

def classify_query_intent(query):
    prompt = f"""
    请判断以下学术查询属于哪一类意图（综述、方法对比、引用追溯、趋势分析、其他）：
    查询内容：{query}
    只输出类别名称。
    """
    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content.strip()

# 示例
intent = classify_query_intent("请比较BERT和GPT在文本生成上的异同")
print(intent)  # 输出：方法对比
