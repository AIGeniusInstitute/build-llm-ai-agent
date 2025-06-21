import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# LLM and Graph setup
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("Doubao_Seed_16_Flash")
llm = ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url)


def rewrite_query(query, intent):
    prompt = f"""
    当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    针对学术检索，帮我将以下查询改写为更具体、可检索的表达：
    查询内容：{query}
    查询意图：{intent}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# 示例
new_query = rewrite_query("近半年最新深度学习综述", "综述")
print(new_query)  # 输出：2025年1月至2025年6月期间的最新深度学习领域综述性论文
