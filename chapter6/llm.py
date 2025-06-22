# llm.py
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 环境与模型初始化
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("Doubao_15Pro")  # 可替换为实际模型名

llm = ChatOpenAI(
        api_key=api_key, 
        model=model_name, 
        base_url=base_url,
        temperature=0,
    )

if __name__ == '__main__':
    print(llm.invoke("你好"))
