from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# 加载环境变量
load_dotenv()

llm = ChatOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
    model=os.environ.get("Doubao_Seed_16_Flash")
)

print(llm.invoke("莫言真名").content)