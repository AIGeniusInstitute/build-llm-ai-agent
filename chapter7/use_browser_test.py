import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
from browser_use import Agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.environ['Doubao_Seed_16'],
    api_key=os.environ['OPENAI_API_KEY'],
    base_url=os.environ['OPENAI_BASE_URL'],
)

# 当前日期: yyyy-MM-dd 格式
current_date = datetime.now().strftime('%Y-%m-%d')  # 获取当前日期并格式化


async def main():
    agent = Agent(
        task=f"{current_date} Search Model Context Protocol (MCP)  on github.com ",
        llm=llm,
    )

    await agent.run(max_steps=10)


if __name__ == '__main__':
    asyncio.run(main())
