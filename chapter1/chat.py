import argparse
import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from read_link import read_link as read_link_tool
from search_api import search_by_tavily as search_tool

tools = [
    Tool(
        name="search",
        func=search_tool,
        description="用于互联网搜索"
    ),
    Tool(
        name="read_link",
        func=read_link_tool,
        description="用于读取链接内容"
    ),
]

# Load environment variables
load_dotenv()

# 2. 初始化LLM
model = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    model=os.environ["Doubao_15Pro"],
    max_tokens=10000,
    temperature=0.0,
)

# 3. 初始化Agent
agent = create_react_agent(
    tools=tools,
    model=model,
)


# 4. 执行 Agent，处理用户输入
async def chat(input):
    """
    执行 Agent，处理用户输入
    :param input: 用户输入
    :return:
    """
    # 流式处理响应
    async for event, chunk_data in agent.astream(
            {"messages": [{"role": "user", "content": input}]},
            config={"recursion_limit": 10},
            stream_mode=["updates", "messages", "custom"],
            debug=False
    ):
        if event == "messages":
            # chunk_data is a list of BaseMessage instances.
            # We are interested in AIMessage instances, which represent the AI's output.
            for message_item in chunk_data:
                if isinstance(message_item, AIMessage) and hasattr(message_item, "content") and message_item.content:
                    print(message_item.content, end="", flush=True)


def main():
    """
    主函数，解析命令行参数并执行 Agent
    :param input_text: 用户输入的文本
    :return:
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Chat with an AI Agent.")

    # Usage:
    # python chat.py "什么是AI Agent？"
    parser.add_argument("input", type=str, help="The text to send to the AI agent.")
    args = parser.parse_args()
    asyncio.run(chat(args.input))


if __name__ == "__main__":
    main()
