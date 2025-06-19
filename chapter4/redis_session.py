import os

import redis
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis import RedisSaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict
from typing_extensions import Annotated

# 加载环境变量
load_dotenv()

# --- LLM 和 Prompt 设置 ---
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("Doubao_Seed_16")

llm = ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | llm


# --- LangGraph 定义 ---
def call_model(state: dict):
    messages = state['messages']
    response = chain.invoke({"messages": messages})
    return {"messages": [response]}


# --- Redis 连接和 LangGraph Checkpointer ---
# 初始化Redis连接
redis_client = redis.Redis.from_url("redis://localhost:6379")

checkpointer = RedisSaver(redis_client=redis_client)


# --- Graph 构建 ---
# 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]


workflow = StateGraph(State)
workflow.add_node("model", call_model)
workflow.add_edge("__start__", "model")
workflow.add_edge("model", END)

# For the first time, you need to create an index
checkpointer.setup()

# build graph
app = workflow.compile(checkpointer=checkpointer)


def chat_with_redis(user_input, session_id):
    config = {"configurable": {"thread_id": session_id}}
    response = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    ai_reply = response['messages'][-1].content
    return ai_reply


def print_session_state(session_id: str):
    print(f"Session {session_id} state: {redis_client.get(f'thread_state:{session_id}')}")


# --- 主程序入口 ---
if __name__ == '__main__':
    session_1 = "session_1_redis"
    session_2 = "session_2_redis"

    print("--- 开始会话 1 ---")
    print(f"用户: 你好，我叫小明。")
    print(f"AI: {chat_with_redis('你好，我叫小明。', session_1)}")
    print_session_state(session_1)

    print(f"\n用户: 我刚才说了我叫什么名字吗？")
    print(f"AI: {chat_with_redis('我刚才说了我叫什么名字吗？', session_1)}")
    print_session_state(session_1)

    print("\n--- 切换到会话 2 ---")
    print(f"用户: 你好，我叫小红。")
    print(f"AI: {chat_with_redis('你好，我叫小红。', session_2)}")
    print_session_state(session_2)

    print(f"\n用户: 我叫什么名字？")
    print(f"AI: {chat_with_redis('我叫什么名字？', session_2)}")
    print_session_state(session_2)

    print("\n--- 切换回会话 1 ---")
    print(f"用户: 我们之前聊了什么？还记得我叫什么吗？")
    print(f"AI: {chat_with_redis('我们之前聊了什么？还记得我叫什么吗？', session_1)}")
    print_session_state(session_1)

    # 测试会话2
    print("\n--- 开始会话 2 ---")
    print(f"用户: 你好，我叫小绿。")
    print(f"AI: {chat_with_redis('你好，我叫小绿。', session_2)}")
    print_session_state(session_2)

    print(f"\n用户: 我叫什么名字？")
    print(f"AI: {chat_with_redis('我叫什么名字？', session_2)}")
    print_session_state(session_2)

    # 查看所有 key
    print(redis_client.keys("*"))
    print(redis_client.get(f"thread_state:{session_1}"))
    print(redis_client.get(f"thread_state:{session_2}"))

    # 清理测试数据
    # redis_client.delete(f"thread_state:{session_1}")
    # redis_client.delete(f"thread_ts:{session_1}")
    # redis_client.delete(f"thread_state:{session_2}")
    # redis_client.delete(f"thread_ts:{session_2}")
    # print("\n--- 已清理测试会话数据 ---")
