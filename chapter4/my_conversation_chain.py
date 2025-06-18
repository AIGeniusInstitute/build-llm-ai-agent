import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# LLM and Graph setup
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("Doubao_Seed_16_Flash")

llm = ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url)

# Define the graph
workflow = StateGraph(MessagesState)

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

workflow.add_node("model", call_model)
workflow.add_edge("__start__", "model")
workflow.add_edge("model", END)

# Add memory to the graph
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def chat_with_agent(user_message: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    input_message = HumanMessage(content=user_message)
    response = app.invoke({"messages": [input_message]}, config=config)
    ai_reply = response['messages'][-1].content
    return ai_reply


if __name__ == '__main__':
    session_1 = "123"
    session_2 = "456"

    print(f"Session {session_1}: {chat_with_agent('你好,我叫 Agent X', session_1)}")
    print(f"Session {session_1}: {chat_with_agent('你是谁', session_1)}")
    print(f"Session {session_1}: {chat_with_agent('请问我叫什么?', session_1)}")

    print("------------------Switching to Session 2------------------")

    print(f"Session {session_2}: {chat_with_agent('请问我叫什么?', session_2)}")
    print(f"Session {session_2}: {chat_with_agent('你好,我叫 Agent Y', session_2)}")
    print(f"Session {session_2}: {chat_with_agent('请问我叫什么?', session_2)}")

    print("------------------Switching back to Session 1------------------")
    print(f"Session {session_1}: {chat_with_agent('还记得我叫什么吗?', session_1)}")