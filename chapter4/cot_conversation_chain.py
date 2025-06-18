# cot_conversation_chain.py
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState

load_dotenv()

# LLM设置
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("Doubao_Seed_16")

llm = ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url)

# 定义Chain of Thought提示模板
COT_SYSTEM_TEMPLATE = """你是一个思维清晰的AI助手,擅长应用思维链（Chain of Thought）解决问题。
当回答用户问题时，请遵循以下思考步骤：

1. 首先，分析问题的关键点和需要考虑的因素
2. 然后，列出可能的解决方案或思路
3. 接着，评估每种方案的优缺点
4. 最后，给出你认为最合适的答案和理由

请在回答前使用"思考过程："标记你的推理步骤，然后使用"回答："标记你的最终答案。
"""

# 创建CoT提示模板
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", COT_SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="messages"),
])

chain = cot_prompt | llm


def call_model(state: MessagesState):
    response = chain.invoke(state)
    return {"messages": [response]}


# LangGraph 设置
workflow = StateGraph(MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge("__start__", "model")
workflow.add_edge("model", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def chat(user_input: str, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    response = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    return response['messages'][-1].content


# 示例使用
if __name__ == '__main__':
    session_1 = "123"
    session_2 = "456"

    # 测试会话1
    print(f"Session {session_1}: {chat('如何评估一个LLM模型的好坏？', session_1)}")
    print(f"Session {session_1}: {chat('能否给我一个具体的例子？', session_1)}")

    # 测试会话2
    print("\n------------------Switching to Session 2------------------\n")
    print(f"Session {session_2}: {chat('解释一下LLM 推理为什么会出现幻觉', session_2)}")
    print(f"Session {session_2}: {chat('这与概率论有什么关系？', session_2)}")

    # 回到会话1
    print("\n------------------Switching back to Session 1------------------\n")
    print(f"Session {session_1}: {chat('我们之前讨论的是什么话题？', session_1)}")
