import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool(
    name_or_callable="CalendarInvite",
    description="调度日历会议邀请"
)
def send_calendar_invite(email: str, task: str, time: str):
    # 调用企业日历API逻辑
    print(f"向 {email} 发送了：{task} 时间：{time}")


# Load environment variables
load_dotenv()

model = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    model=os.environ["Doubao_15Pro"],
)

agent = create_react_agent(
    tools=[send_calendar_invite],
    model=model,
    version="v2",
)


def chat(input):
    # 构建输入
    messages = {
        "messages": [{"role": "user", "content": input}]}

    # 调用Agent
    output = agent.invoke(
        messages,
        config={"recursion_limit": 10},
        debug=False
    )

    # 输出结果
    for chunk_data in output:
        for message in chunk_data:
            if isinstance(message, AIMessage):
                print(message.content)


if __name__ == "__main__":
    input = "我需要在明天下午5点与John Smith进行一次会议，邮箱：JohnSmith@gmail.com，请安排一个日历邀请。"
    chat(input)
