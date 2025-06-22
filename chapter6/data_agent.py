# data_agent.py
import ast

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import Tool

from llm import llm

# 加载API密钥
load_dotenv()

# 初始化数据库
db = SQLDatabase.from_uri("mysql+mysqlconnector://root:88888888@localhost:3306/salesdb")


# 可视化工具
def plot_tool(input_str: str):
    """根据输入数据绘制图表，并将其保存为 sales_trend.png 文件。输入应为一个包含绘图所需参数的字典字符串。"""

    print(f"input_str========={input_str}")

    try:
        input_dict = ast.literal_eval(input_str)

        if isinstance(input_dict, str):
            input_dict = ast.literal_eval(input_dict)

    except (ValueError, SyntaxError) as e:
        return f"无法解析输入字符串: {input_str}. 错误: {e}"

    if not isinstance(input_dict, dict):
        return f"解析后的输入不是一个字典。输入: {input_str}"

    print(f"input_dict========={input_dict}")

    data = input_dict.get('data')
    chart_type = input_dict.get('chart_type', 'line')
    title = input_dict.get('title', 'Chart')
    x_label = input_dict.get('x_label', 'X')
    y_label = input_dict.get('y_label', 'Y')

    if not data or not isinstance(data, list):
        return "输入数据格式错误或数据为空。"

    try:
        # 假设 data 是 (x, y) 元组的列表
        df = pd.DataFrame(data, columns=[x_label, y_label])
    except ValueError:
        return f"无法将数据转换为DataFrame。请确保数据是元组列表，且列数匹配。数据: {data}"

    plt.figure(figsize=(10, 6))

    if chart_type == 'bar':
        plt.bar(df[x_label], df[y_label])
    else:
        plt.plot(df[x_label], df[y_label], marker='o')

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 将图表保存到文件
    filename = "sales_trend.png"
    plt.savefig(filename)
    plt.close()

    return f"成功将图表保存为 {filename}"


# 工具列表
tools = [
    Tool(
        name="数据可视化",
        func=plot_tool,
        description='用于将数据可视化。输入应该是一个字典字符串，其中包含绘图所需的所有参数，例如：`{"data": [("2024-01", 12000)], "chart_type": "bar", "title": "Sales"}`。'
    )
]

# 初始化SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    extra_tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    user_query = "请分析今年每个月的销售额，并用柱状图展示"
    result = agent_executor.invoke({"input": user_query})
    print(result)
