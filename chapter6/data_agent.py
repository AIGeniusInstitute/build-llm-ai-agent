# data_agent.py
import ast
import io

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import Tool, AgentType
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

from llm import llm

# 加载API密钥
load_dotenv()

# 初始化数据库
db = SQLDatabase.from_uri("mysql+mysqlconnector://root:88888888@localhost:3306/salesdb")


# 可视化工具
def plot_tool(query_result: str):
    """将查询结果字符串转换为图表, 并将图表保存为 sales_trend.png 文件。"""
    # query_result为字符串，需转为DataFrame
    try:
        data = ast.literal_eval(query_result)
        df = pd.DataFrame(data)
    except (ValueError, SyntaxError):
        try:
            # Fallback to reading as CSV
            df = pd.read_csv(io.StringIO(query_result))
        except Exception:
            return f"无法解析查询结果: {query_result}"

    if df.empty or len(df.columns) < 2:
        return "数据为空或列数不足，无法绘图。"

    plt.figure(figsize=(8, 5))
    x_data = df.iloc[:, 0]
    y_data = df.iloc[:, 1]
    plt.plot(x_data, y_data, marker='o')
    plt.title("Sales Trend")
    plt.xlabel(str(df.columns[0]))
    plt.ylabel(str(df.columns[1]))
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a file instead of returning bytes
    plt.savefig("sales_trend.png")
    plt.close()

    return "成功将图表保存为 sales_trend.png"


# 工具列表
tools = [
    Tool(
        name="数据可视化",
        func=plot_tool,
        description='''用于将数据可视化。此工具将绘制输入数据的图表，并将其保存为名为 "sales_trend.png" 的文件。输入应该是一个字符串，表示一个列表的元组，例如 "[('2024-01', 1000), ('2024-02', 1500)]"。'''
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
    user_query = "请分析今年每个月的销售额变化趋势，并用折线图展示"
    result = agent_executor.invoke({"input": user_query})
    print(result)
