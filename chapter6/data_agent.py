import base64
import io
import ast

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
    """将查询结果字符串转换为图表。"""
    # query_result为字符串，需转为DataFrame
    try:
        df = pd.read_csv(io.StringIO(query_result))
    except Exception:
        # 兼容直接传入DataFrame
        try:
            # 使用 ast.literal_eval 更安全
            data = ast.literal_eval(query_result)
            df = pd.DataFrame(data)
        except (ValueError, SyntaxError):
            return f"无法解析查询结果: {query_result}"

    if df.empty:
        return "查询结果为空，无法绘图。"

    if len(df.columns) < 2:
        return f"数据列数少于2，无法绘图。"

    plt.figure(figsize=(8, 5))
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], marker='o')
    plt.title("销售趋势")
    plt.xlabel(str(df.columns[0]))
    plt.ylabel(str(df.columns[1]))
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


# 工具列表
tools = [
    Tool(
        name="数据可视化",
        func=plot_tool,
        description='''将查询结果转为图表。输入应该是一个字符串，表示一个列表的元组，例如 "[('2024-01', 1000), ('2024-02', 1500)]"。'''
    )
]

# 初始化SQL Agent
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    extra_tools=tools,
    verbose=True
)

if __name__ == "__main__":
    user_query = "请分析今年每个月的销售额变化趋势，并用折线图展示"
    result = agent_executor.invoke({"input": user_query})
    print(result)
