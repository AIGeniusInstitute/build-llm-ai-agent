# data_agent.py
import ast

import numpy as np
from sklearn.linear_model import LinearRegression

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


def predict_tool(input_str: str):
    """根据历史数据使用线性回归进行预测。输入应该是一个包含历史数据的字典列表字符串。"""
    try:
        data = ast.literal_eval(input_str)
        if isinstance(data, str):
            data = ast.literal_eval(data)
    except (ValueError, SyntaxError) as e:
        return f"无法解析输入字符串: {input_str}. 错误: {e}"

    if not isinstance(data, list) or not data or not all(isinstance(d, dict) for d in data):
        return "输入数据格式错误，应为非空字典列表。"

    # 动态识别 x 和 y 的键
    first_item_keys = list(data[0].keys())
    if len(first_item_keys) < 2:
        return "数据字典中至少需要两个键。"

    # 假设第一个键是x，第二个键是y
    x_key, y_key = first_item_keys[0], first_item_keys[1]

    try:
        x_values_raw = [d[x_key] for d in data]
        y_values = np.array([d[y_key] for d in data])

        # 检查x值是否为字符串，如果是，则转换为数值
        if isinstance(x_values_raw[0], str):
            try:
                # 尝试将日期字符串转换为时间戳
                x_values = pd.to_datetime(x_values_raw).astype(np.int64) // 10**9
                x_values = x_values.to_numpy().reshape(-1, 1)
                is_date = True
            except (ValueError, TypeError):
                # 如果不是日期，则使用索引作为x值
                x_values = np.arange(len(x_values_raw)).reshape(-1, 1)
                is_date = False
        else:
            x_values = np.array(x_values_raw).reshape(-1, 1)
            is_date = False

    except (KeyError, TypeError):
        return f"无法从数据中提取有效的数值。请确保 '{x_key}' 和 '{y_key}' 的值是数字。"

    model = LinearRegression().fit(x_values, y_values)
    
    if is_date:
        # 预测下一个时间点
        last_timestamp = pd.to_datetime(x_values_raw[-1]).timestamp()
        # 假设下一个时间点是一个月后
        next_timestamp = last_timestamp + 30 * 24 * 3600 
        next_x_numeric = np.array([[next_timestamp]])
        next_x_label = pd.to_datetime(next_timestamp, unit='s').strftime('%Y-%m-%d')
    else:
        next_x_numeric = np.array([[x_values.max() + 1]])
        next_x_label = next_x_numeric[0][0]

    prediction = model.predict(next_x_numeric)[0]

    return f"基于历史数据，预测当 {x_key} 为 {next_x_label} 时, {y_key} 的值为: {prediction:.2f}"


# 工具列表
tools = [
    Tool(
        name="数据可视化",
        func=plot_tool,
        description='用于将数据可视化。输入应该是一个字典字符串，其中包含绘图所需的所有参数，例如：`{"data": [("2024-01", 12000)], "chart_type": "bar", "title": "Sales"}`。'
    ),
    Tool(
        name="数据预测",
        func=predict_tool,
        description='用于根据历史数据进行线性回归预测。输入应该是一个字典列表的字符串，例如：`[{"季度": 1, "销售额": 32000}, {"季度": 2, "销售额": 35000}]`。'
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
