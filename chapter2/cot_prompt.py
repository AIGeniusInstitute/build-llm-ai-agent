import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(
    # 环境变量中配置您的API Key
    api_key=os.environ["OPENAI_API_KEY"],
    # 替换为您需要调用的模型服务Base Url
    base_url=os.environ["OPENAI_BASE_URL"],
)

cot_prompt = (
    "你是一位数学老师。请一步步思考并解答："
    "如果一辆车以每小时60公里的速度行驶2小时，"
    "它一共行驶了多少公里？"
)

completion = client.chat.completions.create(
    # 替换为您的模型端点
    model=os.environ["Doubao_15Pro"],
    messages=[
        {"role": "user", "content": cot_prompt}
    ]
)

print(completion.choices[0].message.content)
