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

stream = client.chat.completions.create(
    # 替换为您的模型端点
    model=os.environ["Doubao_Seed_16"],
    messages=[
        {"role": "system", "content": "你是AI人工智能助手"},
        {"role": "user", "content": "人生的意义是什么？"},
    ],
    temperature=0.1,
    stream=True,
)

for chunk in stream:

    if not chunk.choices:
        continue

    delta = chunk.choices[0].delta

    # 思考过程
    if delta and hasattr(delta, 'reasoning_content'):
        print(delta.reasoning_content, end="")

    # 最终结果
    if delta and hasattr(delta, 'content'):
        print(delta.content, end="")
