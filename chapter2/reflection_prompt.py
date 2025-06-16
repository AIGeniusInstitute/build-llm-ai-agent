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

reflection_prompt = (
    "请解答以下问题，并在回答后自我检查推理是否有误："
    "问题：地球到月球的平均距离是多少公里？"
)

completion = client.chat.completions.create(
    # 替换为您的模型端点
    model=os.environ["Doubao_15Pro"],
    messages=[
        {"role": "user", "content": reflection_prompt}
    ]
)

print(completion.choices[0].message.content)
