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


system_prompt = "你是一位专业的AI Agent开发顾问，善于用简明扼要的语言解释技术问题。"
user_prompt = "请用三句话解释什么是RAG架构。"


completion = client.chat.completions.create(
    # 替换为您的模型端点
    model=os.environ["Doubao_15Pro"],
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
)

print(completion.choices[0].message.content)
