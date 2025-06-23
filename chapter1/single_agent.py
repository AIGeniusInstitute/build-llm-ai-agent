import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

model = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    model=os.environ["Doubao_15Pro"],
    max_tokens=10000,
    temperature=0.0,
)

def chat(input):
    message = model.invoke(input)
    res = message.content.strip()
    print(res)
    return res

if __name__ == "__main__":
    chat("AI Agent vs. Agentic AI")
