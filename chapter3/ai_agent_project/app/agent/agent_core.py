import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("Doubao_Seed_16_Flash")

llm = ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url)
prompt = ChatPromptTemplate.from_template("你是一个助手，请回答：{question}")


# Agent核心逻辑封装
def answer_question(question: str) -> str:
    chain = prompt | llm  # Use RunnableSequence
    result = chain.invoke({"question": question})
    return result.content
