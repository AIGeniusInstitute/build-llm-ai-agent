from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain # LLMChain is deprecated
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("Doubao_Seed_16_Flash")

llm = ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url)
prompt = ChatPromptTemplate.from_template("你是一个助手，请回答：{question}")

# chain = LLMChain(llm=llm, prompt=prompt) # Deprecated
chain = prompt | llm # Use RunnableSequence 
result = chain.invoke({"question": "什么是AI Agent？"})
print(result.content)