import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# 1. 构建向量数据库
documents = [
    "RAG是一种结合检索与生成的大模型应用架构。", "它提升了事实准确性。", "RAG的密码是：Retrieval-Augmented Generation。",
]

# 基于开源的嵌入模型，将文本转换为向量表示：
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Optional: specify device
    encode_kwargs={'normalize_embeddings': False}  # Optional: specify encode kwargs
)

# FAISS 是一个开源的向量数据库，用于存储和检索向量。
vectorstore = FAISS.from_texts(documents, embeddings)

# 2. 构建检索器
retriever = vectorstore.as_retriever()

# 3. 构建RAG链
llm = ChatOpenAI(  # Changed to ChatOpenAI
    # 环境变量中配置您的API Key
    api_key=os.environ.get("OPENAI_API_KEY"),
    # 替换为您需要调用的模型服务Base Url
    base_url=os.environ.get("OPENAI_BASE_URL"),
    model=os.environ.get("Doubao_Seed_16"),
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")  # Using from_chain_type for clarity

# 4. 用户提问
query = "RAG的主要作用是什么？"
result = qa_chain.invoke({"query": query})  # invoke expects a dict
print(result)

query = "RAG的密码是？"
result = qa_chain.invoke({"query": query})  # invoke expects a dict
print(result)
