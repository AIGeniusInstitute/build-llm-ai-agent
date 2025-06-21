# cot_paper_chain.py
import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from arxiv_api import get_articles

# 1. 环境与模型初始化
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("Doubao_Seed_16")  # 可替换为实际模型名

llm = ChatOpenAI(api_key=api_key, model=model_name, base_url=base_url)


# 2. 步骤1：检索相关论文
def retrieve_papers_step(query, max_results=10):
    """
    调用arxiv API检索相关论文，返回论文摘要列表
    """
    return get_articles(query, max_results)


# 3. 步骤2：摘要与分块
def split_and_collect_abstracts(papers, chunk_size=300, chunk_overlap=50):
    """
    对论文摘要进行分块，便于后续大模型处理
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []
    for paper in papers:
        abstract_chunks = splitter.split_text(paper.summary)
        for chunk in abstract_chunks:
            # 保留标题信息，便于后续引用
            chunks.append(f"《{paper.title}》: {chunk}")
    return chunks


# 4. 步骤3：多论文内容汇总
def summarize_papers_step(paper_chunks):
    """
    调用大模型对多篇论文内容进行汇总，生成综述
    """
    prompt = "请根据以下多篇论文的内容，生成一段简明的综述，突出主要创新点和研究趋势：\n"
    for chunk in paper_chunks:
        prompt += f"- {chunk}\n"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# 5. 步骤4：基于综述生成最终答案
answer_prompt = PromptTemplate(
    input_variables=["summary", "question"],
    template=(
        "请根据以下内容，回答用户提出的问题。\n"
        "内容综述：{summary}\n"
        "用户问题：{question}\n"
        "请结合综述内容，给出详细、准确的学术答案，并标注不确定性或引用相关论文。"
    )
)

# RunnableSequence Chain
answer_chain = answer_prompt | llm


# 6. 串联推理链
def multi_step_reasoning(query):
    # 步骤1：检索论文
    papers = retrieve_papers_step(query)

    if not papers:
        return "未检索到相关论文，请尝试更换问题或关键词。"

    # 步骤2：摘要分块
    paper_chunks = split_and_collect_abstracts(papers)

    # 步骤3：内容汇总
    summary = summarize_papers_step(paper_chunks)

    # 步骤4：生成答案
    answer = answer_chain.invoke({"summary": summary, "question": query})

    return answer


# 7. 示例使用
if __name__ == '__main__':
    user_query = "AI Agent"
    result = multi_step_reasoning(user_query)
    print("【最终答案】")
    print(result.content.strip())
