import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 原始文本数据
documents = [
    {"id": 1, "title": "关于AI Agent的定义", "text": "AI Agent是一种能够自主感知、决策和行动的智能体。"},
    {"id": 2, "title": "LLM的核心技术", "text": "大型语言模型（LLM）是当前AI Agent的基础。"},
    {"id": 3, "title": "RAG系统简介", "text": "检索增强生成（RAG）结合了检索与生成的能力。"}
]

# 2. 加载嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. 生成文本向量
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts, show_progress_bar=True)

# 4. 保存embeddings.npy
np.save('embeddings.npy', embeddings)

# 5. 准备并保存metadata.json
metadata = []
for doc in documents:
    meta = {
        "id": doc["id"],
        "title": doc["title"],
        "text": doc["text"]
    }
    metadata.append(meta)

with open('metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("已生成 embeddings.npy 和 metadata.json")

# 加载论文分块向量和元数据
embeddings = np.load("embeddings.npy")  # shape: (num_papers, embedding_dim)
with open("metadata.json", "r") as f:
    metadata = json.load(f)  # list of dicts, each with 'paper_id'等字段

# 加载FAISS索引
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 加载语义编码模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 引用网络示例
reference_graph = {
    "arxiv:1234.5678": ["arxiv:2345.6789", "arxiv:3456.7890"],
    "arxiv:2345.6789": ["arxiv:4567.8901"]
}


def hybrid_search(query, faiss_index, metadata, reference_graph, top_k=5):
    """
    query: str, 用户输入的检索语句
    faiss_index: 已训练好的faiss索引
    metadata: list, 每个元素为dict，包含'paper_id'等
    reference_graph: dict, paper_id -> list of paper_id
    top_k: int, 语义检索返回的核心论文数
    """
    # 1. 语义检索
    query_vec = model.encode([query]).astype('float32')  # shape: (1, dim)
    D, I = faiss_index.search(query_vec, top_k)  # I: (1, top_k) 检索到的索引
    core_paper_ids = [metadata[i]['id'] for i in I[0]]

    # 2. 引用网络扩展
    related_papers = set(core_paper_ids)
    for pid in core_paper_ids:
        # 下游引用
        related_papers.update(reference_graph.get(pid, []))
        # 上游被引用
        for k, v in reference_graph.items():
            if pid in v:
                related_papers.add(k)

    return list(related_papers)


# 示例调用
papers = hybrid_search("AI Agent的定义", index, metadata, reference_graph)
print(papers)
