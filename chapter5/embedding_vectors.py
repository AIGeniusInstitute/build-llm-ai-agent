import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 加载向量化模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 假设chunks为论文分块后的文本列表
chunks = [
    """
    Thanks to advances in large language models, a new type of software agent,
    the artificial intelligence (AI) agent, has entered the marketplace. Companies
    such as OpenAI, Google, Microsoft, and Salesforce promise their AI Agents will
    """,
    """
    go from generating passive text to executing tasks. Instead of a travel
    itinerary, an AI Agent would book all aspects of your trip. Instead of
    generating text or images for social media post, an AI Agent would post the
    content across a host of social media outlets. The potential power of AI Agents
    """,
    """
    has fueled legal scholars' fears that AI Agents will enable rogue commerce,
    human manipulation, rampant defamation, and intellectual property harms. These
    scholars are calling for regulation before AI Agents cause havoc.
    """,
    """
    This Article addresses the concerns around AI Agents head on. It shows that
    core aspects of how one piece of software interacts with another creates ways
    to discipline AI Agents so that rogue, undesired actions are unlikely, perhaps
    more so than rules designed to govern human agents. It also develops a way to
    leverage the computer-science approach to value-alignment to improve a user's
    ability to take action to prevent or correct AI Agent operations. That approach
    offers and added benefit of helping AI Agents align with norms around user-AI
    """
]
paper_id = "arxiv:1234.5678"

# 向量化
vectors = model.encode(chunks, normalize_embeddings=True)
vectors = np.array(vectors).astype('float32')  # FAISS要求float32类型

# 构建索引（假设向量维度为384）
dimension = vectors.shape[1]
index = faiss.IndexFlatIP(dimension)  # 余弦相似度（已归一化）
index.add(vectors)

# 存储元数据（如paper_id、chunk_id等）
metadata = [{"paper_id": paper_id, "chunk_id": i, "text": chunk} for i, chunk in enumerate(chunks)]

# 保存索引与元数据
faiss.write_index(index, "arxiv_faiss.index")
with open("arxiv_faiss_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

# 加载索引与元数据
index = faiss.read_index("arxiv_faiss.index")
with open("arxiv_faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
# 示例查询
query = "What is the main idea of the paper?"
query_vector = model.encode([query], normalize_embeddings=True)
query_vector = np.array(query_vector).astype('float32')
# 搜索
k = 2  # 返回最相似的2个结果
distances, indices = index.search(query_vector, k)
print("Top 3 similar chunks:")
for i, idx in enumerate(indices[0]):
    print(f"Rank {i + 1}: {metadata[idx]['text']}")
