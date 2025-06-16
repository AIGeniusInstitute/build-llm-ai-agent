from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()


# 计算余弦相似度
def cosine_similarity(vector1: list[float], vector2: list[float]):
    """
    计算两个向量的余弦相似度。
    :param vector1: 第一个向量 [1,2,3]
    :param vector2: 第二个向量 [4,5,6]
    :return: 余弦相似度
    """
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = sum(a ** 2 for a in vector1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vector2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)


# 基于开源的嵌入模型，将文本转换为向量表示：
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Optional: specify device
    encode_kwargs={'normalize_embeddings': False}  # Optional: specify encode kwargs
)

text1 = "RAG是一种结合检索与生成的大模型应用架构。它提升了事实准确性。"
text2 = "RAG是一种结合检索与生成的大模型应用架构。RAG的全称是：Retrieval-Augmented Generation。"
splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)

chunks1 = splitter.split_text(text1)
chunks2 = splitter.split_text(text2)

print(chunks1)
print(chunks2)

vectors1 = embeddings.embed_documents(chunks1)
vectors2 = embeddings.embed_documents(chunks2)

print(len(vectors1))
print(len(vectors2))

text1_vector = embeddings.embed_query(text1)
text2_vector = embeddings.embed_query(text2)
print(cosine_similarity(text1_vector, text2_vector))
