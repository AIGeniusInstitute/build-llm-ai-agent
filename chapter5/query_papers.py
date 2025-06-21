from fastapi import FastAPI
from pydantic import BaseModel

from arxiv_api import get_articles, Article

app = FastAPI()


class PaperQuery(BaseModel):
    question: str
    top_k: int = 5


@app.post("/api/v1/query", response_model=list[Article])
def query_papers(query: PaperQuery):
    """
    接收用户自然语言问题，返回相关论文列表。
    """
    return get_articles(query.question, query.top_k)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
