# chat.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class Query(BaseModel):
    prompt: str

@app.post("/chat")
def chat(query: Query):
    # 这里可以调用LLM
    return {"response": f"Echo: {query.prompt}"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)