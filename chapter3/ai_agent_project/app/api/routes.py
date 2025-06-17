from fastapi import APIRouter
from app.agent.agent_core import answer_question

router = APIRouter()

@router.post("/chat")
def chat_api(query: dict):
    return {"response": answer_question(query["prompt"])}