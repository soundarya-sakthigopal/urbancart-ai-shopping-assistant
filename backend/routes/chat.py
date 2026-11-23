
from fastapi import APIRouter, HTTPException
from backend.schemas.chat import ChatRequest, ChatResponse
from src.services.agent_service import invoke_agent

router = APIRouter(prefix="/api", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        response = invoke_agent(
            message=request.message,
            thread_id=request.thread_id
        )

        return ChatResponse(
            response=response,
            thread_id=request.thread_id
        )

    except Exception as e:
        print(f"Chat API error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process your request. Please try again later."
        )
