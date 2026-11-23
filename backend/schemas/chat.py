
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="Customer message"
    )

    thread_id: str = Field(
        ...,
        min_length=1,
        description="Unique conversation thread ID"
    )


class ChatResponse(BaseModel):
    response: str
    thread_id: str
