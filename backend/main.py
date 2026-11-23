from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.routes.chat import router as chat_router
from src.rag.rag_initializer import initialize_rag_components


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize RAG components once when API starts
    print("Initializing UrbanCart RAG components...")

    initialize_rag_components()
    print("UrbanCart RAG initialization completed!")

    yield


app = FastAPI(
    title="UrbanCart AI Shopping Assistant API",
    description="Backend API for the UrbanCart AI Shopping Assistant",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "message": "UrbanCart AI Shopping Assistant API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


app.include_router(chat_router)