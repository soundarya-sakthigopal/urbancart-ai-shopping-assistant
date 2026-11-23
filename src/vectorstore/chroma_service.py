
from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings.embedding_service import get_embedding_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DIR = PROJECT_ROOT / "data" / "vectorstore"

COLLECTION_NAME = "urbancart_knowledge_base"


def get_vector_store():

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    return vector_store
