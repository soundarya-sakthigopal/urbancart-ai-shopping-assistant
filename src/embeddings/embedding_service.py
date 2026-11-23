
from langchain_openai import OpenAIEmbeddings

from src.config.settings import OPENAI_API_KEY

EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding_model():

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )

    return embeddings
