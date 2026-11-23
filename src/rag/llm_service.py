
from langchain_openai import ChatOpenAI
from src.config.settings import OPENAI_API_KEY


CHAT_MODEL = "gpt-4o-mini"


def get_chat_model():
    """
    Initialize the chat model used for RAG answer generation.
    """

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0
    )

    return llm
