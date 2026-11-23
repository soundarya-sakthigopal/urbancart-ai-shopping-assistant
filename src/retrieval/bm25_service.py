
from langchain_community.retrievers import BM25Retriever
import re

_bm25_retriever = None
_bm25_documents = []
def preprocess_text(text: str):
    """
    Normalize text for case-insensitive BM25 matching.
    """

    return re.findall(r"[a-z0-9]+", text.lower())
def create_bm25_retriever(documents, top_k=10):
    """
    Build a BM25 retriever from prepared product
    and policy documents.
    """

    if not documents:
        raise ValueError("Documents cannot be empty.")

    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    retriever = BM25Retriever.from_documents(
        documents=documents,
        k=top_k,
        preprocess_func=preprocess_text
    )

    return retriever
def initialize_bm25_retriever(documents, top_k=10):
    """
    Initialize or refresh the BM25 index.
    """

    global _bm25_retriever, _bm25_documents

    _bm25_retriever = create_bm25_retriever(
        documents=documents,
        top_k=top_k
    )

    _bm25_documents = documents.copy()

    return _bm25_retriever

def get_bm25_retriever():
    """
    Return the initialized BM25 retriever.
    """

    if _bm25_retriever is None:
        raise RuntimeError(
            "BM25 retriever is not initialized. "
            "Initialize it during application startup."
        )

    return _bm25_retriever
def retrieve_bm25_documents(
    query: str,
    top_k: int = 10,
    metadata_filter: dict = None
):
    """
    Retrieve BM25 documents with metadata filtering
    across the complete indexed document collection.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    if metadata_filter is not None and not isinstance(metadata_filter, dict):
        raise ValueError("metadata_filter must be a dictionary.")

    retriever = get_bm25_retriever()

    query_tokens = preprocess_text(query.strip())

    # Calculate BM25 scores for the complete collection
    scores = retriever.vectorizer.get_scores(query_tokens)

    # Apply metadata filtering before selecting top results
    eligible_indexes = [
        index
        for index, document in enumerate(_bm25_documents)
        if all(
            document.metadata.get(key) == value
            for key, value in (metadata_filter or {}).items()
        )
    ]

    # Rank only eligible documents using their BM25 scores
    ranked_indexes = sorted(
        eligible_indexes,
        key=lambda index: scores[index],
        reverse=True
    )

    return [
        _bm25_documents[index]
        for index in ranked_indexes[:top_k]
    ]
