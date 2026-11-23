from src.vectorstore.chroma_service import get_vector_store
from langchain_core.documents import Document
def retrieve_documents(
    query: str,
    top_k: int = 10,
    metadata_filter: dict = None
):
    """
    Retrieve documents from Chroma using dense similarity search.
    Supports optional metadata filtering.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be an integer between 1 and 50.")

    if metadata_filter is not None and not isinstance(metadata_filter, dict):
        raise ValueError("metadata_filter must be a dictionary.")

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query=query.strip(),
        k=top_k,
        filter=metadata_filter
    )

    return results
def retrieve_by_sku(sku: str):
    """
    Retrieve a product using exact SKU metadata lookup.
    No embedding generation required.
    """

    if not isinstance(sku, str) or not sku.strip():
        raise ValueError("SKU cannot be empty.")

    vector_store = get_vector_store()

    results = vector_store.get(
        where={"sku": sku.strip()},
        include=["documents", "metadatas"]
    )

    documents = []

    for content, metadata in zip(
        results["documents"],
        results["metadatas"]
    ):
        documents.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    return documents