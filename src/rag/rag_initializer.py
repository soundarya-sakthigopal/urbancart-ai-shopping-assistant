
from src.ingestion.prepare_documents import prepare_all_documents
from src.ingestion.vectorstore_ingestion import (
    validate_documents,
    deduplicate_documents,
    prepare_documents_for_indexing
)
from src.retrieval.bm25_service import initialize_bm25_retriever


def initialize_rag_components():
    """
    Prepare documents and initialize the in-memory BM25 index.
    Does not modify Chroma.
    """

    documents = prepare_all_documents()

    valid_documents, rejected_documents = validate_documents(
        documents
    )

    unique_documents, duplicate_rejections = deduplicate_documents(
        valid_documents
    )

    prepared_documents, document_ids = prepare_documents_for_indexing(
        unique_documents
    )

    initialize_bm25_retriever(prepared_documents)
    

    print("BM25 retriever initialized successfully.")
    print(f"BM25 document count: {len(prepared_documents)}")
    print(f"Rejected documents: {len(rejected_documents)}")
    print(f"Duplicate/conflicting SKU rejections: {len(duplicate_rejections)}")

    return True
