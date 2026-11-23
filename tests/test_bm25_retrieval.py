
from src.ingestion.prepare_documents import prepare_all_documents
from src.ingestion.vectorstore_ingestion import (
    validate_documents,
    deduplicate_documents
)
from src.retrieval.bm25_service import create_bm25_retriever
from src.ingestion.vectorstore_ingestion import (
    prepare_documents_for_indexing
)

# Prepare documents
documents = prepare_all_documents()

# Validate documents
valid_documents, rejected_documents = validate_documents(documents)

# Remove duplicate product documents
unique_documents, duplicate_rejections = deduplicate_documents(
    valid_documents
)

unique_documents, document_ids = prepare_documents_for_indexing(
    unique_documents
)

print("Documents available for BM25:", len(unique_documents))

# Build BM25 index
bm25_retriever = create_bm25_retriever(
    documents=unique_documents,
    top_k=5
)

# Search using product name and brand
query = "UC-ELE-916693"

results = bm25_retriever.invoke(query)

print("\n--- BM25 Search Results ---")

for rank, document in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print("SKU:", document.metadata.get("sku"))
    print("Product:", document.page_content)
    print("Source:", document.metadata.get("source"))
