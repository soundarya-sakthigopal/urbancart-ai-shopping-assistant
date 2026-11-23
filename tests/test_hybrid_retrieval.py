from src.ingestion.prepare_documents import prepare_all_documents
from src.ingestion.vectorstore_ingestion import (
    validate_documents,
    deduplicate_documents,
    prepare_documents_for_indexing
)
from src.retrieval.bm25_service import initialize_bm25_retriever
from src.retrieval.hybrid_retrieval_service import hybrid_retrieve
from src.retrieval.retrieval_service import retrieve_documents
from src.retrieval.bm25_service import get_bm25_retriever
from src.retrieval.retrieval_service import retrieve_by_sku
from src.retrieval.hybrid_retrieval_service import is_sku_query
# Prepare documents
documents = prepare_all_documents()

valid_documents, rejected_documents = validate_documents(documents)

unique_documents, duplicate_rejections = deduplicate_documents(
    valid_documents
)

# Assign consistent document IDs
prepared_documents, document_ids = prepare_documents_for_indexing(
    unique_documents
)

# Initialize BM25 once
initialize_bm25_retriever(
    documents=prepared_documents,
    top_k=10
)
# Test exact SKU retrieval
sku_results = retrieve_by_sku("UC-ELE-916693")

print("\n--- Exact SKU Retrieval ---")

for document in sku_results:
    print("SKU:", document.metadata.get("sku"))
    print("Product:", document.page_content)
# Perform hybrid retrieval
query = "Show me a Zaffira wireless mouse"
# Inspect dense retrieval
dense_results = retrieve_documents(query=query, top_k=5)

print("\n--- Dense Retrieval ---")
for rank, doc in enumerate(dense_results, start=1):
    print(rank, doc.metadata.get("sku"))

# Inspect BM25 retrieval
bm25_results = get_bm25_retriever().invoke(query)

print("\n--- BM25 Retrieval ---")
for rank, doc in enumerate(bm25_results, start=1):
    print(rank, doc.metadata.get("sku"))
results = hybrid_retrieve(
    query=query,
    top_k=5
)

print("\n--- Hybrid Retrieval Results ---")

for rank, document in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print("Document ID:", document.metadata.get("document_id"))
    print("SKU:", document.metadata.get("sku"))
    print("Product:", document.page_content)
    print("Source:", document.metadata.get("source"))

# Test hybrid retrieval with category filtering
filtered_results = hybrid_retrieve(
    query=query,
    top_k=5,
    metadata_filter={"category": "Electronics"}
)

print("\n--- Hybrid Retrieval with Electronics Filter ---")

for rank, document in enumerate(filtered_results, start=1):
    print(f"\nRank: {rank}")
    print("SKU:", document.metadata.get("sku"))
    print("Category:", document.metadata.get("category"))
    print("Source:", document.metadata.get("source"))
# Test 1: Exact SKU query
sku_query = "UC-ELE-916693"

print("\n--- SKU Pattern Detection ---")
print("Is SKU:", is_sku_query(sku_query))

sku_results = hybrid_retrieve(
    query=sku_query,
    top_k=5
)

print("\n--- Exact SKU Search ---")
for document in sku_results:
    print("SKU:", document.metadata.get("sku"))


# Test 2: Normal product query
normal_query = "Show me a Zaffira wireless mouse"

print("\n--- Normal Query Detection ---")
print("Is SKU:", is_sku_query(normal_query))

normal_results = hybrid_retrieve(
    query=normal_query,
    top_k=5
)

print("\n--- Normal Hybrid Search ---")
for rank, document in enumerate(normal_results, start=1):
    print(rank, document.metadata.get("sku"))
# Exact SKU validation
print("\n--- Exact SKU Validation ---")

test_skus = [
    "UC-ELE-916693",  # Existing SKU
    "UC-ELE-999999"   # Nonexistent SKU
]

for sku in test_skus:
    results = hybrid_retrieve(query=sku)

    print(f"\nSKU: {sku}")
    print("Results found:", len(results))

    for document in results:
        print("Product SKU:", document.metadata.get("sku"))


# Normal product query
print("\n--- Normal Product Query ---")

results = hybrid_retrieve(
    query="Show me a Zaffira wireless mouse",
    top_k=5
)

for rank, document in enumerate(results, start=1):
    print(rank, document.metadata.get("sku"))
from src.retrieval.bm25_service import retrieve_bm25_documents
# BM25 metadata filtering
print("\n--- BM25 Electronics Filter ---")

filtered_bm25_results = retrieve_bm25_documents(
    query="Show me a Zaffira wireless mouse",
    top_k=5,
    metadata_filter={"category": "Electronics"}
)

for rank, document in enumerate(filtered_bm25_results, start=1):
    print(
        rank,
        document.metadata.get("sku"),
        document.metadata.get("category")
    )


# Hybrid metadata filtering
print("\n--- Hybrid Electronics Filter ---")

filtered_hybrid_results = hybrid_retrieve(
    query="Show me a Zaffira wireless mouse",
    top_k=5,
    metadata_filter={"category": "Electronics"}
)

for rank, document in enumerate(filtered_hybrid_results, start=1):
    print(
        rank,
        document.metadata.get("sku"),
        document.metadata.get("category")
    )