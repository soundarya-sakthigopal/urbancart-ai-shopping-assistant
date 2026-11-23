from typing import List, Dict
from langchain_core.documents import Document
import re
from src.retrieval.retrieval_service import (
    retrieve_documents,
    retrieve_by_sku
)
from src.retrieval.bm25_service import retrieve_bm25_documents
from src.retrieval.query_rewriter import (
    rewrite_query,
    get_policy_sources
)
def reciprocal_rank_fusion(
    retrieval_results: List[List[Document]],
    k: int = 60
) -> List[Document]:
    """
    Combine multiple retrieval rankings using
    Reciprocal Rank Fusion (RRF).
    """

    if not isinstance(retrieval_results, list):
        raise ValueError("Retrieval results must be a list.")

    if not isinstance(k, int) or k <= 0:
        raise ValueError("RRF k must be a positive integer.")

    fused_scores: Dict[str, float] = {}
    document_map: Dict[str, Document] = {}

    for results in retrieval_results:
        for rank, document in enumerate(results, start=1):

            document_id = document.metadata.get("document_id")

            if not document_id:
                raise ValueError(
                    "Every retrieved document must have a document_id."
                )

            rrf_score = 1 / (k + rank)

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0) + rrf_score
            )

            document_map[document_id] = document

    ranked_documents = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        document_map[document_id]
        for document_id, score in ranked_documents
    ]
def hybrid_retrieve(
    query: str,
    top_k: int = 10,
    rrf_k: int = 60,
    metadata_filter: dict = None
) -> List[Document]:
    """
    Retrieve documents using dense and BM25 search,
    then combine rankings using RRF.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    query = query.strip()

    # Perform exact SKU lookup only when query matches SKU format
    if is_sku_query(query):
        exact_sku_results = retrieve_by_sku(query)

        # Return exact match if SKU exists
        if exact_sku_results:
            return exact_sku_results

        # Do not perform semantic retrieval for an unknown SKU
        return []
        # Expand query with policy-related terminology
    expanded_query = rewrite_query(query)

    # Dense retrieval uses the original query.
    # This avoids an additional embedding API call.
    dense_results = retrieve_documents(
    query=query,
    top_k=top_k,
    metadata_filter=metadata_filter
    )

    # BM25 searches using the original query.
    bm25_original_results = retrieve_bm25_documents(
    query=query,
    top_k=top_k,
    metadata_filter=metadata_filter
    )

    # Run a second BM25 search only if query expansion changed it.
    if expanded_query != query.lower().strip():
        bm25_expanded_results = retrieve_bm25_documents(
        query=expanded_query,
        top_k=top_k,
        metadata_filter=metadata_filter
        )
    else:
        bm25_expanded_results = []

    # Retrieve policy documents based on detected intent
    policy_results = retrieve_relevant_policies(
    query=query,
    top_k=top_k
    )
    # Identify policy sources relevant to the customer query
    allowed_policy_sources = get_policy_sources(query)

    # Remove unrelated policies from the other retrieval results
    dense_results = filter_irrelevant_policies(
    dense_results,
    allowed_policy_sources
    )

    bm25_original_results = filter_irrelevant_policies(
    bm25_original_results,
    allowed_policy_sources
    )

    bm25_expanded_results = filter_irrelevant_policies(
    bm25_expanded_results,
    allowed_policy_sources
    )
    policy_results = filter_irrelevant_policies(
    policy_results,
    allowed_policy_sources
    )
    # Reciprocal Rank Fusion
    hybrid_results = reciprocal_rank_fusion(
    retrieval_results=[
        dense_results,
        bm25_original_results,
        bm25_expanded_results,
        policy_results
    ],
    k=rrf_k
)
    return hybrid_results[:top_k]

def is_sku_query(query: str) -> bool:
    """
    Check whether the query is an exact UrbanCart SKU.
    Format: UC-CATEGORY-123456
    """

    return bool(
        re.fullmatch(
            r"UC-[A-Z]+-\d{6}",
            query.strip(),
            flags=re.IGNORECASE
        )
    )
def retrieve_relevant_policies(
    query: str,
    top_k: int = 5
) -> List[Document]:
    """
    Retrieve policy documents relevant to the
    detected customer intent.
    """

    policy_sources = get_policy_sources(query)

    if not policy_sources:
        return []

    policy_results = []

    for source in policy_sources:
        expanded_query = rewrite_query(query)

        documents = retrieve_bm25_documents(
            query=expanded_query,
            top_k=top_k,
            metadata_filter={"source": source}
        )

        policy_results.extend(documents)

    return policy_results
def filter_irrelevant_policies(
    documents: List[Document],
    allowed_policy_sources: List[str]
) -> List[Document]:
    """
    Remove unrelated policy documents while
    preserving all product catalogue documents.
    """

    if not allowed_policy_sources:
        return documents

    filtered_documents = []

    for document in documents:
        source = document.metadata.get("source", "")

        if source.endswith("_policy.txt"):
            if source not in allowed_policy_sources:
                continue

        filtered_documents.append(document)

    return filtered_documents