
from langchain_core.documents import Document

from src.evaluation.retrieval_evaluator import (
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg
)


def test_recall_at_k():
    retrieved_documents = [
        Document(
            page_content="Product details",
            metadata={"source": "product_catalogue"}
        ),
        Document(
            page_content="Return policy",
            metadata={"source": "return_refund_policy.txt"}
        ),
        Document(
            page_content="Warranty policy",
            metadata={"source": "warranty_policy.txt"}
        )
    ]

    expected_sources = ["return_refund_policy.txt"]

    result = calculate_recall_at_k(
        retrieved_documents,
        expected_sources
    )

    assert result == 1.0


def test_mrr():
    retrieved_documents = [
        Document(
            page_content="Product details",
            metadata={"source": "product_catalogue"}
        ),
        Document(
            page_content="Shipping policy",
            metadata={"source": "shipping_delivery_policy.txt"}
        ),
        Document(
            page_content="Return policy",
            metadata={"source": "return_refund_policy.txt"}
        )
    ]

    expected_sources = ["return_refund_policy.txt"]

    result = calculate_mrr(
        retrieved_documents,
        expected_sources
    )

    assert result == 1 / 3


def test_ndcg():
    retrieved_documents = [
        Document(
            page_content="Product details",
            metadata={"source": "product_catalogue"}
        ),
        Document(
            page_content="Return policy",
            metadata={"source": "return_refund_policy.txt"}
        ),
        Document(
            page_content="Warranty policy",
            metadata={"source": "warranty_policy.txt"}
        ),
        Document(
            page_content="Duplicate return policy chunk",
            metadata={"source": "return_refund_policy.txt"}
        )
    ]

    relevance_scores = {
        "return_refund_policy.txt": 3,
        "product_catalogue": 2,
        "warranty_policy.txt": 1
    }

    result = calculate_ndcg(
        retrieved_documents,
        relevance_scores,
        k=3
    )

    assert 0.0 <= result <= 1.0


def test_empty_expected_sources():
    assert calculate_recall_at_k([], []) == 0.0
    assert calculate_mrr([], []) == 0.0


def test_invalid_ndcg_inputs():
    assert calculate_ndcg([], {}, k=5) == 0.0
    assert calculate_ndcg([], {"return_refund_policy.txt": 3}, k=0) == 0.0
