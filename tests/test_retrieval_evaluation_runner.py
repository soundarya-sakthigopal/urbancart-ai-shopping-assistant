
from langchain_core.documents import Document

from src.evaluation import retrieval_evaluator


def test_evaluate_retrieval_dataset_without_api(
    monkeypatch,
    tmp_path
):

    # Mock BM25 initialization.
    monkeypatch.setattr(
        retrieval_evaluator,
        "initialize_rag_components",
        lambda: True
    )

    # Mock hybrid retrieval.
    def mock_hybrid_retrieve(query, top_k):
        return [
            Document(
                page_content="Mock relevant policy document",
                metadata={
                    "source": "return_refund_policy.txt",
                    "document_id": "test_doc_001"
                }
            )
        ]

    monkeypatch.setattr(
        retrieval_evaluator,
        "hybrid_retrieve",
        mock_hybrid_retrieve
    )

    report = retrieval_evaluator.evaluate_retrieval_dataset(
    top_k=5,
    output_path=tmp_path / "mock_report.json"
)

    assert report["evaluated_rag_questions"] > 0

    assert report["summary"]["average_recall_at_5"] >= 0.0
    assert report["summary"]["average_mrr"] >= 0.0
    assert report["summary"]["average_ndcg"] >= 0.0

    assert len(report["results"]) == report["evaluated_rag_questions"]

    print("\nMock evaluation runner test passed!")
