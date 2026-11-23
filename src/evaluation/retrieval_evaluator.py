import json
from pathlib import Path
from statistics import mean

from src.retrieval.hybrid_retrieval_service import hybrid_retrieve
from src.rag.rag_initializer import initialize_rag_components
def calculate_recall_at_k(
    retrieved_documents: list,
    expected_sources: list
) -> float:
    """
    Calculate Recall@K.

    Measures how many expected source documents
    were retrieved within the top K results.
    """

    if not expected_sources:
        return 0.0

    retrieved_sources = {
        document.metadata.get("source")
        for document in retrieved_documents
        if document.metadata.get("source")
    }

    expected_source_set = set(expected_sources)

    relevant_retrieved = (
        retrieved_sources & expected_source_set
    )

    return len(relevant_retrieved) / len(expected_source_set)

def calculate_mrr(
    retrieved_documents: list,
    expected_sources: list
) -> float:
    """
    Calculate Mean Reciprocal Rank for one query.

    Returns the reciprocal rank of the first
    relevant retrieved document.
    """

    if not expected_sources:
        return 0.0

    expected_source_set = set(expected_sources)

    for rank, document in enumerate(retrieved_documents, start=1):
        source = document.metadata.get("source")

        if source in expected_source_set:
            return 1.0 / rank

    return 0.0


import math


def calculate_ndcg(
    retrieved_documents: list,
    relevance_scores: dict,
    k: int = 5
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.

    Evaluates the ranking quality of retrieved documents
    using source-level relevance scores.

    Duplicate chunks from the same source are counted once.
    """

    if k <= 0 or not relevance_scores:
        return 0.0

    # Collect unique retrieved sources in ranking order.
    actual_scores = []
    seen_sources = set()

    for document in retrieved_documents:
        source = document.metadata.get("source")

        if not source or source in seen_sources:
            continue

        seen_sources.add(source)

        if len(actual_scores) == k:
            break

        score = relevance_scores.get(source, 0)
        actual_scores.append(score)

    # Calculate Discounted Cumulative Gain.
    def dcg(scores):
        return sum(
            (2 ** score - 1) / math.log2(rank + 1)
            for rank, score in enumerate(scores, start=1)
        )

    # Ideal ranking: highest relevance first.
    ideal_scores = sorted(
        relevance_scores.values(),
        reverse=True
    )[:k]

    actual_dcg = dcg(actual_scores)
    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg

def evaluate_retrieval_dataset(
    top_k: int = 5,
    output_path: Path = None
) -> dict:
    """
    Evaluate UrbanCart hybrid retrieval using
    the golden evaluation dataset.

    Only RAG questions with expected sources
    are included in retrieval metrics.
    """

    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "data"
        / "evaluation"
        / "golden_dataset.json"
    )

    report_path = output_path or (
    project_root
    / "data"
    / "evaluation"
    / "retrieval_evaluation_report.json"
)

    with dataset_path.open(encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("Golden dataset must be a list.")

    # Select RAG questions with known expected sources.
    rag_questions = []

    for item in dataset:
        if item.get("expected_tool") != "search_catalog_and_policies":
            continue

        expected_source = item.get("expected_source")

        if isinstance(expected_source, str) and expected_source.strip():
            expected_sources = [expected_source.strip()]
        elif isinstance(expected_source, list):
            expected_sources = [
                source.strip()
                for source in expected_source
                if isinstance(source, str) and source.strip()
            ]
        else:
            expected_sources = []

        if expected_sources:
            rag_questions.append((item, expected_sources))

    if not rag_questions:
        raise ValueError("No eligible RAG questions found.")

    # Initialize BM25 only once.
    initialize_rag_components()

    evaluation_results = []

    for item, expected_sources in rag_questions:

        question = item["question"]

        retrieved_documents = hybrid_retrieve(
            query=question,
            top_k=top_k
        )

        recall = calculate_recall_at_k(
            retrieved_documents,
            expected_sources
        )

        mrr = calculate_mrr(
            retrieved_documents,
            expected_sources
        )

        # Source-level relevance labels.
        relevance_scores = {
            source: 3
            for source in expected_sources
        }

        ndcg = calculate_ndcg(
            retrieved_documents,
            relevance_scores,
            k=top_k
        )

        retrieved_sources = [
            document.metadata.get("source")
            for document in retrieved_documents
            if document.metadata.get("source")
        ]

        evaluation_results.append({
            "id": item["id"],
            "category": item["category"],
            "question": question,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            f"recall_at_{top_k}": recall,
            "mrr": mrr,
            "ndcg": ndcg
        })

    # Calculate average scores across all evaluated questions.
    summary = {
        f"average_recall_at_{top_k}": mean(
            result[f"recall_at_{top_k}"]
            for result in evaluation_results
        ),
        "average_mrr": mean(
            result["mrr"]
            for result in evaluation_results
        ),
        "average_ndcg": mean(
            result["ndcg"]
            for result in evaluation_results
        )
    }

    report = {
        "total_dataset_questions": len(dataset),
        "evaluated_rag_questions": len(evaluation_results),
        "top_k": top_k,
        "summary": summary,
        "results": evaluation_results
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\nRetrieval Evaluation Completed")
    print("--------------------------------")
    print("RAG questions evaluated:", len(evaluation_results))
    print(f"Average Recall@{top_k}: {summary[f'average_recall_at_{top_k}']:.4f}")
    print(f"Average MRR: {summary['average_mrr']:.4f}")
    print(f"Average nDCG: {summary['average_ndcg']:.4f}")
    print("Report saved to:", report_path)

    return report
