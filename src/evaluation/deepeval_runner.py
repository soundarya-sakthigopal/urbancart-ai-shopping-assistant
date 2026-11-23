
import json
from pathlib import Path

from src.rag.rag_chain import get_rag_chain
from src.evaluation.deepeval_evaluator import create_deepeval_test_case
from src.evaluation.deepeval_evaluator import (
    create_deepeval_test_case,
    evaluate_deepeval_test_case
)
from src.rag.rag_initializer import initialize_rag_components
GOLDEN_DATASET_PATH = Path("data/evaluation/golden_dataset.json")


def load_rag_test_cases():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return [
        item for item in dataset
        if item.get("expected_tool") == "search_catalog_and_policies"
        and item.get("expected_source")
    ]


def collect_deepeval_test_cases():
    golden_questions = load_rag_test_cases()
    # Initialize vector database, embeddings and BM25 components
    #initialize_rag_components()

    rag_chain = get_rag_chain()

    test_cases = []

    for item in golden_questions:
        question = item["question"]

        result = rag_chain.invoke(question)

        test_case = create_deepeval_test_case(
            question=question,
            actual_output=result["final_response"]["answer"],
            expected_output=item["expected_answer"],
            retrieved_documents=result["documents"]
        )

        test_cases.append(test_case)

    return test_cases

def run_deepeval_evaluation(max_questions=None):
    """
    Run RAG evaluation against the golden dataset
    and save the results as a JSON report.
    """

    golden_questions = load_rag_test_cases()
    if max_questions is not None:
        golden_questions = golden_questions[:max_questions]
    # Initialize vector database, embeddings and BM25 components
    initialize_rag_components()
    rag_chain = get_rag_chain()

    evaluation_report = []

    for item in golden_questions:
        question = item["question"]

        # Generate the actual RAG response
        result = rag_chain.invoke(question)

        actual_answer = result["final_response"]["answer"]

        # Build the DeepEval test case
        test_case = create_deepeval_test_case(
            question=question,
            actual_output=actual_answer,
            expected_output=item["expected_answer"],
            retrieved_documents=result["documents"]
        )

        # Evaluate using all three metrics
        metric_results = evaluate_deepeval_test_case(test_case)

        evaluation_report.append({
            "id": item["id"],
            "question": question,
            "actual_answer": actual_answer,
            "expected_answer": item["expected_answer"],
            "retrieved_sources": result["final_response"].get("source", []),
            "confidence_level": result["final_response"].get(
                "confidence_level"
            ),
            "metrics": metric_results
        })

    # Save evaluation report
    report_path = Path(
        "data/evaluation/deepeval_evaluation_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(evaluation_report, file, indent=4, ensure_ascii=False)

    print(f"Evaluated questions: {len(evaluation_report)}")
    print(f"Evaluation report saved to: {report_path}")

    return evaluation_report
