
from types import SimpleNamespace

from src.evaluation import deepeval_runner


def test_collect_deepeval_test_cases_without_api(monkeypatch):

    class MockRAGChain:
        def invoke(self, question):
            return {
                "final_response": {
                    "answer": "Apparel can be returned within 14 days."
                },
                "documents": [
                    SimpleNamespace(
                        page_content="Apparel can be returned within 14 days."
                    )
                ]
            }

    monkeypatch.setattr(
        deepeval_runner,
        "get_rag_chain",
        lambda: MockRAGChain()
    )

    test_cases = deepeval_runner.collect_deepeval_test_cases()

    assert len(test_cases) == 16
    assert test_cases[0].input
    assert test_cases[0].actual_output
    assert test_cases[0].expected_output
    assert test_cases[0].retrieval_context

    print(f"DeepEval test cases created: {len(test_cases)}")
    print("DeepEval runner mock test passed!")

import json


def test_run_deepeval_evaluation_without_api(monkeypatch, tmp_path):
    from src.evaluation import deepeval_runner

    mock_question = {
        "id": "GQ001",
        "question": "What is the return window for a cotton kurti?",
        "expected_answer": "Apparel can be returned within 14 days.",
        "expected_source": "return_refund_policy.txt",
        "expected_tool": "search_catalog_and_policies"
    }

    class MockRAGChain:
        def invoke(self, question):
            return {
                "final_response": {
                    "answer": "Apparel can be returned within 14 days.",
                    "source": ["return_refund_policy.txt"],
                    "confidence_level": "High"
                },
                "documents": []
            }

    mock_metrics = {
        "faithfulness": {
            "score": 0.9,
            "passed": True,
            "reason": "Answer is grounded."
        },
        "answer_relevancy": {
            "score": 0.9,
            "passed": True,
            "reason": "Answer is relevant."
        },
        "answer_correctness": {
            "score": 0.9,
            "passed": True,
            "reason": "Answer is correct."
        }
    }

    monkeypatch.setattr(
        deepeval_runner,
        "load_rag_test_cases",
        lambda: [mock_question]
    )

    monkeypatch.setattr(
        deepeval_runner,
        "get_rag_chain",
        lambda: MockRAGChain()
    )

    monkeypatch.setattr(
        deepeval_runner,
        "create_deepeval_test_case",
        lambda **kwargs: kwargs
    )

    monkeypatch.setattr(
        deepeval_runner,
        "evaluate_deepeval_test_case",
        lambda test_case: mock_metrics
    )

    original_path = deepeval_runner.Path

    monkeypatch.setattr(
        deepeval_runner,
        "Path",
        lambda value: tmp_path / original_path(value)
    )

    report = deepeval_runner.run_deepeval_evaluation(
    max_questions=1
)

    report_file = (
        tmp_path / "data/evaluation/deepeval_evaluation_report.json"
    )

    assert len(report) == 1
    assert report[0]["id"] == "GQ001"
    assert report[0]["metrics"]["faithfulness"]["passed"] is True
    assert report_file.exists()

    with open(report_file, "r", encoding="utf-8") as file:
        saved_report = json.load(file)

    assert saved_report[0]["question"] == mock_question["question"]

    print("DeepEval evaluation report mock test passed!")
