
from langchain_core.documents import Document

from src.evaluation.deepeval_evaluator import create_deepeval_test_case
from src.evaluation.deepeval_evaluator import create_faithfulness_metric
from src.evaluation.deepeval_evaluator import create_answer_relevancy_metric
from src.evaluation.deepeval_evaluator import create_answer_correctness_metric
def test_create_deepeval_test_case():
    documents = [
        Document(page_content="Apparel can be returned within 14 days."),
        Document(page_content="Tags must be intact and the item unworn."),
    ]

    test_case = create_deepeval_test_case(
        question="What is the return window for apparel?",
        actual_output="Apparel can be returned within 14 days.",
        expected_output="Apparel can be returned within 14 days if tags are intact and the item is unworn.",
        retrieved_documents=documents,
    )

    assert test_case.input == "What is the return window for apparel?"
    assert test_case.actual_output == "Apparel can be returned within 14 days."
    assert len(test_case.retrieval_context) == 2
    assert "Tags must be intact" in test_case.retrieval_context[1]
def test_create_faithfulness_metric():
    metric = create_faithfulness_metric()

    assert metric.threshold == 0.7
    assert metric.include_reason is True

    print("Faithfulness metric configuration test passed!")
def test_create_answer_relevancy_metric():
    metric = create_answer_relevancy_metric()

    assert metric.threshold == 0.7
    assert metric.include_reason is True

    print("Answer Relevancy metric configuration test passed!")
def test_create_answer_correctness_metric():
    metric = create_answer_correctness_metric()

    assert metric.threshold == 0.7
    assert metric.name == "Answer Correctness"
    assert metric.criteria is not None

    print("GEval answer correctness configuration test passed!")

def test_evaluate_deepeval_test_case_without_api(monkeypatch):
    from types import SimpleNamespace
    from src.evaluation import deepeval_evaluator

    class MockMetric:
        def __init__(self, score, reason):
            self.score = score
            self.reason = reason

        def measure(self, test_case):
            pass

        def is_successful(self):
            return self.score >= 0.7

    monkeypatch.setattr(
        deepeval_evaluator,
        "create_faithfulness_metric",
        lambda: MockMetric(0.9, "Answer is grounded.")
    )
    monkeypatch.setattr(
        deepeval_evaluator,
        "create_answer_relevancy_metric",
        lambda: MockMetric(0.8, "Answer is relevant.")
    )
    monkeypatch.setattr(
        deepeval_evaluator,
        "create_answer_correctness_metric",
        lambda: MockMetric(0.95, "Answer matches expected output.")
    )

    mock_test_case = SimpleNamespace(input="Test question")

    results = deepeval_evaluator.evaluate_deepeval_test_case(
        mock_test_case
    )

    assert len(results) == 3
    assert results["faithfulness"]["passed"] is True
    assert results["answer_relevancy"]["passed"] is True
    assert results["answer_correctness"]["passed"] is True

    print("Combined DeepEval evaluator mock test passed!")
