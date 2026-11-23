
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def create_deepeval_test_case(
    question: str,
    actual_output: str,
    expected_output: str,
    retrieved_documents: list
) -> LLMTestCase:
    """
    Convert UrbanCart RAG output into a DeepEval test case.
    """

    retrieval_context = [
        document.page_content
        for document in retrieved_documents
        if document.page_content.strip()
    ]

    return LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
    )

def create_faithfulness_metric():
    return FaithfulnessMetric(
        threshold=0.7,
        include_reason=True
    )
def create_answer_relevancy_metric():
    return AnswerRelevancyMetric(
        threshold=0.7,
        include_reason=True
    )
def create_answer_correctness_metric():
    return GEval(
        name="Answer Correctness",
        criteria=(
            "Evaluate whether the actual answer correctly answers "
            "the user's question and matches the expected answer. "
            "Check important facts, policy conditions, and numerical "
            "details. Do not reward unsupported claims."
        ),
        evaluation_params=[
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT
],
        threshold=0.7
    )

def evaluate_deepeval_test_case(test_case):
    """
    Evaluate one RAG test case using all three DeepEval metrics.
    """

    metrics = {
        "faithfulness": create_faithfulness_metric(),
        "answer_relevancy": create_answer_relevancy_metric(),
        "answer_correctness": create_answer_correctness_metric()
    }

    results = {}

    for metric_name, metric in metrics.items():
        metric.measure(test_case)

        results[metric_name] = {
            "score": metric.score,
            "passed": metric.is_successful(),
            "reason": metric.reason
        }

    return results
