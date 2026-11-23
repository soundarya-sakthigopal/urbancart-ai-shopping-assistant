import src.tools.escalation_tool as escalation_module
from src.tools.escalation_tool import (
    ESCALATION_FILE,
    load_escalations,
    escalate_to_support
)
from pathlib import Path
TEST_ESCALATION_FILE = (
    Path(__file__).resolve().parent
    / "test_escalation_queue.json"
)


def setup_test_queue():
    escalation_module.ESCALATION_FILE = TEST_ESCALATION_FILE

    if TEST_ESCALATION_FILE.exists():
        TEST_ESCALATION_FILE.unlink()
def test_successful_escalation():

    setup_test_queue()

    result = escalate_to_support(
        customer_query="My order has not been delivered.",
        reason="Repeated delivery delay",
        conversation_summary=(
            "Customer contacted support multiple times "
            "about the delayed order."
        )
    )

    print("\nSuccessful Escalation Test:")
    print(result)

    assert result["status"] == "success"
    assert result["escalation_status"] == "Pending"
    assert result["escalation_id"]

    escalations = load_escalations()

    assert len(escalations) == 1
    assert escalations[0]["reason"] == "Repeated delivery delay"
def test_unique_escalation_ids():

    setup_test_queue()

    first_result = escalate_to_support(
        customer_query="Order delayed.",
        reason="Delivery delay",
        conversation_summary="Customer reported a delayed order."
    )

    second_result = escalate_to_support(
        customer_query="Payment failed.",
        reason="Payment issue",
        conversation_summary="Customer reported a failed payment."
    )

    print("\nUnique Escalation IDs Test:")
    print(first_result["escalation_id"])
    print(second_result["escalation_id"])

    assert first_result["escalation_id"] != second_result["escalation_id"]

    escalations = load_escalations()

    assert len(escalations) == 2
def test_empty_customer_query():

    setup_test_queue()

    result = escalate_to_support(
        customer_query="",
        reason="Delivery delay",
        conversation_summary="Customer reported a delayed order."
    )

    print("\nEmpty Customer Query Test:")
    print(result)

    assert result["status"] == "error"
    assert result["message"] == "Customer query is required."
def test_empty_reason():

    setup_test_queue()

    result = escalate_to_support(
        customer_query="My order is delayed.",
        reason="",
        conversation_summary="Customer reported a delayed order."
    )

    print("\nEmpty Reason Test:")
    print(result)

    assert result["status"] == "error"
    assert result["message"] == "Escalation reason is required."
def test_empty_conversation_summary():

    setup_test_queue()

    result = escalate_to_support(
        customer_query="My order is delayed.",
        reason="Delivery delay",
        conversation_summary=""
    )

    print("\nEmpty Conversation Summary Test:")
    print(result)

    assert result["status"] == "error"
    assert result["message"] == "Conversation summary is required."
if __name__ == "__main__":

    test_successful_escalation()
    test_unique_escalation_ids()
    test_empty_customer_query()
    test_empty_reason()
    test_empty_conversation_summary()

    if TEST_ESCALATION_FILE.exists():
        TEST_ESCALATION_FILE.unlink()

    print("\nAll escalation tool tests passed.")