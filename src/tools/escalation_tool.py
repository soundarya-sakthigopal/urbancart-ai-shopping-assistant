import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ESCALATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escalation_queue.json"
)

"""
Why JSON?

Escalation records have structured fields such as issue description, reason, timestamp and escalation ID. JSON is convenient for our prototype.

In production, this would typically be stored in a CRM, ticketing system or support platform through an API.

Add this and tell me when done.
"""
def load_escalations():
    """
    Load existing escalation records.
    """

    if not ESCALATION_FILE.exists():
        return []

    with open(ESCALATION_FILE, "r", encoding="utf-8") as file:
        escalations = json.load(file)

    if not isinstance(escalations, list):
        raise ValueError(
            "Escalation queue must contain a JSON list."
        )

    return escalations
def escalate_to_support(
    customer_query: str,
    reason: str,
    conversation_summary: str
) -> dict:
    """
    Create a customer support escalation record.
    """

    if not customer_query or not customer_query.strip():
        return {
            "status": "error",
            "message": "Customer query is required."
        }

    if not reason or not reason.strip():
        return {
            "status": "error",
            "message": "Escalation reason is required."
        }

    if not conversation_summary or not conversation_summary.strip():
        return {
            "status": "error",
            "message": "Conversation summary is required."
        }

    escalations = load_escalations()

    escalation_record = {
        "escalation_id": str(uuid4()),
        "customer_query": customer_query.strip(),
        "reason": reason.strip(),
        "conversation_summary": conversation_summary.strip(),
        "status": "Pending",
        "created_at": datetime.now().isoformat(timespec="seconds")
    }

    escalations.append(escalation_record)

    ESCALATION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        ESCALATION_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            escalations,
            file,
            indent=4,
            ensure_ascii=False
        )

    return {
        "status": "success",
        "escalation_id": escalation_record["escalation_id"],
        "escalation_status": "Pending",
        "message": "Your issue has been escalated to customer support."
    }