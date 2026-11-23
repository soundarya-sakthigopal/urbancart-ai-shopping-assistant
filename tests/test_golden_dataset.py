
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "golden_dataset.json"
)

VALID_TOOLS = {
    "search_catalog_and_policies",
    "check_product_stock",
    "track_customer_order",
    "find_similar_products",
    "create_support_escalation"
}

REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "expected_answer",
    "expected_source",
    "expected_tool"
}


def test_golden_dataset():
    with DATASET_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    assert isinstance(dataset, list), "Dataset must be a list"
    assert len(dataset) == 20, "Expected 20 evaluation questions"

    ids = [item["id"] for item in dataset]

    assert len(ids) == len(set(ids)), "Duplicate question IDs found"

    for item in dataset:
        assert REQUIRED_FIELDS.issubset(item.keys())

        for field in [
            "id",
            "category",
            "question",
            "expected_answer",
            "expected_tool"
        ]:
            assert isinstance(item[field], str)
            assert item[field].strip()

        assert (
            item["expected_source"] is None
            or (
                isinstance(item["expected_source"], str)
                and item["expected_source"].strip()
            )
        )

        assert item["expected_tool"] in VALID_TOOLS

    print("Golden Dataset validation passed!")
    print("Total questions:", len(dataset))
    print("Unique IDs:", len(set(ids)))
