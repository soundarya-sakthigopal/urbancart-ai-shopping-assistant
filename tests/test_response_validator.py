
from langchain_core.documents import Document

from src.rag.response_validator import (
    validate_response_structure,
    validate_sources,
    validate_evidence,
    calculate_confidence,
    validate_and_format_response
)


def test_response_validator():

    documents = [
        Document(
           page_content=(
    "1.1 Apparel & Footwear: Returnable within "
    "14 days of delivery, tags intact, unworn."
),
            metadata={
                "source": "return_refund_policy.txt",
                "sku": None
            }
        )
    ]

    valid_response = {
        "answer": (
            "Apparel can be returned within 14 days "
            "if tags are intact and the item is unworn."
        ),
        "source": [
            "return_refund_policy.txt - Section 1.1"
        ],
        "evidence": [
            "Apparel & Footwear: Returnable within "
            "14 days of delivery, tags intact, unworn."
        ]
    }

    print(
        "Structure:",
        validate_response_structure(valid_response)
    )

    print(
        "Source:",
        validate_sources(valid_response["source"], documents)
    )

    print(
        "Evidence:",
        validate_evidence(
         valid_response["evidence"],
        documents,
        valid_response["source"]
        )
    )

    print(
        "Confidence:",
        calculate_confidence(valid_response, documents)
    )

    print(
        "Final response:",
        validate_and_format_response(valid_response, documents)
    )

def test_invalid_evidence():

    documents = [
        Document(
            page_content=(
                "Apparel & Footwear: Returnable within "
                "14 days of delivery, tags intact, unworn."
            ),
            metadata={
                "source": "return_refund_policy.txt",
                "sku": None
            }
        )
    ]

    invalid_response = {
        "answer": "Apparel can be returned within 30 days.",
        "source": [
            "return_refund_policy.txt - Section 1.1"
        ],
        "evidence": [
            "Apparel can be returned within 30 days."
        ]
    }

    final_response = validate_and_format_response(
        invalid_response,
        documents
    )

    print("\nInvalid Evidence Test:")
    print("Final response:", final_response)

def test_invalid_source():

    documents = [
        Document(
            page_content=(
                "Apparel & Footwear: Returnable within "
                "14 days of delivery, tags intact, unworn."
            ),
            metadata={
                "source": "return_refund_policy.txt",
                "sku": None
            }
        )
    ]

    invalid_response = {
        "answer": "Apparel can be returned within 14 days.",
        "source": [
            "warranty_policy.txt - Section 1.1"
        ],
        "evidence": [
            "Apparel & Footwear: Returnable within "
            "14 days of delivery, tags intact, unworn."
        ]
    }

    final_response = validate_and_format_response(
        invalid_response,
        documents
    )

    print("\nInvalid Source Test:")
    print("Final response:", final_response)


def test_cross_source_evidence_mismatch():

    documents = [
        Document(
            page_content=(
                "Apparel & Footwear: Returnable within "
                "14 days of delivery, tags intact, unworn."
            ),
            metadata={
                "source": "return_refund_policy.txt",
                "sku": None
            }
        ),
        Document(
            page_content=(
                "Electronics carry manufacturer warranty "
                "for 6-24 months."
            ),
            metadata={
                "source": "warranty_policy.txt",
                "sku": None
            }
        )
    ]

    invalid_response = {
        "answer": "Apparel has a 6-24 month warranty.",
        "source": [
            "return_refund_policy.txt - Section 1.1"
        ],
        "evidence": [
            "Electronics carry manufacturer warranty "
            "for 6-24 months."
        ]
    }

    final_response = validate_and_format_response(
        invalid_response,
        documents
    )

    print("\nCross-source Evidence Mismatch Test:")
    print("Final response:", final_response)

def test_return_policy_evidence_with_quote_variations():

    policy = Document(
        page_content='''1. RETURN ELIGIBILITY
1.1 Apparel & Footwear: Returnable within 14 days of delivery, tags intact, unworn.
1.2 Electronics: Returnable within 7 days of delivery, original packaging and accessories required.
1.3 Home & Kitchen: Returnable within 10 days of delivery if unused and in original packaging.
1.4 Beauty & Personal Care: Non-returnable once the seal is opened, for hygiene reasons.
2. NON-RETURNABLE ITEMS
2.1 Innerwear, cosmetics with broken seals, and customized products cannot be returned.
2.2 Items marked "Final Sale" on the product page are not eligible for return or exchange.''',
        metadata={
            "source": "return_refund_policy.txt",
            "sku": None
        }
    )

    evidence = [
        "1. RETURN ELIGIBILITY",
        "1.1 Apparel & Footwear: Returnable within 14 days of delivery, tags intact, unworn.",
        "1.2 Electronics: Returnable within 7 days of delivery, original packaging and accessories required.",
        "1.3 Home & Kitchen: Returnable within 10 days of delivery if unused and in original packaging.",
        "1.4 Beauty & Personal Care: Non-returnable once the seal is opened, for hygiene reasons.",
        "2. NON-RETURNABLE ITEMS",
        "2.1 Innerwear, cosmetics with broken seals, and customized products cannot be returned.",
        "2.2 Items marked 'Final Sale' on the product page are not eligible for return or exchange."
    ]

    sources = ["return_refund_policy.txt", "Section 1"]

    assert validate_evidence(evidence, [policy], sources) is True

if __name__ == "__main__":
    test_response_validator()
    test_invalid_evidence()
    test_invalid_source()
    test_cross_source_evidence_mismatch()



