
import re


FALLBACK_ANSWER = "I couldn't find this in our catalog/policy"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace('"', "").replace("'", "")
    return re.sub(r"\s+", " ", text).strip()


def validate_response_structure(response: dict) -> bool:
    """Validate the expected JSON response structure."""

    if not isinstance(response, dict):
        return False

    required_keys = {"answer", "source", "evidence"}

    if set(response.keys()) != required_keys:
        return False

    return (
        isinstance(response["answer"], str)
        and isinstance(response["source"], list)
        and isinstance(response["evidence"], list)
        and all(isinstance(item, str) for item in response["source"])
        and all(isinstance(item, str) for item in response["evidence"])
    )


def sections_exist(section_text: str, documents: list) -> bool:
    """Check whether referenced section numbers exist in retrieved documents."""

    section_match = re.fullmatch(
        r"\d+(?:\.\d+)*(?:\s*,\s*\d+(?:\.\d+)*)*",
        section_text.strip()
    )

    if not section_match:
        return False

    section_numbers = [
        number.strip()
        for number in section_text.split(",")
    ]

    available_content = [
        normalize_text(document.page_content)
        for document in documents
    ]

    for section_number in section_numbers:
        pattern = (
    rf"(?<![\d.]){re.escape(section_number)}"
    rf"(?:\.(?!\d))?"
    rf"(?![\d.])"
)

        if not any(
            re.search(pattern, content)
            for content in available_content
        ):
            return False

    return True


def validate_sources(
    response_sources: list[str],
    retrieved_documents: list
) -> bool:
    """Validate filenames, SKUs and policy section references."""

    if not response_sources:
        return False

    available_sources = set()
    available_skus = set()

    for document in retrieved_documents:
        metadata = document.metadata

        source = metadata.get("source")
        sku = metadata.get("sku")

        if source:
            available_sources.add(source)

        if sku:
            available_skus.add(sku)

    for cited_source in response_sources:
        cited_source = cited_source.strip()

        if cited_source in available_sources:
            continue

        if cited_source in available_skus:
            continue

        combined_match = re.fullmatch(
            r"(.+?)\s*[-,]\s*Section\s+(.+)",
            cited_source,
            re.IGNORECASE
        )

        if combined_match:
            filename = combined_match.group(1).strip()
            section_text = combined_match.group(2).strip()

            matching_documents = [
                document
                for document in retrieved_documents
                if document.metadata.get("source") == filename
            ]

            if (
                matching_documents
                and sections_exist(section_text, matching_documents)
            ):
                continue

        standalone_match = re.fullmatch(
            r"Section\s+(.+)",
            cited_source,
            re.IGNORECASE
        )

        if standalone_match:
            section_text = standalone_match.group(1).strip()

            if sections_exist(section_text, retrieved_documents):
                continue

        return False

    return True


def validate_evidence(
    response_evidence: list[str],
    retrieved_documents: list,
    response_sources: list[str]
) -> bool:
    """Verify that cited evidence exists in the relevant documents."""

    if not response_evidence or not response_sources:
        return False

    cited_documents = []

    for document in retrieved_documents:
        metadata = document.metadata
        source = metadata.get("source")
        sku = metadata.get("sku")

        for citation in response_sources:
            citation = citation.strip()

            # Match source filenames and their combined section citations
            if source and (
                citation == source
                or citation.startswith(source + " - Section")
                or citation.startswith(source + ", Section")
            ):
                cited_documents.append(document)
                break

            # Match product SKU citations
            if sku and sku == citation:
                cited_documents.append(document)
                break

    if not cited_documents:
        return False
  
    document_contents = [
        normalize_text(document.page_content)
        for document in cited_documents
    ]
    
    for evidence in response_evidence:
        normalized_evidence = normalize_text(evidence)

        if not normalized_evidence:
            return False

        if not any(
            normalized_evidence in content
            for content in document_contents
        ):
            return False
    print("CITED DOCUMENT COUNT:", len(cited_documents))

    for index, document in enumerate(cited_documents, start=1):
        print(f"\nCITED DOCUMENT {index}:")
        print(document.page_content)
    
    return True


def calculate_confidence(
    response: dict,
    retrieved_documents: list
) -> str:
    """Calculate confidence using structural, source and evidence checks."""

    if not validate_response_structure(response):
        return "Low"

    if response["answer"].strip() == FALLBACK_ANSWER:
        return "Low"

    if not retrieved_documents:
        return "Low"

    source_valid = validate_sources(
        response["source"],
        retrieved_documents
    )

    evidence_valid = validate_evidence(
        response["evidence"],
        retrieved_documents,
        response["source"]
    )

    if source_valid and evidence_valid:
        return "High"

    return "Low"


def validate_and_format_response(
    response: dict,
    retrieved_documents: list
) -> dict:
    """Validate the LLM response and return the final response."""

    if not validate_response_structure(response):
        return {
            "answer": FALLBACK_ANSWER,
            "source": [],
            "confidence_level": "Low"
        }

    if response["answer"].strip() == FALLBACK_ANSWER:
        return {
            "answer": FALLBACK_ANSWER,
            "source": [],
            "confidence_level": "Low"
        }

    confidence = calculate_confidence(
        response,
        retrieved_documents
    )

    if confidence == "Low":
        return {
            "answer": FALLBACK_ANSWER,
            "source": [],
            "confidence_level": "Low"
        }

    return {
        "answer": response["answer"],
        "source": response["source"],
        "confidence_level": confidence
    }
