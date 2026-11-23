import hashlib
import logging

from langchain_core.documents import Document

from src.ingestion.prepare_documents import prepare_all_documents
from collections import defaultdict
from src.vectorstore.chroma_service import get_vector_store


logger = logging.getLogger(__name__)
import time

def validate_documents(documents):
    """
    Validate documents before vector indexing.

    Returns:
        valid_documents: Documents ready for indexing.
        rejected_documents: Documents that failed validation.
    """

    valid_documents = []
    rejected_documents = []

    for index, document in enumerate(documents):

        errors = []

        # 1. Validate document type
        if not isinstance(document, Document):
            errors.append("Invalid document type")

        else:
            # 2. Validate document content
            if not isinstance(document.page_content, str):
                errors.append("Page content must be a string")

            elif not document.page_content.strip():
                errors.append("Page content is empty")

            # 3. Validate metadata
            metadata = document.metadata

            if not isinstance(metadata, dict):
                errors.append("Metadata must be a dictionary")

            else:
                source = metadata.get("source")

                if not source:
                    errors.append("Missing source metadata")

                elif source == "product_catalogue":
                    if not metadata.get("sku"):
                        errors.append("Product SKU is missing")

                else:
                    if not metadata.get("policy_type"):
                        errors.append("Policy type is missing")

        # 4. Separate valid and rejected documents
        if errors:
            rejected_documents.append({
                "document_index": index,
                "errors": errors
            })

            logger.warning(
                "Document validation failed | index=%s | errors=%s",
                index,
                errors
            )

        else:
            valid_documents.append(document)

    logger.info(
        "Document validation completed | total=%s | valid=%s | rejected=%s",
        len(documents),
        len(valid_documents),
        len(rejected_documents)
    )

    return valid_documents, rejected_documents


def generate_document_id(document, chunk_index=None):
    metadata = document.metadata
    source = metadata.get("source")

    if source == "product_catalogue":
        sku = metadata.get("sku")
        return f"product:{sku}"

    policy_type = metadata.get("policy_type")

    if chunk_index is None:
        raise ValueError("Policy chunk index is required")

    return f"policy:{policy_type}:{chunk_index}"


def generate_content_hash(document):
    content = document.page_content

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()
def assign_document_ids(documents):

    policy_counters = defaultdict(int)
    document_ids = []

    for document in documents:

        source = document.metadata.get("source")

        if source == "product_catalogue":
            document_id = generate_document_id(document)

        else:
            policy_type = document.metadata.get("policy_type")

            chunk_index = policy_counters[policy_type]

            document_id = generate_document_id(
                document,
                chunk_index=chunk_index
            )

            policy_counters[policy_type] += 1

        document_ids.append(document_id)

    return document_ids
def deduplicate_documents(documents):

    product_groups = defaultdict(list)
    policy_documents = []

    for document in documents:

        if document.metadata.get("source") == "product_catalogue":

            sku = document.metadata.get("sku")
            product_groups[sku].append(document)

        else:
            policy_documents.append(document)

    final_documents = []
    rejected_documents = []

    exact_duplicate_groups = 0
    conflicting_duplicate_groups = 0

    for sku, group in product_groups.items():

        if len(group) == 1:
            final_documents.append(group[0])
            continue

        content_versions = {
            generate_content_hash(document)
            for document in group
        }

        if len(content_versions) == 1:

            final_documents.append(group[0])
            exact_duplicate_groups += 1

        else:

            conflicting_duplicate_groups += 1

            rejected_documents.append({
                "sku": sku,
                "reason": "Conflicting documents for duplicate SKU",
                "occurrences": len(group)
            })

    final_documents.extend(policy_documents)

    print("\n--- Deduplication Summary ---")
    print("Input documents:", len(documents))
    print("Documents after deduplication:", len(final_documents))
    print("Exact duplicate groups:", exact_duplicate_groups)
    print("Conflicting duplicate groups:", conflicting_duplicate_groups)
    print("Rejected SKU groups:", len(rejected_documents))

    return final_documents, rejected_documents

def prepare_documents_for_indexing(documents):

    document_ids = assign_document_ids(documents)

    prepared_documents = []

    for document, document_id in zip(documents, document_ids):

        content_hash = generate_content_hash(document)

        document.metadata["document_id"] = document_id
        document.metadata["content_hash"] = content_hash

        prepared_documents.append(document)

    return prepared_documents, document_ids
def identify_documents_for_indexing(
    documents,
    document_ids,
    vector_store
):

    existing_records = vector_store._collection.get(
        ids=document_ids,
        include=["metadatas"]
    )

    existing_hashes = {}

    existing_ids = existing_records.get("ids", [])
    existing_metadatas = existing_records.get("metadatas", [])

    for existing_id, metadata in zip(
        existing_ids,
        existing_metadatas
    ):
        existing_hashes[existing_id] = metadata.get("content_hash")

    new_documents = []
    changed_documents = []
    unchanged_documents = []

    for document, document_id in zip(documents, document_ids):

        current_hash = document.metadata["content_hash"]

        existing_hash = existing_hashes.get(document_id)

        if existing_hash is None:
            new_documents.append((document, document_id))

        elif existing_hash != current_hash:
            changed_documents.append((document, document_id))

        else:
            unchanged_documents.append(document_id)

    return new_documents, changed_documents, unchanged_documents
def index_documents(
    new_documents,
    changed_documents,
    vector_store,
    batch_size=100
):

    documents_to_index = new_documents + changed_documents

    total_indexed = 0

    for start in range(0, len(documents_to_index), batch_size):

        batch = documents_to_index[start:start + batch_size]

        batch_documents = [
            item[0] for item in batch
        ]

        batch_ids = [
            item[1] for item in batch
        ]

        indexed_count = index_batch_with_retry(
                vector_store,
                batch_documents,
                batch_ids
                )

        total_indexed += indexed_count
        logger.info(
            "Indexed batch | batch_size=%s | total_indexed=%s",
            len(batch_documents),
            total_indexed
        )

    return total_indexed
def index_batch_with_retry(
    vector_store,
    documents,
    document_ids,
    max_retries=3
):

    for attempt in range(1, max_retries + 1):

        try:
            vector_store.add_documents(
                documents=documents,
                ids=document_ids
            )

            return len(documents)

        except Exception:

            logger.exception(
                "Indexing batch failed | attempt=%s/%s",
                attempt,
                max_retries
            )

            if attempt == max_retries:
                raise

            wait_time = 2 ** attempt

            logger.info(
                "Retrying batch in %s seconds",
                wait_time
            )

            time.sleep(wait_time)
def identify_stale_documents(
    documents,
    vector_store
):
    """
    Identify previously indexed documents that
    are no longer present in the current source data.
    """

    current_document_ids = {
        document.metadata["document_id"]
        for document in documents
    }

    existing_records = vector_store._collection.get(
        include=["metadatas"]
    )

    stale_ids = []

    existing_ids = existing_records["ids"]
    existing_metadatas = existing_records["metadatas"]

    for existing_id, metadata in zip(
        existing_ids,
        existing_metadatas
    ):
        source = metadata.get("source")

        if (source == "product_catalogue"or (isinstance(source, str)and source.endswith(".txt"))):

            if existing_id not in current_document_ids:
                stale_ids.append(existing_id)

    logger.info(
        "Stale documents identified: %s",
        len(stale_ids)
    )

    return stale_ids    
def delete_stale_documents(
    stale_ids,
    vector_store,
    batch_size=100
):
    """
    Delete stale documents from Chroma in batches.
    """

    if not stale_ids:
        logger.info("No stale documents found.")
        return 0

    total_deleted = 0

    for start in range(0, len(stale_ids), batch_size):
        batch_ids = stale_ids[start:start + batch_size]

        vector_store.delete(ids=batch_ids)

        total_deleted += len(batch_ids)

        logger.info(
            "Deleted stale documents: %s/%s",
            total_deleted,
            len(stale_ids)
        )

    return total_deleted      
def run_vectorstore_ingestion(dry_run=True):

    logger.info("Starting vectorstore ingestion pipeline")

    # 1. Prepare documents
    documents = prepare_all_documents()

    # 2. Validate documents
    valid_documents, rejected_documents = validate_documents(
        documents
    )

    # 3. Deduplicate documents
    unique_documents, duplicate_rejections = deduplicate_documents(
        valid_documents
    )

    # 4. Generate IDs and content hashes
    prepared_documents, document_ids = prepare_documents_for_indexing(
        unique_documents
    )

    # 5. Connect to Chroma
    vector_store = get_vector_store()

    # 6. Identify new, changed and unchanged documents
    new_documents, changed_documents, unchanged_documents = (
        identify_documents_for_indexing(
            prepared_documents,
            document_ids,
            vector_store
        )
    )

    # 7. Identify stale documents
    stale_ids = identify_stale_documents(
        prepared_documents,
        vector_store
    )

    print("\n--- Ingestion Plan ---")
    print("New documents:", len(new_documents))
    print("Changed documents:", len(changed_documents))
    print("Unchanged documents:", len(unchanged_documents))
    print("Stale documents:", len(stale_ids))
    print("Rejected documents:", len(rejected_documents))
    print("Rejected duplicate groups:", len(duplicate_rejections))

    # 8. Dry run: show the plan without modifying Chroma
    if dry_run:
        logger.info("Dry run completed. No vectors were modified.")
        return

    try:
        indexed_count = index_documents(
        new_documents,
        changed_documents,
        vector_store
        )

    except Exception:
        logger.exception(
        "Vector indexing failed. Stale documents will not be deleted."
        )
        raise

    deleted_count = delete_stale_documents(
    stale_ids,
    vector_store
    )

    print("\n--- Ingestion Summary ---")
    print("Successfully indexed:", indexed_count)
    print("Successfully deleted:", deleted_count)

    logger.info("Vectorstore ingestion completed")  
    