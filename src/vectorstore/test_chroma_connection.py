from src.ingestion.prepare_documents import prepare_all_documents
from langchain_core.documents import Document
from src.ingestion.vectorstore_ingestion import (
    validate_documents,
    deduplicate_documents,
    prepare_documents_for_indexing,
    identify_documents_for_indexing,
    identify_stale_documents,
    delete_stale_documents
)
from src.ingestion.vectorstore_ingestion import (
    run_vectorstore_ingestion
)
from src.vectorstore.chroma_service import get_vector_store


def test_incremental_comparison():

    # 1. Prepare source documents
    documents = prepare_all_documents()

    # 2. Validate documents
    valid_documents, rejected_documents = validate_documents(documents)

    # 3. Remove exact duplicate products
    unique_documents, duplicate_rejections = (
        deduplicate_documents(valid_documents)
    )

    # 4. Generate IDs and content hashes
    prepared_documents, document_ids = (
        prepare_documents_for_indexing(unique_documents)
    )

    # 5. Connect to Chroma
    vector_store = get_vector_store()

    # 6. Identify documents requiring indexing
    new_docs, changed_docs, unchanged_docs = (
        identify_documents_for_indexing(
            prepared_documents,
            document_ids,
            vector_store
        )
    )
    stale_ids = identify_stale_documents(
    prepared_documents,
    vector_store
)
    deleted_count = delete_stale_documents(
    [],
    vector_store
)

    print("Deleted documents:", deleted_count)
    print("Stale documents:", len(stale_ids))
    print("\n--- Incremental Comparison ---")
    print("New documents:", len(new_docs))
    print("Changed documents:", len(changed_docs))
    print("Unchanged documents:", len(unchanged_docs))
    print("Rejected documents:", len(rejected_documents))
    print("Rejected duplicate groups:", len(duplicate_rejections))
    
def test_chroma_deletion():

    vector_store = get_vector_store()

    test_id = "test:temporary_deletion"

    test_document = Document(
        page_content="Temporary document for testing Chroma deletion.",
        metadata={
            "source": "test",
            "document_id": test_id
        }
    )

    # Insert temporary document
    vector_store.add_documents(
        documents=[test_document],
        ids=[test_id]
    )

    print("Temporary document inserted.")

    # Verify insertion
    result = vector_store._collection.get(
        ids=[test_id],
        include=["metadatas"]
    )

    print("Before deletion:", len(result["ids"]))

    # Delete using our production function
    deleted_count = delete_stale_documents(
        [test_id],
        vector_store
    )

    print("Deleted count:", deleted_count)

    # Verify deletion
    result = vector_store._collection.get(
        ids=[test_id],
        include=["metadatas"]
    )

    print("After deletion:", len(result["ids"]))
def test_stale_document_detection():

    vector_store = get_vector_store()

    # Prepare the current source documents
    documents = prepare_all_documents()
    valid_documents, _ = validate_documents(documents)
    unique_documents, _ = deduplicate_documents(valid_documents)
    prepared_documents, _ = prepare_documents_for_indexing(
        unique_documents
    )

    # Insert a vector representing an old policy file
    stale_id = "policy:deleted_policy:0"

    stale_document = Document(
        page_content="This is an outdated policy document.",
        metadata={
            "source": "deleted_policy.txt",
            "policy_type": "deleted_policy"
        }
    )

    vector_store.add_documents(
        documents=[stale_document],
        ids=[stale_id]
    )

    # Identify outdated vectors
    stale_ids = identify_stale_documents(
        prepared_documents,
        vector_store
    )

    print("Detected stale IDs:", stale_ids)

    # Clean up the temporary vector
    deleted_count = delete_stale_documents(
        stale_ids,
        vector_store
    )

    print("Deleted stale vectors:", deleted_count)

if __name__ == "__main__":
    run_vectorstore_ingestion(dry_run=True)