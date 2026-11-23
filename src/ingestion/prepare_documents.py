from langchain_core.documents import Document
from src.ingestion.policy_document_builder import (load_policy_documents,split_policy_documents )
from src.ingestion.product_document_builder import (load_processed_catalogue,build_product_documents)
def prepare_product_documents():

    df = load_processed_catalogue()

    product_docs = build_product_documents(df)

    documents = []

    for doc in product_docs:

        document = Document(
            page_content=doc["page_content"],
            metadata=doc["metadata"]
        )

        documents.append(document)

    print("LangChain product documents:", len(documents))

    return documents
def prepare_policy_documents():

    policy_docs = load_policy_documents()

    policy_chunks = split_policy_documents(policy_docs)

    print("LangChain policy documents:", len(policy_chunks))

    return policy_chunks
def prepare_all_documents():

    product_docs = prepare_product_documents()

    policy_docs = prepare_policy_documents()

    all_documents = product_docs + policy_docs

    print("\nTotal documents prepared:", len(all_documents))
    print("Product documents:", len(product_docs))
    print("Policy chunks:", len(policy_docs))

    return all_documents
