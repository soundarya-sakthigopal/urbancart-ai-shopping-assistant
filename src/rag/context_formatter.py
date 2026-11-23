
from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    """
    Format retrieved documents with source metadata
    for RAG context injection.
    """

    if not docs:
        return "No relevant documents found."

    formatted_docs = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata

        sku = metadata.get("sku", "Not applicable")
        source = metadata.get("source", "Unknown source")
        policy_type = metadata.get("policy_type", "Not applicable")
        category = metadata.get("category", "Not applicable")

        formatted_doc = f"""
Document {index}
Source: {source}
SKU: {sku}
Policy Type: {policy_type}
Category: {category}

Content:
{doc.page_content}
"""

        formatted_docs.append(formatted_doc.strip())

    return "\n\n---\n\n".join(formatted_docs)
