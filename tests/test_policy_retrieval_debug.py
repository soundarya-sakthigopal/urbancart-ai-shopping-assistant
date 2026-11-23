
from src.rag.rag_initializer import initialize_rag_components
from src.retrieval.hybrid_retrieval_service import hybrid_retrieve


def test_hybrid_retrieval():

    initialize_rag_components()

    query = "What is the return window for a smartwatch?"

    documents = hybrid_retrieve(
        query=query,
        top_k=5
    )

    print("\nHybrid Retrieved Documents:", len(documents))

    for index, doc in enumerate(documents, start=1):

        print(f"\nDocument {index}")
        print("Source:", doc.metadata.get("source"))
        print("SKU:", doc.metadata.get("sku"))
        print("Content:", doc.page_content)


if __name__ == "__main__":
    test_hybrid_retrieval()
