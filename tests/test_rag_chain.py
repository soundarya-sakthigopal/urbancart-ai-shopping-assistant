
from src.rag.rag_initializer import initialize_rag_components
from src.rag.rag_chain import get_rag_chain


def test_rag_chain():

    initialize_rag_components()

    rag_chain = get_rag_chain()

    #query = "Mera order abhi tak deliver nahi hua. Normally delivery mein kitne din lagte hain?"
    query = "What is the return window for a smartwatch?"
    result = rag_chain.invoke(query)
    print("\nRetrieved Document Sources:")

    for doc in result["documents"]:
        print(doc.metadata.get("source"))
        print(doc.page_content)

    print("\nRaw LLM Response:")
    print(result["response"])

    print("\nValidated Final Response:")
    print(result["final_response"])
    print("\nCustomer Question:")
    print(result["question"])

    print("\nRetrieved Documents:")
    print(len(result["documents"]))
    print("\nRetrieved Document Details:")

    for doc in result["documents"]:
        print("\nSource:", doc.metadata.get("source"))
        print("Policy Type:", doc.metadata.get("policy_type"))
        print("Content:", doc.page_content)
        print("\nRaw LLM Response:")
        print(result["response"])

    print("\nValidated Final Response:")
    print(result["final_response"])


if __name__ == "__main__":
    test_rag_chain()
