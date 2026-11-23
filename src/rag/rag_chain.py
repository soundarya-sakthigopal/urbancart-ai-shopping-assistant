
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough
)
from langchain_core.output_parsers import JsonOutputParser

from src.retrieval.hybrid_retrieval_service import hybrid_retrieve
from src.rag.context_formatter import format_docs
from src.rag.prompt_template import RAG_PROMPT_TEMPLATE
from src.rag.llm_service import get_chat_model
from src.rag.response_validator import validate_and_format_response
from src.rag.response_validator import (
    validate_and_format_response,
    validate_sources,
    validate_evidence
)

def get_rag_chain():
    """
    Build the RAG chain using hybrid retrieval and LCEL.
    Preserve retrieved documents for evidence validation.
    """

    llm = get_chat_model()

    rag_chain = (
        # Convert the customer question into a dictionary
        RunnableLambda(
            lambda question: {"question": question}
        )

        # Retrieve and preserve original documents
               # Retrieve and preserve original documents
        | RunnablePassthrough.assign(
            documents=RunnableLambda(
                lambda inputs: (
                    lambda docs: (
                        print("RAG RETRIEVED DOCUMENTS:", len(docs)),
                        print(
                            "RAG RETRIEVED SOURCES:",
                            [
                                doc.metadata.get("source")
                                for doc in docs
                            ]
                        ),
                        docs
                    )[2]
                )(
                    hybrid_retrieve(
                        query=inputs["question"],
                        top_k=5
                    )
                )
            )
        )

        # Format retrieved documents for the LLM
        | RunnablePassthrough.assign(
            context=RunnableLambda(
                lambda inputs: format_docs(inputs["documents"])
            )
        )

        # Generate and parse the LLM response
        | RunnablePassthrough.assign(
            response=(
                RunnableLambda(
                    lambda inputs: {
                        "question": inputs["question"],
                        "context": inputs["context"]
                    }
                )
                | RAG_PROMPT_TEMPLATE
                | llm
                | JsonOutputParser()
            )
        )

        # Validate the response using original documents
        | RunnablePassthrough.assign(
           final_response=RunnableLambda(
                lambda inputs: (
                print("LLM RAW RESPONSE:", inputs["response"]),
                print(
                    "SOURCE VALID:",
                    validate_sources(
                        inputs["response"]["source"],
                        inputs["documents"]
                    )
                ),
                print(
                    "EVIDENCE VALID:",
                    validate_evidence(
                        inputs["response"]["evidence"],
                        inputs["documents"],
                        inputs["response"]["source"]
                    )
                ),
        validate_and_format_response(
            response=inputs["response"],
            retrieved_documents=inputs["documents"]
        )
    )[-1]
)
        )
    )

    return rag_chain
