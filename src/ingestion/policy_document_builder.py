from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_DIR = PROJECT_ROOT / "data" / "policy_docs"
def load_policy_documents():

    documents = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):

        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "page_content": text,
            "metadata": {
                "source": file_path.name,
                "policy_type": file_path.stem
            }
        })

    print("Policy documents loaded:", len(documents))

    return documents
def split_policy_documents(documents):
    text_splittter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n","\n",".",""," "]

    )
    chunks=text_splittter.create_documents(
        texts=[doc["page_content"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents]
    )
    print("Policy chunks created:", len(chunks))
    return chunks