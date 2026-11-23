import logging

from src.ingestion.vectorstore_ingestion import (
    run_vectorstore_ingestion
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

if __name__ == "__main__":
    run_vectorstore_ingestion(dry_run=False)