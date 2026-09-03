"""
Demo: Chunking with Metadata

Shows how to generate ChunkResult objects for RAG indexing.
"""

import json
import logging

from jet.adapters.llama_cpp.chunking_utils import chunk_texts_with_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    documents = [
        "Python is a versatile programming language used in web development.",
        "Machine learning requires large datasets and computational power. "
        "GPUs accelerate training times significantly compared to CPUs.",
    ]

    doc_ids = ["doc-web-001", "doc-ml-002"]

    logger.info("Starting metadata-rich chunking demo...")

    results = chunk_texts_with_data(
        texts=documents,
        ids=doc_ids,
        chunk_size=48,
        chunk_overlap=4,
        strict_sentences=True,
        show_progress=True,
    )

    logger.info(f"Generated {len(results)} chunk results:")
    for res in results:
        # Pretty print the TypedDict result
        print(json.dumps(res, indent=2))
        print("-" * 40)


if __name__ == "__main__":
    main()
