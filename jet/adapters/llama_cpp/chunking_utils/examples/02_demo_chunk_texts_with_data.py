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
    document = (
        "Python is a versatile programming language used in web development. "
        "Machine learning requires large datasets and computational power. "
        "GPUs accelerate training times significantly compared to CPUs. "
        "Natural language processing is a key application area for deep learning. "
        "Python libraries like PyTorch and TensorFlow enable rapid prototyping. "
        "Modern RAG pipelines split documents into chunks for retrievable context. "
        "You can efficiently index large corpora for LLM-backed search tasks. "
        "Tokenization and chunking strategies affect retrieval recall and precision. "
        "Choosing optimal chunk size often requires some experimentation."
    )

    doc_ids = ["doc-web-001", "doc-ml-002"]

    logger.info("Starting metadata-rich chunking demo...")

    results = chunk_texts_with_data(
        texts=document,
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
