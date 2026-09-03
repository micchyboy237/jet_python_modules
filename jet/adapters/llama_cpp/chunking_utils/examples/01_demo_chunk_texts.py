"""
Demo: Basic Text Chunking

Shows how to split text into token-sized chunks using chunk_texts.
"""

import logging

from jet.adapters.llama_cpp.chunking_utils import chunk_texts

# Configure logging to see internal chunking decisions
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    sample_text = (
        "Artificial intelligence is transforming industries worldwide. "
        "From healthcare to finance, AI models are optimizing processes. "
        "However, ethical considerations remain a significant challenge. "
        "Researchers must balance innovation with safety protocols. "
        "The future depends on responsible development practices."
    )

    logger.info("Starting basic chunking demo...")

    # Chunk with strict sentence boundaries and small size for demonstration
    chunks = chunk_texts(
        texts=sample_text,
        chunk_size=64,  # Max tokens per chunk
        chunk_overlap=8,  # Overlap tokens between chunks
        strict_sentences=True,
        min_chunk_size=10,  # Merge tiny trailing chunks
        show_progress=True,
    )

    logger.info(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)


if __name__ == "__main__":
    main()
