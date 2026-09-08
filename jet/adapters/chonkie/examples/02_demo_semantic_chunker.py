"""Simple demo: SemanticChunker with LlamacppEmbeddings.

Prerequisites:
  - Windows llama.cpp server running with an embedding model loaded
  - LLAMA_CPP_EMBED_URL env var set (or default localhost:8081)
  - pip install chonkie numpy rich
"""

from chonkie import SemanticChunker
from jet.adapters.chonkie.llamacpp_embeddings import LlamacppEmbeddings
from jet.logger import logger


def main() -> None:
    # 1. Initialize embeddings (auto-detects dims + tokenizer)
    logger.info("Initializing LlamacppEmbeddings...")
    embeddings = LlamacppEmbeddings(
        batch_size=64,  # Amortize network RTT over WiFi/LAN
        max_workers=6,  # Safe concurrency ceiling for GTX 1660
        show_progress=True,  # Rich progress bar during embedding
    )
    logger.info(f"Embeddings ready: {embeddings}")

    # 2. Create SemanticChunker using window strategy (faster for remote)
    chunker = SemanticChunker(
        embedding_model=embeddings,
        threshold=0.5,  # Cosine similarity cutoff (0-1)
        chunk_size=512,  # Max tokens per chunk
        similarity_window=3,  # Compare against 3 preceding sentences
        min_sentences_per_chunk=1,
        min_characters_per_sentence=12,
    )
    logger.info(f"Chunker ready: {chunker}")

    # 3. Sample text with clear topic transitions
    text = (
        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. These systems improve their performance "
        "over time without being explicitly programmed. Deep learning is a specialized "
        "form of machine learning that uses neural networks with many layers. "
        "Natural language processing enables computers to understand human language. "
        "It powers applications like chatbots, translation services, and sentiment analysis. "
        "Computer vision allows machines to interpret visual information from images and videos. "
        "This technology is used in autonomous vehicles, medical imaging, and facial recognition. "
        "Reinforcement learning trains agents through trial and error using reward signals. "
        "It has been successfully applied to game playing, robotics, and resource management."
    )

    # 4. Chunk the text
    logger.info(f"Chunking text ({len(text)} chars)...")
    chunks = chunker(text)

    # 5. Display results
    logger.info(f"Produced {len(chunks)} semantic chunks:")
    print("\n" + "=" * 70)
    for i, chunk in enumerate(chunks, 1):
        print(
            f"\n--- Chunk {i} | Tokens: {chunk.token_count} | "
            f"Span: [{chunk.start_index}:{chunk.end_index}] ---"
        )
        print(chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""))
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
