"""
Simple Demo: LlamacppEmbedding
Keep it minimal like demo_chat_stream.
Pre-requisites:
  - Running llama-server with embeddings
  - LLAMA_CPP_EMBED_URL and LLAMA_CPP_EMBED_MODEL env vars set.
"""

import os
import sys

import numpy as np
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.logger import CustomLogger

logger = CustomLogger()


def demo_single_embed():
    """Simple single text embedding."""
    print("=== Demo: Single Text Embedding ===")
    embedder = LlamacppEmbedding(verbose=True, logger=logger)
    text = "The quick brown fox jumps over the lazy dog."
    emb = embedder.embed(text)
    print(f"Input: {text}")
    print(f"Embedding shape: {emb.shape}")
    print(f"First 5: {emb[:5]}")
    print(f"Norm: {np.linalg.norm(emb):.4f}")
    embedder.close()


def demo_batch_embed():
    """Simple batch embedding."""
    print("=== Demo: Batch Embedding ===")
    embedder = LlamacppEmbedding(verbose=True, logger=logger)
    texts = [
        "Machine learning is fun.",
        "Embeddings capture semantics.",
    ]
    embs = embedder.embed(texts, batch_size=2)
    print(f"Batch shape: {embs.shape}")
    embedder.close()


def demo_cosine_sim():
    """Simple cosine similarity."""
    print("=== Demo: Cosine Similarity ===")
    embedder = LlamacppEmbedding(verbose=True, logger=logger)
    text1 = "Hello world"
    text2 = "Hi universe"
    emb1 = embedder.embed(text1)
    emb2 = embedder.embed(text2)
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"Similarity between '{text1}' and '{text2}': {sim:.4f}")
    embedder.close()


if __name__ == "__main__":
    if not os.getenv("LLAMA_CPP_EMBED_URL") or not os.getenv("LLAMA_CPP_EMBED_MODEL"):
        print("Set LLAMA_CPP_EMBED_URL and LLAMA_CPP_EMBED_MODEL")
        sys.exit(1)
    demo_single_embed()
    demo_batch_embed()
    demo_cosine_sim()
    logger.success("Demos completed!")
