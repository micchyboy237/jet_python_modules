"""
Simple Demo: Vector Search with LlamacppEmbedding
Minimal like demo_chat_stream. Few demos only.
"""

import numpy as np
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.logger import CustomLogger

logger = CustomLogger()

DOCS = [
    "Neural networks are inspired by the brain.",
    "Python is great for data science.",
    "Transformers power modern LLMs.",
]

QUERIES = ["What are neural networks?", "Best language for AI?"]


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def demo_simple_search():
    """Simple in-memory vector search."""
    print("=== Demo: Simple Vector Search ===")
    embedder = LlamacppEmbedding(verbose=True, logger=logger)

    # Embed docs
    doc_embs = embedder.embed(DOCS)

    for q in QUERIES:
        q_emb = embedder.embed(q)
        scores = [cosine_sim(q_emb, d_emb) for d_emb in doc_embs]
        best_idx = np.argmax(scores)
        print(f"Query: {q}")
        print(f"Best match: {DOCS[best_idx]} (score: {scores[best_idx]:.4f})")
        print("---")

    embedder.close()


if __name__ == "__main__":
    demo_simple_search()
    logger.success("Vector search demo completed!")
