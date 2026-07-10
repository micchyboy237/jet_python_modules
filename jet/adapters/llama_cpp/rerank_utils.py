import os

import numpy as np
import requests

RERANK_BASE_URL = os.getenv("LLAMA_CPP_RERANK_URL", "http://localhost:8082/v1")
RERANK_URL = RERANK_BASE_URL + "/rerank"
MODEL = os.getenv("LLAMA_CPP_RERANK_MODEL")


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    """
    Call the local llama.cpp reranker and return documents sorted by relevance.
    Each returned dict: {"document": str, "score": float, "index": int}
    """
    payload = {"model": MODEL, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n
    resp = requests.post(RERANK_URL, json=payload, timeout=30)
    resp.raise_for_status()
    results = resp.json()["results"]
    ranked = [
        {
            "document": documents[r["index"]],
            "score": r["relevance_score"],
            "index": r["index"],
        }
        for r in results
    ]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


if __name__ == "__main__":
    from jet.adapters.llama_cpp.embed_utils import embed

    # Same query and docs from embed_utils demo
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    # Embedding-based similarity
    print("=" * 60)
    print("EMBEDDING-BASED SIMILARITY")
    print("=" * 60)
    query_embedding = embed(query)
    doc_embeddings = embed(docs)

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [
        cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings
    ]
    ranked_embed = sorted(zip(similarities, docs), reverse=True)

    print(f"\nQuery: {query}\n")
    for score, doc in ranked_embed:
        print(f"{score:.4f}  {doc}")

    # Reranker-based similarity
    print("\n" + "=" * 60)
    print("RERANKER-BASED SIMILARITY")
    print("=" * 60)
    print(f"\nQuery: {query}\n")
    for r in rerank(query, docs, top_n=5):
        print(f"{r['score']:.4f}  {r['document']}")
