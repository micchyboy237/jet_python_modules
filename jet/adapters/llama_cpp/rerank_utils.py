import os
from typing import TypedDict

import requests
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity

RERANK_BASE_URL = os.getenv("LLAMA_CPP_RERANK_URL", "http://localhost:8082/v1")
RERANK_URL = RERANK_BASE_URL + "/rerank"
MODEL = os.getenv("LLAMA_CPP_RERANK_MODEL")


class RerankResult(TypedDict):
    """Typed structure for a single rerank result."""

    rank: int
    index: int
    score: float
    text: str


def rerank(
    query: str, documents: list[str], top_n: int | None = None
) -> list[RerankResult]:
    """
    Call the local llama.cpp reranker and return documents sorted by relevance.

    Returns:
        List of RerankResult dicts with keys: rank, index, score, text
    """
    payload = {"model": MODEL, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n

    resp = requests.post(RERANK_URL, json=payload, timeout=30)
    resp.raise_for_status()
    results = resp.json()["results"]

    ranked: list[RerankResult] = []
    for rank_position, r in enumerate(results, start=1):
        ranked.append(
            {
                "rank": rank_position,
                "index": r["index"],
                "score": r["relevance_score"],
                "text": documents[r["index"]],
            }
        )

    # Already sorted by relevance from the API, but ensure consistent ordering
    ranked.sort(key=lambda x: x["score"], reverse=True)

    # Re-assign ranks after sorting to ensure they match final order
    for i, item in enumerate(ranked, start=1):
        item["rank"] = i

    return ranked


if __name__ == "__main__":
    from jet.adapters.llama_cpp.embed_utils import embed

    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    print("=" * 60)
    print("EMBEDDING-BASED SIMILARITY")
    print("=" * 60)
    query_embedding = embed(query)
    doc_embeddings = embed(docs)
    similarities = [
        cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings
    ]
    ranked_embed = sorted(zip(similarities, docs), reverse=True)
    print(f"\nQuery: {query}\n")
    for rank, (score, doc) in enumerate(ranked_embed, start=1):
        print(f"#{rank}  {score:.4f}  {doc}")

    print("\n" + "=" * 60)
    print("RERANKER-BASED SIMILARITY")
    print("=" * 60)
    print(f"\nQuery: {query}\n")
    for r in rerank(query, docs, top_n=5):
        print(f"#{r['rank']}  {r['score']:.4f}  {r['text']}")
