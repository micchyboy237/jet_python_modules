import os
from typing import TypedDict

import requests
from jet.adapters.llama_cpp.scoring_utils import sigmoid_normalize_scores

RERANK_BASE_URL = os.getenv("LLAMA_CPP_RERANK_URL", "http://localhost:8082/v1")
RERANK_URL = RERANK_BASE_URL + "/rerank"
MODEL = os.getenv("LLAMA_CPP_RERANK_MODEL")


class RerankResult(TypedDict):
    """Typed structure for a single rerank result."""

    rank: int
    index: int
    score: float  # Normalized 0-1 via sigmoid
    raw_score: float  # Original unbounded cross-encoder score
    text: str


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    normalize_scores: bool = True,
    sigmoid_temperature: float = 1.0,
) -> list[RerankResult]:
    """
    Call the local llama.cpp reranker and return documents sorted by relevance.

    Args:
        query: Search query string.
        documents: List of document strings to rerank.
        top_n: Number of results to return (None = all).
        normalize_scores: If True (default), apply sigmoid normalization.
        sigmoid_temperature: Controls sigmoid steepness (default 1.0).

    Returns:
        List of RerankResult dicts with keys: rank, index, score, raw_score, text
    """
    payload = {"model": MODEL, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n

    resp = requests.post(RERANK_URL, json=payload, timeout=30)
    resp.raise_for_status()
    results = resp.json()["results"]

    # Sort by raw relevance_score descending
    ranked_raw: list[dict] = []
    for r in results:
        ranked_raw.append(
            {
                "index": r["index"],
                "raw_score": r["relevance_score"],
                "text": documents[r["index"]],
            }
        )
    ranked_raw.sort(key=lambda x: x["raw_score"], reverse=True)

    # Normalize scores
    raw_scores = [item["raw_score"] for item in ranked_raw]
    if normalize_scores and raw_scores:
        normalized = sigmoid_normalize_scores(raw_scores, sigmoid_temperature)
    else:
        normalized = raw_scores

    # Build final results with rank
    ranked: list[RerankResult] = []
    for rank_position, (item, norm_score) in enumerate(
        zip(ranked_raw, normalized), start=1
    ):
        ranked.append(
            {
                "rank": rank_position,
                "index": item["index"],
                "score": norm_score,
                "raw_score": item["raw_score"],
                "text": item["text"],
            }
        )

    return ranked


if __name__ == "__main__":
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    print("\n" + "=" * 60)
    print("RERANKER-BASED SIMILARITY")
    print("=" * 60)
    print(f"\nQuery: {query}\n")
    for r in rerank(query, docs, top_n=5):
        print(f"#{r['rank']}  {r['score']:.4f}  {r['raw_score']:.4f}  {r['text']}")
