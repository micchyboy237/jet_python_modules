import os
from typing import Literal, TypedDict

import requests
from jet.adapters.llama_cpp.scoring_utils import sigmoid_normalize_scores
from jet.logger import logger

RERANK_BASE_URL = os.getenv("LLAMA_CPP_RERANK_URL", "http://localhost:8082/v1")
RERANK_URL = RERANK_BASE_URL + "/rerank"
MODEL = os.getenv("LLAMA_CPP_RERANK_MODEL")

# New import for BM25 fallback
from jet.vectors.reranker.bm25 import rerank_bm25 as _bm25_rerank


class RerankResult(TypedDict):
    """Typed structure for a single rerank result."""

    rank: int
    index: int
    score: float
    raw_score: float
    text: str


def _check_rerank_server(timeout: float = 2.0) -> bool:
    """Quick health check for the rerank server."""
    try:
        resp = requests.get(RERANK_BASE_URL, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _rerank_via_model(
    query: str,
    documents: list[str],
    top_n: int | None,
    normalize_scores: bool,
    sigmoid_temperature: float,
) -> list[RerankResult]:
    """Existing model-based reranking logic extracted for clarity."""
    payload = {"model": MODEL, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n

    resp = requests.post(RERANK_URL, json=payload, timeout=30)
    resp.raise_for_status()
    results = resp.json()["results"]

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
    raw_scores = [item["raw_score"] for item in ranked_raw]

    if normalize_scores and raw_scores:
        normalized = sigmoid_normalize_scores(raw_scores, sigmoid_temperature)
    else:
        normalized = raw_scores

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


def _rerank_via_bm25(
    query: str,
    documents: list[str],
    top_n: int | None,
    normalize_scores: bool,
    sigmoid_temperature: float,
) -> list[RerankResult]:
    """BM25-based reranking using existing jet.vectors.reranker.bm25."""
    logger.info("Using BM25 reranking (lexical)")
    _, bm25_results = _bm25_rerank(query=query, documents=documents)

    # Map BM25 SimilarityResult → RerankResult
    # Note: BM25 already returns sorted results with normalized 0-1 scores
    ranked: list[RerankResult] = []
    raw_scores = [r["score"] for r in bm25_results]

    # Apply optional sigmoid normalization on top of BM25's max-normalization
    # This makes BM25 scores comparable to model scores in hybrid pipelines
    if normalize_scores and raw_scores:
        normalized = sigmoid_normalize_scores(raw_scores, sigmoid_temperature)
    else:
        normalized = raw_scores

    for rank_pos, (bm25_res, norm_score) in enumerate(
        zip(bm25_results, normalized), start=1
    ):
        # Find original index by matching text (BM25 doesn't preserve original index)
        original_idx = -1
        for i, doc in enumerate(documents):
            if doc == bm25_res["text"]:
                original_idx = i
                break

        ranked.append(
            {
                "rank": rank_pos,
                "index": original_idx,
                "score": norm_score,
                "raw_score": bm25_res["score"],
                "text": bm25_res["text"],
            }
        )

    if top_n is not None:
        ranked = ranked[:top_n]

    return ranked


def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    normalize_scores: bool = True,
    sigmoid_temperature: float = 1.0,
    method: Literal["auto", "model", "bm25"] = "model",
) -> list[RerankResult]:
    """
    Rerank documents using either a cross-encoder model or BM25.

    Args:
        query: Search query string.
        documents: List of document strings to rerank.
        top_n: Number of results to return (None = all).
        normalize_scores: If True (default), apply sigmoid normalization.
        sigmoid_temperature: Controls sigmoid steepness (default 1.0).
        method: Reranking strategy.
            - "model": Force cross-encoder model via llama.cpp server (default).
            - "auto": Try model first, fall back to BM25 on failure.
            - "bm25": Force lexical BM25 reranking (no GPU/server needed).

    Returns:
        List of RerankResult dicts with keys: rank, index, score, raw_score, text
    """
    if not documents:
        return []

    if method == "bm25":
        return _rerank_via_bm25(
            query, documents, top_n, normalize_scores, sigmoid_temperature
        )

    if method == "model":
        return _rerank_via_model(
            query, documents, top_n, normalize_scores, sigmoid_temperature
        )

    # method == "auto"
    if _check_rerank_server():
        try:
            logger.info(f"Rerank server available, using model '{MODEL}'")
            return _rerank_via_model(
                query, documents, top_n, normalize_scores, sigmoid_temperature
            )
        except Exception as e:
            logger.warning(f"Model reranking failed ({e}), falling back to BM25")
            return _rerank_via_bm25(
                query, documents, top_n, normalize_scores, sigmoid_temperature
            )
    else:
        logger.warning("Rerank server unavailable, falling back to BM25")
        return _rerank_via_bm25(
            query, documents, top_n, normalize_scores, sigmoid_temperature
        )


if __name__ == "__main__":
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    # Demonstrate each reranking method explicitly
    methods_to_demo: list[Literal["model", "auto", "bm25"]] = ["model", "auto", "bm25"]

    for method in methods_to_demo:
        print("\n" + "=" * 60)
        print(f"RERANKING METHOD: {method.upper()}")
        print("=" * 60)
        print(f"\nQuery: {query}\n")

        try:
            results = rerank(query, docs, top_n=5, method=method)
            if not results:
                print("  (No results returned)")
            else:
                for r in results:
                    print(
                        f"#{r['rank']}  score={r['score']:.4f}  "
                        f"raw={r['raw_score']:.4f}  idx={r['index']}  "
                        f"{r['text']}"
                    )
        except Exception as e:
            print(f"  ERROR: {e}")
            logger.exception(f"Demo failed for method='{method}'")
