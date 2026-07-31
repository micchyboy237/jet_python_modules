import os
from typing import TypedDict

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity
from jet.logger import logger
from jet.vectors.reranker.bm25 import get_bm25_similarities


class HybridSearchResult(TypedDict):
    """Result from hybrid search combining vector + BM25 reranking."""

    rank: int
    index: int
    score: float  # Final BM25 score
    vector_score: float  # Original cosine similarity score
    text: str


def hybrid_search(
    query: str,
    documents: list[str],
    top_n: int = 5,
    vector_top_k: int = 20,
    return_embeddings: bool = False,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
) -> list[HybridSearchResult] | tuple[list[HybridSearchResult], np.ndarray, np.ndarray]:
    """
    Hybrid search: vector search (cosine similarity) → BM25 reranking.

    Flow:
    1. Embed query and all documents
    2. Rank by cosine similarity, keep top vector_top_k candidates
    3. Rerank those candidates using BM25
    4. Return top_n results with both vector and BM25 scores

    Args:
        query: The search query string.
        documents: List of document strings to search over.
        top_n: Number of final results to return (default: 5).
        vector_top_k: Number of candidates from vector search to pass to BM25 (default: 20).
        return_embeddings: If True, also returns (query_emb, doc_embs).
        query_prefix: Optional prefix for query embedding.
        document_prefix: Optional prefix for document embedding.

    Returns:
        List of HybridSearchResult dicts (rank, index, score, vector_score, text),
        or if return_embeddings=True, a tuple of (results, query_emb, doc_embs).
    """
    if not documents:
        logger.warning("No documents provided for hybrid search")
        return [] if not return_embeddings else ([], np.array([]), np.array([]))

    logger.info(
        f"Hybrid search: query='{query[:80]}...', docs={len(documents)}, "
        f"top_n={top_n}, vector_top_k={vector_top_k}"
    )

    # Resolve prefixes
    resolved_query_prefix = (
        query_prefix
        if query_prefix is not None
        else os.getenv("EMBED_QUERY_PREFIX") or None
    )
    resolved_doc_prefix = (
        document_prefix
        if document_prefix is not None
        else os.getenv("EMBED_DOC_PREFIX") or None
    )

    # Step 1: Embed query and documents
    logger.debug("Step 1: Computing embeddings...")
    query_embedding = embed(query, prefix=resolved_query_prefix)
    doc_embeddings = embed(documents, prefix=resolved_doc_prefix)

    # Step 2: Vector search - compute cosine similarities
    logger.debug("Step 2: Vector search (cosine similarity)...")
    similarities = [
        cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings
    ]

    # Sort by similarity score, keep top vector_top_k
    scored = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
    top_candidates = scored[: min(vector_top_k, len(scored))]

    logger.debug(
        f"Vector search returned {len(top_candidates)} candidates "
        f"(top score: {top_candidates[0][1]:.4f})"
    )

    # Step 3: BM25 reranking on candidates
    logger.debug("Step 3: BM25 reranking candidates...")
    candidate_indices = [idx for idx, _ in top_candidates]
    candidate_docs = [documents[idx] for idx in candidate_indices]

    # BM25 expects query as a list of query terms (extracted internally)
    # We pass the original query; extract_query_candidates handles tokenization
    from jet.vectors.reranker.bm25 import extract_query_candidates

    query_candidates = extract_query_candidates(query)

    bm25_results = get_bm25_similarities(
        query_candidates,
        candidate_docs,
        ids=[str(idx) for idx in candidate_indices],
    )

    # Step 4: Build final results with both scores
    # Map vector scores by original document index
    vector_score_map = {idx: float(score) for idx, score in top_candidates}

    results: list[HybridSearchResult] = []
    for rank, bm25_result in enumerate(bm25_results[:top_n], start=1):
        original_idx = int(bm25_result["id"])
        results.append(
            {
                "rank": rank,
                "index": original_idx,
                "score": bm25_result["score"],
                "vector_score": vector_score_map.get(original_idx, 0.0),
                "text": bm25_result["text"],
            }
        )

    logger.info(f"Hybrid search complete: {len(results)} results returned")

    if return_embeddings:
        return results, query_embedding, doc_embeddings
    return results


if __name__ == "__main__":
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
        "The panda's diet is 99% bamboo.",
        "Giant pandas are native to south central China.",
        "Data science combines statistics and programming.",
        "Ursidae includes pandas, polar bears, and brown bears.",
        "Deep learning revolutionized computer vision tasks.",
    ]

    print("Hybrid Search Results:")
    print("=" * 80)
    results, query_emb, doc_embs = hybrid_search(
        query, docs, return_embeddings=True, top_n=5, vector_top_k=8
    )

    print(f"Query embedding shape: {query_emb.shape}")
    print(f"Document embeddings shape: {doc_embs.shape}")
    print(f"\nQuery: {query}\n")
    print(f"{'Rank':<6} {'Idx':<5} {'BM25':<8} {'Vector':<8} Text")
    print("-" * 80)
    for r in results:
        print(
            f"#{r['rank']:<5} {r['index']:<5} {r['score']:<8.4f} "
            f"{r['vector_score']:<8.4f} {r['text']}"
        )
