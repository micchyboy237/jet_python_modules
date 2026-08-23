"""
Hybrid Search Utilities

Combines fast vector search (cosine similarity) with precise cross-encoder
reranking for high-quality retrieval. The two-stage pipeline:
1. Vector search retrieves top_k candidates (k > final_n)
2. Reranker scores and reorders candidates, returning top_n

Environment Variables:
    EMBED_QUERY_PREFIX  - Prefix added to queries before embedding (optional)
    EMBED_DOC_PREFIX    - Prefix added to documents before embedding (optional)
"""

import os
from typing import Dict, List, TypedDict

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.rerank_utils import rerank
from jet.adapters.llama_cpp.scoring_utils import (
    cosine_similarity,
    sigmoid_normalize_scores,
)
from jet.logger import logger


class HybridSearchResult(TypedDict):
    """A single result from hybrid_search."""

    rank: int
    index: int
    score: float
    text: str
    vector_score: float
    rerank_score_raw: float


def hybrid_search(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    vector_score_threshold: float | None = None,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    normalize_scores: bool = True,
    sigmoid_temperature: float = 1.0,
    doc_embeddings: np.ndarray | list[list[float]] | None = None,
) -> list[HybridSearchResult]:
    """
    Two-stage hybrid search: vector retrieval → cross-encoder reranking.

    Stage 1 (Vector Search):
        Embeds query and documents, computes cosine similarity, and optionally
        filters candidates by a minimum vector score threshold. All passing
        documents proceed to reranking.

    Stage 2 (Reranking):
        Passes all vector-stage candidates through a cross-encoder reranker
        model for precise relevance scoring. Returns top_n final results
        (or all if top_n is None).

    Scores are normalized to 0–1 range using sigmoid by default for better
    interpretability.

    Args:
        query: Search query string.
        documents: List of document strings to search over.
        top_n: Number of final results to return (default: None = all results).
        vector_score_threshold: Optional minimum cosine similarity threshold.
            Documents below this score are excluded before
            reranking. Default: None (no filtering).
            Typical values: 0.3–0.5 for moderate filtering,
            0.5+ for strict filtering.
        query_prefix: Optional prefix for query embedding (e.g., "search_query: ").
            Falls back to EMBED_QUERY_PREFIX env var.
        document_prefix: Optional prefix for document embeddings (e.g., "search_document: ").
            Falls back to EMBED_DOC_PREFIX env var.
            Ignored when doc_embeddings is provided.
        normalize_scores: If True (default), apply sigmoid normalization to rerank scores.
        sigmoid_temperature: Controls sigmoid steepness (default: 1.0).
            Lower = more extreme, Higher = more moderate.
        doc_embeddings: Optional pre-computed document embeddings. When provided,
            skips the document embedding step entirely. Must have the
            same length as documents (one embedding per document).
            Accepts np.ndarray or list[list[float]].
            Use this to avoid re-embedding the same document set
            across multiple queries.

    Returns:
        List of HybridSearchResult dicts sorted by reranker score (descending).
        Each result includes:
        - rank: Final rank after reranking
        - index: Original index in documents list
        - score: Normalized reranker score (0–1, higher = more relevant)
        - text: Document text
        - vector_score: Original vector similarity score (for comparison)
        - rerank_score_raw: Raw reranker score before normalization
    """
    if not documents:
        logger.warning("hybrid_search: empty documents list, returning []")
        return []

    n_docs = len(documents)
    logger.info(
        f"hybrid_search: query='{query[:80]}...', "
        f"n_docs={n_docs}, top_n={top_n}, "
        f"vector_score_threshold={vector_score_threshold}"
    )

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

    logger.info("Stage 1/2: Vector search (embedding + cosine similarity)")

    query_emb = embed(query, prefix=resolved_query_prefix)

    if doc_embeddings is not None:
        if len(doc_embeddings) != n_docs:
            raise ValueError(
                f"doc_embeddings length ({len(doc_embeddings)}) "
                f"does not match documents length ({n_docs})"
            )
        doc_embs = np.asarray(doc_embeddings, dtype=np.float32)
        logger.info(
            f"Using provided doc_embeddings (shape={doc_embs.shape}), "
            "skipping document embedding step"
        )
    else:
        doc_embs = embed(documents, prefix=resolved_doc_prefix)
        logger.info(f"Document embeddings computed (shape={doc_embs.shape})")

    similarities = np.array(
        [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embs]
    )

    if vector_score_threshold is not None:
        passing_mask = similarities >= vector_score_threshold
        candidate_indices = np.where(passing_mask)[0].tolist()
        n_filtered = n_docs - len(candidate_indices)
        logger.info(
            f"Vector score threshold ({vector_score_threshold:.4f}): "
            f"{len(candidate_indices)}/{n_docs} documents pass "
            f"({n_filtered} filtered out)"
        )
        if not candidate_indices:
            logger.warning(
                "No documents passed the vector score threshold, returning empty results"
            )
            return []
    else:
        candidate_indices = list(range(n_docs))

    candidates = [documents[i] for i in candidate_indices]
    candidate_vector_scores = [float(similarities[i]) for i in candidate_indices]

    logger.info(
        f"Vector search complete: {len(candidates)} candidates selected from {n_docs} documents"
    )
    logger.debug(
        "Vector search results (before reranking): "
        + ", ".join(
            [
                f"idx={idx}(vec={score:.4f})"
                for idx, score in zip(candidate_indices, candidate_vector_scores)
            ]
        )
    )
    logger.debug(
        f"Vector score stats: min={min(candidate_vector_scores):.4f}, "
        f"max={max(candidate_vector_scores):.4f}, "
        f"mean={np.mean(candidate_vector_scores):.4f}"
    )

    if not candidates:
        logger.warning("No candidates to rerank, returning empty results")
        return []

    logger.info("Stage 2/2: Cross-encoder reranking")

    rerank_top_n = min(top_n, len(candidates)) if top_n is not None else len(candidates)
    rerank_results = rerank(query, candidates, top_n=rerank_top_n)

    raw_rerank_scores = [rr["score"] for rr in rerank_results]

    if normalize_scores and raw_rerank_scores:
        normalized_scores = sigmoid_normalize_scores(
            raw_rerank_scores,
            temperature=sigmoid_temperature,
        )
        logger.info(
            f"Score normalization applied (sigmoid, temperature={sigmoid_temperature})"
        )
        logger.debug(
            f"Raw scores: [{', '.join(f'{s:.4f}' for s in raw_rerank_scores)}]"
        )
        logger.debug(
            f"Normalized scores: [{', '.join(f'{s:.4f}' for s in normalized_scores)}]"
        )
    else:
        normalized_scores = raw_rerank_scores
        if raw_rerank_scores:
            logger.info("Score normalization disabled, using raw scores")

    final_results: list[HybridSearchResult] = []
    for rank_pos, rr in enumerate(rerank_results, start=1):
        candidate_local_idx = rr["index"]
        original_idx = candidate_indices[candidate_local_idx]
        vector_score = candidate_vector_scores[candidate_local_idx]
        raw_score = rr["score"]
        norm_score = normalized_scores[rank_pos - 1]

        final_results.append(
            {
                "rank": rank_pos,
                "index": original_idx,
                "score": norm_score,
                "text": documents[original_idx],
                "vector_score": vector_score,
                "rerank_score_raw": raw_score,
            }
        )

    logger.info(f"Reranking complete: {len(final_results)} final results returned")

    if normalized_scores:
        logger.debug(
            "Normalized score interpretation: "
            "0.5 = neutral, >0.7 = high relevance, <0.3 = low relevance. "
            "Relative ordering matters more than absolute values."
        )
        logger.debug(
            f"Normalized score stats: min={min(normalized_scores):.4f}, "
            f"max={max(normalized_scores):.4f}, "
            f"mean={np.mean(normalized_scores):.4f}"
        )

    logger.debug(
        "Score comparison (vector → rerank normalized): "
        + ", ".join(
            [
                f"#{r['rank']} idx={r['index']} (vec={r['vector_score']:.4f} → score={r['score']:.4f})"
                for r in final_results
            ]
        )
    )

    return final_results


# ✅ NEW: PDR-aware hybrid search wrapper
def hybrid_search_pdr(
    query: str,
    pdr_result: Dict[str, List[Dict]],
    top_n: int = 5,
    vector_score_threshold: float | None = None,
    model: str | None = None,
    **kwargs,
) -> List[Dict]:
    """Hybrid search over PDR children with automatic parent resolution.

    Searches over child chunks for precise matching, then resolves hits
    to their full parent documents for LLM context injection. Preserves
    rerank ordering while deduplicating by parent.

    Args:
        query: Search query string.
        pdr_result: Output from ParentDocumentChunker.chunk_pdr() containing
            'parents' and 'children' lists with linked IDs.
        top_n: Number of unique parent results to return.
        vector_score_threshold: Optional minimum vector score for child candidates.
        model: Model key for embedding (passed to hybrid_search).
        **kwargs: Additional kwargs forwarded to hybrid_search().

    Returns:
        List of dicts with keys: rank, score, text (full parent),
        child_text (original matching child), parent_id, num_tokens.
    """
    children = pdr_result["children"]
    parent_map = {p["id"]: p for p in pdr_result["parents"]}

    if not children:
        logger.warning("hybrid_search_pdr: no children in pdr_result, returning []")
        return []

    # Stage 1: Search over child chunks only
    child_texts = [c["content"] for c in children]
    child_results = hybrid_search(
        query,
        child_texts,
        top_n=top_n * 2,  # Over-retrieve to account for parent deduplication
        vector_score_threshold=vector_score_threshold,
        **kwargs,
    )

    # Stage 2: Resolve to unique parents, preserving rerank order
    seen_parents: set[str] = set()
    resolved: List[Dict] = []

    for cr in child_results:
        child_idx = cr["index"]
        parent_id = children[child_idx]["parent_id"]

        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent = parent_map[parent_id]
            resolved.append(
                {
                    "rank": len(resolved) + 1,
                    "score": cr["score"],
                    "text": parent["content"],  # ← FULL PARENT CONTEXT
                    "child_text": cr["text"],  # ← ORIGINAL MATCHING CHILD
                    "parent_id": parent_id,
                    "num_tokens": parent.get("num_tokens", 0),
                }
            )

        if len(resolved) >= top_n:
            break

    logger.info(
        "hybrid_search_pdr: %d child hits → %d unique parents resolved",
        len(child_results),
        len(resolved),
    )
    return resolved


if __name__ == "__main__":
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    print("=" * 60)
    print("HYBRID SEARCH RESULTS (CLEAN)")
    print("=" * 60)

    results = hybrid_search(query, docs, normalize_scores=True)

    print(f"\nQuery: {query}\n")
    print("Final ranked results (after reranking):")
    print("Format: #rank  idx  score(0-1)  vector  raw  text")
    print("-" * 60)
    for r in results:
        print(
            f"  #{r['rank']}  idx={r['index']}  "
            f"score={r['score']:.4f}  vector={r['vector_score']:.4f}  raw={r['rerank_score_raw']:.4f}  "
            f"{r['text']}"
        )

    # ─────────────────────────────────────────────
    # PDR DEMO
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HYBRID SEARCH PDR RESULTS")
    print("=" * 60)

    pdr_result = {
        "parents": [
            {
                "id": "p1",
                "content": (
                    "The giant panda (Ailuropoda melanoleuca) is a bear species "
                    "endemic to China. It is characterised by its bold black-and-white "
                    "coat and rotund body. Pandas primarily eat bamboo and can consume "
                    "up to 38 kg of it per day. They live in mountainous regions of "
                    "central China."
                ),
                "num_tokens": 64,
            },
            {
                "id": "p2",
                "content": (
                    "Python is a high-level, general-purpose programming language. "
                    "Its design philosophy emphasises code readability. Python is "
                    "dynamically typed and garbage-collected. It supports multiple "
                    "programming paradigms including structured, object-oriented, "
                    "and functional programming."
                ),
                "num_tokens": 52,
            },
            {
                "id": "p3",
                "content": (
                    "Machine learning is a subset of artificial intelligence. "
                    "It gives systems the ability to learn from data without being "
                    "explicitly programmed. Common techniques include supervised "
                    "learning, unsupervised learning, and reinforcement learning."
                ),
                "num_tokens": 45,
            },
        ],
        "children": [
            # p1 split into 3 smaller chunks
            {
                "id": "c1",
                "parent_id": "p1",
                "content": "The giant panda is a bear species endemic to China.",
            },
            {
                "id": "c2",
                "parent_id": "p1",
                "content": "Pandas eat bamboo and can consume up to 38 kg per day.",
            },
            {
                "id": "c3",
                "parent_id": "p1",
                "content": "Giant pandas live in mountainous regions of central China.",
            },
            # p2 split into 2 chunks
            {
                "id": "c4",
                "parent_id": "p2",
                "content": "Python is a high-level programming language.",
            },
            {
                "id": "c5",
                "parent_id": "p2",
                "content": "Python supports object-oriented and functional programming.",
            },
            # p3 split into 2 chunks
            {
                "id": "c6",
                "parent_id": "p3",
                "content": "Machine learning is a subset of artificial intelligence.",
            },
            {
                "id": "c7",
                "parent_id": "p3",
                "content": "ML systems learn from data without explicit programming.",
            },
        ],
    }

    pdr_query = "What do giant pandas eat and where do they live?"
    pdr_results = hybrid_search_pdr(pdr_query, pdr_result, top_n=2)

    print(f"\nQuery: {pdr_query}\n")
    print(
        "Format: #rank  score  parent_id  tokens  child_text → parent_text (truncated)"
    )
    print("-" * 60)
    for r in pdr_results:
        child_preview = r["child_text"][:60].rstrip()
        parent_preview = r["text"][:80].rstrip()
        print(
            f"  #{r['rank']}  score={r['score']:.4f}  "
            f"parent={r['parent_id']}  tokens={r['num_tokens']}\n"
            f"       child  : {child_preview!r}\n"
            f"       parent : {parent_preview!r}...\n"
        )
