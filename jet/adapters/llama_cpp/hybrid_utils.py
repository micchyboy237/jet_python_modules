"""Hybrid search utilities combining embeddings, reranking, and keyword search."""

from typing import TypedDict

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.rerank_utils import rerank
from rank_bm25 import BM25Okapi


class HybridSearchResult(TypedDict):
    """Result from hybrid_search."""

    document: str
    score: float
    original_index: int


class KeywordHybridResult(TypedDict):
    """Result from hybrid_search_with_keywords."""

    document: str
    score: float
    original_index: int
    keyword_score: float
    embed_score: float


def hybrid_search(
    query: str,
    documents: list[str],
    top_k: int = 10,
    embed_candidates: int = 100,
) -> list[HybridSearchResult]:
    """
    Production search pipeline: Embed for recall, rerank for precision.
    This is the recommended architecture for most use cases.
    Args:
        query: Search query
        documents: List of documents to search
        top_k: Number of final results to return
        embed_candidates: Number of candidates from embedding stage
    Returns:
        List of HybridSearchResult with document, score, and original_index
    """
    query_emb = embed(query)
    doc_embs = embed(documents)
    similarities = [_cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embs]
    candidates_count = min(embed_candidates, len(documents))
    top_indices = np.argsort(similarities)[-candidates_count:][::-1]
    candidate_docs = [documents[i] for i in top_indices]
    reranked = rerank(query, candidate_docs, top_n=top_k)
    return [
        HybridSearchResult(
            document=r["text"],  # Fixed: 'text' instead of 'document'
            score=r["score"],
            original_index=top_indices[r["index"]],
        )
        for r in reranked
    ]


def hybrid_search_with_keywords(
    query: str,
    documents: list[str],
    top_k: int = 10,
    embed_weight: float = 0.5,
    keyword_weight: float = 0.5,
    use_reranker: bool = True,
    embed_candidates: int = 100,
) -> list[KeywordHybridResult]:
    """
    Hybrid search combining keyword (BM25) and semantic (embeddings) search.
    Best for domain-specific vocabulary, exact matching, and cold start problems.
    Args:
        query: Search query
        documents: List of documents to search
        top_k: Number of final results to return
        embed_weight: Weight for embedding similarity (0-1)
        keyword_weight: Weight for keyword similarity (0-1)
        use_reranker: Whether to apply reranker as final stage
        embed_candidates: Number of candidates for reranker stage
    Returns:
        List of KeywordHybridResult with document, scores, and original_index
    """
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.lower().split()
    keyword_scores = bm25.get_scores(tokenized_query)
    query_emb = embed(query)
    doc_embs = embed(documents)
    embed_scores = np.array(
        [_cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embs]
    )
    if keyword_scores.max() > 0:
        keyword_scores = (keyword_scores - keyword_scores.min()) / (
            keyword_scores.max() - keyword_scores.min()
        )
    embed_scores = (embed_scores - embed_scores.min()) / (
        embed_scores.max() - embed_scores.min()
    )
    combined_scores = (embed_weight * embed_scores) + (keyword_weight * keyword_scores)
    candidates_count = min(embed_candidates, len(documents))
    top_indices = np.argsort(combined_scores)[-candidates_count:][::-1]
    candidate_docs = [documents[i] for i in top_indices]
    if use_reranker and len(candidate_docs) > top_k:
        reranked = rerank(query, candidate_docs, top_n=top_k)
        return [
            KeywordHybridResult(
                document=r["text"],  # Fixed: 'text' instead of 'document'
                score=r["score"],
                original_index=int(top_indices[r["index"]]),
                keyword_score=float(keyword_scores[top_indices[r["index"]]]),
                embed_score=float(embed_scores[top_indices[r["index"]]]),
            )
            for r in reranked
        ]
    top_k_indices = top_indices[:top_k]
    return [
        KeywordHybridResult(
            document=documents[i],
            score=float(combined_scores[i]),
            original_index=int(i),
            keyword_score=float(keyword_scores[i]),
            embed_score=float(embed_scores[i]),
        )
        for i in top_k_indices
    ]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    query = "What is a giant panda?"
    documents = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
        "The panda's diet is 99% bamboo despite being classified as carnivores.",
        "Data science combines statistics, computer science, and domain expertise.",
        "China has established over 50 panda reserves in Sichuan province.",
        "JavaScript is commonly used for web development.",
        "Conservation efforts have helped increase wild panda populations.",
    ]
    print("=" * 60)
    print("HYBRID SEARCH (Embed → Rerank)")
    print("=" * 60)
    print(f"Query: {query}\n")
    results = hybrid_search(query, documents, top_k=5, embed_candidates=7)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.4f}] (idx:{r['original_index']}) {r['document']}")
    print("\n" + "=" * 60)
    print("HYBRID SEARCH WITH KEYWORDS (BM25 + Embed + Rerank)")
    print("=" * 60)
    print(f"Query: {query}\n")
    results = hybrid_search_with_keywords(
        query, documents, top_k=5, embed_weight=0.5, keyword_weight=0.5
    )
    for i, r in enumerate(results, 1):
        print(
            f"{i}. [total:{r['score']:.4f}] [kw:{r['keyword_score']:.4f}] [emb:{r['embed_score']:.4f}] (idx:{r['original_index']}) {r['document']}"
        )
