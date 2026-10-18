import os
from typing import TypedDict

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed


class VectorSearchResult(TypedDict):
    """Typed structure for a single vector search result."""

    rank: int
    index: int
    score: float
    text: str


def cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query embedding and document embeddings.

    Args:
        query_emb: Query embedding vector (1D array).
        doc_embs: Document embeddings (2D array, shape: [n_docs, emb_dim]).

    Returns:
        Array of cosine similarity scores (shape: [n_docs]).
    """
    # Normalize embeddings to unit length
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity is the dot product of normalized vectors
    similarities = np.dot(doc_norms, query_norm)
    return similarities


def vector_search(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    return_embeddings: bool = False,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    embed_model: str = EMBED_MODEL,
) -> list[VectorSearchResult] | tuple[list[VectorSearchResult], np.ndarray, np.ndarray]:
    """
    Embed query and documents, then rank by cosine similarity.

    Args:
        query: The search query string.
        documents: List of document strings to search over.
        top_n: Optional limit on number of results to return.
        return_embeddings: If True, also returns (query_emb, doc_embs).
        query_prefix: Prefix for the query. Falls back to EMBED_QUERY_PREFIX env var.
        document_prefix: Prefix for documents. Falls back to EMBED_DOC_PREFIX env var.
        embed_model: Embedding model to use. Defaults to EMBED_MODEL from config.

    Returns:
        List of VectorSearchResult dicts (rank, index, score, text),
        or if return_embeddings=True, a tuple of (results, query_emb, doc_embs).
    """
    if not documents:
        return [] if not return_embeddings else ([], np.array([]), np.array([]))

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

    query_embedding = embed(query, prefix=resolved_query_prefix, model=embed_model)
    doc_embeddings = embed(documents, prefix=resolved_doc_prefix, model=embed_model)

    # Ensure embeddings are numpy arrays
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding, dtype=np.float32)
    if isinstance(doc_embeddings, list):
        doc_embeddings = np.array(doc_embeddings, dtype=np.float32)

    # Handle single document case
    if doc_embeddings.ndim == 1:
        doc_embeddings = doc_embeddings.reshape(1, -1)

    similarities = cosine_similarity(query_embedding, doc_embeddings)

    # Create scored list with original indices
    scored = [(idx, float(score)) for idx, score in enumerate(similarities)]
    scored.sort(key=lambda x: x[1], reverse=True)

    results: list[VectorSearchResult] = []
    for rank, (idx, score) in enumerate(scored, start=1):
        results.append(
            {
                "rank": rank,
                "index": idx,
                "score": score,
                "text": documents[idx],
            }
        )
        if top_n is not None and len(results) >= top_n:
            break

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
    ]

    print("Vector Search Results:")
    print("=" * 60)
    results, query_emb, doc_embs = vector_search(query, docs, return_embeddings=True)
    print(f"Shape of query_embedding: {np.shape(query_emb)}")
    print(f"Size of query_embedding: {np.size(query_emb)}")
    print(f"Shape of doc_embeddings: {np.shape(doc_embs)}")
    print(f"Size of doc_embeddings: {np.size(doc_embs)}")
    print(f"\nQuery: {query}\n")
    print("Documents ranked by similarity:")
    for r in results:
        print(f"#{r['rank']}  {r['score']:.4f}  {r['text']}")
