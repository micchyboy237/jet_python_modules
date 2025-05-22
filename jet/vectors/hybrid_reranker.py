from typing import List, Optional, Tuple, TypedDict
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


class Models(TypedDict):
    embedder: SentenceTransformer
    cross_encoder: CrossEncoder


class RawSearchResults(TypedDict):
    reranked_indices: np.ndarray
    reranked_scores: np.ndarray
    candidate_indices: np.ndarray
    candidate_distances: np.ndarray


class ScoreResults(TypedDict):
    indices: np.ndarray
    distances: np.ndarray
    raw_scores: np.ndarray
    normalized_scores: np.ndarray


class SearchResult(TypedDict):
    rank: int
    doc_idx: int
    distance: float
    raw_score: float
    normalized_score: float
    similarity_score: float  # Add similarity score
    text: str


def load_embed_model() -> SentenceTransformer:
    """Load sentence transformer and cross-encoder models."""
    embedder = SentenceTransformer('all-MiniLM-L12-v2')
    return embedder


def load_rerank_model() -> CrossEncoder:
    """Load sentence transformer and cross-encoder models."""
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    return cross_encoder


def load_models() -> Models:
    """Load sentence transformer and cross-encoder models."""
    embedder = load_embed_model()
    cross_encoder = load_rerank_model()
    return {"embedder": embedder, "cross_encoder": cross_encoder}


def create_index(embedder: SentenceTransformer, documents: List[str]) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    """Create FAISS index with document embeddings."""
    doc_embeddings: np.ndarray = embedder.encode(
        documents, convert_to_tensor=False)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    return index, doc_embeddings


def retrieve_candidates(embedder: SentenceTransformer, index: faiss.IndexFlatL2,
                        query: str, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve top-k candidates using sentence transformer embeddings."""
    query_embedding: np.ndarray = embedder.encode(
        [query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, k=k)
    return indices[0], distances[0]


def rerank_candidates(cross_encoder: CrossEncoder, query: str, documents: List[str],
                      candidate_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rerank candidates using cross-encoder."""
    pairs = [[query, documents[idx]] for idx in candidate_indices]
    scores: np.ndarray = cross_encoder.predict(pairs)
    sorted_indices = np.argsort(scores)[::-1]
    return candidate_indices[sorted_indices], scores[sorted_indices]


def calculate_scores(query: str, documents: List[str], search_results: RawSearchResults) -> ScoreResults:
    candidate_indices = search_results["candidate_indices"]
    candidate_distances = search_results["candidate_distances"]
    reranked_indices = search_results["reranked_indices"]
    reranked_scores = search_results["reranked_scores"]

    # Map reranked scores to original candidate indices
    raw_scores = np.zeros_like(candidate_distances)
    for idx, score in zip(reranked_indices, reranked_scores):
        raw_scores[np.where(candidate_indices == idx)[0]] = score

    # Normalize cross-encoder scores using sigmoid
    normalized_scores = 1 / (1 + np.exp(-raw_scores))

    # Convert L2 distances to similarity scores (e.g., using inverse distance)
    similarity_scores = 1 / (1 + candidate_distances)

    # Sort by raw_scores in descending order
    sort_indices = np.argsort(raw_scores)[::-1]
    sorted_indices = candidate_indices[sort_indices]
    sorted_distances = candidate_distances[sort_indices]
    sorted_raw_scores = raw_scores[sort_indices]
    sorted_normalized_scores = normalized_scores[sort_indices]
    sorted_similarity_scores = similarity_scores[sort_indices]

    return {
        "indices": sorted_indices,
        "distances": sorted_distances,
        "raw_scores": sorted_raw_scores,
        "normalized_scores": sorted_normalized_scores,
        "similarity_scores": sorted_similarity_scores  # Add similarity scores
    }


def search_documents(
    query: str,
    documents: List[str],
    embed_model: Optional[SentenceTransformer] = None,
    rerank_model: Optional[CrossEncoder] = None,
    k: Optional[int] = 10
) -> List[SearchResult]:
    if not embed_model:
        embed_model = load_embed_model()
    if not rerank_model:
        rerank_model = load_rerank_model()
    if k is None:
        k = len(documents)

    # Create index and retrieve candidates
    index, _ = create_index(embed_model, documents)
    candidate_indices, candidate_distances = retrieve_candidates(
        embed_model, index, query, k)

    # Rerank candidates
    reranked_indices, reranked_scores = rerank_candidates(
        rerank_model, query, documents, candidate_indices)

    # Compute scores
    search_results: RawSearchResults = {
        "reranked_indices": reranked_indices,
        "reranked_scores": reranked_scores,
        "candidate_indices": candidate_indices,
        "candidate_distances": candidate_distances
    }
    score_results: ScoreResults = calculate_scores(
        query, documents, search_results)

    # Build final results
    final_results: List[SearchResult] = []
    for rank_idx, (idx, dist, raw_score, norm_score, sim_score) in enumerate(zip(
        score_results["indices"],
        score_results["distances"],
        score_results["raw_scores"],
        score_results["normalized_scores"],
        score_results["similarity_scores"]  # Add similarity scores
    ), start=1):
        final_results.append({
            "rank": rank_idx,
            "doc_idx": int(idx),
            "distance": float(dist),
            "raw_score": float(raw_score),
            "normalized_score": float(norm_score),
            "similarity_score": float(sim_score),  # Include similarity score
            "text": documents[idx]
        })

    return final_results
