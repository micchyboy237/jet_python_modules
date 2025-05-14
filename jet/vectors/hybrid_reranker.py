from typing import List, Tuple, TypedDict
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


class Models(TypedDict):
    embedder: SentenceTransformer
    cross_encoder: CrossEncoder


class SearchResults(TypedDict):
    reranked_indices: np.ndarray
    reranked_scores: np.ndarray
    candidate_indices: np.ndarray
    candidate_distances: np.ndarray


def load_models() -> Models:
    """Load sentence transformer and cross-encoder models."""
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
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
    distances, indices = index.search(query_embedding, k)
    return indices[0], distances[0]


def rerank_candidates(cross_encoder: CrossEncoder, query: str, documents: List[str],
                      candidate_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rerank candidates using cross-encoder."""
    pairs = [[query, documents[idx]] for idx in candidate_indices]
    scores: np.ndarray = cross_encoder.predict(pairs)
    sorted_indices = np.argsort(scores)[::-1]
    return candidate_indices[sorted_indices], scores[sorted_indices]


def search_documents(query: str, documents: List[str], embed_model: SentenceTransformer,
                     rerank_model: CrossEncoder, k: int = 10) -> SearchResults:
    """
    Perform document search with retrieval and reranking.

    Args:
        query: Search query
        documents: List of document strings
        embed_model: Sentence transformer model for initial retrieval
        rerank_model: Cross-encoder model for reranking
        k: Number of candidates to retrieve initially

    Returns:
        SearchResults dictionary containing reranked and candidate indices and scores
    """
    index, _ = create_index(embed_model, documents)
    candidate_indices, candidate_distances = retrieve_candidates(
        embed_model, index, query, k)
    reranked_indices, reranked_scores = rerank_candidates(
        rerank_model, query, documents, candidate_indices)

    return {
        "reranked_indices": reranked_indices,
        "reranked_scores": reranked_scores,
        "candidate_indices": candidate_indices,
        "candidate_distances": candidate_distances
    }


def main() -> None:
    documents: List[str] = [
        "The quick brown fox jumps over the lazy dog.",
        "A fox fled from danger in the forest.",
        "Dogs are loyal and friendly pets.",
        "The cat sleeps on the windowsill.",
        "Foxes are known for their cunning behavior."
    ]

    query: str = "Tell me about foxes."

    print("Loading models...")
    models: Models = load_models()

    print(f"\nPerforming search for query: '{query}'")
    results: SearchResults = search_documents(
        query, documents, models["embedder"], models["cross_encoder"], k=3
    )

    print("\nInitial retrieval results:")
    for idx, dist in zip(results["candidate_indices"], results["candidate_distances"]):
        print(f"Doc {idx}: {documents[idx]} (Distance: {dist:.4f})")

    print("\nReranked results:")
    for idx, score in zip(results["reranked_indices"], results["reranked_scores"]):
        print(f"Doc {idx}: {documents[idx]} (Score: {score:.4f})")


if __name__ == "__main__":
    main()
