import json
import os
from typing import List, TypedDict, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from jet.file.utils import load_file, save_file


class Header(TypedDict):
    """
    Represents a header dictionary with a required text field.

    Fields:
        text: The text content of the header.
    """
    text: str


class PreprocessedText(TypedDict):
    """
    Represents a preprocessed text entry with index and ID.

    Fields:
        text: The text content.
        doc_index: Original index in the input list.
        id: Unique identifier for the text.
    """
    text: str
    doc_index: int
    id: str


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text.
        rank: Rank based on score (1 for highest).
        doc_index: Original index of the text in the input list.
        score: Normalized similarity score.
        text: The compared text (or chunk if long).
        tokens: Number of encoded tokens from text.
        rerank_score: Score from cross-encoder reranking.
        diversity_score: Score from MMR diversity calculation.
        embedding: Optional embedding vector for MMR calculations.
    """
    id: str
    rank: int
    doc_index: int
    score: float
    text: str
    tokens: int
    rerank_score: float
    diversity_score: float
    embedding: Optional[np.ndarray]


def preprocess_texts(headers: List[Header]) -> List[PreprocessedText]:
    """
    Filter out noisy texts (e.g., menus, short texts) from headers.
    Args:
        headers: List of header dicts with 'text' key.
    Returns:
        List of dicts with 'text', 'doc_index', and 'id'.
    """
    exclude_keywords = ["menu", "sign in", "trending", "close"]
    min_words = 10
    return [
        {"text": header["text"], "doc_index": i, "id": f"doc_{i}"}
        for i, header in enumerate(headers)
        if not any(keyword in header["text"].lower() for keyword in exclude_keywords)
        and len(header["text"].split()) >= min_words
    ]


def embed_search(
    query: str,
    texts: List[PreprocessedText],
    model_name: str = "all-MiniLM-L12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    top_k: int = 20
) -> List[SimilarityResult]:
    """
    Perform embedding-based search to retrieve top-k relevant texts.
    Args:
        query: Search query.
        texts: List of dicts with 'text', 'doc_index', and 'id'.
        model_name: Sentence Transformer model.
        device: Device for encoding (mps for M1).
        top_k: Number of candidates to retrieve.
    Returns:
        List of SimilarityResult dicts.
    """
    model = SentenceTransformer(model_name, device=device)
    text_strings = [t["text"] for t in texts]
    query_embedding = model.encode(
        query, convert_to_tensor=True, device=device)
    text_embeddings = model.encode(
        text_strings,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device
    )
    similarities = util.cos_sim(query_embedding, text_embeddings)[
        0].cpu().numpy()
    top_k_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_k_indices, 1):
        tokens = len(model.tokenize([text_strings[idx]])["input_ids"][0])
        results.append({
            "id": texts[idx]["id"],
            "rank": rank,
            "doc_index": texts[idx]["doc_index"],
            "score": float(similarities[idx]),
            "text": text_strings[idx],
            "tokens": tokens,
            "rerank_score": 0.0,
            "diversity_score": 0.0,
            "embedding": text_embeddings[idx].cpu().numpy()
        })
    return results


def rerank_results(
    query: str,
    candidates: List[SimilarityResult],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
) -> List[SimilarityResult]:
    """
    Rerank candidates using a cross-encoder.
    Args:
        query: Search query.
        candidates: List of SimilarityResult dicts.
        model_name: Cross-encoder model.
        device: Device for encoding.
    Returns:
        Reranked list of SimilarityResult dicts with updated rerank_score.
    """
    model = CrossEncoder(model_name, device=device)
    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(
        candidates, key=lambda x: x["rerank_score"], reverse=True)
    for rank, candidate in enumerate(reranked, 1):
        candidate["rank"] = rank
    return reranked


def mmr_diversity(
    candidates: List[SimilarityResult],
    num_results: int = 5,
    lambda_param: float = 0.5
) -> List[SimilarityResult]:
    """
    Select diverse results using Maximal Marginal Relevance.
    Args:
        candidates: List of SimilarityResult dicts with 'embedding'.
        num_results: Number of final results.
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
    Returns:
        List of diverse SimilarityResult dicts with diversity_score.
    """
    selected = []
    candidate_embeddings = np.array([c["embedding"] for c in candidates])

    while len(selected) < num_results and candidates:
        if not selected:
            best_candidate = candidates.pop(0)
            best_candidate["diversity_score"] = best_candidate["rerank_score"]
            selected.append(best_candidate)
        else:
            mmr_scores = []
            selected_embeddings = np.array([c["embedding"] for c in selected])
            for i, candidate in enumerate(candidates):
                relevance = candidate["rerank_score"]
                similarity = np.max(
                    util.cos_sim(
                        torch.tensor(candidate["embedding"]).to(
                            "mps" if torch.backends.mps.is_available() else "cpu"),
                        torch.tensor(selected_embeddings).to(
                            "mps" if torch.backends.mps.is_available() else "cpu")
                    )[0].cpu().numpy()
                )
                mmr_score = lambda_param * relevance - \
                    (1 - lambda_param) * similarity
                mmr_scores.append(mmr_score)
                candidate["diversity_score"] = float(mmr_score)

            best_idx = np.argmax(mmr_scores)
            selected.append(candidates.pop(best_idx))

    for rank, candidate in enumerate(selected, 1):
        candidate["rank"] = rank
    return selected


def search_diverse_context(
    query: str,
    headers: List[Header],
    model_name: str = "all-MiniLM-L12-v2",
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    top_k: int = 20,
    num_results: int = 5,
    lambda_param: float = 0.5
) -> List[SimilarityResult]:
    """
    Search for diverse context data given a query.
    Args:
        query: Search query.
        headers: List of header dicts with 'text'.
        model_name: Sentence Transformer model.
        rerank_model: Cross-encoder model.
        device: Device for encoding.
        top_k: Number of candidates for reranking.
        num_results: Number of final diverse results.
        lambda_param: MMR relevance-diversity trade-off.
    Returns:
        List of SimilarityResult dicts.
    """
    texts = preprocess_texts(headers)
    if not texts:
        return []

    candidates = embed_search(query, texts, model_name, device, top_k)
    reranked = rerank_results(query, candidates, rerank_model, device)
    diverse_results = mmr_diversity(reranked, num_results, lambda_param)
    return diverse_results
