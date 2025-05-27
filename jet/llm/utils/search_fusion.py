import json
from typing import Any, Callable, Union, List, Dict, Optional, Literal, TypedDict, DefaultDict
from jet.code.utils import ProcessedResult, process_markdown_file
from jet.data.utils import generate_key
from jet.features.rag_llm_generation import get_embedding_function
from jet.file.utils import load_file, save_file
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import EmbedModelType, ModelKey, LLMModelType
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.markdown import extract_block_content
import numpy as np
from collections import defaultdict
import faiss
from sentence_transformers import SentenceTransformer
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from bs4 import BeautifulSoup
import trafilatura
import re
# from fast_langdetect import detect
from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize, word_tokenize


class SimilarityResult(TypedDict):
    """
    Represents a single similarity result for a text.

    Fields:
        id: Identifier for the text.
        rank: Rank based on score (1 for highest).
        doc_index: Original index of the text in the input list.
        score: Fused similarity score.
        percent_difference: Percentage difference from the highest score.
        text: The compared text (or chunk if long).
        relevance: Optional relevance score (e.g., from user feedback).
        word_count: Number of words in the text.
    """
    id: str
    rank: Optional[int]
    doc_index: int
    score: float
    percent_difference: Optional[float]
    text: str
    relevance: Optional[float]
    word_count: Optional[int]


def query_similarity_scores(
    query: Union[str, List[str]],
    texts: Union[str, List[str]],
    threshold: float = 0.0,
    model: Union[EmbedModelType, List[EmbedModelType]] = "all-MiniLM-L6-v2",
    fuse_method: Literal["average", "max", "min"] = "average",
    ids: Union[List[str], None] = None,
    metrics: Literal["cosine", "dot", "euclidean"] = "cosine"
) -> List[SimilarityResult]:
    """
    Computes similarity scores for queries against texts using one or more embedding models,
    fusing results into a single sorted list with one result per text.

    For each text and query, scores are averaged across models. Then, for each text,
    the query-specific scores are fused using the specified method ('average', 'max', or 'min').

    Args:
        query: Single query or list of queries.
        texts: Single text or list of texts to compare against.
        threshold: Minimum similarity score to include in results (default: 0.0).
        model: One or more embedding model names (default: "all-MiniLM-L6-v2").
        fuse_method: Fusion method for combining scores ('average', 'max', or 'min') (default: "average").
        ids: Optional list of IDs for texts; must match texts length if provided.
        metrics: Similarity metric to use ('cosine', 'euclidean', 'dot') (default: "cosine").

    Returns:
        List of SimilarityResult, containing one fused result per text,
        sorted by score in descending order with ranks, percent_difference, and doc_index.

    Raises:
        ValueError: If inputs are empty, model is empty, ids length mismatches texts,
                    invalid fuse_method, or invalid metrics.
    """
    if isinstance(query, str):
        query = [query]
    if isinstance(texts, str):
        texts = [texts]
    if isinstance(model, str):
        model = [model]

    if not query or not texts:
        raise ValueError("Both query and texts must be non-empty.")
    if not model:
        raise ValueError("At least one model name must be provided.")
    if ids is not None and len(ids) != len(texts):
        raise ValueError(
            f"Length of ids ({len(ids)}) must match length of texts ({len(texts)})."
        )

    supported_methods = {"average", "max", "min"}
    if fuse_method not in supported_methods:
        raise ValueError(
            f"Fusion method must be one of {supported_methods}; got {fuse_method}."
        )

    supported_metrics = {"cosine", "euclidean", "dot"}
    if metrics not in supported_metrics:
        raise ValueError(
            f"Metrics must be one of {supported_metrics}; got {metrics}."
        )

    text_ids = (
        ids
        if ids is not None
        else [generate_key(text, query[0] if query else None) for text in texts]
    )

    # Collect all results across queries and models
    all_results: List[Dict[str, Any]] = []

    for model_name in model:
        embed_func = get_embedding_function(model_name)

        query_embeddings = np.array(embed_func(query))
        text_embeddings = np.array(embed_func(texts))

        if metrics == "cosine":
            # Normalize embeddings for cosine similarity
            query_norms = np.linalg.norm(
                query_embeddings, axis=1, keepdims=True)
            text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)

            query_embeddings = np.divide(
                query_embeddings,
                query_norms,
                out=np.zeros_like(query_embeddings),
                where=query_norms != 0
            )
            text_embeddings = np.divide(
                text_embeddings,
                text_norms,
                out=np.zeros_like(text_embeddings),
                where=text_norms != 0
            )

            similarity_matrix = np.dot(query_embeddings, text_embeddings.T)
        elif metrics == "dot":
            # Raw dot product without normalization
            similarity_matrix = np.dot(query_embeddings, text_embeddings.T)
        elif metrics == "euclidean":
            # Euclidean distance (lower is better, so we negate and add 1 to make higher better)
            similarity_matrix = np.zeros((len(query), len(texts)))
            for i in range(len(query)):
                for j in range(len(texts)):
                    dist = np.linalg.norm(
                        query_embeddings[i] - text_embeddings[j])
                    similarity_matrix[i, j] = 1 / (1 + dist)

        for i, query_text in enumerate(query):
            scores = similarity_matrix[i]

            mask = scores >= threshold
            filtered_texts = np.array(texts)[mask]
            filtered_ids = np.array(text_ids)[mask]
            filtered_scores = scores[mask]
            filtered_indices = np.arange(
                len(texts))[mask]  # Track original indices

            sorted_indices = np.argsort(filtered_scores)[::-1]
            for idx, j in enumerate(sorted_indices):
                all_results.append({
                    "id": filtered_ids[j],
                    "doc_index": int(filtered_indices[j]),
                    "query": query_text,
                    "text": filtered_texts[j],
                    "score": float(filtered_scores[j]),
                })

    # Fuse results
    fused_results = fuse_all_results(all_results, method=fuse_method)

    # Update fused results to include doc_index
    # Fixed: Use result["id"] instead of result.id
    fused_dict = {result["id"]: result for result in fused_results}
    for result in all_results:
        if result["id"] in fused_dict:
            fused_dict[result["id"]]["doc_index"] = result["doc_index"]

    # Convert dictionaries to SimilarityResult TypedDict
    final_results = [
        {
            "id": result["id"],
            "rank": result["rank"],
            "doc_index": result.get("doc_index", 0),  # Default to 0 if not set
            "score": result["score"],
            "percent_difference": result["percent_difference"],
            "text": result["text"],
            "relevance": None,  # Optional field, not computed here
            "word_count": None  # Optional field, not computed here
        }
        for result in fused_dict.values()
    ]

    return final_results


def fuse_all_results(
    results: List[Dict[str, Any]],
    method: str = "average"
) -> List[SimilarityResult]:
    """
    Fuses similarity results into a single sorted list with one result per text.

    First, averages scores for each text and query across models.
    Then, fuses the query-specific scores for each text using the specified method.

    Args:
        results: List of result dictionaries with id, query, text, and score.
        method: Fusion method ('average', 'max', or 'min').

    Returns:
        List of SimilarityResult, sorted by score with ranks and percent_difference.

    Raises:
        ValueError: If an unsupported fusion method is provided.
    """
    # Step 1: Average scores for each (id, query, text) across models
    query_text_data = defaultdict(lambda: {"scores": [], "text": None})

    for result in results:
        key = (result["id"], result["query"], result["text"])
        query_text_data[key]["scores"].append(result["score"])
        query_text_data[key]["text"] = result["text"]

    query_text_averages = {
        key: {
            "text": data["text"],
            "score": float(sum(data["scores"]) / len(data["scores"]))
        }
        for key, data in query_text_data.items()
    }

    # Step 2: Collect query-specific scores for each (id, text)
    text_data = defaultdict(lambda: {"scores": [], "text": None})

    for (id_, query, text), data in query_text_averages.items():
        text_key = (id_, text)
        text_data[text_key]["scores"].append(data["score"])
        text_data[text_key]["text"] = text

    # Create fused results
    fused_scores = []
    if method == "average":
        fused_scores = [
            {
                "id": key[0],
                "rank": None,
                "score": float(sum(data["scores"]) / len(data["scores"])),
                "percent_difference": None,
                "text": key[1]
            }
            for key, data in text_data.items()
        ]
    elif method == "max":
        fused_scores = [
            {
                "id": key[0],
                "rank": None,
                "score": float(max(data["scores"])),
                "percent_difference": None,
                "text": key[1]
            }
            for key, data in text_data.items()
        ]
    elif method == "min":
        fused_scores = [
            {
                "id": key[0],
                "rank": None,
                "score": float(min(data["scores"])),
                "percent_difference": None,
                "text": key[1]
            }
            for key, data in text_data.items()
        ]
    else:
        raise ValueError(f"Unsupported fusion method: {method}")

    # Sort by score and assign ranks
    sorted_scores = sorted(
        fused_scores, key=lambda x: x["score"], reverse=True)
    for idx, result in enumerate(sorted_scores):
        result["rank"] = idx + 1

    # Calculate percent_difference
    if sorted_scores:
        max_score = sorted_scores[0]["score"]
        if max_score != 0:
            for result in sorted_scores:
                result["percent_difference"] = round(
                    abs(max_score - result["score"]) / max_score * 100, 2
                )
        else:
            for result in sorted_scores:
                result["percent_difference"] = 0.0

    return sorted_scores
