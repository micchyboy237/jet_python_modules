import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
import faiss
from typing import Dict, List, Tuple, Optional, Union, TypedDict
from dataclasses import dataclass
import pickle
import os
from jet.logger import logger
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.model_types import EmbedModelType
from jet.models.tokenizer.base import get_tokenizer_fn


def select_diverse_texts(
    cluster_embeddings: np.ndarray,
    cluster_texts: List[str],
    initial_text_idx: int,
    diversity_threshold: float = 0.8,
    max_diverse_texts: Optional[int] = None
) -> List[str]:
    """Select a diverse subset of texts from a cluster based on embedding similarity.

    Args:
        cluster_embeddings: Array of shape (n_texts, embedding_dim) with embeddings for texts.
        cluster_texts: List of texts in the cluster.
        initial_text_idx: Index of the initial text to include (e.g., most similar to centroid).
        diversity_threshold: Maximum cosine similarity for texts to be considered diverse.
        max_diverse_texts: Maximum number of diverse texts to return. If None, defaults to min(3, len(cluster_texts)).

    Returns:
        List of diverse texts, starting with the text at initial_text_idx.
    """
    logger.debug(
        f"Input embeddings shape: {cluster_embeddings.shape}, dtype: {cluster_embeddings.dtype}")
    logger.debug(f"Input texts: {cluster_texts}")
    logger.debug(
        f"Initial text index: {initial_text_idx}, threshold: {diversity_threshold}, max_diverse_texts: {max_diverse_texts}")

    if len(cluster_texts) == 0 or len(cluster_texts) != cluster_embeddings.shape[0]:
        logger.debug(
            "Empty texts or mismatched embeddings, returning empty list")
        return []
    if initial_text_idx < 0 or initial_text_idx >= len(cluster_texts):
        logger.debug("Invalid initial index, returning empty list")
        return []

    # Set max_diverse_texts dynamically if not provided
    active_max_diverse_texts = min(
        3, len(cluster_texts)) if max_diverse_texts is None else max_diverse_texts
    logger.debug(f"Active max_diverse_texts: {active_max_diverse_texts}")

    diverse_texts = [cluster_texts[initial_text_idx]]
    remaining_indices = [i for i in range(
        len(cluster_texts)) if i != initial_text_idx]
    logger.debug(
        f"Starting with text: {diverse_texts[0]}, remaining indices: {remaining_indices}")

    for i in remaining_indices:
        is_diverse = True
        curr_embedding = cluster_embeddings[i].reshape(1, -1)
        logger.debug(f"Checking text {i}: {cluster_texts[i]}")
        for selected_text in diverse_texts:
            selected_idx = cluster_texts.index(selected_text)
            selected_embedding = cluster_embeddings[selected_idx].reshape(
                1, -1)
            similarity = np.dot(
                curr_embedding, selected_embedding.T).flatten()[0]
            logger.debug(
                f"Similarity between text {i} and selected text {selected_idx} ({selected_text}): {similarity}")
            if similarity > diversity_threshold:
                is_diverse = False
                logger.debug(
                    f"Text {i} rejected: similarity {similarity} > {diversity_threshold}")
                break
        if is_diverse:
            diverse_texts.append(cluster_texts[i])
            logger.debug(f"Text {i} added: {cluster_texts[i]}")
        if len(diverse_texts) >= active_max_diverse_texts:
            logger.debug(
                f"Reached max_diverse_texts ({active_max_diverse_texts}), stopping")
            break

    logger.debug(f"Final diverse texts: {diverse_texts}")
    return diverse_texts
