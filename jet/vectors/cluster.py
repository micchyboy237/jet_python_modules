import json
import os
from typing import List, Optional, Tuple, Union, TypedDict
from jet.file.utils import load_file, save_file
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.metrics import silhouette_score


class ClusterResult(TypedDict):
    text: str
    label: int
    embedding: np.ndarray
    cluster_probability: float
    is_noise: bool
    cluster_size: int


def cluster_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L12-v2",
    batch_size: int = 32,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    reduce_dim: bool = True,
    n_components: int = 10,
    min_cluster_size: int = 5,
    random_state: int = 42
) -> List[ClusterResult]:
    """
    Cluster a list of texts using Sentence Transformers, UMAP, and HDBSCAN.

    Args:
        texts (List[str]): List of texts to cluster.
        model_name (str): Sentence Transformer model name (default: "all-MiniLM-L6-v2").
        batch_size (int): Batch size for encoding (default: 32).
        device (str): Device for Sentence Transformer ("mps" for M1 Mac, "cpu", or "cuda").
        reduce_dim (bool): Whether to apply UMAP dimensionality reduction (default: True).
        n_components (int): Number of dimensions for UMAP reduction (default: 10).
        min_cluster_size (int): Minimum cluster size for HDBSCAN (default: 5).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        List[ClusterResult]: List of dictionaries containing clustering results with:
            - text: Original input text
            - label: Cluster label (-1 for noise)
            - embedding: Text embedding (reduced if reduce_dim=True)
            - cluster_probability: HDBSCAN membership probability
            - is_noise: Whether the point is classified as noise
            - cluster_size: Number of points in the assigned cluster

    Raises:
        ValueError: If texts is empty or invalid parameters are provided.
    """
    if not texts:
        raise ValueError("Input text list cannot be empty.")

    # Step 1: Generate embeddings
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Step 2: Dimensionality reduction (optional)
    if reduce_dim:
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            metric="cosine",
            n_neighbors=15,
            min_dist=0.1
        )
        embeddings = reducer.fit_transform(embeddings)

    # Step 3: Cluster with HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean" if reduce_dim else "cosine",
        cluster_selection_method="eom",
        min_samples=None,
        prediction_data=True
    )
    labels = clusterer.fit_predict(embeddings)
    probabilities = clusterer.probabilities_

    # Step 4: Compute cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))

    # Step 5: Build results
    results: List[ClusterResult] = []
    for i, text in enumerate(texts):
        label = int(labels[i])
        result: ClusterResult = {
            "text": text,
            "label": label,
            "embedding": embeddings[i],
            "cluster_probability": float(probabilities[i]),
            "is_noise": label == -1,
            "cluster_size": int(cluster_sizes.get(label, 0))
        }
        results.append(result)

    return results
