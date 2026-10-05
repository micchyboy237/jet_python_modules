"""
Text clustering using embeddings, UMAP, and HDBSCAN.

Uses llama.cpp embedding backend via jet.adapters.llama_cpp.embed_utils
for consistency with the rest of the codebase.
"""

import warnings
from typing import List, TypedDict

import hdbscan
import numpy as np
import umap
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.logger import logger


class ClusterResult(TypedDict):
    """Result of clustering a single text."""

    text: str
    label: int
    embedding: np.ndarray
    cluster_probability: float
    is_noise: bool
    cluster_size: int


def cluster_texts(
    texts: List[str],
    model_name: str = EMBED_MODEL,
    batch_size: int = 64,
    reduce_dim: bool = True,
    n_components: int = 10,
    min_cluster_size: int = 5,
    random_state: int = 42,
    show_progress: bool = True,
) -> List[ClusterResult]:
    """
    Cluster a list of texts using embeddings, UMAP, and HDBSCAN.

    Uses llama.cpp embedding backend for generating text embeddings.

    Args:
        texts: List of texts to cluster.
        model_name: Embedding model name (default: EMBED_MODEL from config).
            Should be a llama.cpp model key like "nomic-embed:2-moe".
        batch_size: Batch size for embedding API calls (default: 64).
        reduce_dim: Whether to apply UMAP dimensionality reduction (default: True).
        n_components: Number of dimensions for UMAP reduction (default: 10).
        min_cluster_size: Minimum cluster size for HDBSCAN (default: 5).
        random_state: Random seed for reproducibility (default: 42).
        show_progress: Show progress bar during embedding (default: True).

    Returns:
        List of dictionaries containing clustering results with:
            - text: Original input text
            - label: Cluster label (-1 for noise)
            - embedding: Text embedding (reduced if reduce_dim=True)
            - cluster_probability: HDBSCAN membership probability
            - is_noise: Whether the point is classified as noise
            - cluster_size: Number of points in the assigned cluster

    Raises:
        ValueError: If texts is empty.
    """
    if not texts:
        raise ValueError("Input text list cannot be empty.")

    # Generate embeddings using llama.cpp backend
    logger.info(
        f"Generating embeddings for {len(texts)} texts using model: {model_name}"
    )

    embeddings_array = embed(
        text=texts,
        model=model_name,
        return_format="numpy",
        max_workers=6,
        show_progress=show_progress,
        batch_size=batch_size,
        progress_description="Embedding texts for clustering",
    )

    # Ensure 2D array
    if embeddings_array.ndim == 1:
        embeddings_array = embeddings_array.reshape(1, -1)

    logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")

    # Optional dimensionality reduction with UMAP
    if reduce_dim:
        logger.info(
            f"Applying UMAP dimensionality reduction to {n_components} components"
        )

        # Adjust n_neighbors if dataset is smaller than default
        n_neighbors = min(15, len(texts) - 1) if len(texts) > 1 else 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                metric="cosine",
                n_neighbors=n_neighbors,
                min_dist=0.1,
            )
            embeddings_reduced = reducer.fit_transform(embeddings_array)

        logger.info(f"Reduced embeddings shape: {embeddings_reduced.shape}")
    else:
        embeddings_reduced = embeddings_array

    # Cluster with HDBSCAN
    logger.info(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})")

    # Adjust min_cluster_size if dataset is too small
    effective_min_cluster_size = min(min_cluster_size, len(texts))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_cluster_size,
        metric="euclidean" if reduce_dim else "cosine",
        cluster_selection_method="eom",
        min_samples=None,
        prediction_data=True,
    )

    try:
        labels = clusterer.fit_predict(embeddings_reduced)
    except TypeError as e:
        # Handle scikit-learn version incompatibility
        if "force_all_finite" in str(e):
            logger.warning(
                "HDBSCAN compatibility issue detected. "
                "Try upgrading: pip install --upgrade hdbscan scikit-learn"
            )
            raise
        raise

    probabilities = clusterer.probabilities_

    # Calculate cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))

    # Build results
    results: List[ClusterResult] = []
    for i, text in enumerate(texts):
        label = int(labels[i])
        result: ClusterResult = {
            "text": text,
            "label": label,
            "embedding": embeddings_reduced[i],
            "cluster_probability": float(probabilities[i]),
            "is_noise": label == -1,
            "cluster_size": int(cluster_sizes.get(label, 0)),
        }
        results.append(result)

    # Log summary
    noise_count = sum(1 for r in results if r["is_noise"])
    cluster_count = len([l for l in unique_labels if l != -1])
    logger.info(
        f"Clustering complete: {cluster_count} clusters found, "
        f"{noise_count} noise points out of {len(texts)} texts"
    )

    return results


if __name__ == "__main__":
    """Simple demo showing text clustering in action."""
    sample_texts = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "The giant panda is a bear species endemic to China.",
        "Deep learning uses neural networks with many layers.",
        "Pandas eat bamboo and live in mountainous regions.",
        "JavaScript is used for web development.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Natural language processing helps computers understand text.",
        "React is a JavaScript library for building user interfaces.",
        "Computer vision enables machines to interpret visual data.",
    ]

    print("=" * 60)
    print("Text Clustering Demo")
    print("=" * 60)
    print(f"\nInput texts ({len(sample_texts)}):")
    for i, text in enumerate(sample_texts, 1):
        print(f"  {i}. {text}")

    print("\n" + "-" * 60)
    print("Running clustering...")
    print("-" * 60)

    results = cluster_texts(
        texts=sample_texts,
        model_name=EMBED_MODEL,
        batch_size=32,
        reduce_dim=True,
        n_components=5,
        min_cluster_size=2,
        show_progress=True,
    )

    print("\n" + "=" * 60)
    print("Clustering Results")
    print("=" * 60)

    # Group by cluster
    clusters = {}
    for result in results:
        label = result["label"]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(result)

    for label in sorted(clusters.keys()):
        items = clusters[label]
        if label == -1:
            print(f"\n🔇 Noise ({len(items)} items):")
        else:
            print(
                f"\n📦 Cluster {label} ({len(items)} items, "
                f"avg probability: {np.mean([r['cluster_probability'] for r in items]):.2f}):"
            )

        for item in items:
            print(f"  • {item['text'][:80]}...")
            print(f"    Probability: {item['cluster_probability']:.4f}")

    print("\n" + "=" * 60)
    print(
        f"Summary: {len([k for k in clusters.keys() if k != -1])} clusters, "
        f"{len(clusters.get(-1, []))} noise points"
    )
    print("=" * 60)
