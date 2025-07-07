from typing import List, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
try:
    from hdbscan import HDBSCAN
except ImportError:
    HDBSCAN = None  # Handle case where hdbscan is not installed


def group_similar_texts_agglomerative(
    texts: List[str],
    threshold: float = 0.7,
    model_name: str = "all-MiniLM-L6-v2",
    embeddings: Optional[List[np.ndarray]] = None,
) -> List[List[str]]:
    """
    Groups similar texts using AgglomerativeClustering based on cosine similarity.

    Args:
        texts (List[str]): List of input texts to be grouped.
        threshold (float): Similarity threshold for clustering. Default is 0.7.
        model_name (str): Sentence transformer model to use if embeddings not provided.
        embeddings (Optional[List[np.ndarray]]): Precomputed embeddings as NumPy arrays.

    Returns:
        List[List[str]]: List of grouped similar texts, with no duplicates.
    """
    if not texts:
        return []

    # Deduplicate texts while preserving order
    seen_texts = {}
    unique_texts = []
    original_texts = []
    for text in texts:
        if text not in seen_texts:
            seen_texts[text] = True
            unique_texts.append(text.lower())
            original_texts.append(text)

    # Load embeddings if not provided
    if not embeddings:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(unique_texts, convert_to_numpy=True)

    # Ensure embeddings is a 2D NumPy array
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim != 2:
        raise ValueError(
            "Embeddings must be a list of 1D NumPy arrays with consistent dimensions")

    # Compute cosine similarity matrix
    norm = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized_embeddings = embeddings_array / np.maximum(norm, 1e-10)
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=1 - threshold
    ).fit(1 - similarity_matrix)

    # Organize texts into clusters
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        clusters.setdefault(label, []).append(original_texts[idx])

    return list(clusters.values())


def group_similar_texts_kmeans(
    texts: List[str],
    n_clusters: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
    embeddings: Optional[List[np.ndarray]] = None,
) -> List[List[str]]:
    """
    Groups similar texts using KMeans clustering based on embeddings.

    Args:
        texts (List[str]): List of input texts to be grouped.
        n_clusters (int): Number of clusters to form. Default is 3.
        model_name (str): Sentence transformer model to use if embeddings not provided.
        embeddings (Optional[List[np.ndarray]]): Precomputed embeddings as NumPy arrays.

    Returns:
        List[List[str]]: List of grouped similar texts, with no duplicates.
    """
    if not texts:
        return []

    # Deduplicate texts
    seen_texts = {}
    unique_texts = []
    original_texts = []
    for text in texts:
        if text not in seen_texts:
            seen_texts[text] = True
            unique_texts.append(text.lower())
            original_texts.append(text)

    # Load embeddings if not provided
    if not embeddings:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(unique_texts, convert_to_numpy=True)

    # Ensure embeddings is a 2D NumPy array
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim != 2:
        raise ValueError(
            "Embeddings must be a list of 1D NumPy arrays with consistent dimensions")

    # Normalize embeddings for cosine similarity
    norm = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized_embeddings = embeddings_array / np.maximum(norm, 1e-10)

    # Perform clustering
    clustering = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    ).fit(normalized_embeddings)

    # Group texts by cluster
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        clusters.setdefault(label, []).append(original_texts[idx])

    return list(clusters.values())


def group_similar_texts_dbscan(
    texts: List[str],
    eps: float = 0.3,
    min_samples: int = 2,
    model_name: str = "all-MiniLM-L6-v2",
    embeddings: Optional[List[np.ndarray]] = None,
) -> List[List[str]]:
    """
    Groups similar texts using DBSCAN clustering based on cosine distance.

    Args:
        texts (List[str]): List of input texts to be grouped.
        eps (float): Maximum distance between points in a cluster. Default is 0.3.
        min_samples (int): Minimum points to form a cluster. Default is 2.
        model_name (str): Sentence transformer model to use if embeddings not provided.
        embeddings (Optional[List[np.ndarray]]): Precomputed embeddings as NumPy arrays.

    Returns:
        List[List[str]]: List of grouped similar texts, with no duplicates.
    """
    if not texts:
        return []

    # Deduplicate texts
    seen_texts = {}
    unique_texts = []
    original_texts = []
    for text in texts:
        if text not in seen_texts:
            seen_texts[text] = True
            unique_texts.append(text.lower())
            original_texts.append(text)

    # Load embeddings if not provided
    if not embeddings:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(unique_texts, convert_to_numpy=True)

    # Ensure embeddings is a 2D NumPy array
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim != 2:
        raise ValueError(
            "Embeddings must be a list of 1D NumPy arrays with consistent dimensions")

    # Normalize embeddings for cosine distance
    norm = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized_embeddings = embeddings_array / np.maximum(norm, 1e-10)

    # Perform clustering
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine"
    ).fit(normalized_embeddings)

    # Group texts by cluster (exclude noise with label -1)
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label != -1:
            clusters.setdefault(label, []).append(original_texts[idx])

    return list(clusters.values())


def group_similar_texts_hdbscan(
    texts: List[str],
    min_cluster_size: int = 2,
    model_name: str = "all-MiniLM-L6-v2",
    embeddings: Optional[List[np.ndarray]] = None,
) -> List[List[str]]:
    """
    Groups similar texts using HDBSCAN clustering based on cosine distance.

    Args:
        texts (List[str]): List of input texts to be grouped.
        min_cluster_size (int): Minimum points to form a cluster. Default is 2.
        model_name (str): Sentence transformer model to use if embeddings not provided.
        embeddings (Optional[List[np.ndarray]]): Precomputed embeddings as NumPy arrays.

    Returns:
        List[List[str]]: List of grouped similar texts, with no duplicates.
    """
    if not texts:
        return []

    if HDBSCAN is None:
        raise ImportError(
            "HDBSCAN is not installed. Please install the hdbscan package.")

    # Deduplicate texts
    seen_texts = {}
    unique_texts = []
    original_texts = []
    for text in texts:
        if text not in seen_texts:
            seen_texts[text] = True
            unique_texts.append(text.lower())
            original_texts.append(text)

    # Load embeddings if not provided
    if not embeddings:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(unique_texts, convert_to_numpy=True)

    # Ensure embeddings is a 2D NumPy array
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim != 2:
        raise ValueError(
            "Embeddings must be a list of 1D NumPy arrays with consistent dimensions")

    # Normalize embeddings for cosine distance
    norm = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized_embeddings = embeddings_array / np.maximum(norm, 1e-10)

    # Perform clustering
    clustering = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="cosine"
    ).fit(normalized_embeddings)

    # Group texts by cluster (exclude noise with label -1)
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label != -1:
            clusters.setdefault(label, []).append(original_texts[idx])

    return list(clusters.values())


def group_similar_texts_spectral(
    texts: List[str],
    n_clusters: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
    embeddings: Optional[List[np.ndarray]] = None,
) -> List[List[str]]:
    """
    Groups similar texts using SpectralClustering based on cosine similarity.

    Args:
        texts (List[str]): List of input texts to be grouped.
        n_clusters (int): Number of clusters to form. Default is 3.
        model_name (str): Sentence transformer model to use if embeddings not provided.
        embeddings (Optional[List[np.ndarray]]): Precomputed embeddings as NumPy arrays.

    Returns:
        List[List[str]]: List of grouped similar texts, with no duplicates.
    """
    if not texts:
        return []

    # Deduplicate texts
    seen_texts = {}
    unique_texts = []
    original_texts = []
    for text in texts:
        if text not in seen_texts:
            seen_texts[text] = True
            unique_texts.append(text.lower())
            original_texts.append(text)

    # Load embeddings if not provided
    if not embeddings:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(unique_texts, convert_to_numpy=True)

    # Ensure embeddings is a 2D NumPy array
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim != 2:
        raise ValueError(
            "Embeddings must be a list of 1D NumPy arrays with consistent dimensions")

    # Normalize embeddings for cosine similarity
    norm = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized_embeddings = embeddings_array / np.maximum(norm, 1e-10)

    # Perform clustering
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="cosine",
        random_state=42
    ).fit(normalized_embeddings)

    # Group texts by cluster
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        clusters.setdefault(label, []).append(original_texts[idx])

    return list(clusters.values())


def group_similar_texts_gmm(
    texts: List[str],
    n_components: int = 3,
    model_name: str = "all-MiniLM-L6-v2",
    embeddings: Optional[List[np.ndarray]] = None,
) -> List[List[str]]:
    """
    Groups similar texts using GaussianMixture clustering based on embeddings.

    Args:
        texts (List[str]): List of input texts to be grouped.
        n_components (int): Number of clusters to form. Default is 3.
        model_name (str): Sentence transformer model to use if embeddings not provided.
        embeddings (Optional[List[np.ndarray]]): Precomputed embeddings as NumPy arrays.

    Returns:
        List[List[str]]: List of grouped similar texts, with no duplicates.
    """
    if not texts:
        return []

    # Deduplicate texts
    seen_texts = {}
    unique_texts = []
    original_texts = []
    for text in texts:
        if text not in seen_texts:
            seen_texts[text] = True
            unique_texts.append(text.lower())
            original_texts.append(text)

    # Load embeddings if not provided
    if not embeddings:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(unique_texts, convert_to_numpy=True)

    # Ensure embeddings is a 2D NumPy array
    embeddings_array = np.array(embeddings)
    if embeddings_array.ndim != 2:
        raise ValueError(
            "Embeddings must be a list of 1D NumPy arrays with consistent dimensions")

    # Normalize embeddings for cosine similarity
    norm = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized_embeddings = embeddings_array / np.maximum(norm, 1e-10)

    # Perform clustering
    clustering = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42
    ).fit(normalized_embeddings)

    # Group texts by predicted cluster
    labels = clustering.predict(normalized_embeddings)
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(original_texts[idx])

    return list(clusters.values())
