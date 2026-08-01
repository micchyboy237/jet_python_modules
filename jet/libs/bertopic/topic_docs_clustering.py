"""
BERTopic Pipeline with Local Embeddings

Provides a reusable function for running BERTopic with pre-computed local embeddings,
configurable UMAP, HDBSCAN, and TfidfVectorizer parameters.

Features:
- Probabilistic outlier threshold to filter low-confidence assignments
- Topic size floor to dissolve meaningless micro-clusters
- Topic quality scoring (coherence + keyword diversity)
- Adaptive vectorizer parameters

Usage:
    python -m jet.libs.bertopic.topic_docs_clustering
"""

from collections import Counter
from typing import List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
from jet.libs.bertopic.monkey_patches.add_check_array import init_patch
from numpy.typing import NDArray

init_patch()

from bertopic import BERTopic
from hdbscan import HDBSCAN
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.logger import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

# ---------------------------------------------------------------------------
# Typed Definitions
# ---------------------------------------------------------------------------


class UMAPParams(TypedDict, total=False):
    """Parameters for UMAP dimensionality reduction."""

    n_neighbors: int
    n_components: int
    min_dist: float
    metric: str
    random_state: int
    low_memory: bool
    init: Union[str, NDArray[np.float64]]


class HDBSCANParams(TypedDict, total=False):
    """Parameters for HDBSCAN clustering."""

    min_cluster_size: int
    metric: str
    cluster_selection_method: str
    prediction_data: bool
    min_samples: Optional[int]
    cluster_selection_epsilon: float


class TfidfVectorizerParams(TypedDict, total=False):
    """Parameters for TfidfVectorizer keyword extraction."""

    stop_words: Union[str, List[str]]
    ngram_range: Tuple[int, int]
    max_features: int
    sublinear_tf: bool
    min_df: Union[int, float]
    max_df: Union[int, float]
    norm: str


class BERTopicParams(TypedDict, total=False):
    """Parameters for BERTopic model.

    Extended parameters:
        outlier_threshold: Minimum topic probability to retain assignment.
            Documents below this threshold are reassigned to outliers (-1).
            Requires calculate_probabilities=True. Default: None (disabled).
        min_topic_floor: Minimum number of documents for a topic to be retained.
            Topics smaller than this are dissolved into outliers (-1).
            Default: None (disabled).
    """

    calculate_probabilities: bool
    min_topic_size: int
    top_n_words: int
    nr_topics: Union[int, str]
    outlier_threshold: Optional[float]
    min_topic_floor: Optional[int]


class TopicResult(TypedDict):
    """Structured topic results from the pipeline.

    Quality metrics:
        coherence_score: Mean c-TF-IDF score of top keywords (higher = more coherent).
            None for outliers.
        keyword_diversity: Ratio of unique keywords to total (higher = less redundant).
            None for outliers.
    """

    topic_id: int
    name: str
    keywords: List[str]
    size: int
    documents: List[str]
    keyword_scores: List[Tuple[str, float]]
    coherence_score: Optional[float]
    keyword_diversity: Optional[float]


class BERTopicPipelineResult(TypedDict):
    """Complete results from the BERTopic pipeline."""

    topic_model: BERTopic
    topics: List[int]
    probabilities: Optional[NDArray[np.float64]]
    topic_info: pd.DataFrame
    topic_results: List[TopicResult]
    embeddings: NDArray[np.float32]


# ---------------------------------------------------------------------------
# Default Configurations
# ---------------------------------------------------------------------------

DEFAULT_UMAP_PARAMS: UMAPParams = {
    "n_neighbors": 15,
    "n_components": 5,
    "min_dist": 0.0,
    "metric": "cosine",
    "random_state": 42,
}

DEFAULT_HDBSCAN_PARAMS: HDBSCANParams = {
    "min_cluster_size": 2,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
    "prediction_data": True,
}

DEFAULT_VECTORIZER_PARAMS: TfidfVectorizerParams = {
    "stop_words": "english",
    "ngram_range": (1, 2),
    "max_features": 10000,
    "sublinear_tf": True,
    "min_df": 1,
    "max_df": 0.9,
}

DEFAULT_BERTOPIC_PARAMS: BERTopicParams = {
    "calculate_probabilities": True,  # Enabled to support outlier threshold
    "outlier_threshold": None,  # None = no probability filtering
    "min_topic_floor": None,  # None = keep all topic sizes
}


# ---------------------------------------------------------------------------
# Post-Processing Functions
# ---------------------------------------------------------------------------


def _filter_low_confidence_assignments(
    topics: List[int],
    probs: Optional[np.ndarray],
    threshold: Optional[float],
) -> Tuple[List[int], int]:
    """
    Reassign low-confidence topic assignments to outliers (-1).

    Documents whose maximum topic probability falls below the threshold
    are moved to the outlier category. This helps prevent off-topic
    documents from being force-classified into the nearest cluster.

    Args:
        topics: Original topic assignments from BERTopic
        probs: Topic probability matrix of shape (n_docs, n_topics) or
               (n_docs,) for single-topic probabilities
        threshold: Minimum probability to retain a topic assignment.
                   If None, no filtering is applied.

    Returns:
        Tuple of (filtered_topics, n_reassigned)
    """
    if threshold is None or probs is None:
        return topics, 0

    # Get max probability per document
    if probs.ndim > 1:
        max_probs = probs.max(axis=1)
    else:
        max_probs = probs

    filtered = [t if p >= threshold else -1 for t, p in zip(topics, max_probs)]

    n_reassigned = sum(1 for a, b in zip(topics, filtered) if a != b)
    if n_reassigned > 0:
        logger.info(
            "Outlier threshold (%.2f): reassigned %d/%d documents to outliers",
            threshold,
            n_reassigned,
            len(topics),
        )

    return filtered, n_reassigned


def _reassign_tiny_topics(
    topics: List[int],
    min_floor: Optional[int],
) -> Tuple[List[int], int]:
    """
    Reassign documents in topics smaller than min_floor to outliers (-1).

    Micro-clusters with very few documents are often noise or over-fragmentation
    artifacts. Dissolving them into outliers improves topic interpretability.

    Args:
        topics: Original topic assignments
        min_floor: Topics with fewer documents than this are dissolved.
                   If None, no reassignment is performed.

    Returns:
        Tuple of (filtered_topics, n_topics_dissolved)
    """
    if min_floor is None:
        return topics, 0

    topic_counts = Counter(t for t in topics if t != -1)
    small_topics = {t for t, c in topic_counts.items() if c < min_floor}

    if not small_topics:
        return topics, 0

    filtered = [-1 if t in small_topics else t for t in topics]

    n_docs_moved = sum(1 for a, b in zip(topics, filtered) if a != b)
    logger.info(
        "Topic size floor (%d): dissolved %d topic(s) with < %d docs "
        "(%d documents moved to outliers)",
        min_floor,
        len(small_topics),
        min_floor,
        n_docs_moved,
    )

    return filtered, len(small_topics)


def _calculate_topic_quality(
    topic_id: int,
    keywords: List[str],
    keyword_scores: List[Tuple[str, float]],
    documents: List[str],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate quality metrics for a topic.

    Args:
        topic_id: Topic identifier (-1 for outliers)
        keywords: List of top keywords
        keyword_scores: List of (keyword, c-TF-IDF score) tuples
        documents: Documents assigned to this topic

    Returns:
        Tuple of (coherence_score, keyword_diversity).
        Returns (None, None) for outliers.
    """
    if topic_id == -1 or not keywords or not keyword_scores:
        return None, None

    # Coherence: mean of top-N keyword c-TF-IDF scores
    # Higher scores indicate more internally consistent topics
    scores = [s for _, s in keyword_scores[:10]]
    coherence = float(np.mean(scores)) if scores else 0.0

    # Diversity: ratio of unique keyword stems to total keywords
    # Higher values indicate less redundant keyword lists
    unique_keywords = set(k.lower().strip() for k in keywords if k.strip())
    diversity = len(unique_keywords) / max(len(keywords), 1)

    return round(coherence, 6), round(diversity, 4)


# ---------------------------------------------------------------------------
# Reusable Pipeline Function
# ---------------------------------------------------------------------------


def run_bertopic_pipeline(
    documents: List[str],
    embedding_model: Optional[str] = None,
    umap_params: Optional[UMAPParams] = None,
    hdbscan_params: Optional[HDBSCANParams] = None,
    vectorizer_params: Optional[TfidfVectorizerParams] = None,
    bertopic_params: Optional[BERTopicParams] = None,
    show_progress: bool = True,
    verbose: bool = True,
) -> BERTopicPipelineResult:
    """
    Run BERTopic pipeline with pre-computed local embeddings.

    This function handles the complete topic modeling workflow:
    1. Generate embeddings using llama.cpp
    2. Configure UMAP dimensionality reduction
    3. Configure HDBSCAN clustering
    4. Configure TF-IDF vectorization for keyword extraction
    5. Fit BERTopic model
    6. Apply post-processing (outlier threshold, topic size floor)
    7. Calculate topic quality metrics
    8. Extract and structure results

    Args:
        documents: List of text documents to analyze
        embedding_model: Name of the embedding model (defaults to EMBED_MODEL config)
        umap_params: UMAP configuration parameters
        hdbscan_params: HDBSCAN configuration parameters
        vectorizer_params: TfidfVectorizer configuration parameters
        bertopic_params: BERTopic configuration parameters including:
            - outlier_threshold: Min probability to stay in a topic (default: None)
            - min_topic_floor: Min docs for a topic to be retained (default: None)
        show_progress: Show progress bar during embedding
        verbose: Print detailed progress and results

    Returns:
        BERTopicPipelineResult containing:
            - topic_model: Fitted BERTopic model
            - topics: Topic assignments for each document (post-processed)
            - probabilities: Topic probabilities (if calculated)
            - topic_info: DataFrame with topic summary
            - topic_results: List of structured topic results with quality scores
            - embeddings: Document embeddings matrix

    Raises:
        ValueError: If documents list is empty
        RuntimeError: If embedding or model fitting fails

    Example:
        >>> docs = ["Document one text...", "Document two text..."]
        >>> # Basic usage
        >>> result = run_bertopic_pipeline(docs)
        >>>
        >>> # With outlier threshold and topic floor
        >>> result = run_bertopic_pipeline(
        ...     docs,
        ...     bertopic_params={
        ...         "outlier_threshold": 0.3,
        ...         "min_topic_floor": 3,
        ...     }
        ... )
        >>>
        >>> # Inspect quality scores
        >>> for topic in result['topic_results']:
        ...     if topic['topic_id'] != -1:
        ...         print(f"{topic['name']}: coherence={topic['coherence_score']:.4f}")
    """
    if not documents:
        raise ValueError("No documents provided for topic modeling.")

    # Merge with defaults
    umap_cfg = {**DEFAULT_UMAP_PARAMS, **(umap_params or {})}
    hdbscan_cfg = {**DEFAULT_HDBSCAN_PARAMS, **(hdbscan_params or {})}
    vectorizer_cfg = {**DEFAULT_VECTORIZER_PARAMS, **(vectorizer_params or {})}
    bertopic_cfg = {**DEFAULT_BERTOPIC_PARAMS, **(bertopic_params or {})}

    # Extract post-processing params before passing to BERTopic
    outlier_threshold = bertopic_cfg.pop("outlier_threshold", None)
    min_topic_floor = bertopic_cfg.pop("min_topic_floor", None)

    # Get embedding model
    target_model = embedding_model or EMBED_MODEL

    # -----------------------------------------------------------------------
    # Step 1: Generate Local Embeddings
    # -----------------------------------------------------------------------
    if verbose:
        print("--- Step 1: Generating Local Embeddings ---")
        logger.info(
            f"Encoding {len(documents)} documents using local model: {target_model}"
        )

    embeddings = embed(
        text=documents,
        model=target_model,
        return_format="numpy",
        show_progress=show_progress,
    )

    if verbose:
        print(f"Generated embedding matrix shape: {embeddings.shape}")

    # -----------------------------------------------------------------------
    # Step 2: Configure BERTopic Pipeline
    # -----------------------------------------------------------------------
    if verbose:
        print("\n--- Step 2: Configuring BERTopic Pipeline ---")

    # Step 2a: Dimension reduction
    umap_model = UMAP(**umap_cfg)
    if verbose:
        print(
            f"UMAP: n_neighbors={umap_cfg['n_neighbors']}, "
            f"n_components={umap_cfg['n_components']}, "
            f"metric={umap_cfg['metric']}"
        )

    # Step 2b: Density-based clustering
    hdbscan_model = HDBSCAN(**hdbscan_cfg)
    if verbose:
        print(
            f"HDBSCAN: min_cluster_size={hdbscan_cfg['min_cluster_size']}, "
            f"metric={hdbscan_cfg['metric']}, "
            f"selection_method={hdbscan_cfg['cluster_selection_method']}"
        )

    # Step 2c: TF-IDF tokenization for keywords
    vectorizer_model = TfidfVectorizer(**vectorizer_cfg)
    if verbose:
        print(
            f"TfidfVectorizer: ngram_range={vectorizer_cfg['ngram_range']}, "
            f"sublinear_tf={vectorizer_cfg['sublinear_tf']}, "
            f"min_df={vectorizer_cfg['min_df']}, "
            f"max_df={vectorizer_cfg['max_df']}"
        )

    # -----------------------------------------------------------------------
    # Step 3: Fit Topic Model
    # -----------------------------------------------------------------------
    if verbose:
        print("\n--- Step 3: Fitting Topic Model ---")

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        **bertopic_cfg,
    )

    topics, probs = topic_model.fit_transform(documents, embeddings)

    if verbose:
        print(
            f"Discovered {len(set(topics))} unique topics "
            f"(including outliers: {-1 in set(topics)})"
        )

    # -----------------------------------------------------------------------
    # Step 3b: Post-Processing
    # -----------------------------------------------------------------------
    n_post_filtered = 0

    # Apply outlier probability threshold
    if outlier_threshold is not None:
        if verbose:
            print(f"\n--- Post-Processing: Outlier Threshold = {outlier_threshold} ---")
        topics, n_reassigned = _filter_low_confidence_assignments(
            topics, probs, outlier_threshold
        )
        n_post_filtered += n_reassigned

    # Apply topic size floor
    if min_topic_floor is not None:
        if verbose:
            print(f"\n--- Post-Processing: Topic Size Floor = {min_topic_floor} ---")
        topics, n_dissolved = _reassign_tiny_topics(topics, min_topic_floor)
        n_post_filtered += n_dissolved

    if verbose and n_post_filtered > 0:
        new_topic_count = len(set(topics))
        print(
            f"Post-processing complete: {new_topic_count} topics remain "
            f"({n_post_filtered} topics dissolved/filtered)"
        )

    # -----------------------------------------------------------------------
    # Step 4: Extract Topic Info
    # -----------------------------------------------------------------------
    if verbose:
        print("\n--- Step 4: Extracting Topic Information ---")

    topic_info = topic_model.get_topic_info()

    if verbose:
        print(topic_info[["Topic", "Count", "Name"]].to_string(index=False))

    # -----------------------------------------------------------------------
    # Step 5: Build Structured Results
    # -----------------------------------------------------------------------
    if verbose:
        print("\n--- Step 5: Building Structured Results ---")

    topic_results: List[TopicResult] = []

    for topic_id in sorted(set(topics)):
        # Get keywords with scores
        topic_keywords = topic_model.get_topic(topic_id)
        if topic_keywords:
            keywords = [word for word, _ in topic_keywords]
            keyword_scores = topic_keywords
        else:
            keywords = []
            keyword_scores = []

        # Get documents for this topic
        cluster_docs = [doc for doc, t in zip(documents, topics) if t == topic_id]

        # Calculate quality metrics
        coherence, diversity = _calculate_topic_quality(
            topic_id, keywords, keyword_scores, cluster_docs
        )

        # Get topic name from topic_info
        if topic_id == -1:
            name = "Outliers"
        else:
            info_row = topic_info[topic_info["Topic"] == topic_id]
            name = (
                info_row["Name"].iloc[0] if not info_row.empty else f"Topic_{topic_id}"
            )

        topic_results.append(
            {
                "topic_id": topic_id,
                "name": name,
                "keywords": keywords,
                "size": len(cluster_docs),
                "documents": cluster_docs,
                "keyword_scores": keyword_scores,
                "coherence_score": coherence,
                "keyword_diversity": diversity,
            }
        )

    # Sort by size (descending), but keep outliers last
    outliers = [t for t in topic_results if t["topic_id"] == -1]
    regular_topics = sorted(
        [t for t in topic_results if t["topic_id"] != -1],
        key=lambda x: x["size"],
        reverse=True,
    )
    topic_results = regular_topics + outliers

    if verbose:
        for topic in topic_results:
            if topic["topic_id"] == -1:
                print(f"\n❌ Outliers / Unclustered Docs ({topic['size']} docs):")
            else:
                quality_str = ""
                if topic["coherence_score"] is not None:
                    quality_str = (
                        f" | coherence={topic['coherence_score']:.4f}, "
                        f"diversity={topic['keyword_diversity']:.2f}"
                    )
                print(
                    f"\n⚡ {topic['name']} (Topic {topic['topic_id']}, "
                    f"{topic['size']} docs{quality_str}):"
                )
            print(f"   Keywords: {', '.join(topic['keywords'][:10])}")
            for doc in topic["documents"][:3]:
                print(f"   - {doc[:200]}{'...' if len(doc) > 200 else ''}")
            if len(topic["documents"]) > 3:
                print(f"   ... and {len(topic['documents']) - 3} more documents")

    return {
        "topic_model": topic_model,
        "topics": [int(t) for t in topics],
        "probabilities": probs,
        "topic_info": topic_info,
        "topic_results": topic_results,
        "embeddings": embeddings,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Demo: Run BERTopic pipeline with sample documents.

    Demonstrates:
    1. Default parameters with quality scoring
    2. Outlier threshold to filter low-confidence assignments
    3. Topic size floor to dissolve micro-clusters
    """
    from jet.libs.bertopic.examples.doc_samples import DOCS_LG

    print("=" * 70)
    print("BERTopic Pipeline Demo with Quality Scoring & Post-Processing")
    print("=" * 70)

    # Example 1: Default parameters with quality scoring
    print("\n" + "=" * 70)
    print("Example 1: Default Parameters with Quality Metrics")
    print("=" * 70)

    result = run_bertopic_pipeline(
        documents=DOCS_LG,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("Quality Summary:")
    for topic in result["topic_results"]:
        if topic["topic_id"] != -1:
            print(
                f"  {topic['name']}: "
                f"coherence={topic['coherence_score']:.4f}, "
                f"diversity={topic['keyword_diversity']:.2f}, "
                f"size={topic['size']}"
            )

    # Example 2: With outlier threshold
    print("\n" + "=" * 70)
    print("Example 2: Outlier Threshold (0.3) + Topic Floor (3)")
    print("=" * 70)

    filtered_result = run_bertopic_pipeline(
        documents=DOCS_LG,
        bertopic_params={
            "outlier_threshold": 0.3,
            "min_topic_floor": 3,
        },
        verbose=True,
    )

    print(
        f"\nFiltered pipeline: {len(filtered_result['topic_results'])} topics "
        f"({sum(1 for t in filtered_result['topic_results'] if t['topic_id'] == -1)} outliers)"
    )
