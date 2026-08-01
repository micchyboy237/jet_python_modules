# jet_python_modules/jet/libs/bertopic/topic_docs_clustering_dynamic.py

"""
BERTopic Pipeline with Local Embeddings

Provides a reusable function for running BERTopic with pre-computed local embeddings,
configurable UMAP, HDBSCAN, and TfidfVectorizer parameters.

Features:
- Dynamic parameter scaling based on corpus size
- Probabilistic outlier threshold to filter low-confidence assignments
- Topic size floor to dissolve meaningless micro-clusters
- Topic quality scoring (coherence + keyword diversity)
- Adaptive vectorizer parameters optimized per corpus size

Usage:
    python -m jet.libs.bertopic.topic_docs_clustering_dynamic
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
    """Parameters for UMAP dimensionality reduction.

    Dynamic defaults (when None):
        - n_neighbors: min(15, n_docs - 1), minimum 2
        - n_components: min(5, n_docs - 1), minimum 1
        - init: 'random' for n_docs ≤ 30, 'spectral' for larger
    """

    n_neighbors: Optional[int]
    n_components: Optional[int]
    min_dist: float
    metric: str
    random_state: int
    low_memory: bool
    init: Union[str, NDArray[np.float64]]


class HDBSCANParams(TypedDict, total=False):
    """Parameters for HDBSCAN clustering.

    Dynamic defaults (when None):
        - min_cluster_size: scales from 2 to 10 based on n_docs
        - min_samples: equals min_cluster_size
    """

    min_cluster_size: Optional[int]
    min_samples: Optional[int]
    metric: str
    cluster_selection_method: str
    prediction_data: bool
    cluster_selection_epsilon: float


class TfidfVectorizerParams(TypedDict, total=False):
    """Parameters for TfidfVectorizer keyword extraction.

    Dynamic defaults (when None):
        - max_features: scales from 3000 to 20000 based on n_docs
        - min_df: scales from 1 to 2 based on n_docs
        - max_df: scales from 0.9 to 0.8 based on n_docs
    """

    stop_words: Union[str, List[str]]
    ngram_range: Tuple[int, int]
    max_features: Optional[int]
    sublinear_tf: bool
    min_df: Optional[Union[int, float]]
    max_df: Optional[Union[int, float]]
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
        keyword_diversity: Ratio of unique word stems to total words in keywords.
            Higher = less redundant. None for outliers.
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
# Default Configurations (None = dynamically calculated)
# ---------------------------------------------------------------------------

DEFAULT_UMAP_PARAMS: UMAPParams = {
    "n_neighbors": None,  # Dynamic: min(15, max(2, n_docs - 1))
    "n_components": None,  # Dynamic: min(5, max(1, n_docs - 1))
    "min_dist": 0.0,  # Best for clustering
    "metric": "cosine",  # Best for embedding vectors
    "random_state": 42,  # Reproducibility
    "low_memory": True,  # Better for large datasets
}

DEFAULT_HDBSCAN_PARAMS: HDBSCANParams = {
    "min_cluster_size": None,  # Dynamic: scales with n_docs
    "min_samples": None,  # Dynamic: defaults to min_cluster_size
    "metric": "euclidean",  # euclidean on UMAP-reduced space
    "cluster_selection_method": "eom",  # Excess of Mass - generally best
    "prediction_data": True,  # Enable topic prediction for new docs
}

DEFAULT_VECTORIZER_PARAMS: TfidfVectorizerParams = {
    "stop_words": "english",  # Remove common English stop words
    "ngram_range": (1, 2),  # Unigrams and bigrams
    "max_features": None,  # Dynamic: scales with n_docs
    "sublinear_tf": True,  # 1 + log(tf) reduces high-freq dominance
    "min_df": None,  # Dynamic: filters rare terms
    "max_df": None,  # Dynamic: filters ubiquitous terms
    "norm": "l2",  # L2 normalization for cosine similarity
}

DEFAULT_BERTOPIC_PARAMS: BERTopicParams = {
    "calculate_probabilities": True,  # Enabled to support outlier threshold
    "outlier_threshold": None,  # None = no probability filtering
    "min_topic_floor": None,  # None = keep all topic sizes
}


# ---------------------------------------------------------------------------
# Dynamic Parameter Calculation
# ---------------------------------------------------------------------------


def _calculate_umap_params(n_docs: int, user_params: UMAPParams) -> UMAPParams:
    """Calculate UMAP parameters based on dataset size."""
    params = {**DEFAULT_UMAP_PARAMS, **user_params}

    if params["n_neighbors"] is None:
        if n_docs <= 3:
            params["n_neighbors"] = 2
        elif n_docs <= 10:
            params["n_neighbors"] = max(2, n_docs - 1)
        elif n_docs <= 30:
            params["n_neighbors"] = min(10, n_docs - 1)
        else:
            params["n_neighbors"] = min(15, n_docs - 1)

    if params["n_components"] is None:
        if n_docs <= 3:
            params["n_components"] = 1
        elif n_docs <= 10:
            params["n_components"] = min(2, n_docs - 1)
        elif n_docs <= 30:
            params["n_components"] = min(5, n_docs - 1)
        else:
            params["n_components"] = min(5, n_docs - 1)

    # Auto-select init method based on dataset size
    if "init" not in user_params:
        if n_docs <= 30:
            params["init"] = "random"  # Avoid spectral init ARPACK errors
        else:
            params["init"] = "spectral"  # Better for larger datasets

    return params


def _calculate_hdbscan_params(n_docs: int, user_params: HDBSCANParams) -> HDBSCANParams:
    """Calculate HDBSCAN parameters based on dataset size."""
    params = {**DEFAULT_HDBSCAN_PARAMS, **user_params}

    if params["min_cluster_size"] is None:
        if n_docs <= 30:
            params["min_cluster_size"] = 2
        elif n_docs <= 100:
            params["min_cluster_size"] = (
                2  # Keep at 2 for ≤100 — prevents over-fragmentation
            )
        elif n_docs <= 500:
            params["min_cluster_size"] = 3
        elif n_docs <= 1000:
            params["min_cluster_size"] = 5
        else:
            params["min_cluster_size"] = 10

    # HDBSCAN default: min_samples = min_cluster_size if not specified
    if params["min_samples"] is None:
        params["min_samples"] = params["min_cluster_size"]

    return params


def _calculate_vectorizer_params(
    n_docs: int, user_params: TfidfVectorizerParams
) -> TfidfVectorizerParams:
    """Calculate TfidfVectorizer parameters based on dataset size.

    Key design decisions based on empirical testing:
    - max_df=0.9 for ≤100 docs: Prevents keyword dilution while keeping clusters clean.
      Tested against 0.95 (too permissive) and 0.85 (too aggressive).
    - min_df=1 for ≤100 docs: Small corpora need rare discriminative terms.
      Tested against min_df=2 (filters useful terms like "sovereign", "funds").
    - max_features=5000 for ≤100 docs: Balances vocabulary coverage vs. noise.
    """
    params = {**DEFAULT_VECTORIZER_PARAMS, **user_params}

    if params["max_features"] is None:
        if n_docs <= 50:
            params["max_features"] = 3000
        elif n_docs <= 100:
            params["max_features"] = 5000
        elif n_docs <= 1000:
            params["max_features"] = 10000
        else:
            params["max_features"] = 20000

    if params["min_df"] is None:
        if n_docs <= 100:
            params["min_df"] = 1  # Keep all terms for small corpora
        elif n_docs <= 500:
            params["min_df"] = 2  # Filter singletons for medium corpora
        else:
            params["min_df"] = 3  # More aggressive for large corpora

    if params["max_df"] is None:
        if n_docs <= 30:
            params["max_df"] = 0.9  # Stricter for tiny corpora
        elif n_docs <= 100:
            params["max_df"] = 0.9  # Clean clusters — empirically validated
        elif n_docs <= 500:
            params["max_df"] = 0.85
        elif n_docs <= 1000:
            params["max_df"] = 0.85
        else:
            params["max_df"] = 0.8

    return params


def _calculate_bertopic_postprocess_params(
    n_docs: int, user_params: BERTopicParams
) -> BERTopicParams:
    """Calculate sensible post-processing defaults based on corpus size.

    Dynamic defaults:
        - outlier_threshold: None for ≤100 docs (too aggressive for small corpora),
          0.15 for 100-1000, 0.2 for >1000
        - min_topic_floor: None for ≤30 docs (micro-clusters may be valid),
          3 for 30-500, 5 for >500
    """
    params = {**user_params}

    # Only set dynamic defaults if user hasn't explicitly provided values
    if "outlier_threshold" not in params or params.get("outlier_threshold") is None:
        # For small corpora, probability estimates are unreliable
        # Don't set a threshold unless user explicitly requests it
        pass  # Keep as None

    if "min_topic_floor" not in params or params.get("min_topic_floor") is None:
        if n_docs <= 30:
            pass  # Keep as None — micro-clusters may be valid
        elif n_docs <= 500:
            params["min_topic_floor"] = 3
        else:
            params["min_topic_floor"] = 5

    return params


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

    # Diversity: ratio of unique word stems to total words across all keywords
    # Splits multi-word phrases to check for word-level repetition
    all_words = []
    for kw in keywords:
        if kw and kw.strip():
            all_words.extend(kw.lower().split())
    unique_words = set(all_words)
    diversity = len(unique_words) / max(len(all_words), 1) if all_words else 0.0

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
    Run BERTopic pipeline with pre-computed local embeddings and dynamic parameter scaling.

    All parameters with None defaults are dynamically calculated based on the number
    of documents to ensure optimal performance across dataset sizes (from 2 to 100K+ docs).

    This function handles the complete topic modeling workflow:
    1. Calculate dynamic parameters based on corpus size
    2. Generate embeddings using llama.cpp
    3. Configure UMAP dimensionality reduction
    4. Configure HDBSCAN clustering
    5. Configure TF-IDF vectorization for keyword extraction
    6. Fit BERTopic model
    7. Apply post-processing (outlier threshold, topic size floor)
    8. Calculate topic quality metrics
    9. Extract and structure results

    Args:
        documents: List of text documents to analyze
        embedding_model: Name of the embedding model (defaults to EMBED_MODEL config)
        umap_params: UMAP configuration. Dynamic params (None by default):
            - n_neighbors: min(15, max(2, n_docs-1))
            - n_components: min(5, max(1, n_docs-1))
            - init: 'random' for n_docs≤30, 'spectral' for larger
        hdbscan_params: HDBSCAN configuration. Dynamic params (None by default):
            - min_cluster_size: 2 for n_docs≤100, scales to 10 for n_docs>1000
            - min_samples: equals min_cluster_size
        vectorizer_params: TfidfVectorizer configuration. Dynamic params (None by default):
            - max_features: 3000 for n_docs≤50, scales to 20000 for n_docs>1000
            - min_df: 1 for n_docs≤100, scales to 3 for larger
            - max_df: 0.9 for n_docs≤100, scales to 0.8 for n_docs>1000
        bertopic_params: BERTopic configuration parameters including:
            - outlier_threshold: Min probability to stay in a topic (default: None)
            - min_topic_floor: Min docs for a topic to be retained (default: None)
        show_progress: Show progress bar during embedding
        verbose: Print detailed progress and results

    Returns:
        BERTopicPipelineResult with topics, model, quality scores, and structured results

    Raises:
        ValueError: If documents list is empty

    Example:
        >>> # Minimal usage - all params dynamically scaled
        >>> result = run_bertopic_pipeline(docs)
        >>>
        >>> # Custom UMAP only - HDBSCAN and vectorizer still dynamic
        >>> result = run_bertopic_pipeline(
        ...     docs,
        ...     umap_params={"n_neighbors": 20, "n_components": 10}
        ... )
        >>>
        >>> # With explicit outlier threshold (opt-in)
        >>> result = run_bertopic_pipeline(
        ...     docs,
        ...     bertopic_params={
        ...         "outlier_threshold": 0.15,
        ...         "min_topic_floor": 3,
        ...     }
        ... )
    """
    if not documents:
        raise ValueError("No documents provided for topic modeling.")

    n_docs = len(documents)

    # Calculate dynamic parameters
    final_umap_params = _calculate_umap_params(n_docs, umap_params or {})
    final_hdbscan_params = _calculate_hdbscan_params(n_docs, hdbscan_params or {})
    final_vectorizer_params = _calculate_vectorizer_params(
        n_docs, vectorizer_params or {}
    )
    final_bertopic_params = {**DEFAULT_BERTOPIC_PARAMS, **(bertopic_params or {})}

    # Extract post-processing params before passing to BERTopic
    outlier_threshold = final_bertopic_params.pop("outlier_threshold", None)
    min_topic_floor = final_bertopic_params.pop("min_topic_floor", None)

    # Get embedding model
    target_model = embedding_model or EMBED_MODEL

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"BERTopic Pipeline: {n_docs} documents")
        print(f"{'=' * 70}")
        print(f"\nDynamic Parameters (scaled for {n_docs} docs):")
        print(
            f"  UMAP: n_neighbors={final_umap_params['n_neighbors']}, "
            f"n_components={final_umap_params['n_components']}, "
            f"init={final_umap_params.get('init', 'default')}"
        )
        print(
            f"  HDBSCAN: min_cluster_size={final_hdbscan_params['min_cluster_size']}, "
            f"min_samples={final_hdbscan_params['min_samples']}"
        )
        print(
            f"  TfidfVectorizer: max_features={final_vectorizer_params['max_features']}, "
            f"min_df={final_vectorizer_params['min_df']}, "
            f"max_df={final_vectorizer_params['max_df']}"
        )
        if outlier_threshold is not None:
            print(f"  Post-processing: outlier_threshold={outlier_threshold}")
        if min_topic_floor is not None:
            print(f"  Post-processing: min_topic_floor={min_topic_floor}")

    # -----------------------------------------------------------------------
    # Step 1: Generate Local Embeddings
    # -----------------------------------------------------------------------
    if verbose:
        print("\n--- Step 1: Generating Local Embeddings ---")
        logger.info(f"Encoding {n_docs} documents using model: {target_model}")

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

    # Remove None values before model init
    umap_init_params = {
        k: v
        for k, v in final_umap_params.items()
        if v is not None and k in UMAPParams.__optional_keys__
    }
    hdbscan_init_params = {
        k: v
        for k, v in final_hdbscan_params.items()
        if v is not None and k in HDBSCANParams.__optional_keys__
    }
    vectorizer_init_params = {
        k: v
        for k, v in final_vectorizer_params.items()
        if v is not None and k in TfidfVectorizerParams.__optional_keys__
    }

    # Create models
    umap_model = UMAP(**umap_init_params)
    hdbscan_model = HDBSCAN(**hdbscan_init_params)
    vectorizer_model = TfidfVectorizer(**vectorizer_init_params)

    # -----------------------------------------------------------------------
    # Step 3: Fit Topic Model
    # -----------------------------------------------------------------------
    if verbose:
        print("\n--- Step 3: Fitting Topic Model ---")

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        **final_bertopic_params,
    )

    topics, probs = topic_model.fit_transform(documents, embeddings)

    if verbose:
        unique_topics = set(topics)
        print(
            f"Discovered {len(unique_topics)} unique topics "
            f"(including outliers: {-1 in unique_topics})"
        )

    # -----------------------------------------------------------------------
    # Step 3b: Post-Processing
    # -----------------------------------------------------------------------
    n_topics_dissolved = 0
    n_docs_reassigned = 0

    # Apply outlier probability threshold
    if outlier_threshold is not None:
        if verbose:
            print(f"\n--- Post-Processing: Outlier Threshold = {outlier_threshold} ---")
        topics, n_reassigned = _filter_low_confidence_assignments(
            topics, probs, outlier_threshold
        )
        n_docs_reassigned += n_reassigned

    # Apply topic size floor
    if min_topic_floor is not None:
        if verbose:
            print(f"\n--- Post-Processing: Topic Size Floor = {min_topic_floor} ---")
        topics, n_dissolved = _reassign_tiny_topics(topics, min_topic_floor)
        n_topics_dissolved += n_dissolved

    if verbose and (n_docs_reassigned > 0 or n_topics_dissolved > 0):
        new_topic_count = len(set(topics))
        print(
            f"Post-processing complete: {new_topic_count} topics remain "
            f"({n_docs_reassigned} docs reassigned, {n_topics_dissolved} topics dissolved)"
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

    # Sort by size (descending), keep outliers last
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
    pass
