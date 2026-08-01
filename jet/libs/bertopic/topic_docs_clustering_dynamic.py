"""
BERTopic Pipeline with Local Embeddings

Provides a reusable function for running BERTopic with pre-computed local embeddings,
configurable UMAP, HDBSCAN, and TfidfVectorizer parameters.

Usage:
    python -m jet.libs.bertopic.bertopic_pipeline
"""

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
        - min_samples: min_cluster_size (HDBSCAN default behavior)
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
        - max_features: scales from 1000 to 10000 based on n_docs
        - min_df: scales from 1 to 3 based on n_docs
        - max_df: scales from 0.95 to 0.8 based on n_docs
    """

    stop_words: Union[str, List[str]]
    ngram_range: Tuple[int, int]
    max_features: Optional[int]
    sublinear_tf: bool
    min_df: Optional[Union[int, float]]
    max_df: Optional[Union[int, float]]
    norm: str


class BERTopicParams(TypedDict, total=False):
    """Parameters for BERTopic model."""

    calculate_probabilities: bool
    min_topic_size: int
    top_n_words: int
    nr_topics: Union[int, str]


class TopicResult(TypedDict):
    """Structured topic results from the pipeline."""

    topic_id: int
    name: str
    keywords: List[str]
    size: int
    documents: List[str]
    keyword_scores: List[Tuple[str, float]]


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
    "max_features": None,  # Dynamic: scales with vocabulary
    "sublinear_tf": True,  # 1 + log(tf) reduces high-freq dominance
    "min_df": None,  # Dynamic: filters rare terms
    "max_df": None,  # Dynamic: filters ubiquitous terms
    "norm": "l2",  # L2 normalization for cosine similarity
}

DEFAULT_BERTOPIC_PARAMS: BERTopicParams = {
    "calculate_probabilities": False,  # False for speed, True for doc-topic probabilities
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
    params = {**DEFAULT_HDBSCAN_PARAMS, **user_params}

    if params["min_cluster_size"] is None:
        if n_docs <= 30:
            params["min_cluster_size"] = 2  # Small corpora need small clusters
        elif n_docs <= 100:
            params["min_cluster_size"] = 2  # Changed from 3 - too large
        elif n_docs <= 500:
            params["min_cluster_size"] = 3
        elif n_docs <= 1000:
            params["min_cluster_size"] = 5
        else:
            params["min_cluster_size"] = 10

    if params["min_samples"] is None:
        params["min_samples"] = params["min_cluster_size"]

    return params


def _calculate_vectorizer_params(
    n_docs: int, user_params: TfidfVectorizerParams
) -> TfidfVectorizerParams:
    params = {**DEFAULT_VECTORIZER_PARAMS, **user_params}

    if params["max_features"] is None:
        if n_docs <= 50:
            params["max_features"] = 3000  # Lower for very small corpora
        elif n_docs <= 100:
            params["max_features"] = 5000
        elif n_docs <= 1000:
            params["max_features"] = 10000
        else:
            params["max_features"] = 20000

    if params["min_df"] is None:
        if n_docs <= 30:
            params["min_df"] = 1  # Keep all terms for small corpora
        elif n_docs <= 100:
            params["min_df"] = 1  # Changed from 2 - too aggressive
        else:
            params["min_df"] = 2  # Only filter for larger corpora

    if params["max_df"] is None:
        if n_docs <= 30:
            params["max_df"] = 1.0  # Don't filter at all
        elif n_docs <= 100:
            params["max_df"] = 0.95  # Changed from 0.85 - too strict
        elif n_docs <= 1000:
            params["max_df"] = 0.9
        else:
            params["max_df"] = 0.85

    return params


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

    Args:
        documents: List of text documents to analyze
        embedding_model: Name of the embedding model (defaults to EMBED_MODEL config)
        umap_params: UMAP configuration. Dynamic params (None by default):
            - n_neighbors: min(15, max(2, n_docs-1))
            - n_components: min(5, max(1, n_docs-1))
            - init: 'random' for n_docs≤30, 'spectral' for larger
        hdbscan_params: HDBSCAN configuration. Dynamic params (None by default):
            - min_cluster_size: 2 for n_docs≤20, scales to 10 for n_docs>1000
            - min_samples: equals min_cluster_size
        vectorizer_params: TfidfVectorizer configuration. Dynamic params (None by default):
            - max_features: 1000 for n_docs≤10, scales to 20000 for n_docs>1000
            - min_df: 1 for n_docs≤10, scales to 3 for n_docs>100
            - max_df: 0.95 for n_docs≤10, scales to 0.7 for n_docs>1000
        bertopic_params: BERTopic configuration parameters
        show_progress: Show progress bar during embedding
        verbose: Print detailed progress and results

    Returns:
        BERTopicPipelineResult with topics, model, and structured results

    Raises:
        ValueError: If documents list is empty

    Example:
        >>> # Minimal usage - all params dynamically scaled
        >>> result = run_bertopic_pipeline(docs)

        >>> # Custom UMAP only - HDBSCAN and vectorizer still dynamic
        >>> result = run_bertopic_pipeline(
        ...     docs,
        ...     umap_params={"n_neighbors": 20, "n_components": 10}
        ... )

        >>> # Override dynamic behavior
        >>> result = run_bertopic_pipeline(
        ...     docs,
        ...     hdbscan_params={"min_cluster_size": 5},  # Force fixed value
        ...     vectorizer_params={"min_df": 2, "max_df": 0.9}  # Force fixed values
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
    # Step 2: Configure Models
    # -----------------------------------------------------------------------
    if verbose:
        print("\n--- Step 2: Configuring BERTopic Pipeline ---")

    # Remove None values and typed_dict-specific keys before model init
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
                print(
                    f"\n⚡ {topic['name']} (Topic {topic['topic_id']}, {topic['size']} docs):"
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
    1. Default dynamic scaling
    2. Custom parameter overrides
    3. Mixed dynamic/custom configurations
    """
    from jet.libs.bertopic.examples.doc_samples import DOCS_LG

    print("=" * 70)
    print("BERTopic Pipeline Demo with Dynamic Parameter Scaling")
    print("=" * 70)

    # Example 1: Fully dynamic (all None defaults)
    print("\n" + "=" * 70)
    print("Example 1: Fully Dynamic Parameters")
    print("=" * 70)

    result = run_bertopic_pipeline(
        documents=DOCS_LG,
        verbose=True,
    )

    print(f"\nFully dynamic pipeline found {len(result['topic_results'])} topics")

    # Example 2: Custom UMAP, dynamic HDBSCAN and vectorizer
    print("\n" + "=" * 70)
    print("Example 2: Custom UMAP, Dynamic HDBSCAN & Vectorizer")
    print("=" * 70)

    custom_result = run_bertopic_pipeline(
        documents=DOCS_LG,
        umap_params={
            "n_neighbors": 10,
            "n_components": 3,
        },
        # HDBSCAN and vectorizer will be dynamically calculated
        verbose=True,
    )

    print(f"\nCustom UMAP pipeline found {len(custom_result['topic_results'])} topics")

    # Example 3: Override dynamic behavior with fixed values
    print("\n" + "=" * 70)
    print("Example 3: Override Dynamic with Fixed Values")
    print("=" * 70)

    fixed_result = run_bertopic_pipeline(
        documents=DOCS_LG,
        hdbscan_params={
            "min_cluster_size": 5,  # Force fixed cluster size
        },
        vectorizer_params={
            "min_df": 2,  # Force fixed min_df
            "max_df": 0.85,  # Force fixed max_df
        },
        bertopic_params={
            "calculate_probabilities": True,
        },
        verbose=True,
    )

    print(f"\nFixed params pipeline found {len(fixed_result['topic_results'])} topics")
