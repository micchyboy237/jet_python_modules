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
    "calculate_probabilities": False,
}


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
    6. Extract and structure results

    Args:
        documents: List of text documents to analyze
        embedding_model: Name of the embedding model (defaults to EMBED_MODEL config)
        umap_params: UMAP configuration parameters
        hdbscan_params: HDBSCAN configuration parameters
        vectorizer_params: TfidfVectorizer configuration parameters
        bertopic_params: BERTopic configuration parameters
        show_progress: Show progress bar during embedding
        verbose: Print detailed progress and results

    Returns:
        BERTopicPipelineResult containing:
            - topic_model: Fitted BERTopic model
            - topics: Topic assignments for each document
            - probabilities: Topic probabilities (if calculated)
            - topic_info: DataFrame with topic summary
            - topic_results: List of structured topic results
            - embeddings: Document embeddings matrix

    Raises:
        ValueError: If documents list is empty
        RuntimeError: If embedding or model fitting fails

    Example:
        >>> docs = ["Document one text...", "Document two text..."]
        >>> result = run_bertopic_pipeline(docs)
        >>> for topic in result['topic_results']:
        ...     print(f"{topic['name']}: {', '.join(topic['keywords'])}")
    """
    if not documents:
        raise ValueError("No documents provided for topic modeling.")

    # Merge with defaults
    umap_cfg = {**DEFAULT_UMAP_PARAMS, **(umap_params or {})}
    hdbscan_cfg = {**DEFAULT_HDBSCAN_PARAMS, **(hdbscan_params or {})}
    vectorizer_cfg = {**DEFAULT_VECTORIZER_PARAMS, **(vectorizer_params or {})}
    bertopic_cfg = {**DEFAULT_BERTOPIC_PARAMS, **(bertopic_params or {})}

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
                print(
                    f"\n⚡ {topic['name']} (Topic {topic['topic_id']}, {topic['size']} docs):"
                )
            print(f"   Keywords: {', '.join(topic['keywords'][:10])}")
            for doc in topic["documents"][:3]:  # Show first 3 docs
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
    
    Uses a larger document set (DOCS_LG) to demonstrate the full pipeline
    with all default parameters.
    """
    from jet.libs.bertopic.examples.doc_samples import DOCS_LG

    print("=" * 70)
    print("BERTopic Pipeline Demo with Local Embeddings")
    print("=" * 70)

    # Run pipeline with defaults
    result = run_bertopic_pipeline(
        documents=DOCS_LG,
        verbose=True,
    )

    # Access results
    print("\n" + "=" * 70)
    print("Pipeline Complete! Access results via returned dictionary:")
    print(f"  - topic_model: {type(result['topic_model']).__name__}")
    print(f"  - topics: List of {len(result['topics'])} assignments")
    print(f"  - topic_info: DataFrame with {len(result['topic_info'])} rows")
    print(f"  - topic_results: {len(result['topic_results'])} structured topics")
    print(f"  - embeddings: Shape {result['embeddings'].shape}")

    # Example: Custom parameters
    print("\n" + "=" * 70)
    print("Example with Custom Parameters:")
    print("=" * 70)

    custom_result = run_bertopic_pipeline(
        documents=DOCS_LG,
        umap_params={
            "n_neighbors": 10,
            "n_components": 3,
            "min_dist": 0.1,
        },
        hdbscan_params={
            "min_cluster_size": 3,
            "cluster_selection_method": "leaf",
        },
        vectorizer_params={
            "ngram_range": (1, 3),  # Include trigrams
            "min_df": 2,
            "max_df": 0.8,
        },
        bertopic_params={
            "calculate_probabilities": True,
            "min_topic_size": 3,
        },
        verbose=True,
    )

    print(f"\nCustom pipeline found {len(custom_result['topic_results'])} topics")
