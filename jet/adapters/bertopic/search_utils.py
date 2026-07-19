import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from bertopic import BERTopic

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_topics(
    topic_model: BERTopic, search_term: str, top_n: int = 5, verbose: bool = True
) -> Tuple[List[int], List[float]]:
    """
    Find top similar topics to a search term.

    Returns: (similar_topics, similarities)
    """
    if (
        not hasattr(topic_model, "topic_embeddings_")
        or topic_model.topic_embeddings_ is None
    ):
        logger.error("Model not fitted or missing topic embeddings.")
        raise ValueError("Fit the model with an embedding_model first.")

    start = time.time()
    similar_topics, similarities = topic_model.find_topics(search_term, top_n=top_n)

    if verbose:
        logger.info(
            f"find_topics completed in {time.time() - start:.2f}s | Query: '{search_term}'"
        )
        for tid, sim in zip(similar_topics[:3], similarities[:3]):  # Top 3 summary
            logger.info(f"  Topic {tid}: similarity={sim:.4f}")

    return similar_topics, similarities


def find_topics_with_data(
    topic_model: BERTopic,
    search_term: str,
    docs: Optional[List[str]] = None,
    top_n: int = 5,
    include_reps: bool = True,
    max_reps: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enhanced find_topics: returns rich DataFrame with topics, similarity,
    top words, and optional representative docs.
    """
    start = time.time()

    similar_topics, similarities = find_topics(
        topic_model, search_term, top_n=top_n, verbose=False
    )

    data = []
    for tid, sim in zip(similar_topics, similarities):
        row: Dict[str, Any] = {
            "Topic": tid,
            "Similarity": round(sim, 4),
            "Top_Words": [w for w, _ in topic_model.get_topic(tid)[:10]]
            if tid != -1
            else [],
        }

        if include_reps and docs is not None:
            reps = topic_model.get_representative_docs(tid)
            row["Representative_Docs"] = reps[:max_reps] if reps else []

        data.append(row)

    df = pd.DataFrame(data)

    if verbose:
        logger.info(
            f"find_topics_with_data completed in {time.time() - start:.2f}s | "
            f"Query: '{search_term}' | Found {len(df)} topics"
        )
        logger.info(f"\nTop results:\n{df.head(5)}")

    return df


def explore_hierarchy(
    topic_model: BERTopic,
    docs: List[str],
    use_ctfidf: bool = True,
    linkage: Optional[str] = None,  # e.g., 'ward', 'single'
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build and explore topic hierarchy. Returns hierarchical_topics DataFrame.
    """
    if not docs or len(docs) == 0:
        logger.error("docs list is required and cannot be empty.")
        raise ValueError("Provide original documents.")

    start = time.time()

    # Custom linkage if requested
    from scipy.cluster import hierarchy as sch

    linkage_func = None
    if linkage:
        linkage_func = lambda x: sch.linkage(x, linkage, optimal_ordering=True)

    hier_df = topic_model.hierarchical_topics(
        docs, use_ctfidf=use_ctfidf, linkage_function=linkage_func
    )

    if verbose:
        duration = time.time() - start
        logger.info(
            f"explore_hierarchy completed in {duration:.2f}s | "
            f"Merges: {len(hier_df)} | use_ctfidf={use_ctfidf}"
        )
        logger.info(
            f"\nTop hierarchy merges:\n{hier_df.head(8)[['Parent_ID', 'Parent_Name', 'Distance']]}"
        )

        # Print tree summary
        try:
            tree_str = topic_model.get_topic_tree(hier_df)
            logger.info(f"\nTopic Tree Preview (first 500 chars):\n{tree_str[:500]}...")
        except Exception as e:
            logger.warning(f"Tree preview failed: {e}")

    return hier_df
