from typing import Any, Callable, List, Optional, Tuple, TypedDict, Union

import nltk
import numpy as np
from jet.adapters.llama_cpp.config import DEFAULT_EMBED_MODEL
from jet.logger import logger
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK stopwords
nltk.download("stopwords", quiet=True)


class TopicDistribution(TypedDict):
    """Type for topic-word distributions: topic_id -> list of (word, prob) tuples."""

    __annotations__ = {"topic_id": List[Tuple[str, float]]}


class QueryResult(TypedDict):
    """Type for query results: list of topic_ids and relevance scores."""

    topic_ids: List[int]
    probabilities: List[float]


def extract_topics_without_query(
    docs: List[str],
    embedding_model: Union[
        str, Callable[[List[str], str], np.ndarray]
    ] = DEFAULT_EMBED_MODEL,
    nr_topics: Optional[str] = None,
    min_topic_size: int = 10,
    **kwargs: Any,
) -> Tuple[np.ndarray, TopicDistribution]:
    """
    Extract topics from documents without a query (unsupervised).

    Args:
        docs: List of input documents (strings).
        embedding_model: Embedding model name or callable.
        nr_topics: Number of topics.
        min_topic_size: Min cluster size.
        **kwargs: Passed to BERTopic (e.g., top_k_words).

    Returns:
        Tuple of (topic_assignments: np.ndarray, topics: TopicDistribution).
        topic_assignments[i] is the topic ID for docs[i]; -1 for outliers.

    Example:
        topics, probs = extract_topics_without_query(["doc1 text", "doc2 text"])
    """
    from jet.adapters.bertopic import BERTopic

    if not docs:
        raise ValueError("Documents list cannot be empty.")

    model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        **kwargs,
    )

    topics, probs = model.fit_transform(docs)
    topic_info = model.get_topics()

    return topics, topic_info


def extract_topics_with_query(
    docs: List[str],
    query: str,
    top_k: int = 5,
    embedding_model: Union[
        str, Callable[[List[str], str], np.ndarray]
    ] = DEFAULT_EMBED_MODEL,
    nr_topics: Optional[str] = None,
    min_topic_size: int = 10,
    **kwargs: Any,
) -> Tuple[np.ndarray, QueryResult]:
    """
    Fit model on documents, then find top topics matching the query.

    Args:
        docs: List of input documents.
        query: Search query string (e.g., "climate change").
        top_k: Number of top matching topics to return.
        embedding_model: Embedding model name or callable.
        nr_topics: Number of topics.
        min_topic_size: Min cluster size.
        **kwargs: Passed to BERTopic.

    Returns:
        Tuple of (topic_assignments: np.ndarray, query_result: QueryResult).
        query_result contains top topic_ids and probabilities for the query.

    Example:
        topics, result = extract_topics_with_query(
            ["doc1 text", "doc2 text"], query="environment"
        )
    """
    from jet.adapters.bertopic import BERTopic

    if not docs:
        raise ValueError("Documents list cannot be empty.")
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        **kwargs,
    )

    topics, _ = model.fit_transform(docs)
    similar_topics, similarities = model.find_topics(query, top_n=top_k)

    query_result: QueryResult = {
        "topic_ids": similar_topics,
        "probabilities": similarities,
    }

    return topics, query_result


def get_vectorizer(total_docs: Optional[int] = None) -> CountVectorizer:
    """Create a CountVectorizer with stopword removal and dynamic df thresholds.

    Args:
        total_docs: Total number of documents. If None, uses conservative defaults
                    that still work with very small document sets.

    Returns:
        CountVectorizer: Configured vectorizer with safe min_df and max_df.
    """
    stop_words = list(set(stopwords.words("english")))

    if total_docs is None or total_docs <= 0:
        # Conservative defaults for unknown/small doc counts
        min_df = 1
        max_df = 1.0
    else:
        # min_df: never require more than total_docs, minimum 1
        min_df = min(2, max(1, total_docs - 1))

        # max_df: fraction-based, but always >= min_df to prevent ValueError
        max_df_frac = min(0.95, max(0.5, (total_docs - 1) / total_docs))
        max_df_abs = max(min_df + 1, int(max_df_frac * total_docs))
        max_df = min(1.0, max_df_abs / total_docs)

    logger.info(
        f"Configuring vectorizer with min_df={min_df}, max_df={max_df} "
        f"(total_docs={total_docs})"
    )

    return CountVectorizer(stop_words=stop_words, min_df=min_df, max_df=max_df)
