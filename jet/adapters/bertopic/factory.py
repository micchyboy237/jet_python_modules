"""
BERTopic Factory Module

Provides reusable factory functions and classes for BERTopic integration
with llama.cpp embedding servers.

Reuses jet.adapters.llama_cpp.embed_utils for concurrent embedding.

Key components:
- LlamaCppEmbedder: BERTopic-compatible embedder wrapping llama.cpp server
- create_bertopic_embedder: Factory function to create the embedder
- create_topic_model: Factory function to create a configured BERTopic model
- extract_topics: High-level function to extract topics from documents
"""

import logging
import os
from typing import List, Optional, TypedDict

import numpy as np
import pandas as pd
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_DIMS,
    EMBED_MODEL,
)
from jet.adapters.llama_cpp.embed_utils import embed as embed_batch
from jet.adapters.llama_cpp.factory import get_embedding_client
from numpy.typing import NDArray
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from bertopic import BERTopic
from bertopic.backend import BaseEmbedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

DOC_PREFIX = os.environ.get("BERTopic_DOC_PREFIX", "search_document: ")
BATCH_SIZE = int(os.environ.get("BERTopic_BATCH_SIZE", "32"))
MAX_RETRIES = int(os.environ.get("BERTopic_MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Typed Definitions
# ---------------------------------------------------------------------------


class Topic(TypedDict):
    """Structured representation of a BERTopic topic."""

    topic_id: int
    name: str
    keywords: List[str]
    size: int
    representative_doc: str


class TopicExtractionResult(TypedDict):
    """Complete result from topic extraction."""

    topics: List[Topic]
    topic_labels: List[int]
    topic_info: pd.DataFrame
    embeddings: NDArray[np.float32]


# ---------------------------------------------------------------------------
# LlamaCppEmbedder - BERTopic-compatible Embedding Backend
# ---------------------------------------------------------------------------


class LlamaCppEmbedder(BaseEmbedder):
    """
    BERTopic-compatible wrapper around a local llama.cpp OpenAI-compatible
    embeddings endpoint.

    Reuses jet.adapters.llama_cpp.embed_utils for concurrent embedding with:
    - ThreadPoolExecutor-based parallelism
    - Automatic deduplication
    - Progress tracking
    - Batch optimization

    Implements the BaseEmbedder interface required by BERTopic, allowing
    seamless integration with locally hosted embedding models.

    Attributes:
        client: OpenAI client connected to llama.cpp server
        model: Name of the embedding model
        dims: Expected embedding dimensions
    """

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        dims: Optional[int] = None,
        doc_prefix: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        max_workers: Optional[int] = None,
        show_progress: bool = False,
    ):
        """
        Initialize the llama.cpp embedder.

        Args:
            client: OpenAI client (created if not provided)
            model: Model name (defaults to EMBED_MODEL config)
            dims: Embedding dimensions (defaults to EMBED_DIMS config)
            doc_prefix: Task prefix for documents (defaults to DOC_PREFIX)
            batch_size: Batch size for embedding (defaults to BATCH_SIZE)
            max_retries: Max retry attempts (defaults to MAX_RETRIES)
            max_workers: Number of worker threads for concurrent embedding
            show_progress: Whether to show progress bar during embedding
        """
        super().__init__()
        self.client = client or get_embedding_client()
        self.model = model or EMBED_MODEL
        self.dims = dims or EMBED_DIMS
        self.doc_prefix = doc_prefix or DOC_PREFIX
        self.batch_size = batch_size or BATCH_SIZE
        self.max_retries = max_retries or MAX_RETRIES
        self.max_workers = max_workers
        self.show_progress = show_progress

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """
        Embed a list of documents using llama.cpp server with concurrency.

        Reuses jet.adapters.llama_cpp.embed_utils.embed() which provides:
        - Concurrent batch processing via ThreadPoolExecutor
        - Automatic text deduplication
        - Progress tracking with Rich
        - Network RTT optimization

        Args:
            documents: List of text documents to embed
            verbose: Whether to log progress information

        Returns:
            Numpy array of embeddings with shape (n_documents, dims)

        Raises:
            RuntimeError: If embedding fails after max retries
        """
        if not documents:
            return np.array([], dtype=np.float32).reshape(0, self.dims)

        # Apply prefix to all documents
        prefixed_docs = [f"{self.doc_prefix}{doc}" for doc in documents]

        logger.info(
            "Embedding %d documents with prefix '%s'...",
            len(documents),
            self.doc_prefix if self.doc_prefix else "(none)",
        )

        try:
            # Use embed_utils.embed for concurrent processing
            # Note: embed() accepts 'text' parameter (not 'texts')
            embeddings = embed_batch(
                text=prefixed_docs,
                model=self.model,
                max_workers=self.max_workers if self.max_workers is not None else 6,
                show_progress=self.show_progress or verbose,
                return_format="numpy",
                batch_size=self.batch_size,
                progress_description="Embedding documents for BERTopic",
            )

            # Ensure correct shape
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            logger.info(
                "Embedded %d documents -> shape %s",
                len(documents),
                embeddings.shape,
            )

            if embeddings.shape[1] != self.dims:
                logger.warning(
                    "Embedding dim mismatch: server returned %d dims, "
                    "EMBED_DIMS says %d. Check your model/env var.",
                    embeddings.shape[1],
                    self.dims,
                )

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.error("Embedding failed: %s", e)
            raise RuntimeError(f"Failed to embed documents: {e}") from e


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_bertopic_embedder(
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    dims: Optional[int] = None,
    max_workers: Optional[int] = None,
    show_progress: bool = False,
    **kwargs,
) -> LlamaCppEmbedder:
    """
    Create a BERTopic-compatible embedder for llama.cpp server.

    Args:
        client: OpenAI client (auto-created if not provided)
        model: Model name (defaults to EMBED_MODEL env var)
        dims: Embedding dimensions (defaults to EMBED_DIMS env var)
        max_workers: Number of worker threads for concurrent embedding
        show_progress: Whether to show progress bar during embedding
        **kwargs: Additional arguments passed to LlamaCppEmbedder

    Returns:
        Configured LlamaCppEmbedder instance

    Example:
        embedder = create_bertopic_embedder()
        topic_model = BERTopic(embedding_model=embedder)
    """
    return LlamaCppEmbedder(
        client=client or get_embedding_client(),
        model=model or EMBED_MODEL,
        dims=dims or EMBED_DIMS,
        max_workers=max_workers,
        show_progress=show_progress,
        **kwargs,
    )


def create_topic_model(
    embedder: Optional[BaseEmbedder] = None,
    min_topic_size: int = 10,
    top_n_words: int = 5,
    verbose: bool = False,
    use_tfidf: bool = True,
    **kwargs,
) -> BERTopic:
    """
    Create a configured BERTopic model with the llama.cpp embedder.

    Args:
        embedder: Embedding backend (auto-created if not provided)
        min_topic_size: Minimum documents per topic
        top_n_words: Number of keywords per topic
        verbose: Enable BERTopic verbose output
        use_tfidf: Use TfidfVectorizer instead of CountVectorizer for better
                  topic word discrimination (recommended for most use cases)
        **kwargs: Additional BERTopic configuration

    Returns:
        Configured BERTopic model instance

    Example:
        topic_model = create_topic_model(min_topic_size=15)
        topics, embeddings = topic_model.fit_transform(documents)
    """
    if embedder is None:
        embedder = create_bertopic_embedder()

    # Configure vectorizer - TfidfVectorizer generally gives cleaner topics
    vectorizer_model = None
    if use_tfidf:
        vectorizer_model = TfidfVectorizer(
            stop_words="english",  # Remove common English words
            ngram_range=(1, 2),  # Include unigrams and bigrams
            max_features=10000,  # Limit vocabulary size
            sublinear_tf=True,  # Use 1+log(tf) scaling
            min_df=2,  # Ignore terms that appear in < 2 docs
            max_df=0.85,  # Ignore terms that appear in > 85% of docs
        )

    return BERTopic(
        embedding_model=embedder,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        verbose=verbose,
        **kwargs,
    )


def extract_topics(
    documents: List[str],
    embedder: Optional[BaseEmbedder] = None,
    min_topic_size: int = 3,
    top_n_words: int = 5,
    verbose: bool = False,
    max_workers: Optional[int] = None,
    show_progress: bool = False,
) -> TopicExtractionResult:
    """
    Extract topics from documents using BERTopic with llama.cpp embeddings.

    Uses concurrent embedding via jet.adapters.llama_cpp.embed_utils for
    faster processing of large document sets.
    """
    if embedder is None:
        embedder = create_bertopic_embedder(
            max_workers=max_workers,
            show_progress=show_progress,
        )

    topic_model = create_topic_model(
        embedder=embedder,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        verbose=verbose,
    )

    logger.info("Starting topic extraction with %d documents...", len(documents))
    topics, embeddings = topic_model.fit_transform(documents)
    topic_info = topic_model.get_topic_info()

    topics_list: List[Topic] = []

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])

        # Skip outlier topic
        if topic_id == -1:
            continue

        # Get keywords from Representation column
        keywords = row["Representation"]
        if isinstance(keywords, str):
            keywords = [kw.strip() for kw in keywords.split(",")]
        elif isinstance(keywords, list):
            keywords = [str(kw).strip() for kw in keywords]
        else:
            keywords = []

        # Get representative document
        rep_docs = topic_model.get_representative_docs(topic_id)
        rep_doc = rep_docs[0] if rep_docs else ""

        topics_list.append(
            {
                "topic_id": topic_id,
                "name": row.get("Name", f"Topic_{topic_id}"),
                "keywords": keywords,
                "size": int(row["Count"]),
                "representative_doc": rep_doc,
            }
        )

    return {
        "topics": topics_list,
        "topic_labels": [int(t) for t in topics],
        "topic_info": topic_info,
        "embeddings": embeddings,
    }


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def sanity_check_embedder(embedder: Optional[LlamaCppEmbedder] = None) -> bool:
    """
    Verify the embedding server is reachable and working.

    Args:
        embedder: Embedder to test (auto-created if not provided)

    Returns:
        True if check passes

    Raises:
        Exception: If the server is not reachable or misconfigured
    """
    if embedder is None:
        embedder = create_bertopic_embedder()

    logger.info("Running embedding server sanity check...")

    try:
        test_vec = embedder.embed(["connectivity check"], verbose=False)
        logger.info("Sanity check OK: got vector of shape %s", test_vec.shape)
        return True

    except Exception as exc:
        logger.error(
            "Could not reach/embed via %s. Confirm llama-server is running "
            "with --embeddings enabled and reachable from this machine.",
            EMBED_BASE_URL,
        )
        raise
