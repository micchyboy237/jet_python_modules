"""
BERTopic Factory Module

Provides reusable factory functions and classes for BERTopic integration
with llama.cpp embedding servers.

Key components:
- LlamaCppEmbedder: BERTopic-compatible embedder wrapping llama.cpp server
- create_bertopic_embedder: Factory function to create the embedder
- create_topic_model: Factory function to create a configured BERTopic model
- extract_topics: High-level function to extract topics from documents
"""

import logging
import os
import time
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_DIMS,
    EMBED_MODEL,
)
from jet.adapters.llama_cpp.factory import get_embedding_client
from jet.adapters.llama_cpp.token_utils import (
    detokenize,
    tokenize,
)
from numpy.typing import NDArray
from openai import OpenAI

from bertopic import BERTopic
from bertopic.backend import BaseEmbedder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

DOC_PREFIX = os.environ.get("BERTopic_DOC_PREFIX", "search_document: ")
BATCH_SIZE = int(os.environ.get("BERTopic_BATCH_SIZE", "32"))
MAX_RETRIES = int(os.environ.get("BERTopic_MAX_RETRIES", "3"))
MAX_MODEL_TOKENS = int(os.environ.get("LLAMA_CPP_CTX_SIZE", "512"))
SAFETY_MARGIN_TOKENS = int(os.environ.get("BERTopic_SAFETY_MARGIN_TOKENS", "16"))
TOKEN_BUDGET = MAX_MODEL_TOKENS - SAFETY_MARGIN_TOKENS

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
        token_budget: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize the llama.cpp embedder.

        Args:
            client: OpenAI client (created if not provided)
            model: Model name (defaults to EMBED_MODEL config)
            dims: Embedding dimensions (defaults to EMBED_DIMS config)
            doc_prefix: Task prefix for documents (defaults to DOC_PREFIX)
            token_budget: Max tokens per document (defaults to TOKEN_BUDGET)
            batch_size: Batch size for embedding (defaults to BATCH_SIZE)
            max_retries: Max retry attempts (defaults to MAX_RETRIES)
        """
        super().__init__()
        self.client = client or get_embedding_client()
        self.model = model or EMBED_MODEL
        self.dims = dims or EMBED_DIMS
        self.doc_prefix = doc_prefix or DOC_PREFIX
        self.token_budget = token_budget or TOKEN_BUDGET
        self.batch_size = batch_size or BATCH_SIZE
        self.max_retries = max_retries or MAX_RETRIES

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """
        Embed a list of documents using llama.cpp server.

        Args:
            documents: List of text documents to embed
            verbose: Whether to log progress information

        Returns:
            Numpy array of embeddings with shape (n_documents, dims)

        Raises:
            RuntimeError: If embedding fails after max retries
        """
        all_embeddings: List[List[float]] = []
        n_batches = (len(documents) + self.batch_size - 1) // self.batch_size

        # Prepare documents (apply prefix + truncate if needed)
        prepared_all, n_truncated = self._prepare_inputs(documents)

        if n_truncated:
            logger.warning(
                "%d/%d documents exceeded the %d-token model budget and were "
                "truncated (via the server's own tokenizer) before embedding.",
                n_truncated,
                len(documents),
                self.token_budget,
            )

        # Process in batches
        for i in range(0, len(prepared_all), self.batch_size):
            batch_num = i // self.batch_size + 1
            batch = prepared_all[i : i + self.batch_size]

            if verbose:
                logger.info(
                    "Embedding batch %d/%d (%d texts)...",
                    batch_num,
                    n_batches,
                    len(batch),
                )

            all_embeddings.extend(self._embed_batch_with_retry(batch, batch_num))

        embeddings = np.array(all_embeddings, dtype=np.float32)

        logger.info(
            "Embedded %d documents -> shape %s",
            len(documents),
            embeddings.shape,
        )

        # Validate dimensions
        if embeddings.shape[1] != self.dims:
            logger.warning(
                "Embedding dim mismatch: server returned %d dims, "
                "EMBED_DIMS says %d. Check your model/env var.",
                embeddings.shape[1],
                self.dims,
            )

        return embeddings

    def _prepare_inputs(self, documents: List[str]) -> Tuple[List[str], int]:
        """
        Apply task prefix and truncate documents exceeding token budget.

        Uses the llama.cpp server's tokenizer for accurate token counting
        and truncation.

        Args:
            documents: Raw document texts

        Returns:
            Tuple of (prepared_texts, n_truncated)
        """
        prepared = []
        n_truncated = 0

        for doc in documents:
            text = f"{self.doc_prefix}{doc}"
            tokens = tokenize(text)["tokens"]

            if len(tokens) > self.token_budget:
                truncated_tokens = tokens[: self.token_budget]
                text = detokenize(truncated_tokens)["content"]
                n_truncated += 1

            prepared.append(text)

        return prepared, n_truncated

    def _embed_batch_with_retry(
        self, batch: List[str], batch_num: int
    ) -> List[List[float]]:
        """
        Embed a single batch with retry logic.

        Args:
            batch: List of prepared document texts
            batch_num: Batch number for logging

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model,
                    encoding_format="float",
                )
                return [item.embedding for item in response.data]

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Batch %d failed (attempt %d/%d): %s",
                    batch_num,
                    attempt,
                    self.max_retries,
                    exc,
                )
                time.sleep(1.5 * attempt)

        raise RuntimeError(
            f"Batch {batch_num} failed after {self.max_retries} attempts"
        ) from last_error


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_bertopic_embedder(
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    dims: Optional[int] = None,
    **kwargs,
) -> LlamaCppEmbedder:
    """
    Create a BERTopic-compatible embedder for llama.cpp server.

    Args:
        client: OpenAI client (auto-created if not provided)
        model: Model name (defaults to EMBED_MODEL env var)
        dims: Embedding dimensions (defaults to EMBED_DIMS env var)
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
        **kwargs,
    )


def create_topic_model(
    embedder: Optional[BaseEmbedder] = None,
    min_topic_size: int = 10,
    top_n_words: int = 5,
    verbose: bool = False,
    **kwargs,
) -> BERTopic:
    """
    Create a configured BERTopic model with the llama.cpp embedder.

    Args:
        embedder: Embedding backend (auto-created if not provided)
        min_topic_size: Minimum documents per topic
        top_n_words: Number of keywords per topic
        verbose: Enable BERTopic verbose output
        **kwargs: Additional BERTopic configuration

    Returns:
        Configured BERTopic model instance

    Example:
        topic_model = create_topic_model(min_topic_size=15)
        topics, embeddings = topic_model.fit_transform(documents)
    """
    if embedder is None:
        embedder = create_bertopic_embedder()

    return BERTopic(
        embedding_model=embedder,
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
) -> TopicExtractionResult:
    """Extract topics from documents using BERTopic with llama.cpp embeddings."""
    if embedder is None:
        embedder = create_bertopic_embedder()

    topic_model = create_topic_model(
        embedder=embedder,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        verbose=verbose,
    )

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
