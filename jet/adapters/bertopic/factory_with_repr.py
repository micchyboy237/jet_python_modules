"""
BERTopic Factory - Enhanced Representation Module
Extends the base BERTopic factory with improved topic representation models:
- KeyBERTInspired for better topic labeling
- Stop word removal for cleaner keywords
- Bigram support for phrase detection
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
import time
from typing import List, Optional, Tuple, TypedDict

import numpy as np
import pandas as pd
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_DIMS,
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.embed_utils import embed as embed_batch
from jet.adapters.llama_cpp.factory import get_embedding_client
from numpy.typing import NDArray
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.representation import KeyBERTInspired

logger = logging.getLogger(__name__)

QUERY_PREFIX = EMBED_QUERY_PREFIX
DOC_PREFIX = EMBED_DOC_PREFIX
BATCH_SIZE = int(os.environ.get("BERTopic_BATCH_SIZE", "32"))
MAX_RETRIES = int(os.environ.get("BERTopic_MAX_RETRIES", "3"))
MAX_MODEL_TOKENS = int(os.environ.get("LLAMA_CPP_CTX_SIZE", "512"))
SAFETY_MARGIN_TOKENS = int(os.environ.get("BERTopic_SAFETY_MARGIN_TOKENS", "16"))
TOKEN_BUDGET = MAX_MODEL_TOKENS - SAFETY_MARGIN_TOKENS


class Topic(TypedDict):
    """Structured representation of a BERTopic topic."""

    topic_id: int
    name: str
    keywords: List[str]
    size: int
    representative_docs: List[str]


class TopicExtractionResult(TypedDict):
    """Complete result from topic extraction."""

    topics: List[Topic]
    topic_labels: List[int]
    topic_info: pd.DataFrame
    embeddings: NDArray[np.float32]


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
        token_budget: Optional[int] = None,
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
            token_budget: Max tokens per document (defaults to TOKEN_BUDGET)
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
        self.token_budget = token_budget or TOKEN_BUDGET
        self.batch_size = batch_size or BATCH_SIZE
        self.max_retries = max_retries or MAX_RETRIES
        self.max_workers = max_workers
        self.show_progress = show_progress

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """
        Embed a list of documents using llama.cpp server with concurrency.

        Reuses jet.adapters.llama_cpp.embed_utils.embed_batch() which provides:
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
            # Use embed_utils.embed_batch for concurrent processing
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
    remove_stop_words: bool = True,
    use_keybert: bool = True,
    use_tfidf: bool = True,
    verbose: bool = False,
    **kwargs,
) -> BERTopic:
    """
    Create a configured BERTopic model.

    Args:
        embedder: Embedding backend (auto-created if not provided)
        min_topic_size: Minimum documents per topic
        top_n_words: Number of keywords per topic
        remove_stop_words: Remove English stop words for cleaner keywords
        use_keybert: Use KeyBERT-inspired representation for better topics
        use_tfidf: Use TF-IDF vectorizer instead of CountVectorizer
        verbose: Enable progress logging
        **kwargs: Additional arguments passed to BERTopic

    Returns:
        Configured BERTopic model
    """
    if embedder is None:
        embedder = create_bertopic_embedder()

    vectorizer_model = None
    if remove_stop_words:
        if use_tfidf:
            vectorizer_model = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
            )
        else:
            vectorizer_model = CountVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=10000,
            )

    representation_model = None
    if use_keybert:
        representation_model = KeyBERTInspired()

    return BERTopic(
        embedding_model=embedder,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=verbose,
        **kwargs,
    )


def extract_topics(
    documents: List[str],
    embedder: Optional[BaseEmbedder] = None,
    min_topic_size: int = 3,
    top_n_words: int = 5,
    remove_stop_words: bool = True,
    use_keybert: bool = True,
    verbose: bool = False,
    n_representative_docs: Optional[int] = None,
    max_workers: Optional[int] = None,
    show_progress: bool = False,
) -> TopicExtractionResult:
    """
    Extract topics from documents using BERTopic with llama.cpp embeddings.

    This is the main high-level function for topic extraction. It handles
    the complete pipeline: embedding, topic modeling, and result formatting.

    Uses concurrent embedding via jet.adapters.llama_cpp.embed_utils for
    faster processing of large document sets.

    Args:
        documents: List of text documents to analyze
        embedder: Embedding backend (auto-created if not provided)
        min_topic_size: Minimum documents per topic
        top_n_words: Number of keywords per topic
        remove_stop_words: Remove English stop words for cleaner keywords
        use_keybert: Use KeyBERT-inspired representation for better topics
        verbose: Enable progress logging
        n_representative_docs: Max representative docs per topic.
            None (default) returns all available. Set to an int to cap.
        max_workers: Number of worker threads for concurrent embedding
        show_progress: Whether to show progress bar during embedding

    Returns:
        TopicExtractionResult containing structured topic data, sorted by size desc

    Example:
        docs = ["Document one text...", "Document two text..."]
        # Return all representative docs
        result = extract_topics(docs)
        # Return at most 5 representative docs
        result = extract_topics(docs, n_representative_docs=5)
        for topic in result['topics']:
            print(f"{topic['name']}: {topic['representative_docs'][:2]}")
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
        remove_stop_words=remove_stop_words,
        use_keybert=use_keybert,
        verbose=verbose,
    )

    logger.info("Starting topic extraction with %d documents...", len(documents))
    topic_labels, embeddings = topic_model.fit_transform(documents)
    topic_info = topic_model.get_topic_info()

    # Build topic-to-document indices
    topic_doc_indices: dict[int, list[int]] = {}
    for doc_idx, topic_id in enumerate(topic_labels):
        if topic_id == -1:
            continue
        if topic_id not in topic_doc_indices:
            topic_doc_indices[topic_id] = []
        topic_doc_indices[topic_id].append(doc_idx)

    logger.debug(
        "Topic document indices built: %s",
        {k: len(v) for k, v in topic_doc_indices.items()},
    )

    topics_list: List[Topic] = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        # Extract keywords
        keywords = row["Representation"]
        if isinstance(keywords, str):
            keywords = [kw.strip() for kw in keywords.split(",")]
        elif isinstance(keywords, list):
            keywords = [str(kw).strip() for kw in keywords]
        else:
            keywords = []

        # Get representative documents
        doc_indices = topic_doc_indices.get(topic_id, [])
        if doc_indices:
            all_rep_docs = [documents[idx] for idx in doc_indices]

            # Try to sort by probability if available
            try:
                doc_info = topic_model.get_document_info(documents)
                topic_doc_info = doc_info[doc_info["Topic"] == topic_id]
                if "Probability" in topic_doc_info.columns:
                    topic_doc_info = topic_doc_info.sort_values(
                        "Probability", ascending=False
                    )
                    all_rep_docs = topic_doc_info["Document"].tolist()
                    logger.debug(
                        "Topic %d: sorted %d docs by probability scores",
                        topic_id,
                        len(all_rep_docs),
                    )
            except Exception as e:
                logger.debug(
                    "Topic %d: couldn't sort by probability, using assignment order: %s",
                    topic_id,
                    e,
                )

            logger.debug(
                "Topic %d: fetched %d docs from topic assignment",
                topic_id,
                len(all_rep_docs),
            )
        else:
            all_rep_docs = []
            logger.warning(
                "Topic %d: no documents found in topic assignment",
                topic_id,
            )

        # Cap representative docs if requested
        if n_representative_docs is not None:
            rep_docs = all_rep_docs[:n_representative_docs]
            logger.debug(
                "Topic %d: %d docs available, capped to %d",
                topic_id,
                len(all_rep_docs),
                n_representative_docs,
            )
        else:
            rep_docs = all_rep_docs
            logger.debug(
                "Topic %d: returning all %d documents in topic",
                topic_id,
                len(all_rep_docs),
            )

        topics_list.append(
            {
                "topic_id": topic_id,
                "name": row.get("Name", f"Topic_{topic_id}"),
                "keywords": keywords,
                "size": int(row["Count"]),
                "representative_docs": rep_docs,
            }
        )

    # Sort topics by size (descending)
    topics_list.sort(key=lambda t: t["size"], reverse=True)

    logger.info(
        "Topics sorted by size (descending): %s",
        [f"Topic {t['topic_id']} (size={t['size']})" for t in topics_list],
    )

    return {
        "topics": topics_list,
        "topic_labels": [int(t) for t in topic_labels],
        "topic_info": topic_info,
        "embeddings": embeddings,
    }


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


def find_topics(
    topic_model: BERTopic,
    search_term: str,
    top_n: int = 5,
    verbose: bool = True,
) -> Tuple[List[int], List[float]]:
    """
    Find topics most similar to a search term using BERTopic's embedding space.
    Lightweight semantic search over topic centroids.
    """
    start = time.time()
    if not hasattr(topic_model, "find_topics"):
        logger.error(
            "BERTopic model does not support find_topics (embedding_model required)."
        )
        raise AttributeError("Model must be fitted with embedding_model.")

    similar_topics, similarities = topic_model.find_topics(search_term, top_n=top_n)

    if verbose:
        elapsed = time.time() - start
        logger.info(
            "find_topics completed in %.2fs | Query: '%s' | Top %d topics",
            elapsed,
            search_term,
            top_n,
        )
        topic_info = topic_model.get_topic_info()
        for rank, (tid, sim) in enumerate(zip(similar_topics, similarities), start=1):
            if tid == -1:
                logger.info(
                    "  #%d Topic %d (Outliers): similarity=%.4f", rank, tid, sim
                )
                continue
            info_row = topic_info[topic_info["Topic"] == tid]
            name = info_row["Name"].iloc[0] if not info_row.empty else f"Topic_{tid}"
            size = int(info_row["Count"].iloc[0]) if not info_row.empty else 0
            rep_docs = topic_model.get_representative_docs(tid) or []
            logger.info(
                "  #%d Topic %d: '%s' | similarity=%.4f | size=%d docs | representative_samples=%d",
                rank,
                tid,
                name,
                sim,
                size,
                len(rep_docs),
            )

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
    Enhanced topic search: returns DataFrame with similarity, topic name,
    size, top words, representative-doc count, and optional representative
    documents (capped at `max_reps`).
    """
    start = time.time()
    similar_topics, similarities = find_topics(
        topic_model, search_term, top_n=top_n, verbose=False
    )

    topic_info = topic_model.get_topic_info()
    data = []
    for tid, sim in zip(similar_topics, similarities):
        info_row = topic_info[topic_info["Topic"] == tid]
        name = info_row["Name"].iloc[0] if not info_row.empty else f"Topic_{tid}"
        size = int(info_row["Count"].iloc[0]) if not info_row.empty else 0
        all_reps = topic_model.get_representative_docs(tid) or [] if tid != -1 else []

        row = {
            "Topic": tid,
            "Name": name,
            "Similarity": round(sim, 4),
            "Size": size,
            "Top_Words": [w for w, _ in topic_model.get_topic(tid)[:10]]
            if tid != -1
            else [],
            "Representative_Docs_Count": len(all_reps),
        }

        if include_reps and docs is not None:
            row["Representative_Docs"] = all_reps[:max_reps]

        data.append(row)

    df = pd.DataFrame(data)

    if verbose:
        elapsed = time.time() - start
        logger.info(
            "find_topics_with_data completed in %.2fs | Query: '%s' | %d results",
            elapsed,
            search_term,
            len(df),
        )
        for _, r in df.iterrows():
            shown = len(r["Representative_Docs"]) if "Representative_Docs" in r else 0
            logger.info(
                "  Topic %d: '%s' | similarity=%.4f | size=%d docs | keywords=%s | "
                "representative_samples_total=%d (showing %d)",
                r["Topic"],
                r["Name"],
                r["Similarity"],
                r["Size"],
                ", ".join(r["Top_Words"][:5]),
                r["Representative_Docs_Count"],
                shown,
            )

    return df


def explore_hierarchy(
    topic_model: BERTopic,
    docs: List[str],
    use_ctfidf: bool = True,
    linkage: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build topic hierarchy and return merge DataFrame.
    Useful for understanding parent-child topic relationships.
    """
    if not docs:
        logger.error("docs list required for hierarchical_topics.")
        raise ValueError("Provide original documents.")

    start = time.time()
    linkage_func = None
    if linkage:
        from scipy.cluster import hierarchy as sch

        linkage_func = lambda x: sch.linkage(x, linkage, optimal_ordering=True)

    hier_df = topic_model.hierarchical_topics(
        docs, use_ctfidf=use_ctfidf, linkage_function=linkage_func
    )

    if verbose:
        logger.info(
            "explore_hierarchy completed in %.2fs | Merges: %d | use_ctfidf=%s",
            time.time() - start,
            len(hier_df),
            use_ctfidf,
        )
        logger.info(
            "\nTop merges:\n%s",
            hier_df.head(8)[["Parent_ID", "Parent_Name", "Distance"]].to_string(
                index=False
            ),
        )
        try:
            tree_preview = topic_model.get_topic_tree(hier_df)[:600]
            logger.info("\nHierarchy Tree Preview:\n%s...", tree_preview)
        except Exception as e:
            logger.warning("Tree preview failed: %s", e)

    return hier_df
