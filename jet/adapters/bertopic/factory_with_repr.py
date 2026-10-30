"""
BERTopic Factory - Enhanced Representation Module
Extends the base BERTopic factory with improved topic representation models:
- KeyBERTInspired for better topic labeling
- MaximalMarginalRelevance for diverse keywords
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

import os
import time
from typing import Dict, List, Optional, Tuple, TypedDict

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
from rich.console import Console
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

console = Console()

QUERY_PREFIX = EMBED_QUERY_PREFIX
DOC_PREFIX = EMBED_DOC_PREFIX
BATCH_SIZE = int(os.environ.get("BERTopic_BATCH_SIZE", "32"))
MAX_RETRIES = int(os.environ.get("BERTopic_MAX_RETRIES", "3"))
MAX_MODEL_TOKENS = int(os.environ.get("LLAMA_CPP_CTX_SIZE", "512"))
SAFETY_MARGIN_TOKENS = int(os.environ.get("BERTopic_SAFETY_MARGIN_TOKENS", "16"))
TOKEN_BUDGET = MAX_MODEL_TOKENS - SAFETY_MARGIN_TOKENS


class KeywordScore(TypedDict):
    """A single keyword with its relevance score."""

    word: str
    score: float


class DocScore(TypedDict):
    """A representative document with its probability/similarity score."""

    text: str
    score: float


class Topic(TypedDict):
    """Structured representation of a BERTopic topic with scoring info."""

    topic_id: int
    name: str
    keywords: List[str]
    keyword_scores: List[KeywordScore]
    size: int
    representative_docs: List[str]
    doc_scores: List[DocScore]


class TopicExtractionResult(TypedDict):
    """Complete result from topic extraction with full scoring info."""

    topics: List[Topic]
    topic_labels: List[int]
    topic_info: pd.DataFrame
    embeddings: NDArray[np.float32]
    # Optional: Full probability matrix if calculated
    probabilities: Optional[NDArray[np.float32]]


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

        prefixed_docs = [f"{self.doc_prefix}{doc}" for doc in documents]

        try:
            embeddings = embed_batch(
                text=prefixed_docs,
                model=self.model,
                max_workers=self.max_workers if self.max_workers is not None else 6,
                show_progress=self.show_progress or verbose,
                return_format="numpy",
                batch_size=self.batch_size,
                progress_description="Embedding documents for BERTopic",
            )
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            if embeddings.shape[1] != self.dims:
                console.print(
                    f"[yellow]Warning: Embedding dim mismatch: server returned {embeddings.shape[1]} dims, "
                    f"EMBED_DIMS says {self.dims}. Check your model/env var.[/yellow]"
                )
            return embeddings.astype(np.float32)
        except Exception as e:
            console.print(f"[red]Embedding failed: {e}[/red]")
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
    use_mmr: bool = False,
    mmr_diversity: float = 0.3,
    use_tfidf: bool = True,
    calculate_probabilities: bool = True,
    n_gram_range: Tuple[int, int] = (1, 2),
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
        use_mmr: Use Maximal Marginal Relevance for diverse keywords
        mmr_diversity: Diversity parameter for MMR (0-1)
        use_tfidf: Use TF-IDF vectorizer instead of CountVectorizer
        calculate_probabilities: Calculate topic probabilities for each document
        n_gram_range: The n-gram range for the vectorizer
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
                ngram_range=n_gram_range,
                max_features=10000,
                sublinear_tf=True,
            )
        else:
            vectorizer_model = CountVectorizer(
                stop_words="english",
                ngram_range=n_gram_range,
                max_features=10000,
            )

    representation_model = None
    if use_keybert:
        kb_model = KeyBERTInspired(top_n_words=top_n_words)
        if use_mmr:
            # Chain MMR after KeyBERT for diversity
            mmr_model = MaximalMarginalRelevance(
                diversity=mmr_diversity, top_n_words=top_n_words
            )
            representation_model = [kb_model, mmr_model]
        else:
            representation_model = kb_model
    elif use_mmr:
        # MMR usually works best on top of c-TF-IDF or another representation
        mmr_model = MaximalMarginalRelevance(
            diversity=mmr_diversity, top_n_words=top_n_words
        )
        representation_model = mmr_model

    return BERTopic(
        embedding_model=embedder,
        min_topic_size=min_topic_size,
        top_n_words=top_n_words,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=calculate_probabilities,
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
    use_mmr: bool = False,
    mmr_diversity: float = 0.3,
    n_representative_docs: Optional[int] = None,
    max_workers: Optional[int] = None,
    n_gram_range: Tuple[int, int] = (1, 2),
    verbose: bool = False,
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
        use_mmr: Use Maximal Marginal Relevance for diverse keywords
        mmr_diversity: Diversity parameter for MMR (0-1)
        n_representative_docs: Max representative docs per topic.
            None (default) returns all available. Set to an int to cap.
        max_workers: Number of worker threads for concurrent embedding
        n_gram_range: The n-gram range for the vectorizer
        verbose: Enable progress logging
        show_progress: Whether to show progress bar during embedding
    Returns:
        TopicExtractionResult containing structured topic data with scores, sorted by size desc
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
        use_mmr=use_mmr,
        mmr_diversity=mmr_diversity,
        n_gram_range=n_gram_range,
        verbose=verbose,
    )

    console.print("[yellow]Starting topic extraction...[/yellow]")
    topic_labels, probs = topic_model.fit_transform(documents)

    topic_info = topic_model.get_topic_info()

    # Build mapping of topic ID to document indices
    topic_doc_indices: Dict[int, List[int]] = {}
    for doc_idx, topic_id in enumerate(topic_labels):
        if topic_id == -1:
            continue
        if topic_id not in topic_doc_indices:
            topic_doc_indices[topic_id] = []
        topic_doc_indices[topic_id].append(doc_idx)

    topics_list: List[Topic] = []

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        # Get Keywords and Scores
        raw_keywords = topic_model.get_topic(topic_id)
        keywords = []
        keyword_scores = []

        if isinstance(raw_keywords, list):
            for word, score in raw_keywords:
                if word:  # Filter empty strings
                    keywords.append(word)
                    keyword_scores.append({"word": word, "score": float(score)})

        # Get Representative Docs and Scores
        doc_indices = topic_doc_indices.get(topic_id, [])
        rep_docs_text = []
        rep_docs_scores = []

        if doc_indices:
            # Try to get sorted docs by probability if available
            try:
                doc_info_df = topic_model.get_document_info(documents)
                topic_doc_info = doc_info_df[doc_info_df["Topic"] == topic_id]

                if "Probability" in topic_doc_info.columns:
                    topic_doc_info = topic_doc_info.sort_values(
                        "Probability", ascending=False
                    )
                    sorted_docs = topic_doc_info["Document"].tolist()
                    sorted_probs = topic_doc_info["Probability"].tolist()
                else:
                    # Fallback to assignment order
                    sorted_docs = [documents[idx] for idx in doc_indices]
                    sorted_probs = [0.0] * len(sorted_docs)

            except Exception:
                sorted_docs = [documents[idx] for idx in doc_indices]
                sorted_probs = [0.0] * len(sorted_docs)

            # Apply limit if requested
            if n_representative_docs is not None:
                sorted_docs = sorted_docs[:n_representative_docs]
                sorted_probs = sorted_probs[:n_representative_docs]

            rep_docs_text = sorted_docs
            rep_docs_scores = [
                {"text": t, "score": float(s)}
                for t, s in zip(sorted_docs, sorted_probs)
            ]

        topics_list.append(
            {
                "topic_id": topic_id,
                "name": row.get("Name", f"Topic_{topic_id}"),
                "keywords": keywords,
                "keyword_scores": keyword_scores,
                "size": int(row["Count"]),
                "representative_docs": rep_docs_text,
                "doc_scores": rep_docs_scores,
            }
        )

    topics_list.sort(key=lambda t: t["size"], reverse=True)

    console.print(f"[green]Extracted {len(topics_list)} topics.[/green]")

    return {
        "topics": topics_list,
        "topic_labels": [int(t) for t in topic_labels],
        "topic_info": topic_info,
        "embeddings": topic_model.embedding_model.embed(documents)
        if hasattr(topic_model.embedding_model, "embed")
        else np.array([]),
        "probabilities": probs,
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

    console.print("[yellow]Running embedding server sanity check...[/yellow]")
    try:
        test_vec = embedder.embed(["connectivity check"], verbose=False)
        console.print(
            f"[green]Sanity check OK: got vector of shape {test_vec.shape}[/green]"
        )
        return True
    except Exception as exc:
        console.print(
            f"[red]Could not reach/embed via {EMBED_BASE_URL}. Confirm llama-server is running "
            "with --embeddings enabled and reachable from this machine.[/red]"
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
        console.print(
            "[red]BERTopic model does not support find_topics (embedding_model required).[/red]"
        )
        raise AttributeError("Model must be fitted with embedding_model.")

    similar_topics, similarities = topic_model.find_topics(search_term, top_n=top_n)

    if verbose:
        elapsed = time.time() - start
        console.print(
            f"[dim]find_topics completed in {elapsed:.2f}s | Query: '{search_term}' | Top {top_n} topics[/dim]"
        )
        topic_info = topic_model.get_topic_info()
        for rank, (tid, sim) in enumerate(zip(similar_topics, similarities), start=1):
            if tid == -1:
                console.print(f"  #{rank} Topic {tid} (Outliers): similarity={sim:.4f}")
                continue
            info_row = topic_info[topic_info["Topic"] == tid]
            name = info_row["Name"].iloc[0] if not info_row.empty else f"Topic_{tid}"
            size = int(info_row["Count"].iloc[0]) if not info_row.empty else 0
            rep_docs = topic_model.get_representative_docs(tid) or []
            console.print(
                f"  #{rank} Topic {tid}: '{name}' | similarity={sim:.4f} | size={size} docs | representative_samples={len(rep_docs)}"
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
        console.print(
            f"[dim]find_topics_with_data completed in {elapsed:.2f}s | Query: '{search_term}' | {len(df)} results[/dim]"
        )
        for _, r in df.iterrows():
            shown = len(r["Representative_Docs"]) if "Representative_Docs" in r else 0
            console.print(
                f"  Topic {r['Topic']}: '{r['Name']}' | similarity={r['Similarity']:.4f} | size={r['Size']} docs | keywords={', '.join(r['Top_Words'][:5])} | "
                f"representative_samples_total={r['Representative_Docs_Count']} (showing {shown})"
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
        console.print("[red]docs list required for hierarchical_topics.[/red]")
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
        console.print(
            f"[dim]explore_hierarchy completed in {time.time() - start:.2f}s | Merges: {len(hier_df)} | use_ctfidf={use_ctfidf}[/dim]"
        )
        console.print(
            f"\nTop merges:\n{hier_df.head(8)[['Parent_ID', 'Parent_Name', 'Distance']].to_string(index=False)}"
        )
        try:
            tree_preview = topic_model.get_topic_tree(hier_df)[:600]
            console.print(f"\nHierarchy Tree Preview:\n{tree_preview}...")
        except Exception as e:
            console.print(f"[yellow]Tree preview failed: {e}[/yellow]")
    return hier_df
