# jet_python_modules/jet/adapters/bertopic/embeddings.py
import numpy as np
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.embed_utils import embed_batch
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.logger import logger

from bertopic.backend import BaseEmbedder


class BERTopicLlamacppEmbedder(BaseEmbedder):
    """BERTopic embedder that reuses jet.adapters.llama_cpp.embed_utils.embed_batch
    (instead of a dedicated LlamacppEmbedding client class).

    Overrides `embed_words` and `embed_documents` (instead of the generic
    `embed`) so that query-side text (search terms passed to `find_topics`,
    BERTopic's method="word" path) and document-side text (the corpus being
    fit/transformed, method="document" path) each get the correct prefix,
    matching what retrieval-style embedding models expect
    (e.g. "search_query: " vs "search_document: ").
    """

    def __init__(
        self,
        embedding_model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
        max_workers: int = 6,
        batch_size: int | None = 32,
        query_prefix: str = EMBED_QUERY_PREFIX,
        doc_prefix: str = EMBED_DOC_PREFIX,
    ):
        """Initialize the embedder.

        Args:
            embedding_model: Model key to use. Defaults to config.EMBED_MODEL
                (env var LLAMA_CPP_EMBED_MODEL).
            max_workers: Parallel worker threads used by embed_batch.
            batch_size: Batch size used by embed_batch.
            query_prefix: Prefix prepended to text embedded via
                `embed_words` (the `find_topics` search-term path).
                Defaults to config.EMBED_QUERY_PREFIX (env var
                EMBED_QUERY_PREFIX).
            doc_prefix: Prefix prepended to text embedded via
                `embed_documents` (the corpus fit/transform path).
                Defaults to config.EMBED_DOC_PREFIX (env var
                EMBED_DOC_PREFIX).

        Note:
            The embed server URL comes from config.EMBED_BASE_URL
            (env var LLAMA_CPP_EMBED_URL). embed_utils builds its OpenAI
            client once at import time, so it can't be overridden per
            instance anymore.
        """
        super().__init__()
        self.model = embedding_model
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        logger.info(
            f"BERTopicLlamacppEmbedder ready | model={self.model} "
            f"base_url={EMBED_BASE_URL} "
            f"query_prefix={self.query_prefix!r} doc_prefix={self.doc_prefix!r}"
        )

    def _embed(
        self, texts: list[str], prefix: str, kind: str, verbose: bool
    ) -> np.ndarray:
        """Shared embed path used by both embed_words and embed_documents.

        Args:
            texts: Raw text to embed (prefix not yet applied).
            prefix: Prefix to prepend to every text (may be "").
            kind: "query", "document", or "generic" — used only for logs.
            verbose: Controls the verbosity of the process.

        Returns:
            Embeddings with shape (n, m) where n is len(texts) and m is
            the embedding size.
        """
        if not texts:
            logger.debug(f"No {kind} texts to embed, returning empty array")
            return np.array([])

        prefixed_texts = [f"{prefix}{t}" for t in texts] if prefix else list(texts)

        logger.info(
            f"Embedding {len(texts)} {kind} text(s) with model={self.model} "
            f"prefix={prefix!r}"
        )
        embeddings = embed_batch(
            texts=prefixed_texts,
            model=self.model,
            max_workers=self.max_workers,
            show_progress=verbose,
            return_format="numpy",
            batch_size=self.batch_size,
        )
        logger.debug(f"Produced {kind} embeddings with shape {embeddings.shape}")
        return embeddings

    def embed_words(self, words: list[str], verbose: bool = False) -> np.ndarray:
        """Embed query-side text (e.g. `find_topics` search terms).

        BERTopic reaches this via method="word" in `_extract_embeddings`,
        which is what backs `find_topics(search_term=...)`. Despite the
        name "words", this is the query/search-term path — it gets
        `query_prefix`, not a per-token prefix.

        Args:
            words: Search terms / query strings to be embedded.
            verbose: Controls the verbosity of the process.

        Returns:
            Query embeddings with shape (n, m).
        """
        return self._embed(words, self.query_prefix, "query", verbose)

    def embed_documents(self, document: list[str], verbose: bool = False) -> np.ndarray:
        """Embed corpus documents.

        BERTopic reaches this via method="document" in `_extract_embeddings`,
        which backs `fit`/`fit_transform`/`transform` on the document corpus.

        Args:
            document: Documents to be embedded.
            verbose: Controls the verbosity of the process.

        Returns:
            Document embeddings with shape (n, m).
        """
        return self._embed(document, self.doc_prefix, "document", verbose)

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        """Fallback kept for BaseEmbedder API compatibility (e.g. the
        `hasattr(self.embedding_model, "embed_images")` branch in
        `_extract_embeddings`, which this class does not implement, so it
        will never actually be hit in practice).

        No prefix is applied here since the caller's intent (query vs.
        document) is unknown at this entry point. `embed_words` /
        `embed_documents` cover every normal fit/transform/find_topics flow.

        Args:
            documents: Documents or words to be embedded.
            verbose: Controls the verbosity of the process.

        Returns:
            Embeddings with shape (n, m).
        """
        logger.warning(
            "BERTopicLlamacppEmbedder.embed() called directly with no prefix; "
            "expected embed_words/embed_documents to be used instead."
        )
        return self._embed(documents, "", "generic", verbose)
