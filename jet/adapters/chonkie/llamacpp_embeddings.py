# jet_python_modules/jet/adapters/chonkie/llamacpp_embeddings.py
"""Chonkie ``BaseEmbeddings`` adapter backed by a llama.cpp OpenAI-compatible server.

This lets any Chonkie component that accepts a ``BaseEmbeddings`` instance
(``SemanticChunker``, ``SDPMChunker``, ``LateChunker``, ``EmbeddingsRefinery``,
handshakes, etc.) use a local or remote llama.cpp embedding server, reusing
the existing pooling / batching / retry logic in
``jet.adapters.llama_cpp.embed_utils`` instead of duplicating it here.
"""

from typing import Any, Optional

import numpy as np
from jet.adapters.llama_cpp.config import (
    EMBED_DIMS,
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.embed_utils import embed as _llamacpp_embed
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.adapters.llama_cpp.token_utils import get_tokenizer as _get_llamacpp_tokenizer
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.logger import logger

from chonkie.embeddings import BaseEmbeddings


class LlamacppEmbeddings(BaseEmbeddings):
    """Chonkie-compatible embeddings using a llama.cpp `/v1/embeddings` server.

    Thin wrapper: all networking, batching, threading and retries are
    delegated to ``jet.adapters.llama_cpp.embed_utils.embed``.
    """

    def __init__(
        self,
        model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
        dimension: Optional[int] = None,
        query_prefix: str = EMBED_QUERY_PREFIX,
        doc_prefix: str = EMBED_DOC_PREFIX,
        max_workers: int = 6,
        batch_size: Optional[int] = 64,
        show_progress: bool = False,
    ) -> None:
        """Initialize the llama.cpp embeddings adapter.

        Args:
            model: llama.cpp embedding model key. Defaults to config.EMBED_MODEL.
            dimension: Known embedding dimension. If ``None``, it is resolved
                lazily (on first access) from the live server, falling back
                to ``config.EMBED_DIMS`` if the server can't be reached.
            query_prefix: Prefix applied by ``embed_query()`` (asymmetric models).
            doc_prefix: Prefix applied by ``embed()`` / ``embed_batch()``.
            max_workers: Thread pool size hint for batch embedding.
            batch_size: Texts per network request when embedding a batch.
            show_progress: Whether to render a progress bar for batches.
        """
        super().__init__()
        self.model = model
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.show_progress = show_progress

        self._dimension: Optional[int] = dimension  # resolved lazily if None
        self._tokenizer: Any = None  # cached lazily

        logger.info(
            f"LlamacppEmbeddings initialized (model={self.model!r}, "
            f"doc_prefix={self.doc_prefix!r}, query_prefix={self.query_prefix!r})"
        )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text as a document (uses ``doc_prefix``)."""
        logger.debug(f"Embedding single text (len={len(text)}) with model={self.model}")
        vector = _llamacpp_embed(
            text,
            model=self.model,
            return_format="numpy",
            prefix=self.doc_prefix or None,
        )
        return np.asarray(vector, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts as documents (uses ``doc_prefix``)."""
        if not texts:
            logger.debug("embed_batch called with empty list, returning []")
            return []
        logger.info(f"Embedding batch of {len(texts)} texts with model={self.model}")
        vectors = _llamacpp_embed(
            texts,
            model=self.model,
            return_format="numpy",
            max_workers=self.max_workers,
            show_progress=self.show_progress,
            batch_size=self.batch_size,
            prefix=self.doc_prefix or None,
        )
        result = [np.asarray(v, dtype=np.float32) for v in vectors]
        logger.debug(f"Finished embedding batch: {len(result)} vectors returned")
        return result

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single text as a query (uses ``query_prefix`` instead of ``doc_prefix``).

        Not part of the ``BaseEmbeddings`` contract — a convenience for
        asymmetric embedding models (e.g. nomic-embed) used outside Chonkie's
        chunkers, such as retrieval-time query embedding.
        """
        logger.debug(f"Embedding query text (len={len(text)}) with model={self.model}")
        vector = _llamacpp_embed(
            text,
            model=self.model,
            return_format="numpy",
            prefix=self.query_prefix or None,
        )
        return np.asarray(vector, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Embedding vector dimension, resolved lazily and cached."""
        if self._dimension is None:
            self._dimension = self._resolve_dimension()
        return self._dimension

    def _resolve_dimension(self) -> int:
        """Try to read the live embedding dimension from the server, else fall back."""
        try:
            info = get_model_ctx_embd_size(self.model)
            dims = info.get("embd_dims", 0)
            if dims:
                logger.info(f"Resolved embedding dimension from server: {dims}")
                return dims
            logger.warning(
                f"Server returned no embd_dims for model={self.model!r}, "
                f"falling back to config EMBED_DIMS={EMBED_DIMS}"
            )
        except Exception as e:
            logger.warning(
                f"Could not resolve embedding dimension from server ({e}); "
                f"falling back to config EMBED_DIMS={EMBED_DIMS}"
            )
        return EMBED_DIMS

    def get_tokenizer(self) -> Any:
        """Return the (lazily loaded, cached) tokenizer for this embedding model."""
        if self._tokenizer is None:
            logger.debug(f"Loading tokenizer for model={self.model}")
            self._tokenizer = _get_llamacpp_tokenizer(self.model)
        return self._tokenizer

    def __repr__(self) -> str:
        """Return a string representation of the LlamacppEmbeddings instance."""
        return (
            f"{self.__class__.__name__}(model={self.model!r}, "
            f"dimension={self._dimension})"
        )
