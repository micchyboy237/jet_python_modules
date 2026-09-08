"""Chonkie-compatible embeddings adapter for llama.cpp servers.

Bridges jet.adapters.llama_cpp utilities with chonkie.embeddings.BaseEmbeddings,
enabling use of local/remote GGUF embedding models with SemanticChunker,
LateChunker, and AutoEmbeddings.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_DIMS, EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed as jet_embed
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.logger import logger

from chonkie.embeddings import BaseEmbeddings, EmbeddingsRegistry


class LlamacppEmbeddings(BaseEmbeddings):
    """Embeddings handler backed by a llama.cpp OpenAI-compatible server.

    Leverages jet.adapters.llama_cpp.embed_utils for optimized batched embedding
    with thread-pool parallelism, deduplication, and progress reporting.

    Args:
        model: Model identifier (must match a loaded model on the server).
            Defaults to LLAMA_CPP_EMBED_MODEL env var or "nomic-embed:2-moe".
        base_url: Override the embedding server URL. If None, uses the
            LLAMA_CPP_EMBED_URL / LLAMA_CPP_EMBED_HOST env vars.
        dimension: Expected embedding dimension. If None, auto-detected from
            the server's /v1/models metadata, falling back to LLAMA_CPP_EMBED_DIMS.
        prefix: Optional prefix prepended to every text before embedding
            (e.g., "Represent this sentence: ").
        batch_size: Texts per API batch request. Default 64.
        max_workers: Thread pool size for concurrent batch requests. Default 6.
        show_progress: Show Rich progress bar during batch embedding.
        tokenizer_model: HF model ID for the tokenizer. If None, inferred from
            the model key via jet.adapters.llama_cpp.model_utils.
    """

    def __init__(
        self,
        model: str = EMBED_MODEL,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        prefix: Optional[str] = None,
        batch_size: int = 64,
        max_workers: int = 6,
        show_progress: bool = True,
        tokenizer_model: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url
        self.prefix = prefix
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.show_progress = show_progress

        # Resolve dimension: explicit > server metadata > config default
        if dimension is not None:
            self._dimension = dimension
        else:
            self._dimension = self._detect_dimension()

        # Resolve tokenizer
        self._tokenizer_model = tokenizer_model
        self._tokenizer: Any = None

        logger.info(
            f"LlamacppEmbeddings initialized: model={model}, "
            f"dim={self._dimension}, batch_size={batch_size}"
        )

    def _detect_dimension(self) -> int:
        """Try to get embedding dims from server metadata, fall back to config."""
        try:
            info = get_model_ctx_embd_size(self.model, base_url=self.base_url)
            dims = info.get("embd_dims", 0)
            if dims > 0:
                logger.debug(f"Auto-detected embedding dimension: {dims}")
                return dims
        except Exception as e:
            logger.warning(
                f"Could not auto-detect dimensions for '{self.model}': {e}. "
                f"Falling back to config default ({EMBED_DIMS})."
            )
        return EMBED_DIMS

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension

    def get_tokenizer(self) -> Any:
        """Return a HF tokenizer matching the embedding model.

        Lazily loaded and cached. Uses tokenizer_model override if provided,
        otherwise attempts to map the llama.cpp model key to an HF ID.
        """
        if self._tokenizer is None:
            model_id = self._tokenizer_model or self.model
            try:
                self._tokenizer = get_tokenizer(model_id)
                logger.debug(f"Loaded tokenizer for '{model_id}'")
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer for '{model_id}': {e}. "
                    "Token counting may be inaccurate."
                )
                # Return a minimal fallback so Chonkie doesn't crash
                self._tokenizer = _FallbackCharTokenizer()
        return self._tokenizer

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            np.ndarray of shape (dimension,) with dtype float32.
        """
        result = jet_embed(
            text=text,
            model=self.model,
            return_format="numpy",
            prefix=self.prefix,
        )
        # jet_embed returns np.ndarray for str input with return_format="numpy"
        if isinstance(result, np.ndarray):
            return result.astype(np.float32)
        return np.array(result, dtype=np.float32)

    async def aembed(self, text: str) -> np.ndarray:
        """Async wrapper around embed()."""
        return await asyncio.to_thread(self.embed, text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts using optimized batched API calls.

        Delegates to jet.adapters.llama_cpp.embed_utils.embed which handles:
        - ThreadPoolExecutor parallelism
        - Per-request batching (batch_size)
        - Duplicate text deduplication
        - Rich progress reporting
        - Server connectivity verification

        Args:
            texts: List of text strings to embed.

        Returns:
            List of np.ndarray vectors, one per input text.
        """
        if not texts:
            return []

        result = jet_embed(
            text=texts,
            model=self.model,
            return_format="numpy",
            max_workers=self.max_workers,
            show_progress=self.show_progress,
            batch_size=self.batch_size,
            prefix=self.prefix,
        )

        # jet_embed returns np.ndarray of shape (N, dim) for list input
        if isinstance(result, np.ndarray):
            return [result[i].astype(np.float32) for i in range(len(result))]

        # Fallback: list of lists
        return [np.array(r, dtype=np.float32) for r in result]

    async def aembed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Async wrapper around embed_batch()."""
        return await asyncio.to_thread(self.embed_batch, texts)

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        tokenizer = self.get_tokenizer()
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return len(text)  # char-level fallback

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts."""
        tokenizer = self.get_tokenizer()
        counts = []
        for text in texts:
            try:
                counts.append(len(tokenizer.encode(text, add_special_tokens=False)))
            except Exception:
                counts.append(len(text))
        return counts

    @classmethod
    def is_available(cls) -> bool:
        """Check if the llama.cpp embedding server is reachable."""
        try:
            from jet.adapters.llama_cpp.factory import get_embedding_client

            client = get_embedding_client()
            client.models.list()
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return (
            f"LlamacppEmbeddings(model='{self.model}', "
            f"dimension={self._dimension}, "
            f"batch_size={self.batch_size})"
        )


class _FallbackCharTokenizer:
    """Minimal char-level tokenizer used when HF tokenizer loading fails."""

    def encode(self, text: str, **kwargs) -> list[int]:
        return list(range(len(text)))

    def decode(self, tokens: list[int], **kwargs) -> str:
        return ""


# ---------------------------------------------------------------------------
# Optional: Register with Chonkie's AutoEmbeddings registry
# Usage: AutoEmbeddings.get_embeddings("llamacpp://nomic-embed:2-moe")
# ---------------------------------------------------------------------------
try:
    # Register provider alias so "llamacpp://..." URIs resolve correctly
    EmbeddingsRegistry.register_provider("llamacpp", LlamacppEmbeddings)

    # Register pattern for direct string matching (e.g., "llamacpp:nomic-embed:2-moe")
    EmbeddingsRegistry.register_pattern(r"^llamacpp[:/]", LlamacppEmbeddings)

    # Register type so passing a LlamacppEmbeddings instance to AutoEmbeddings.wrap() works
    EmbeddingsRegistry.register_types("LlamacppEmbeddings", LlamacppEmbeddings)

    logger.debug("Registered LlamacppEmbeddings with AutoEmbeddings registry")
except Exception as e:
    logger.warning(f"Could not register LlamacppEmbeddings: {e}")
