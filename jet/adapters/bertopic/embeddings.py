import numpy as np
from jet.adapters.llama_cpp.config import (
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
)
from jet.adapters.llama_cpp.embed_utils import embed as jet_embed
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.logger import logger

from bertopic.backend import BaseEmbedder


class BERTopicLlamacppEmbedder(BaseEmbedder):
    """BERTopic embedder that targets specific prefixes for documents and queries
    by overriding embed_documents and embed_words.
    """

    def __init__(
        self,
        embedding_model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
        max_workers: int = 6,
        batch_size: int | None = 32,
    ):
        super().__init__()
        self.model = embedding_model
        self.max_workers = max_workers
        self.batch_size = batch_size

        logger.info(
            f"BERTopicLlamacppEmbedder ready | model={self.model} | "
            f"query_prefix='{EMBED_QUERY_PREFIX}' | doc_prefix='{EMBED_DOC_PREFIX}'"
        )

    def embed_documents(
        self, documents: list[str], verbose: bool = False
    ) -> np.ndarray:
        """Embed document sequences using the defined document prefix."""
        if not documents:
            logger.debug("No documents to embed, returning empty array")
            return np.array([])

        logger.info(
            f"Embedding {len(documents)} document(s) with prefix='{EMBED_DOC_PREFIX}'"
        )
        return jet_embed(
            text=documents,
            model=self.model,
            return_format="numpy",
            max_workers=self.max_workers,
            show_progress=verbose,
            batch_size=self.batch_size,
            progress_description="Embedding documents",
            prefix=EMBED_DOC_PREFIX if EMBED_DOC_PREFIX else None,
        )

    def embed_words(self, words: list[str], verbose: bool = False) -> np.ndarray:
        """Embed search terms or words using the defined query prefix."""
        if not words:
            logger.debug("No words to embed, returning empty array")
            return np.array([])

        logger.info(
            f"Embedding {len(words)} word(s)/query terms with prefix='{EMBED_QUERY_PREFIX}'"
        )
        return jet_embed(
            text=words,
            model=self.model,
            return_format="numpy",
            max_workers=self.max_workers,
            show_progress=verbose,
            batch_size=self.batch_size,
            progress_description="Embedding words",
            prefix=EMBED_QUERY_PREFIX if EMBED_QUERY_PREFIX else None,
        )

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        """Fallback method handling direct raw matrix extractions if invoked outside context."""
        if not documents:
            logger.debug("No texts to embed, returning empty array")
            return np.array([])

        logger.info(f"Embedding {len(documents)} raw item(s) without context prefix.")
        return jet_embed(
            text=documents,
            model=self.model,
            return_format="numpy",
            max_workers=self.max_workers,
            show_progress=verbose,
            batch_size=self.batch_size,
        )
