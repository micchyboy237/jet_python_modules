import numpy as np
from jet.adapters.llama_cpp.config import EMBED_BASE_URL, EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed_batch
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.logger import logger

from bertopic.backend import BaseEmbedder


class BERTopicLlamacppEmbedder(BaseEmbedder):
    """BERTopic embedder that reuses jet.adapters.llama_cpp.embed_utils.embed_batch
    (instead of a dedicated LlamacppEmbedding client class).
    """

    def __init__(
        self,
        embedding_model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
        max_workers: int = 6,
        batch_size: int | None = 32,
    ):
        """Initialize the embedder.

        Args:
            embedding_model: Model key to use. Defaults to config.EMBED_MODEL
                (env var LLAMA_CPP_EMBED_MODEL).
            max_workers: Parallel worker threads used by embed_batch.
            batch_size: Batch size used by embed_batch.

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
        logger.info(
            f"BERTopicLlamacppEmbedder ready | model={self.model} "
            f"base_url={EMBED_BASE_URL}"
        )

    def embed(self, documents: list[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of documents/words into an n-dimensional matrix of embeddings.

        Args:
            documents: A list of documents or words to be embedded.
            verbose: Controls the verbosity of the process.

        Returns:
            Embeddings with shape (n, m) where n is the number of documents/words
            and m is the embedding size.
        """
        if not documents:
            logger.debug("No documents to embed, returning empty array")
            return np.array([])
        logger.info(f"Embedding {len(documents)} document(s) with model={self.model}")
        embeddings = embed_batch(
            texts=documents,
            model=self.model,
            max_workers=self.max_workers,
            show_progress=verbose,
            return_format="numpy",
            batch_size=self.batch_size,
        )
        logger.debug(f"Produced embeddings with shape {embeddings.shape}")
        return embeddings
