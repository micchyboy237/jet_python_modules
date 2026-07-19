from typing import List, Union

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.logger import logger

from keybert.backend import BaseEmbedder


class KeyBERTLlamacppEmbedder(BaseEmbedder):
    """KeyBERT embedder using local llama.cpp server for generating embeddings."""

    def __init__(self, embedding_model: LLAMACPP_EMBED_KEYS = None):
        """Initialize the embedder utilizing the native embed utility.

        Args:
            embedding_model: Valid llama.cpp embedding model key name. Defaults to EMBED_MODEL config.
        """
        super().__init__()
        self.embedding_model_name = embedding_model or EMBED_MODEL or "nomic-embed:1.5"
        logger.info(
            f"KeyBERTLlamacppEmbedder initialized with model: {self.embedding_model_name}"
        )

    def embed(
        self, documents: Union[str, List[str], np.ndarray], verbose: bool = True
    ) -> np.ndarray:
        """Embed a list/array of documents or words into an n-dimensional numpy matrix."""
        # Normalize incoming format types safely (handling numpy arrays / iterables)
        if isinstance(documents, np.ndarray):
            input_docs = documents.tolist()
        elif isinstance(documents, str):
            input_docs = [documents]
        else:
            input_docs = list(documents)

        if len(input_docs) == 0:
            return np.array([])

        if not all(isinstance(doc, str) for doc in input_docs):
            raise ValueError(
                "All elements inside the documents sequence must be strings"
            )

        logger.debug(
            f"Requesting embeddings for {len(input_docs)} item(s) via model: {self.embedding_model_name}"
        )

        embeddings = embed(
            text=input_docs,
            model=self.embedding_model_name,
            return_format="numpy",
            show_progress=verbose,
        )

        return embeddings
