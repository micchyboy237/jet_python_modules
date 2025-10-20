from typing import List
import numpy as np
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from keybert.backend import BaseEmbedder

class KeyBERTLlamacppEmbedder(BaseEmbedder):
    """KeyBERT embedder using LlamacppEmbedding for generating embeddings."""
    
    def __init__(self, embedding_model: str = "embeddinggemma", base_url: str = "http://shawn-pc.local:8081/v1", use_cache: bool = False):
        """Initialize the embedder with LlamacppEmbedding model.
        
        Args:
            embedding_model: The model name for LlamacppEmbedding.
            base_url: Base URL for the llama-server API.
        """
        super().__init__(
            embedding_model=LlamacppEmbedding(model=embedding_model, base_url=base_url, use_cache=use_cache, use_dynamic_batch_sizing=True),
            word_embedding_model=None
        )
    
    def embed(self, documents: List[str], verbose: bool = True) -> np.ndarray:
        """Embed a list of n documents/words into an n-dimensional matrix of embeddings.

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        # Check if documents is a list and is empty
        if isinstance(documents, list) and len(documents) == 0:
            return np.array([])
        # Handle single string input by converting to list
        input_docs = [documents] if isinstance(documents, str) else documents
        # Validate inputs
        if not all(isinstance(doc, str) for doc in input_docs):
            raise ValueError("All documents must be strings")
        embeddings = self.embedding_model.get_embeddings(
            inputs=input_docs,
            return_format="numpy",
            show_progress=verbose,
        )
        # Ensure embeddings is a NumPy array
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings
