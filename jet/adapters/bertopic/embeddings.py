from typing import List
import numpy as np
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from bertopic.backend import BaseEmbedder

class BERTopicLlamacppEmbedder(BaseEmbedder):
    """BERTopic embedder using LlamacppEmbedding for generating embeddings."""
    
    def __init__(self, embedding_model: str = "embeddinggemma", base_url: str = "http://shawn-pc.local:8081/v1"):
        """Initialize the embedder with LlamacppEmbedding model.
        
        Args:
            embedding_model: The model name for LlamacppEmbedding.
            base_url: Base URL for the llama-server API.
        """
        super().__init__(
            embedding_model=LlamacppEmbedding(model=embedding_model, base_url=base_url, use_cache=True),
            word_embedding_model=None
        )
    
    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """Embed a list of documents/words into an n-dimensional matrix of embeddings.
        
        Args:
            documents: A list of documents or words to be embedded.
            verbose: Controls the verbosity of the process.
        
        Returns:
            Embeddings with shape (n, m) where n is the number of documents/words
            and m is the embedding size.
        """
        if not documents:
            return np.array([])
        embeddings = self.embedding_model.get_embeddings(
            inputs=documents,
            return_format="numpy",
            show_progress=verbose
        )
        return np.array(embeddings)