from openai import OpenAI
from typing import List, Union, Literal, Callable
import numpy as np
from tqdm import tqdm

GenerateEmbeddingsReturnType = Union[List[List[float]], np.ndarray]

class LlamacppEmbedding:
    """A client for generating embeddings via llama-server using OpenAI API."""
    
    def __init__(self, base_url: str = "http://shawn-pc.local:8080/v1", model_path: str = "embeddinggemma-300M-Q8_0.gguf"):
        """Initialize the client with server URL and model path."""
        self.client = OpenAI(base_url=base_url, api_key="no-key-required")
        self.model_path = model_path

    def get_embedding(self, input_text: str, return_format: Literal["numpy", "list"] = "numpy") -> GenerateEmbeddingsReturnType:
        """Generate embedding for a single text input."""
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")
        result = self.get_embeddings([input_text], return_format=return_format)
        return result[0]

    def get_embeddings(self, inputs: List[str], return_format: Literal["numpy", "list"] = "numpy", batch_size: int = 32, show_progress: bool = False) -> List[GenerateEmbeddingsReturnType]:
        """Generate embeddings for a list of text inputs in batches."""
        if not inputs or not all(isinstance(i, str) and i for i in inputs):
            raise ValueError("inputs must be a non-empty list of non-empty strings")
        
        embeddings = []
        progress_bar = tqdm(range(0, len(inputs), batch_size), desc="Processing batches", disable=not show_progress)
        for i in progress_bar:
            batch = inputs[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_path,
                    input=batch
                )
                batch_embeddings = [d.embedding for d in response.data]
                if return_format == "numpy":
                    batch_embeddings = [np.array(emb) for emb in batch_embeddings]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch {i // batch_size + 1}: {e}")
                raise
        
        return embeddings

    def get_embedding_function(self) -> Callable[[str], GenerateEmbeddingsReturnType]:
        """Return a callable function that generates embeddings for a single text input."""
        def embedding_function(input_text: str, return_format: Literal["numpy", "list"] = "numpy") -> GenerateEmbeddingsReturnType:
            return self.get_embedding(input_text, return_format=return_format)
        return embedding_function
