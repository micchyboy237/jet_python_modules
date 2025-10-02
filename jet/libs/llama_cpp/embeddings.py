from openai import OpenAI
from typing import List, Union, Literal, Callable
import numpy as np
from tqdm import tqdm

GenerateEmbeddingsReturnType = Union[List[List[float]], np.ndarray]

class LlamacppEmbedding:
    """A client for generating embeddings via llama-server using OpenAI API."""
    
    def __init__(self, model: str = "embeddinggemma-300M-Q8_0.gguf", base_url: str = "http://shawn-pc.local:8080/v1"):
        """Initialize the client with server URL and model path."""
        self.client = OpenAI(base_url=base_url, api_key="no-key-required")
        self.model = model

    def get_embeddings(self, inputs: Union[str, List[str]], return_format: Literal["numpy", "list"] = "numpy", batch_size: int = 32, show_progress: bool = False) -> GenerateEmbeddingsReturnType:
        """Generate embeddings for a single text or list of text inputs in batches."""
        # Normalize inputs to list
        input_list = [inputs] if isinstance(inputs, str) else inputs
        if not input_list or not all(isinstance(i, str) and i for i in input_list):
            raise ValueError("inputs must be a non-empty string or list of non-empty strings")
        
        embeddings = []
        progress_bar = tqdm(range(0, len(input_list), batch_size), desc="Processing batches", disable=not show_progress)
        for i in progress_bar:
            batch = input_list[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [d.embedding for d in response.data]
                if return_format == "numpy":
                    batch_embeddings = [np.array(emb) for emb in batch_embeddings]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch {i // batch_size + 1}: {e}")
                raise
        
        # Return single embedding if input was a string, else return list
        return embeddings[0] if isinstance(inputs, str) else embeddings

    def get_embedding_function(self, return_format: Literal["numpy", "list"] = "numpy", batch_size: int = 32, show_progress: bool = False) -> Callable[[Union[str, List[str]]], GenerateEmbeddingsReturnType]:
        """Return a callable function that generates embeddings for a single text or list of texts."""
        def embedding_function(inputs: Union[str, List[str]]) -> GenerateEmbeddingsReturnType:
            return self.get_embeddings(inputs, return_format=return_format, batch_size=batch_size, show_progress=show_progress)
        return embedding_function
