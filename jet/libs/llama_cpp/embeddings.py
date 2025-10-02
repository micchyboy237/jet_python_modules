from openai import OpenAI
from typing import List, Union, Literal
import numpy as np

GenerateEmbeddingsReturnType = Union[List[List[float]], np.ndarray]

class LlamacppEmbedding:
    """A client for generating embeddings via llama-server using OpenAI API."""
    
    def __init__(self, base_url: str = "http://shawn-pc.local:8080/v1", model_path: str = ""):
        """Initialize the client with server URL and model path."""
        self.client = OpenAI(base_url=base_url, api_key="no-key-required")
        self.model_path = model_path

    def get_embedding(self, input_text: str, return_format: Literal["numpy", "list"] = "numpy") -> GenerateEmbeddingsReturnType:
        """Generate embedding for a single text input."""
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")
        result = self.get_embeddings([input_text], return_format=return_format)
        return result[0]

    def get_embeddings(self, inputs: List[str], return_format: Literal["numpy", "list"] = "numpy") -> List[GenerateEmbeddingsReturnType]:
        """Generate embeddings for a list of text inputs."""
        if not inputs or not all(isinstance(i, str) and i for i in inputs):
            raise ValueError("inputs must be a non-empty list of non-empty strings")
        
        try:
            response = self.client.embeddings.create(
                model=self.model_path,
                input=inputs
            )
            embeddings = [d.embedding for d in response.data]
            if return_format == "numpy":
                embeddings = [np.array(emb) for emb in embeddings]
            
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise
