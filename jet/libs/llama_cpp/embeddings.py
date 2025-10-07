import time
import numpy as np
from openai import OpenAI
from typing import Iterator, List, Union, Literal, Callable
from tqdm import tqdm

from jet.libs.llama_cpp.models import resolve_model_value

GenerateEmbeddingsReturnType = Union[List[List[float]], np.ndarray]

class LlamacppEmbedding:
    """A client for generating embeddings via llama-server using OpenAI API."""
    
    def __init__(self, model: str = "embeddinggemma", base_url: str = "http://shawn-pc.local:8081/v1"):
        """Initialize the client with server URL and model path."""
        self.client = OpenAI(base_url=base_url, api_key="no-key-required")
        self.model = resolve_model_value(model)

    def __call__(self, inputs: Union[str, List[str]], return_format: Literal["numpy", "list"] = "numpy", batch_size: int = 16, show_progress: bool = False) -> GenerateEmbeddingsReturnType:
        """Make the instance callable to generate embeddings, equivalent to get_embeddings."""
        return self.get_embeddings(inputs, return_format=return_format, batch_size=batch_size, show_progress=show_progress)

    def get_embeddings(self, inputs: Union[str, List[str]], return_format: Literal["numpy", "list"] = "numpy", batch_size: int = 16, show_progress: bool = False) -> GenerateEmbeddingsReturnType:
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

    def get_embedding_function(self, return_format: Literal["numpy", "list"] = "numpy", batch_size: int = 16, show_progress: bool = False) -> Callable[[Union[str, List[str]]], GenerateEmbeddingsReturnType]:
        """Return a callable function that generates embeddings for a single text or list of texts."""
        def embedding_function(inputs: Union[str, List[str]]) -> GenerateEmbeddingsReturnType:
            return self.get_embeddings(inputs, return_format=return_format, batch_size=batch_size, show_progress=show_progress)
        return embedding_function

    def get_embeddings_stream(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 16,
        max_retries: int = 3,
        show_progress: bool = False
    ) -> Iterator[GenerateEmbeddingsReturnType]:
        """Stream embeddings batch by batch using the OpenAI API client.
        
        Yields one batch of embeddings at a time instead of collecting all.
        
        Args:
            inputs: A single text or list of texts to generate embeddings for.
            return_format: Format of the returned embeddings ("numpy" or "list").
            batch_size: Number of texts to process in each batch.
            max_retries: Maximum number of retry attempts for failed API calls.
            show_progress: Whether to display a progress bar.
        
        Yields:
            List of embeddings for each batch (as list or numpy array based on return_format).
        
        Raises:
            ValueError: If inputs are invalid (empty or not strings).
            Exception: If API call fails after max retries.
        """
        input_list = [inputs] if isinstance(inputs, str) else inputs
        if not input_list or not all(isinstance(i, str) and i for i in input_list):
            raise ValueError("inputs must be a non-empty string or list of non-empty strings")
        
        progress_bar = tqdm(range(0, len(input_list), batch_size), desc="Streaming batches", disable=not show_progress)
        
        for i in progress_bar:
            batch = input_list[i:i + batch_size]
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    batch_embeddings = [d.embedding for d in response.data]
                    if return_format == "numpy":
                        batch_embeddings = np.array(batch_embeddings)
                    yield batch_embeddings
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Error generating embeddings for batch {i // batch_size + 1} after {max_retries} attempts: {e}")
                        raise
                    time.sleep(2 ** attempt)
