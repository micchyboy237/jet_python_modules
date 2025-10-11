import numpy as np
from openai import OpenAI
from typing import Iterator, List, Union, Literal, Callable, Optional
from tqdm import tqdm

from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.models import resolve_model_value
from jet.models.utils import get_context_size
from jet.logger import logger

GenerateEmbeddingsReturnType = Union[List[List[float]], np.ndarray]

class InputTooLargeError(ValueError):
    """Custom exception for inputs exceeding the maximum allowed length."""
    def __init__(self, long_input_indexes: List[int], max_input_length: int):
        self.long_input_indexes = long_input_indexes
        self.max_input_length = max_input_length
        super().__init__(f"Inputs at indexes {long_input_indexes} are too long (> {max_input_length} tokens). Please reduce input size or increase server physical batch size.")

class LlamacppEmbedding:
    """A client for generating embeddings via llama-server using OpenAI API."""
    
    def __init__(self, model: str = "embeddinggemma", base_url: str = "http://shawn-pc.local:8081/v1", max_retries: int = 3):
        """Initialize the client with server URL, model path, and max retries."""
        self.client = OpenAI(base_url=base_url, api_key="no-key-required", max_retries=max_retries)
        self.model = resolve_model_value(model)
        self.max_retries = max_retries

    def __call__(self, inputs: Union[str, List[str]], return_format: Literal["numpy", "list"] = "numpy", batch_size: int = 32, show_progress: bool = True) -> GenerateEmbeddingsReturnType:
        """Make the instance callable to generate embeddings, equivalent to get_embeddings."""
        return self.get_embeddings(inputs, return_format=return_format, batch_size=batch_size, show_progress=show_progress)

    def get_embeddings(self, inputs: Union[str, List[str]], return_format: Literal["numpy", "list"] = "numpy", batch_size: int = 32, show_progress: bool = True, max_input_length: Optional[int] = None) -> GenerateEmbeddingsReturnType:
        """Generate embeddings for a single text or list of text inputs in batches."""
        # Normalize inputs to list
        input_list = [inputs] if isinstance(inputs, str) else inputs
        
        # Filter out invalid inputs (non-strings or empty strings)
        valid_inputs = [i for i in input_list if isinstance(i, str) and i.strip()]
        invalid_inputs = [i for i in input_list if not (isinstance(i, str) and i.strip())]
        
        # Log invalid inputs for debugging
        if invalid_inputs:
            logger.warning(f"Warning: Skipped {len(invalid_inputs)} invalid inputs: {invalid_inputs}")
        
        # Check if there are any valid inputs after filtering
        if not valid_inputs:
            raise ValueError("No valid inputs provided: inputs must be a non-empty string or list of non-empty strings")
        
        # Determine max input length
        max_length = max_input_length if max_input_length is not None else get_context_size(self.model)
        if max_length <= 0:
            logger.warning(f"Warning: Invalid max_input_length ({max_length}) from get_context_size; falling back to 512")
            max_length = 512
        
        # Check for inputs exceeding max length
        token_counts: List[int] = token_counter(valid_inputs, self.model, prevent_total=True)
        long_inputs = [(count, idx) for idx, count in enumerate(token_counts) if count > max_length]
        if long_inputs:
            long_input_indexes = [idx for _, idx in long_inputs]
            logger.error(f"Error: Found {len(long_inputs)} inputs exceeding max length ({max_length} tokens): indexes {long_input_indexes}")
            raise InputTooLargeError(long_input_indexes, max_length)
        
        embeddings = []
        progress_bar = tqdm(range(0, len(valid_inputs), batch_size), desc="Processing batches", disable=not show_progress)
        for i in progress_bar:
            batch = valid_inputs[i:i + batch_size]
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
                logger.error(f"Error generating embeddings for batch {i // batch_size + 1}: {e}")
                raise
        
        if return_format == "numpy":
            embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings

    def get_embedding_function(self, return_format: Literal["numpy", "list"] = "numpy", batch_size: int = 32, show_progress: bool = True) -> Callable[[Union[str, List[str]]], GenerateEmbeddingsReturnType]:
        """Return a callable function that generates embeddings for a single text or list of texts."""
        def embedding_function(inputs: Union[str, List[str]]) -> GenerateEmbeddingsReturnType:
            return self.get_embeddings(inputs, return_format=return_format, batch_size=batch_size, show_progress=show_progress)
        return embedding_function

    def get_embeddings_stream(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Iterator[GenerateEmbeddingsReturnType]:
        """Stream embeddings batch by batch using the OpenAI API client.
        
        Yields one batch of embeddings at a time instead of collecting all.
        
        Args:
            inputs: A single text or list of texts to generate embeddings for.
            return_format: Format of the returned embeddings ("numpy" or "list").
            batch_size: Number of texts to process in each batch.
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
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [d.embedding for d in response.data]
                if return_format == "numpy":
                    batch_embeddings = np.array(batch_embeddings)
                yield batch_embeddings
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i // batch_size + 1} after {self.max_retries} attempts: {e}")
                raise
