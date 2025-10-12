import numpy as np
from openai import OpenAI
from typing import Iterator, List, Union, Literal, Callable, Optional
from tqdm import tqdm
from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.models import resolve_model_value
from jet.models.utils import get_context_size, get_embedding_size
from jet.models.embeddings.utils import calculate_dynamic_batch_size
from jet.models.embeddings.cache import EmbeddingCache
from jet.logger import logger

GenerateEmbeddingsReturnType = Union[List[List[float]], np.ndarray]

class InputTooLargeError(ValueError):
    """Custom exception for inputs exceeding the maximum allowed length."""
    def __init__(self, long_input_indexes: List[int], max_input_length: int):
        self.long_input_indexes = long_input_indexes
        self.max_input_length = max_input_length
        super().__init__(f"Inputs at indexes {long_input_indexes} are too long (> {max_input_length} tokens). Please reduce input size or increase server physical batch size.")

class LlamacppEmbedding:
    """A client for generating embeddings via llama-server using OpenAI API with modern caching."""
    
    def __init__(
        self,
        model: str = "embeddinggemma",
        base_url: str = "http://shawn-pc.local:8081/v1",
        max_retries: int = 3,
        cache_backend: Literal["memory", "file", "sqlite"] = "sqlite",
        cache_ttl: Optional[int] = None,
        cache_max_size: int = 10000,
        use_cache: bool = True,
        use_dynamic_batch_sizing: bool = False
    ):
        """Initialize the client with server URL, model path, and cache settings."""
        self.client = OpenAI(base_url=base_url, api_key="no-key-required", max_retries=max_retries)
        self.model = resolve_model_value(model)
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.use_dynamic_batch_sizing = use_dynamic_batch_sizing
        self.cache = EmbeddingCache(
            backend=cache_backend,
            max_size=cache_max_size,
            ttl=cache_ttl,
            namespace=f"llama_{self.model}"
        )

    def __call__(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None
    ) -> GenerateEmbeddingsReturnType:
        """Make the instance callable to generate embeddings, equivalent to get_embeddings."""
        return self.get_embeddings(
            inputs,
            return_format=return_format,
            batch_size=batch_size,
            show_progress=show_progress,
            use_cache=use_cache if use_cache is not None else self.use_cache,
            use_dynamic_batch_sizing=use_dynamic_batch_sizing if use_dynamic_batch_sizing is not None else self.use_dynamic_batch_sizing
        )

    def get_embeddings(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        max_input_length: Optional[int] = None,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None
    ) -> GenerateEmbeddingsReturnType:
        """Generate embeddings with caching and optional dynamic batch sizing."""
        use_cache = use_cache if use_cache is not None else self.use_cache
        use_dynamic_batch_sizing = use_dynamic_batch_sizing if use_dynamic_batch_sizing is not None else self.use_dynamic_batch_sizing

        input_list = [inputs] if isinstance(inputs, str) else inputs
        valid_inputs = [i for i in input_list if isinstance(i, str) and i.strip()]
        invalid_inputs = [i for i in input_list if not (isinstance(i, str) and i.strip())]
        
        if invalid_inputs:
            logger.warning(f"Warning: Skipped {len(invalid_inputs)} invalid inputs: {invalid_inputs}")
        if not valid_inputs:
            raise ValueError("No valid inputs provided: inputs must be a non-empty string or list of non-empty strings")
        
        embedding_size = get_embedding_size(self.model)
        context_size = get_context_size(self.model)
        max_length = max_input_length if max_input_length is not None else context_size
        
        if max_length <= 0:
            logger.warning(f"Warning: Invalid max_input_length ({max_length}) from get_context_size; falling back to 512")
            max_length = 512
        
        token_counts: List[int] = token_counter(valid_inputs, self.model, prevent_total=True)

        # Log detailed input statistics
        logger.info(
            f"Embedding stats -> model: {self.model}, "
            f"embedding_size: {embedding_size}, "
            f"context_size: {context_size}, "
            f"max_length: {max_length}"
        )
        logger.debug(f"\nInputs: {len(input_list)}")
        logger.debug(f"Tokens\nmax: {max(token_counts)}\nmin: {min(token_counts)}")

        long_inputs = [(count, idx) for idx, count in enumerate(token_counts) if count > max_length]
        
        if long_inputs:
            long_input_indexes = [idx for _, idx in long_inputs]
            logger.error(f"Error: Found {len(long_inputs)} inputs exceeding max length ({max_length} tokens): indexes {long_input_indexes}")
            raise InputTooLargeError(long_input_indexes, max_length)
        
        # Apply dynamic batch sizing if enabled
        if use_dynamic_batch_sizing:
            dynamic_batch_size = calculate_dynamic_batch_size(
                token_counts=token_counts,
                embedding_size=embedding_size,
                context_size=context_size
            )
            batch_size = dynamic_batch_size
            logger.debug(f"Dynamic batch sizing enabled. Using batch_size: {batch_size}")
        else:
            logger.debug(f"Using static batch_size: {batch_size}")
        
        # Cache check
        if use_cache:
            cache_key = self.cache._generate_key(valid_inputs)
            cached = self.cache.get(cache_key)
            if cached is not None:
                if return_format == "numpy":
                    result = np.array(cached, dtype=np.float32)
                else:
                    result = cached
                logger.debug(f"Cache hit for {len(valid_inputs)} texts (key: {cache_key[:16]}...)")
                return result[0] if isinstance(inputs, str) else result
            logger.debug(f"Cache miss for {len(valid_inputs)} texts (key: {cache_key[:16]}...). Computing...")
        
        embeddings = []
        progress_bar = tqdm(range(0, len(valid_inputs), batch_size), desc="Processing batches", disable=not show_progress)
        
        for i in progress_bar:
            batch = valid_inputs[i:i + batch_size]
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                batch_embeddings = [d.embedding for d in response.data]
                if return_format == "numpy":
                    batch_embeddings = [np.array(emb) for emb in batch_embeddings]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i // batch_size + 1}: {e}")
                raise
        
        final_embeddings = embeddings if return_format != "numpy" else np.array(embeddings, dtype=np.float32)
        
        if use_cache:
            self.cache.set(cache_key, final_embeddings.tolist() if return_format == "numpy" else final_embeddings)
            logger.info(f"Cached embeddings for {len(valid_inputs)} texts (key: {cache_key[:16]}...)")
        
        return final_embeddings[0] if isinstance(inputs, str) else final_embeddings

    def get_embedding_function(
        self,
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None
    ) -> Callable[[Union[str, List[str]]], GenerateEmbeddingsReturnType]:
        """Return callable with caching and optional dynamic batch sizing."""
        def embedding_function(inputs: Union[str, List[str]]) -> GenerateEmbeddingsReturnType:
            return self.get_embeddings(
                inputs,
                return_format=return_format,
                batch_size=batch_size,
                show_progress=show_progress,
                use_cache=use_cache if use_cache is not None else self.use_cache,
                use_dynamic_batch_sizing=use_dynamic_batch_sizing if use_dynamic_batch_sizing is not None else self.use_dynamic_batch_sizing
            )
        return embedding_function

    def get_embeddings_stream(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        max_input_length: Optional[int] = None,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None
    ) -> Iterator[GenerateEmbeddingsReturnType]:
        """Stream with per-batch caching and optional dynamic batch sizing."""
        use_cache = use_cache if use_cache is not None else self.use_cache
        use_dynamic_batch_sizing = use_dynamic_batch_sizing if use_dynamic_batch_sizing is not None else self.use_dynamic_batch_sizing

        input_list = [inputs] if isinstance(inputs, str) else inputs
        valid_inputs = [i for i in input_list if isinstance(i, str) and i.strip()]
        
        if not valid_inputs:
            raise ValueError("inputs must be a non-empty string or list of non-empty strings")
        
        embedding_size = get_embedding_size(self.model)
        context_size = get_context_size(self.model)
        max_length = max_input_length if max_input_length is not None else context_size
        
        if max_length <= 0:
            logger.warning(f"Warning: Invalid max_input_length ({max_length}) from get_context_size; falling back to 512")
            max_length = 512

        token_counts: List[int] = token_counter(valid_inputs, self.model, prevent_total=True)

        # Log detailed input statistics
        logger.info(
            f"Embedding stats -> model: {self.model}, "
            f"embedding_size: {embedding_size}, "
            f"context_size: {context_size}, "
            f"max_length: {max_length}"
        )
        logger.debug(f"\nInputs: {len(input_list)}")
        logger.debug(f"Tokens\nmax: {max(token_counts)}\nmin: {min(token_counts)}")

        long_inputs = [(count, idx) for idx, count in enumerate(token_counts) if count > max_length]
        
        if long_inputs:
            long_input_indexes = [idx for _, idx in long_inputs]
            logger.error(f"Error: Found {len(long_inputs)} inputs exceeding max length ({max_length} tokens): indexes {long_input_indexes}")
            raise InputTooLargeError(long_input_indexes, max_length)
        
        # Apply dynamic batch sizing if enabled
        if use_dynamic_batch_sizing:
            dynamic_batch_size = calculate_dynamic_batch_size(
                token_counts=token_counts,
                embedding_size=embedding_size,
                context_size=context_size
            )
            batch_size = dynamic_batch_size
            logger.debug(f"Dynamic batch sizing enabled for stream. Using batch_size: {batch_size}")
        
        # Full cache check for streaming
        if use_cache:
            cache_key = self.cache._generate_key(valid_inputs)
            cached = self.cache.get(cache_key)
            if cached is not None:
                result = np.array(cached, dtype=np.float32) if return_format == "numpy" else cached
                logger.debug(f"Cache hit in stream for {len(valid_inputs)} texts")
                for i in range(0, len(result), batch_size):
                    yield result[i:i + batch_size]
                return
        
        progress_bar = tqdm(range(0, len(valid_inputs), batch_size), desc="Streaming batches", disable=not show_progress)
        
        for i in progress_bar:
            batch = valid_inputs[i:i + batch_size]
            batch_key = self.cache._generate_key(batch)
            
            if use_cache:
                cached_batch = self.cache.get(batch_key)
                if cached_batch is not None:
                    if return_format == "numpy":
                        yield np.array(cached_batch, dtype=np.float32)
                    else:
                        yield cached_batch
                    continue
            
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                batch_embeddings = [d.embedding for d in response.data]
                if return_format == "numpy":
                    batch_embeddings = np.array(batch_embeddings, dtype=np.float32)
                
                if use_cache:
                    self.cache.set(batch_key, batch_embeddings.tolist() if return_format == "numpy" else batch_embeddings)
                
                yield batch_embeddings
            except Exception as e:
                logger.error(f"Error in stream batch {i // batch_size + 1}: {e}")
                raise

    def close(self) -> None:
        """Close cache (e.g., SQLite conn)."""
        self.cache.close()
