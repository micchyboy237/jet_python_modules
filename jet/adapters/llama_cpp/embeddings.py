import numpy as np
from openai import OpenAI
from typing import Iterator, List, Union, Literal, Callable, Optional
from tqdm import tqdm
from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_TYPES
from jet.adapters.llama_cpp.utils import resolve_model_value
from jet.models.utils import get_context_size, get_embedding_size
from jet.models.embeddings.utils import calculate_dynamic_batch_size
from jet.models.embeddings.cache import EmbeddingCache
from jet.logger import CustomLogger

GenerateEmbeddingsReturnType = Union[List[List[float]], np.ndarray]

class InputTooLargeError(ValueError):
    """Custom exception for inputs exceeding the maximum allowed length."""
    def __init__(self, long_input_indexes: List[int], max_input_length: int):
        self.long_input_indexes = long_input_indexes
        self.max_input_length = max_input_length
        super().__init__(f"Inputs at indexes {long_input_indexes} are too long (> {max_input_length} tokens). Please reduce input size or increase server physical batch size.")

class LlamacppEmbedding:
    """
    Initialize the Llama.cpp embedding client.

    This client communicates with a `llama-server` instance exposing an
    OpenAI-compatible `/v1/embeddings` API and supports optional caching
    and dynamic batch sizing.

    Args:
        model (LLAMACPP_EMBED_TYPES):
            Embedding model identifier or alias resolved via
            `resolve_model_value`. Must be compatible with llama.cpp
            embedding endpoints.

        base_url (str):
            Base URL of the llama-server OpenAI-compatible API
            (e.g. "http://localhost:8081/v1").

        max_retries (int):
            Maximum number of retry attempts for failed API requests.

        cache_backend (Literal["memory", "file", "sqlite"]):
            Cache storage backend:
            - "memory": in-process LRU cache
            - "file": compressed pickle file cache
            - "sqlite": persistent SQLite-backed cache

        cache_ttl (Optional[int]):
            Time-to-live (in seconds) for cached embeddings.
            If None, cached values never expire.

        cache_max_size (int):
            Maximum number of embedding entries stored in cache
            before LRU eviction.

        use_cache (bool):
            Whether to enable embedding caching by default.

        use_dynamic_batch_sizing (bool):
            If True, batch size is automatically calculated based on
            token counts, embedding size, and model context window.

        verbose (bool):
            Enable informational and debug logging.

        logger (Optional[CustomLogger]):
            Optional custom logger instance. If not provided,
            a default `CustomLogger` is created.
    """
    def __init__(
        self,
        model: LLAMACPP_EMBED_TYPES = "embeddinggemma",
        base_url: str = "http://shawn-pc.local:8081/v1",
        max_retries: int = 3,
        cache_backend: Literal["memory", "file", "sqlite"] = "sqlite",
        cache_ttl: Optional[int] = None,
        cache_max_size: int = 10000,
        use_cache: bool = False,
        use_dynamic_batch_sizing: bool = False,
        verbose: bool = True,
        logger: Optional[CustomLogger] = None,
    ):
        self.client = OpenAI(base_url=base_url, api_key="no-key-required", max_retries=max_retries)
        self.model = resolve_model_value(model)
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.use_dynamic_batch_sizing = use_dynamic_batch_sizing
        self.verbose = verbose
        self.cache = EmbeddingCache(
            backend=cache_backend,
            max_size=cache_max_size,
            ttl=cache_ttl,
            namespace=f"llama_{self.model}"
        )

        self._logger = logger or CustomLogger()

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

    def encode(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        max_input_length: Optional[int] = None,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None
    ) -> GenerateEmbeddingsReturnType:
        return self.get_embeddings(
            inputs,
            return_format=return_format,
            batch_size=batch_size,
            show_progress=show_progress,
            max_input_length=max_input_length,
            use_cache=use_cache,
            use_dynamic_batch_sizing=use_dynamic_batch_sizing,
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
            self._logger.warning(f"Warning: Skipped {len(invalid_inputs)} invalid inputs: {invalid_inputs}")
        if not valid_inputs:
            raise ValueError("No valid inputs provided: inputs must be a non-empty string or list of non-empty strings")
        
        embedding_size = get_embedding_size(self.model)
        context_size = get_context_size(self.model)
        max_length = max_input_length if max_input_length is not None else context_size
        
        if max_length <= 0:
            self._logger.warning(f"Warning: Invalid max_input_length ({max_length}) from get_context_size; falling back to 512")
            max_length = 512
        elif max_length <= 0:
            max_length = 512
        
        token_counts: List[int] = token_counter(valid_inputs, self.model, prevent_total=True)
        
        # Log detailed input statistics
        if self.verbose:
            self._logger.info(
                f"Embedding stats -> model: {self.model}, "
                f"embedding_size: {embedding_size}, "
                f"context_size: {context_size}, "
                f"max_length: {max_length}"
            )
            self._logger.debug(f"\nInputs: {len(input_list)}")
            self._logger.debug(f"Tokens\nmax: {max(token_counts)}\nmin: {min(token_counts)}")

        long_inputs = [(count, idx) for idx, count in enumerate(token_counts) if count > max_length]
        
        if long_inputs:
            long_input_indexes = [idx for _, idx in long_inputs]
            self._logger.error(f"Error: Found {len(long_inputs)} inputs exceeding max length ({max_length} tokens): indexes {long_input_indexes}")
            raise InputTooLargeError(long_input_indexes, max_length)
        
        # Apply dynamic batch sizing if enabled
        if use_dynamic_batch_sizing:
            dynamic_batch_size = calculate_dynamic_batch_size(
                token_counts=token_counts,
                embedding_size=embedding_size,
                context_size=context_size
            )
            batch_size = dynamic_batch_size
            if self.verbose:
                self._logger.debug(f"Dynamic batch sizing enabled. Using batch_size: {batch_size}")
        else:
            if self.verbose:
                self._logger.debug(f"Using static batch_size: {batch_size}")
        
        # Cache check
        if use_cache:
            cache_key = self.cache._generate_key(valid_inputs)
            cached = self.cache.get(cache_key)
            if cached is not None:
                if return_format == "numpy":
                    result = np.array(cached, dtype=np.float32)
                else:
                    result = cached
                if self.verbose:
                    self._logger.debug(f"Cache hit for {len(valid_inputs)} texts (key: {cache_key[:16]}...)")
                return result
            if self.verbose:
                self._logger.debug(f"Cache miss for {len(valid_inputs)} texts (key: {cache_key[:16]}...). Computing...")
        
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
                self._logger.error(f"Error generating embeddings for batch {i // batch_size + 1}: {e}")
                raise
        
        final_embeddings = embeddings if return_format != "numpy" else np.array(embeddings, dtype=np.float32)
        
        if use_cache:
            self.cache.set(cache_key, final_embeddings.tolist() if return_format == "numpy" else final_embeddings)
            if self.verbose:
                self._logger.info(f"Cached embeddings for {len(valid_inputs)} texts (key: {cache_key[:16]}...)")
        
        return final_embeddings

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
        """Stream embeddings with per-text caching, batch processing only misses."""
        use_cache = use_cache if use_cache is not None else self.use_cache
        use_dynamic_batch_sizing = use_dynamic_batch_sizing if use_dynamic_batch_sizing is not None else self.use_dynamic_batch_sizing

        input_list = [inputs] if isinstance(inputs, str) else inputs
        valid_inputs = [i for i in input_list if isinstance(i, str) and i.strip()]
        if not valid_inputs:
            raise ValueError("inputs must be a non-empty string or list of non-empty strings")

        # ── same validation / stats as before ──
        embedding_size = get_embedding_size(self.model)
        context_size = get_context_size(self.model)
        max_length = max_input_length if max_input_length is not None else context_size
        if max_length <= 0:
            max_length = 512

        token_counts: List[int] = token_counter(valid_inputs, self.model, prevent_total=True)

        long_inputs = [idx for idx, cnt in enumerate(token_counts) if cnt > max_length]
        if long_inputs:
            raise InputTooLargeError(long_inputs, max_length)

        if use_dynamic_batch_sizing:
            batch_size = calculate_dynamic_batch_size(
                token_counts=token_counts,
                embedding_size=embedding_size,
                context_size=context_size
            )
            if self.verbose:
                self._logger.debug(f"Dynamic batch size: {batch_size}")

        # ── Per-text cache check ─────────────────────────────────────────────
        cache_hits = 0
        miss_indices: List[int] = []
        miss_texts: List[str] = []
        cached_embeddings: List[Optional[Union[List[float], np.ndarray]]] = [None] * len(valid_inputs)

        if use_cache:
            for idx, text in enumerate(valid_inputs):
                key = self.cache._generate_key([text])[0]  # single item → gets one key
                emb = self.cache.get(key)
                if emb is not None:
                    cache_hits += 1
                    if return_format == "numpy":
                        cached_embeddings[idx] = np.array(emb, dtype=np.float32)
                    else:
                        cached_embeddings[idx] = emb
                else:
                    miss_indices.append(idx)
                    miss_texts.append(text)

            if self.verbose:
                self._logger.info(
                    f"Cache stats: {cache_hits} hits / {len(valid_inputs) - cache_hits} misses "
                    f"({cache_hits / len(valid_inputs):.0%} hit rate)"
                )
        else:
            miss_indices = list(range(len(valid_inputs)))
            miss_texts = valid_inputs

        # ── Compute missing embeddings in batches ─────────────────────────────
        computed_embeddings: List[Union[List[float], np.ndarray]] = []

        if miss_texts:
            if self.verbose and cache_hits > 0:
                self._logger.info(f"Computing {len(miss_texts)} missing embeddings...")

            progress_bar = tqdm(
                range(0, len(miss_texts), batch_size),
                desc="Embedding misses",
                disable=not show_progress
            )

            for start in progress_bar:
                end = start + batch_size
                batch_texts = miss_texts[start:end]
                batch_original_indices = miss_indices[start:end]

                try:
                    resp = self.client.embeddings.create(model=self.model, input=batch_texts)
                    batch_embs = [d.embedding for d in resp.data]

                    if return_format == "numpy":
                        batch_embs_np = np.array(batch_embs, dtype=np.float32)
                        computed_embeddings.extend(batch_embs_np)
                        for i, orig_idx in enumerate(batch_original_indices):
                            cached_embeddings[orig_idx] = batch_embs_np[i]
                            if use_cache:
                                key = self.cache._generate_key([batch_texts[i]])[0]
                                self.cache.set(key, batch_embs[i])
                    else:
                        computed_embeddings.extend(batch_embs)
                        for i, orig_idx in enumerate(batch_original_indices):
                            cached_embeddings[orig_idx] = batch_embs[i]
                            if use_cache:
                                key = self.cache._generate_key([batch_texts[i]])[0]
                                self.cache.set(key, batch_embs[i])

                except Exception as e:
                    self._logger.error(f"Failed to embed batch {start // batch_size + 1}: {e}")
                    raise

        # ── Now stream the final embeddings in original order ────────────────
        all_embeddings = cached_embeddings  # already in correct order

        progress_bar = tqdm(
            range(0, len(all_embeddings), batch_size),
            desc="Yielding stream",
            disable=not show_progress
        )

        for start in progress_bar:
            batch = all_embeddings[start : start + batch_size]

            if return_format == "numpy":
                yield np.array(batch, dtype=np.float32)   # shape: (bs, dim)
            else:
                yield batch                               # list of lists

    def close(self) -> None:
        """Close cache (e.g., SQLite conn)."""
        self.cache.close()
