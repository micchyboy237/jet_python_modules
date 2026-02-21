import os
from collections.abc import Callable, Iterator
from typing import (
    Literal,
    Union,
)

import numpy as np
from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.parallel_embeddings import embed_batch, embed_single
from jet.adapters.llama_cpp.types import (
    LLAMACPP_EMBED_KEYS,
    GenerateEmbeddingsReturnType,
)
from jet.adapters.llama_cpp.utils import resolve_model_key
from jet.logger import CustomLogger
from jet.models.embeddings.cache import EmbeddingCache
from jet.models.embeddings.utils import calculate_dynamic_batch_size
from jet.models.utils import get_context_size, get_embedding_size
from openai import OpenAI
from rich.console import Console
from tqdm import tqdm

console = Console()

SERVER_URL = os.getenv("LLAMA_CPP_EMBED_URL")
MODEL_NAME: LLAMACPP_EMBED_KEYS = os.getenv("LLAMA_CPP_EMBED_MODEL")

client = OpenAI(
    base_url=SERVER_URL,
    api_key="not-needed-for-local",  # llama.cpp ignores this
    # max_retries=3,
    timeout=300.0,  # 5 mins timeout (in seconds)
)


class InputTooLargeError(ValueError):
    """Custom exception for inputs exceeding the maximum allowed length."""

    def __init__(self, long_input_indexes: list[int], max_input_length: int):
        self.long_input_indexes = long_input_indexes
        self.max_input_length = max_input_length
        super().__init__(
            f"Inputs at indexes {long_input_indexes} are too long (> {max_input_length} tokens). "
            "Please reduce input size or increase server physical batch size."
        )


class LlamacppEmbedding:
    """
    Initialize the Llama.cpp embedding client.

    This client communicates with a `llama-server` instance exposing an
    OpenAI-compatible `/v1/embeddings` API and supports optional caching
    and dynamic batch sizing.

    Args:
        model (LLAMACPP_EMBED_KEYS):
            Embedding model identifier or alias resolved via
            `resolve_model_key`. Must be compatible with llama.cpp
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
        model: LLAMACPP_EMBED_KEYS = os.getenv("LLAMA_CPP_EMBED_MODEL"),
        base_url: str | None = os.getenv("LLAMA_CPP_EMBED_URL"),
        max_retries: int = 3,
        cache_backend: Literal["memory", "file", "sqlite"] = "sqlite",
        cache_ttl: int | None = None,
        cache_max_size: int = 10000,
        use_cache: bool = False,
        use_dynamic_batch_sizing: bool = False,
        verbose: bool = True,
        max_workers: int = 6,
        logger: CustomLogger | None = None,
    ):
        if not base_url:
            raise ValueError(
                "base_url must be provided. Set the LLAMA_CPP_EMBED_URL environment variable or pass base_url explicitly."
            )
        self.model = resolve_model_key(model)
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.use_dynamic_batch_sizing = use_dynamic_batch_sizing
        self.verbose = verbose
        self.max_workers = max_workers
        self.cache = EmbeddingCache(
            backend=cache_backend,
            max_size=cache_max_size,
            ttl=cache_ttl,
            namespace=f"llama_{self.model}",
        )

        self._logger = logger or CustomLogger()

    def __call__(
        self,
        inputs: str | list[str],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: bool | None = None,
        use_dynamic_batch_sizing: bool | None = None,
    ) -> GenerateEmbeddingsReturnType:
        """Make the instance callable to generate embeddings, equivalent to get_embeddings."""
        return self.get_embeddings(
            inputs,
            return_format=return_format,
            batch_size=batch_size,
            show_progress=show_progress,
            use_cache=use_cache if use_cache is not None else self.use_cache,
            use_dynamic_batch_sizing=use_dynamic_batch_sizing,
        )

    def embed(
        self,
        inputs: str | list[str],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        max_input_length: int | None = None,
        use_cache: bool | None = None,
        use_dynamic_batch_sizing: bool | None = None,
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
        inputs: str | list[str],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        max_input_length: int | None = None,
        use_cache: bool | None = None,
        use_dynamic_batch_sizing: bool | None = None,
    ) -> GenerateEmbeddingsReturnType:
        if isinstance(inputs, str):
            return embed_single(
                text=inputs,
                model=self.model,
                return_format=return_format,
            )

        return embed_batch(
            texts=inputs,
            model=self.model,
            max_workers=self.max_workers,
            show_progress=show_progress,
            return_format=return_format,
            batch_size=batch_size,
        )

    def get_embedding_function(
        self,
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: bool | None = None,
        use_dynamic_batch_sizing: bool | None = None,
    ) -> Callable[[str | list[str]], GenerateEmbeddingsReturnType]:
        """Return callable with caching and optional dynamic batch sizing."""

        def embedding_function(
            inputs: str | list[str],
        ) -> GenerateEmbeddingsReturnType:
            return self.get_embeddings(
                inputs,
                return_format=return_format,
                batch_size=batch_size,
                show_progress=show_progress,
                use_cache=use_cache if use_cache is not None else self.use_cache,
                use_dynamic_batch_sizing=use_dynamic_batch_sizing
                if use_dynamic_batch_sizing is not None
                else self.use_dynamic_batch_sizing,
            )

        return embedding_function

    def get_embeddings_stream(
        self,
        inputs: str | list[str],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        max_input_length: int | None = None,
        use_cache: bool | None = None,
        use_dynamic_batch_sizing: bool | None = None,
    ) -> Iterator[GenerateEmbeddingsReturnType]:
        """
        Stream embeddings with per-text caching, yielding all cache hits first (batched, ordered),
        then streaming batches of misses immediately as they are computed and cached.
        """
        use_cache = use_cache if use_cache is not None else self.use_cache
        use_dynamic_batch_sizing = (
            use_dynamic_batch_sizing
            if use_dynamic_batch_sizing is not None
            else self.use_dynamic_batch_sizing
        )

        input_list = [inputs] if isinstance(inputs, str) else inputs
        valid_inputs = [i for i in input_list if isinstance(i, str) and i.strip()]
        if not valid_inputs:
            raise ValueError(
                "inputs must be a non-empty string or list of non-empty strings"
            )

        embedding_size = get_embedding_size(self.model)
        context_size = get_context_size(self.model)
        max_length = max_input_length if max_input_length is not None else context_size
        if max_length <= 0:
            max_length = 512

        token_counts = token_counter(valid_inputs, self.model, prevent_total=True)
        long_inputs = [idx for idx, cnt in enumerate(token_counts) if cnt > max_length]
        if long_inputs:
            raise InputTooLargeError(long_inputs, max_length)

        if use_dynamic_batch_sizing:
            batch_size = calculate_dynamic_batch_size(
                token_counts=token_counts,
                embedding_size=embedding_size,
                context_size=context_size,
            )
            if self.verbose:
                self._logger.debug(f"Dynamic batch size: {batch_size}")

        # ------------------ Phase 1: Per-text cache scan ----------------------
        cached_embeddings: list[list[float] | np.ndarray | None] = [None] * len(
            valid_inputs
        )
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        if use_cache:
            for idx, text in enumerate(valid_inputs):
                key = self.cache._generate_key(text)
                emb = self.cache.get(key)
                if emb is not None:
                    # Migration / safety for old buggy multi-vector cache entries
                    if (
                        isinstance(emb, list)
                        and emb
                        and isinstance(emb[0], (list, tuple))
                    ):
                        emb = emb[0]  # take first — or you could raise / skip
                    cached_embeddings[idx] = (
                        np.array(emb, dtype=np.float32)
                        if return_format == "numpy"
                        else emb
                    )
                else:
                    miss_indices.append(idx)
                    miss_texts.append(text)
        else:
            miss_indices = list(range(len(valid_inputs)))
            miss_texts = valid_inputs

        # ------------- Phase 2: Yield all cache hits (batched, ordered) -----------------
        cached_pairs = [
            (idx, emb) for idx, emb in enumerate(cached_embeddings) if emb is not None
        ]
        for start in range(0, len(cached_pairs), batch_size):
            batch = cached_pairs[start : start + batch_size]
            # Each batch: [(orig_idx, emb), ...]
            if not batch:
                continue
            _, batch_ems = zip(*batch)
            if return_format == "numpy":
                yield self.transform_data(np.array(batch_ems, dtype=np.float32))
            else:
                yield list(batch_ems)

        # ----------- Phase 3: For misses, compute + yield each batch immediately ---------
        if miss_texts:
            progress_bar = tqdm(
                range(0, len(miss_texts), batch_size),
                desc="Embedding misses",
                disable=not show_progress,
            )

            for start in progress_bar:
                end = start + batch_size
                batch_texts = miss_texts[start:end]
                batch_indices = miss_indices[start:end]
                try:
                    response = client.embeddings.create(
                        model=self.model, input=batch_texts
                    )
                    batch_embs = [d.embedding for d in response.data]
                    if return_format == "numpy":
                        batch_np = np.array(batch_embs, dtype=np.float32)
                        yield self.transform_data(batch_np)
                        for i, orig_idx in enumerate(batch_indices):
                            # Optionally update cache
                            if use_cache:
                                key = self.cache._generate_key(batch_texts[i])
                                self.cache.set(key, batch_embs[i])
                    else:
                        yield batch_embs
                        for i, orig_idx in enumerate(batch_indices):
                            if use_cache:
                                key = self.cache._generate_key(batch_texts[i])
                                self.cache.set(key, batch_embs[i])
                except Exception as e:
                    self._logger.error(
                        f"Failed to embed batch {start // batch_size + 1}: {e}"
                    )
                    raise

    def transform_data(
        self,
        embeddings: list[float] | np.ndarray,
        truncate_dim: int | None = None,
    ) -> np.ndarray:
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        if truncate_dim is not None and embeddings.shape[-1] > truncate_dim:
            embeddings = embeddings[:truncate_dim]
        return embeddings

    def close(self) -> None:
        """Close cache (e.g., SQLite conn)."""
        self.cache.close()

    def reset_cache(self, force: bool = False) -> "LlamacppEmbedding":
        """
        Clear/reset all cached embeddings.

        Args:
            force: Reserved for future use (confirmation / selective reset modes)

        Returns:
            self (for method chaining)
        """
        self.cache.reset()
        if getattr(self, "verbose", False):
            self._logger.info(f"Embedding cache reset (backend: {self.cache.backend})")
        return self

    def embed_parallel(
        self,
        inputs: list[str],
        show_progress: bool = True,
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int | None = 32,  # sensible default
        progress_description: str = "Embedding texts",
    ) -> Union[list[list[float]], np.ndarray]:
        """
        Embed multiple texts in parallel using ThreadPoolExecutor + batching.
        """
        return embed_batch(
            texts=inputs,
            show_progress=show_progress,
            return_format=return_format,
            batch_size=batch_size,
            progress_description=progress_description,
        )
