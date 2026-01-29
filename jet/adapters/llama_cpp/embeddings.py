import numpy as np
from openai import OpenAI
from typing import (
    Iterator,
    List,
    Tuple,
    Union,
    Literal,
    Callable,
    Optional,
    TypedDict,
)
from tqdm import tqdm
from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_TYPES
from jet.adapters.llama_cpp.utils import resolve_model_value
from jet.models.utils import get_context_size, get_embedding_size
from jet.models.embeddings.utils import calculate_dynamic_batch_size
from jet.models.embeddings.cache import EmbeddingCache
from jet.logger import CustomLogger


# ────────────────────────────────────────────────────────────────────────────────
# Type Definitions
# ────────────────────────────────────────────────────────────────────────────────

EmbeddingVector = Union[List[float], np.ndarray]
EmbeddingBatch = List[EmbeddingVector]
EmbeddingOutput = Union[
    EmbeddingBatch, np.ndarray
]  # list of vectors or single 2D array
EmbeddingInput = Union[str, List[str]]


class EmbeddingResultItem(TypedDict):
    object: Literal["embedding"]
    embedding: List[float]
    index: int


class EmbeddingResponse(TypedDict):
    object: Literal["list"]
    data: List[EmbeddingResultItem]
    model: str
    usage: dict[str, int]


class SearchResultType(TypedDict):
    index: int
    text: str
    score: float


GenerateEmbeddingsReturnType = EmbeddingOutput  # kept for backward compatibility


def cosine_similarity(vec1: EmbeddingVector, vec2: EmbeddingVector) -> float:
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


class InputTooLargeError(ValueError):
    """Custom exception for inputs exceeding the maximum allowed length."""

    def __init__(self, long_input_indexes: List[int], max_input_length: int):
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
        self.client = OpenAI(
            base_url=base_url, api_key="no-key-required", max_retries=max_retries
        )
        self.model = resolve_model_value(model)
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.use_dynamic_batch_sizing = use_dynamic_batch_sizing
        self.verbose = verbose
        self.cache = EmbeddingCache(
            backend=cache_backend,
            max_size=cache_max_size,
            ttl=cache_ttl,
            namespace=f"llama_{self.model}",
        )

        self._logger = logger or CustomLogger()

    def __call__(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None,
    ) -> GenerateEmbeddingsReturnType:
        """Make the instance callable to generate embeddings, equivalent to get_embeddings."""
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

    def encode(
        self,
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        max_input_length: Optional[int] = None,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None,
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
        use_dynamic_batch_sizing: Optional[bool] = None,
    ) -> GenerateEmbeddingsReturnType:
        """Generate embeddings with caching and optional dynamic batch sizing."""
        use_cache = use_cache if use_cache is not None else self.use_cache
        use_dynamic_batch_sizing = (
            use_dynamic_batch_sizing
            if use_dynamic_batch_sizing is not None
            else self.use_dynamic_batch_sizing
        )

        input_list = [inputs] if isinstance(inputs, str) else inputs
        valid_inputs = [i for i in input_list if isinstance(i, str) and i.strip()]
        invalid_inputs = [
            i for i in input_list if not (isinstance(i, str) and i.strip())
        ]

        if invalid_inputs:
            self._logger.warning(
                f"Warning: Skipped {len(invalid_inputs)} invalid inputs: {invalid_inputs}"
            )
        if not valid_inputs:
            raise ValueError(
                "No valid inputs provided: inputs must be a non-empty string or list of non-empty strings"
            )

        embedding_size = get_embedding_size(self.model)
        context_size = get_context_size(self.model)
        max_length = max_input_length if max_input_length is not None else context_size

        if max_length <= 0:
            self._logger.warning(
                f"Warning: Invalid max_input_length ({max_length}) from get_context_size; falling back to 512"
            )
            max_length = 512
        elif max_length <= 0:
            max_length = 512

        token_counts: List[int] = token_counter(
            valid_inputs, self.model, prevent_total=True
        )

        # Log detailed input statistics
        if self.verbose:
            self._logger.info(
                f"Embedding stats -> model: {self.model}, "
                f"embedding_size: {embedding_size}, "
                f"context_size: {context_size}, "
                f"max_length: {max_length}"
            )
            self._logger.debug(f"\nInputs: {len(input_list)}")
            self._logger.debug(
                f"Tokens\nmax: {max(token_counts)}\nmin: {min(token_counts)}"
            )

        long_inputs = [
            (count, idx) for idx, count in enumerate(token_counts) if count > max_length
        ]

        if long_inputs:
            long_input_indexes = [idx for _, idx in long_inputs]
            self._logger.error(
                f"Error: Found {len(long_inputs)} inputs exceeding max length ({max_length} tokens): indexes {long_input_indexes}"
            )
            raise InputTooLargeError(long_input_indexes, max_length)

        # Apply dynamic batch sizing if enabled
        if use_dynamic_batch_sizing:
            dynamic_batch_size = calculate_dynamic_batch_size(
                token_counts=token_counts,
                embedding_size=embedding_size,
                context_size=context_size,
            )
            batch_size = dynamic_batch_size
            if self.verbose:
                self._logger.debug(
                    f"Dynamic batch sizing enabled. Using batch_size: {batch_size}"
                )
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
                    self._logger.debug(
                        f"Cache hit for {len(valid_inputs)} texts (key: {cache_key[:16]}...)"
                    )
                return result
            if self.verbose:
                self._logger.debug(
                    f"Cache miss for {len(valid_inputs)} texts (key: {cache_key[:16]}...). Computing..."
                )

        embeddings = []
        progress_bar = tqdm(
            range(0, len(valid_inputs), batch_size),
            desc="Processing batches",
            disable=not show_progress,
        )

        for i in progress_bar:
            batch = valid_inputs[i : i + batch_size]
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                batch_embeddings = [d.embedding for d in response.data]
                if return_format == "numpy":
                    batch_embeddings = [np.array(emb) for emb in batch_embeddings]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                self._logger.error(
                    f"Error generating embeddings for batch {i // batch_size + 1}: {e}"
                )
                raise

        final_embeddings = (
            embeddings
            if return_format != "numpy"
            else np.array(embeddings, dtype=np.float32)
        )
        final_embeddings = self.transform_data(final_embeddings)

        if use_cache:
            self.cache.set(
                cache_key,
                final_embeddings.tolist()
                if return_format == "numpy"
                else final_embeddings,
            )
            if self.verbose:
                self._logger.info(
                    f"Cached embeddings for {len(valid_inputs)} texts (key: {cache_key[:16]}...)"
                )

        return final_embeddings

    def get_embedding_function(
        self,
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None,
    ) -> Callable[[Union[str, List[str]]], GenerateEmbeddingsReturnType]:
        """Return callable with caching and optional dynamic batch sizing."""

        def embedding_function(
            inputs: Union[str, List[str]],
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
        inputs: Union[str, List[str]],
        return_format: Literal["numpy", "list"] = "numpy",
        batch_size: int = 32,
        show_progress: bool = True,
        max_input_length: Optional[int] = None,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None,
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
        cached_embeddings: List[Optional[Union[List[float], np.ndarray]]] = [
            None
        ] * len(valid_inputs)
        miss_indices: List[int] = []
        miss_texts: List[str] = []

        if use_cache:
            for idx, text in enumerate(valid_inputs):
                key = self.cache._generate_key([text])[0]
                emb = self.cache.get(key)
                if emb is not None:
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
                    response = self.client.embeddings.create(
                        model=self.model, input=batch_texts
                    )
                    batch_embs = [d.embedding for d in response.data]
                    if return_format == "numpy":
                        batch_np = np.array(batch_embs, dtype=np.float32)
                        yield self.transform_data(batch_np)
                        for i, orig_idx in enumerate(batch_indices):
                            # Optionally update cache
                            if use_cache:
                                key = self.cache._generate_key([batch_texts[i]])[0]
                                self.cache.set(key, batch_embs[i])
                    else:
                        yield batch_embs
                        for i, orig_idx in enumerate(batch_indices):
                            if use_cache:
                                key = self.cache._generate_key([batch_texts[i]])[0]
                                self.cache.set(key, batch_embs[i])
                except Exception as e:
                    self._logger.error(
                        f"Failed to embed batch {start // batch_size + 1}: {e}"
                    )
                    raise

    def transform_data(
        self,
        embeddings: Union[List[float], np.ndarray],
        truncate_dim: Optional[int] = None,
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

    def search(
        self,
        query: str,
        documents: List[str],
        *,
        top_k: Optional[int] = None,
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None,
    ) -> List[SearchResultType]:
        """
        Semantic search — computes query + all documents embeddings in **one call**.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not documents:
            return []

        # Combine query + documents → single embedding call
        all_texts = [query] + documents

        all_embs_list = self.get_embeddings(
            all_texts,
            return_format="list",
            batch_size=batch_size,
            show_progress=show_progress,
            use_cache=use_cache,
            use_dynamic_batch_sizing=use_dynamic_batch_sizing,
        )

        # First embedding belongs to the query
        query_emb = all_embs_list[0]
        doc_embs = all_embs_list[1:]

        results: List[SearchResultType] = []
        for i, (text, emb) in enumerate(zip(documents, doc_embs), start=0):
            score = cosine_similarity(query_emb, emb)
            item: SearchResultType = {"index": i, "text": text, "score": score}
            results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        if top_k is not None:
            results = results[:top_k]

        return results

    def search_stream(
        self,
        query: str,
        documents: List[str],
        *,
        top_k: int = 10,
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: Optional[bool] = None,
        use_dynamic_batch_sizing: Optional[bool] = None,
        yield_all: bool = False,  # if True → yield every result as soon as computed
        min_score_threshold: float = -1.0,  # optional early filtering
    ) -> Iterator[SearchResultType]:
        """
        Streaming semantic search — yields best matches as soon as they are ready.

        Behavior:
          - First yields all cache hits (in original document order)
          - Then computes missing embeddings in batches → yields results as soon as
            each batch is finished (sorted within batch by score)
          - Final yield contains remaining unsorted tail if needed

        Args:
            query: search query
            documents: list of texts to search in
            top_k: how many best results to eventually return (default 10)
            batch_size: embedding batch size for uncached documents
            show_progress: show tqdm progress bar for embedding computation
            use_cache / use_dynamic_batch_sizing: same as in get_embeddings
            yield_all: if True, yields EVERY result as soon as computed (not sorted)
                       if False (default), tries to yield better results first
            min_score_threshold: only yield results >= this cosine score

        Yields:
            SearchResultType items one by one (best first when possible)
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not documents:
            return

        use_cache = use_cache if use_cache is not None else self.use_cache
        use_dynamic = (
            use_dynamic_batch_sizing
            if use_dynamic_batch_sizing is not None
            else self.use_dynamic_batch_sizing
        )

        # ─── Prepare ─────────────────────────────────────────────────────────────
        n_docs = len(documents)

        # We'll keep track of (original_index, text, score) for results
        # None score = not yet computed
        results: List[Optional[Tuple[int, str, float]]] = [None] * n_docs

        # ─── Phase 1: Quick cache lookup for all documents ───────────────────────
        miss_indices: List[int] = []
        miss_texts: List[str] = []

        query_emb: Optional[Union[List[float], np.ndarray]] = None

        if use_cache:
            # Cache query
            q_key = self.cache._generate_key([query])[0]
            cached_query = self.cache.get(q_key)
            if cached_query is not None:
                query_emb = cached_query

            # Cache documents
            for i, doc in enumerate(documents):
                if not doc.strip():
                    continue
                key = self.cache._generate_key([doc])[0]
                emb = self.cache.get(key)
                if emb is not None:
                    score = cosine_similarity(
                        query_emb if query_emb is not None else [], emb
                    )
                    results[i] = (i, doc, score)
                else:
                    miss_indices.append(i)
                    miss_texts.append(doc)
        else:
            miss_indices = list(range(n_docs))
            miss_texts = documents[:]

        # If query wasn't cached → we must compute it anyway
        if query_emb is None:
            # We'll compute query + first batch together if possible
            pass

        # ─── Yield already cached & scored documents ─────────────────────────────
        cached_results = [r for r in results if r is not None]
        if cached_results:
            cached_results.sort(key=lambda x: x[2], reverse=True)  # sort by score desc
            for idx, text, score in cached_results:
                if score >= min_score_threshold:
                    yield {"index": idx, "text": text, "score": score}

        # ─── If top_k already reached from cache ─────────────────────────────────
        yielded_so_far = sum(1 for r in cached_results if r[2] >= min_score_threshold)
        if yielded_so_far >= top_k and not yield_all:
            return

        # ─── Phase 2: Compute missing embeddings (including query if needed) ─────
        to_embed = miss_texts
        to_embed_indices = miss_indices

        # If query not cached → prepend it
        query_needs_compute = query_emb is None
        if query_needs_compute:
            to_embed = [query] + to_embed
            # We'll treat query as virtual index -1
            to_embed_indices = [-1] + to_embed_indices

        if not to_embed:
            # Everything was cached → sort remaining and finish
            remaining = [r for r in results if r is not None]
            remaining.sort(key=lambda x: x[2], reverse=True)
            for i, text, score in remaining[yielded_so_far:]:
                if score >= min_score_threshold:
                    yield {"index": i, "text": text, "score": score}
            return

        # Adjust batch size dynamically if requested
        if use_dynamic:
            token_counts = token_counter(to_embed, self.model, prevent_total=True)
            batch_size = calculate_dynamic_batch_size(
                token_counts=token_counts,
                embedding_size=get_embedding_size(self.model),
                context_size=get_context_size(self.model),
            )

        # ─── Stream computation ──────────────────────────────────────────────────
        yielded = yielded_so_far
        partial_results: List[Tuple[int, str, float]] = []

        pbar = tqdm(
            range(0, len(to_embed), batch_size),
            desc="Embedding & scoring stream",
            disable=not show_progress,
            total=len(to_embed),
        )

        for start in pbar:
            end = start + batch_size
            batch_texts = to_embed[start:end]
            batch_orig_indices = to_embed_indices[start:end]

            try:
                resp = self.client.embeddings.create(
                    model=self.model, input=batch_texts
                )
                batch_embs = [d.embedding for d in resp.data]

                # ─── Special handling if query is in this batch ─────────────────
                query_local_idx = None
                if query_needs_compute and -1 in batch_orig_indices:
                    query_local_idx = batch_orig_indices.index(-1)
                    query_emb = batch_embs[query_local_idx]
                    # Cache query if enabled
                    if use_cache:
                        self.cache.set(q_key, query_emb)
                    # Remove query from batch for scoring
                    batch_embs = [
                        e for i, e in zip(batch_orig_indices, batch_embs) if i != -1
                    ]
                    batch_orig_indices = [i for i in batch_orig_indices if i != -1]

                # Now score documents
                for doc_idx, doc_text, doc_emb in zip(
                    batch_orig_indices, batch_texts, batch_embs
                ):
                    score = cosine_similarity(query_emb, doc_emb)

                    # Cache if enabled
                    if use_cache:
                        key = self.cache._generate_key([doc_text])[0]
                        self.cache.set(key, doc_emb)

                    # Store & consider yielding
                    item = (doc_idx, doc_text, score)
                    partial_results.append(item)

                    if yield_all:
                        if score >= min_score_threshold:
                            yield {"index": doc_idx, "text": doc_text, "score": score}
                            yielded += 1
                            if yielded >= top_k:
                                return

            except Exception as e:
                self._logger.error(f"Embedding batch failed: {e}")
                raise

        # ─── Final sort & yield remaining best results ───────────────────────────
        partial_results.sort(key=lambda x: x[2], reverse=True)

        for idx, text, score in partial_results:
            if score >= min_score_threshold:
                yield {"index": idx, "text": text, "score": score}
                yielded += 1
                if not yield_all and yielded >= top_k:
                    break
