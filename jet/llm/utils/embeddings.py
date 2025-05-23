import time
import logging
from typing import Optional, List
import zlib
import pickle
from pathlib import Path
import os
import json
import threading
from typing import Callable, Union, List
from jet.data.utils import hash_text
from functools import lru_cache
from jet.token.token_utils import get_model_max_tokens, split_texts, token_counter, truncate_texts
from jet.transformers.formatters import format_json
import numpy as np

from jet.logger import logger
from jet.logger.timer import time_it
import sentence_transformers
from tqdm import tqdm
from jet.llm.models import DEFAULT_SF_EMBED_MODEL, OLLAMA_EMBED_MODELS, OLLAMA_MODEL_CONTEXTS, OLLAMA_MODEL_EMBEDDING_TOKENS, OLLAMA_MODEL_NAMES
from sentence_transformers import SentenceTransformer
import requests
from typing import Any, Optional, Callable, Sequence, Union, List, TypedDict
from chromadb import Documents, EmbeddingFunction, Embeddings
from jet.llm.ollama.config import (
    large_embed_model,
    base_url,

)
from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()


class MemoryEmbedding(TypedDict):
    vector: List[float]  # Embedding vector
    metadata: str        # Any associated metadata (if applicable)


# GenerateEmbeddingsReturnType = Union[MemoryEmbedding, List[MemoryEmbedding]]
GenerateEmbeddingsReturnType = list[float] | list[list[float]]

GenerateMultipleReturnType = Callable[[
    Union[str, List[str]]], List[MemoryEmbedding]]


# Cache file configuration
CACHE_FILE = "embedding_cache.pkl"  # Switch to pickle for compression
CACHE_DIR = os.path.expanduser("~/.cache/jet_python_modules")
Path(CACHE_DIR).mkdir(exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_FILE)

# Thread-safe lock for file access
_cache_lock = threading.Lock()

# In-memory cache to reduce file I/O
_memory_cache = {}
MEMORY_CACHE_MAX_SIZE = 10000  # Max entries in memory cache
MEMORY_CACHE_PRUNE_RATIO = 0.8  # Prune to 80% when full


def load_cache() -> dict:
    """Load cache from file, return empty dict if file doesn't exist or is invalid."""
    global _memory_cache
    if _memory_cache:
        logger.debug(
            f"Using in-memory cache with {len(_memory_cache)} entries.")
        return _memory_cache

    try:
        with open(CACHE_PATH, 'rb') as f:
            # Decompress and load pickle
            compressed_data = f.read()
            data = zlib.decompress(compressed_data)
            cache = pickle.loads(data)
            logger.debug(
                f"Loaded cache from {CACHE_PATH} with {len(cache)} entries.")
            _memory_cache = cache
            return cache
    except (FileNotFoundError, pickle.PickleError, zlib.error, IOError) as e:
        logger.debug(
            f"Failed to load cache from {CACHE_PATH}: {e}. Starting with empty cache.")
        _memory_cache = {}
        return {}


def save_cache(cache: dict):
    """Save cache to file with compression and pruning."""
    try:
        if len(cache) > MEMORY_CACHE_MAX_SIZE:
            sorted_items = sorted(cache.items(), key=lambda x: x[0])
            pruned_size = int(MEMORY_CACHE_MAX_SIZE * MEMORY_CACHE_PRUNE_RATIO)
            pruned_cache = dict(sorted_items[:pruned_size])
            logger.info(
                f"Pruned cache to {len(pruned_cache)} entries to manage size.")
        else:
            pruned_cache = cache

        data = pickle.dumps(pruned_cache)
        compressed_data = zlib.compress(data, level=6)

        with open(CACHE_PATH, 'wb') as f:
            f.write(compressed_data)

        logger.debug(
            f"Saved cache to {CACHE_PATH} with {len(pruned_cache)} entries.")
    except (pickle.PickleError, zlib.error, IOError) as e:
        logger.error(f"Failed to save cache to {CACHE_PATH}: {e}")


def get_embedding_function(
    model_name: str,
    batch_size: int = 64
) -> Callable[[str | list[str]], list[float] | list[list[float]]]:
    """Retrieve embeddings with in-memory and file-based caching."""
    embed_func = initialize_embed_function(model_name, batch_size)

    def generate_cache_key(input_text: str | list[str]) -> str:
        """Generate a cache key based on model name, batch size, and input text."""
        if isinstance(input_text, str):
            text_hash = hash_text(input_text)
        else:
            text_hash = hash_text("".join(sorted(input_text)))
        return f"embed:{model_name}:{batch_size}:{text_hash}"

    def embedding_function(input_text: str | list[str]) -> list[float] | list[list[float]]:
        """Compute embeddings with caching."""
        single_input = isinstance(input_text, str)
        text_count = 1 if single_input else len(input_text)
        text_summary = (
            input_text[:100] + ("..." if len(input_text) > 100 else "")
            if single_input
            else ", ".join(t[:30] for t in input_text[:3])[:100] + ("..." if len(input_text) > 3 else "")
        )

        # Generate cache key
        cache_key = generate_cache_key(input_text)

        # Check in-memory cache
        with _cache_lock:
            if cache_key in _memory_cache:
                # logger.success(
                #     f"Memory cache hit for {'1 text' if single_input else f'{text_count} texts'} "
                #     f"(key: {cache_key}): {text_summary}"
                # )
                return _memory_cache[cache_key]

        # Check file cache
        with _cache_lock:
            cache = load_cache()
            if cache_key in cache:
                _memory_cache[cache_key] = cache[cache_key]
                # logger.success(
                #     f"File cache hit for {'1 text' if single_input else f'{text_count} texts'} "
                #     f"(key: {cache_key}, file: {CACHE_PATH}): {text_summary}"
                # )
                return cache[cache_key]

        logger.warning(
            f"Cache miss for {'1 text' if single_input else f'{text_count} texts'} "
            f"(key: {cache_key}, file: {CACHE_PATH}): {text_summary}. Computing embeddings..."
        )

        # Compute embeddings
        input_texts = [input_text] if single_input else input_text
        computed_embeddings = embed_func(input_texts)

        # Store in caches
        with _cache_lock:
            _memory_cache[cache_key] = computed_embeddings[0] if single_input else computed_embeddings
            cache[cache_key] = computed_embeddings[0] if single_input else computed_embeddings
            save_cache(cache)
            logger.info(
                f"Cached embeddings for {'1 text' if single_input else f'{text_count} texts'} "
                f"(key: {cache_key}, model: {model_name}, batch_size: {batch_size}, file: {CACHE_PATH})."
            )

        return computed_embeddings[0] if single_input else computed_embeddings

    return embedding_function


class OllamaEmbeddingFunction():
    def __init__(
            self,
            model_name: str = large_embed_model,
            batch_size: int = 32,
            key: str = "",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.key = key

    def __call__(self, input: str | list[str]) -> list[float] | list[list[float]]:
        logger.info(f"Generating Ollama embeddings...")
        logger.debug(f"Model: {self.model_name}")
        logger.debug(f"Max Context: {OLLAMA_MODEL_CONTEXTS[self.model_name]}")
        logger.debug(
            f"Embeddings Dim: {OLLAMA_MODEL_EMBEDDING_TOKENS[self.model_name]}")
        logger.debug(f"Texts: {len(input)}")
        logger.debug(f"Batch size: {self.batch_size}")
        logger.info(
            f"Total batches: {len(input) // self.batch_size + bool(len(input) % self.batch_size)}")

        def func(query: str | list[str]): return generate_embeddings(
            model=self.model_name,
            text=query,
            url=base_url,
            key=self.key,
        )

        batch_embeddings = generate_multiple(input, func, self.batch_size)

        if isinstance(input, str):
            return batch_embeddings[0]
        return batch_embeddings


class SFEmbeddingFunction():
    def __init__(self, model_name: str = DEFAULT_SF_EMBED_MODEL, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def __getattr__(self, name):
        """Delegate attribute/method calls to self.model if not found in SFEmbeddingFunction."""
        return getattr(self.model, name)

    def tokenize(self, documents: List[str]) -> List[List[str]]:
        # Tokenize documents into words using the sentence transformer tokenizer
        return [self.model.tokenize(doc) for doc in documents]

    def calculate_tokens(self, documents: List[str]) -> List[int]:
        # Calculate the token count for each document using the model's tokenizer
        tokenized = self.tokenize(documents)
        return [len(tokens) for tokens in tokenized]

    @time_it(function_name="generate_sf_batch_embeddings")
    def __call__(self, input: str | list[str]) -> list[float] | list[list[float]]:
        base_input = input

        logger.info(f"Generating SF embeddings...")
        logger.debug(f"Model: {self.model_name}")
        logger.debug(f"Texts: {len(input) if isinstance(input, list) else 1}")
        logger.debug(f"Batch size: {self.batch_size}")

        all_embeddings = self.model.encode(
            base_input, convert_to_tensor=True, show_progress_bar=False)

        return all_embeddings.tolist()


class SFRerankingFunction:
    def __init__(self, model_name: str = DEFAULT_SF_EMBED_MODEL, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = sentence_transformers.CrossEncoder(model_name)

    def __getattr__(self, name):
        """Delegate attribute/method calls to self.model if not found in SFEmbeddingFunction."""
        return getattr(self.model, name)

    def tokenize(self, documents: List[str]) -> List[List[str]]:
        """Tokenize documents using the sentence transformer tokenizer."""
        return [self.model.tokenize(doc) for doc in documents]

    def calculate_tokens(self, documents: List[str]) -> List[int]:
        """Calculate token count for each document using the model's tokenizer."""
        tokenized = self.tokenize(documents)
        return [len(tokens) for tokens in tokenized]

    @time_it(function_name="generate_sf_batch_embeddings")
    def __call__(self, input: str | list[str]) -> list[float] | list[list[float]]:
        logger.info(f"Generating SF embeddings...")
        logger.debug(f"Model: {self.model_name}")
        logger.debug(f"Texts: {len(input)}")
        logger.debug(f"Batch size: {self.batch_size}")

        if isinstance(input, str):
            input = [input]
        # Tokenize the input and calculate token counts for each document
        token_counts = self.calculate_tokens(input)

        batched_input = []
        current_batch = []
        current_token_count = 0

        # Split the input into batches based on the batch_size
        for doc, token_count in zip(input, token_counts):
            # Start a new batch when the batch size limit is reached
            if len(current_batch) >= self.batch_size:
                batched_input.append(current_batch)
                current_batch = [doc]
                current_token_count = token_count
            else:
                current_batch.append(doc)
                current_token_count += token_count

        if current_batch:  # Don't forget to add the last batch
            batched_input.append(current_batch)

        # Embed each batch using encode and decode methods
        all_embeddings = []
        for batch in batched_input:
            # Encode documents into embeddings
            embeddings = self.model.encode(batch, show_progress_bar=True)
            all_embeddings.extend(embeddings)

        return all_embeddings


global_embed_model_func = None
global_embed_model_name = None
global_embed_batch_size = 32


def initialize_embed_function(
    model_name: str, batch_size: int
) -> Callable[[str | list[str]], list[float] | list[list[float]]]:
    """Initialize and cache embedding functions globally."""
    global global_embed_model_func, global_embed_model_name, global_embed_batch_size

    if not global_embed_model_func or global_embed_model_name != model_name or global_embed_batch_size != batch_size:
        use_ollama = model_name in OLLAMA_EMBED_MODELS.__args__

        if use_ollama:
            embed_func = OllamaEmbeddingFunction(model_name, batch_size)
        else:
            embed_func = SFEmbeddingFunction(model_name, batch_size)

        global_embed_model_func = embed_func
        global_embed_model_name = model_name
        global_embed_batch_size = batch_size

    return global_embed_model_func


# def get_embedding_function(
#     model_name: str,
#     batch_size: int = 64
# ) -> Callable[[str | list[str]], list[float] | list[list[float]]]:
#     """Retrieve embeddings directly without using cache."""
#     embed_func = initialize_embed_function(model_name, batch_size)

#     def embedding_function(input_text: str | list[str]):
#         """Compute embeddings directly without caching."""
#         single_input = False
#         if isinstance(input_text, str):
#             input_text = [input_text]
#             single_input = True

#         # Compute embeddings for all inputs
#         computed_embeddings = embed_func(input_text)

#         return computed_embeddings[0] if single_input else computed_embeddings

#     return embedding_function


def get_reranking_function(
    model_name: str | OLLAMA_EMBED_MODELS,
    batch_size: int = 32,
) -> Callable[[str | list[str]], list[float] | list[list[float]]]:
    use_ollama = model_name in OLLAMA_EMBED_MODELS.__args__
    if use_ollama:
        return OllamaEmbeddingFunction(model_name=model_name, batch_size=batch_size)
    else:
        return SFRerankingFunction(model_name=model_name, batch_size=batch_size)


def ollama_embedding_function(texts, model) -> list[float] | list[list[float]]:
    if isinstance(texts, str):
        texts = [texts]

    from jet.llm.ollama.base import OllamaEmbedding

    embed_model = OllamaEmbedding(model_name=model)
    results = embed_model.get_general_text_embedding(texts)
    return results


def get_ollama_embedding_function(
    model: OLLAMA_EMBED_MODELS,
    batch_size: int = 32,
):
    return OllamaEmbeddingFunction(model_name=model, batch_size=batch_size)


def generate_multiple(
    query: str | list[str],
    func: Callable[[str | list[str]], list],  # Replace with the correct type
    batch_size: int = 32,
) -> list[list[float]]:
    if isinstance(query, list):
        embeddings = []
        pbar = tqdm(range(0, len(query), batch_size),
                    desc="Generating embeddings")
        for i in pbar:
            pbar.set_description(
                f"Generating embeddings batch {i // batch_size + 1}")
            embeddings.extend(func(query[i: i + batch_size]))
        return embeddings
    else:
        embeddings = func(query)
        return embeddings


def generate_embeddings(
    model: str,
    text: Union[str, list[str]],
    **kwargs
) -> GenerateEmbeddingsReturnType:
    url = kwargs.get("url", "")
    key = kwargs.get("key", "")

    text = [text] if isinstance(text, str) else text
    embeddings = generate_ollama_batch_embeddings(
        **{"model": model, "texts": text, "url": url, "key": key}
    )

    return embeddings[0] if isinstance(text, str) else embeddings


@time_it
def generate_ollama_batch_embeddings(
    model: OLLAMA_MODEL_NAMES,
    texts: list[str],
    url: str,
    key: str = "",
    max_tokens: Optional[int | float] = None,
    max_retries: int = 3
) -> list[list[float]]:
    if not max_tokens:
        max_tokens = 0.5

    model_max_tokens: int = get_model_max_tokens(model)

    if isinstance(max_tokens, float) and max_tokens < 1:
        max_tokens = int(model_max_tokens * max_tokens)
    else:
        max_tokens = int(max_tokens or model_max_tokens)

    token_counts: list[int] = token_counter(texts, model, prevent_total=True)

    # Identify texts that exceed the max token limit
    exceeded_texts = [
        (text, count) for text, count in zip(texts, token_counts) if count > model_max_tokens
    ]

    if exceeded_texts:
        logger.warning(
            "Some texts exceed the model's max token limit:\n" +
            "\n".join(
                f"- {count} tokens: {text[:50].replace("\n", " ")}..." for text, count in exceeded_texts)
        )
        # raise ValueError(
        #     f"{len(exceeded_texts)} texts exceed max token limit")
        # texts = split_texts(texts, model, max_tokens, 100)
        texts = truncate_texts(texts, model, max_tokens)

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{url}/api/embed",
                headers=headers,
                json={"model": model, "input": texts},
                timeout=300  # Set a timeout for reliability
            )
            r.raise_for_status()
            data = r.json()

            if "embeddings" in data:
                return data["embeddings"]
            else:
                logger.error("No embeddings found in response.")
                raise ValueError("Invalid response: missing embeddings")

        except requests.RequestException as e:
            logger.error(
                f"Attempt {attempt + 1} failed with error: {e}, Response text: {r.text if 'r' in locals() else 'No response'}")
            logger.error(f"\nModel: {model}")
            logger.error(f"\nTexts:\n{len(texts)}")
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
