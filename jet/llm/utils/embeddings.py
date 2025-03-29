from jet.cache.joblib.utils import load_persistent_cache, save_persistent_cache, ttl_cache
from jet.data.utils import generate_key, hash_text
from jet.token.token_utils import get_model_max_tokens, split_texts, token_counter, truncate_texts
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

    @time_it(function_name="generate_ollama_batch_embeddings")
    def __call__(self, input: str | list[str]) -> list[float] | list[list[float]]:
        logger.info(f"Generating Ollama embeddings...")
        logger.debug(f"Model: {self.model_name}")
        logger.debug(f"Max Context: {OLLAMA_MODEL_CONTEXTS[self.model_name]}")
        logger.debug(
            f"Embeddings Dim: {OLLAMA_MODEL_EMBEDDING_TOKENS[self.model_name]}")
        logger.debug(f"Texts: {len(input)}")
        logger.debug(f"Batch size: {self.batch_size}")

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

        # if isinstance(input, str):
        #     input = [input]

        # # Tokenize the input and calculate token counts for each document
        # token_counts = self.calculate_tokens(input)

        # batched_input = []
        # current_batch = []
        # current_token_count = 0

        # # Split the input into batches based on the batch_size
        # for doc, token_count in zip(input, token_counts):
        #     # Start a new batch when the batch size limit is reached
        #     if len(current_batch) >= self.batch_size:
        #         batched_input.append(current_batch)
        #         current_batch = [doc]
        #         current_token_count = token_count
        #     else:
        #         current_batch.append(doc)
        #         current_token_count += token_count

        # if current_batch:  # Don't forget to add the last batch
        #     batched_input.append(current_batch)

        # # Embed each batch using encode and decode methods
        # all_embeddings = []
        # for batch in batched_input:
        #     # Encode documents into embeddings
        #     embeddings = self.model.encode(batch, show_progress_bar=False)
        #     all_embeddings.extend(embeddings)

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


def get_embedding_function(
    model_name: str, batch_size: int = 64
) -> Callable[[str | list[str]], list[float] | list[list[float]]]:
    """Retrieve or cache embeddings using TTLCache and persistent storage."""

    # ✅ Load cache from embeddings.pkl
    cache_file = "embeddings.pkl"
    load_persistent_cache(cache_file)

    embed_func = initialize_embed_function(model_name, batch_size)

    def cached_embedding(input_text: str | list[str]):
        """Fetch embedding from cache or compute if not cached."""
        single_input = False
        if isinstance(input_text, str):
            input_text = [input_text]
            single_input = True

        results = [None] * len(input_text)
        missing_texts, missing_indices = [], []

        # ✅ Check TTLCache first
        for idx, text in enumerate(input_text):
            text_key = generate_key(model_name, text)

            if text_key in ttl_cache:
                results[idx] = ttl_cache[text_key]
                logger.success(f"Cache hit: Embedding {idx} - {text[:30]}")
            else:
                missing_texts.append(text)
                missing_indices.append(idx)
                logger.debug(f"Cache miss: Embedding {idx} - {text[:30]}")

        # ✅ Compute missing embeddings
        if missing_texts:
            computed_embeddings = embed_func(missing_texts)

            for idx, (text, embedding) in zip(missing_indices, zip(missing_texts, computed_embeddings)):
                text_key = generate_key(model_name, text)
                ttl_cache[text_key] = embedding  # ✅ Store in TTLCache
                results[idx] = embedding

            # ✅ Save updated cache to embeddings.pkl
            save_persistent_cache(cache_file)

        return results[0] if single_input else results

    return cached_embedding


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


def generate_ollama_batch_embeddings(
    model: OLLAMA_MODEL_NAMES,
    texts: list[str], url: str,
    key: str = "",
    max_tokens: Optional[int | float] = None,
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

    try:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

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
        logger.error(f"Request error: {e}")
        raise e
