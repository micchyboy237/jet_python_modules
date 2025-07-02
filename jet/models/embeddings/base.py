from functools import lru_cache
import time
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from tokenizers import Tokenizer
from typing import Callable, Literal, Union, List, Optional, TypeAlias
import psutil
import math
from tqdm import tqdm
import torch
from jet.data.utils import generate_key
from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry
from jet.models.utils import resolve_model_value
from jet.models.model_types import EmbedModelType
from jet.logger import logger
import mlx.core as mx
from jet.models.tokenizer.base import get_tokenizer, tokenize
from jet.models.tokenizer.utils import calculate_batch_size

EmbeddingOutput: TypeAlias = Union[List[int], List[List[int]],
                                   List[float], List[List[float]], np.ndarray, 'torch.Tensor', 'mx.array']


# Add at top of file with other imports

# Global model caches
_embed_model_cache: dict[str, SentenceTransformer] = {}
_rerank_model_cache: dict[str, CrossEncoder] = {}


def load_embed_model(model: EmbedModelType, truncate_dim: Optional[int] = None) -> SentenceTransformer:
    # _cache_key = generate_key(model, truncate_dim)
    # model_id = resolve_model_value(model)
    # if _cache_key in _embed_model_cache:
    #     logger.info(
    #         f"Reusing cached embedding model: {model_id} (cache key: {_cache_key})")
    #     return _embed_model_cache[_cache_key]
    # try:
    #     logger.info(f"Loading embedding model on CPU (onnx): {model_id}")
    #     model_instance = SentenceTransformer(
    #         model_id, device="cpu", backend="onnx", truncate_dim=truncate_dim,
    #         model_kwargs={'file_name': 'model.onnx', 'subfolder': 'onnx'})
    # except Exception as e:
    #     logger.warning(f"Falling back to MPS for embed model due to: {e}")
    #     model_instance = SentenceTransformer(model_id, device="mps")
    # _embed_model_cache[_cache_key] = model_instance
    # return model_instance
    instance = SentenceTransformerRegistry()
    return instance.load_model(model, truncate_dim)


def load_rerank_model(model: EmbedModelType) -> CrossEncoder:
    model_id = resolve_model_value(model)
    if model_id in _rerank_model_cache:
        logger.info(f"Reusing cached rerank model: {model_id}")
        return _rerank_model_cache[model_id]
    try:
        logger.info(f"Loading rerank model on CPU (onnx): {model_id}")
        model = CrossEncoder(model_id, device="cpu", backend="onnx")
    except Exception as e:
        logger.warning(f"Falling back to MPS for rerank model due to: {e}")
        model = CrossEncoder(model_id, device="mps")
    _rerank_model_cache[model_id] = model
    return model


def last_token_pool(last_hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
    left_padding = mx.sum(attention_mask[:, -1]) == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = mx.sum(attention_mask, axis=1) - 1
    batch_size = last_hidden_states.shape[0]
    indices = mx.stack([mx.arange(batch_size), sequence_lengths], axis=1)
    return last_hidden_states[indices[:, 0], indices[:, 1]]


def generate_multiple(
    query: Union[str, List[str]],
    func: Callable[[Union[str, List[str]]], EmbeddingOutput],
    batch_size: int,
    return_format: Literal["list", "numpy", "torch", "mlx"] = "list"
) -> EmbeddingOutput:
    if return_format not in ["list", "numpy", "torch", "mlx"]:
        raise ValueError(
            "return_format must be 'list', 'numpy', 'torch', or 'mlx'")
    if isinstance(query, list):
        embeddings = []
        for i in range(0, len(query), batch_size):
            batch_result = func(query[i:i + batch_size])
            embeddings.extend(batch_result if return_format ==
                              "list" else batch_result.tolist())
        if return_format == "numpy":
            return np.array(embeddings, dtype=np.float32)
        elif return_format == "torch":
            return torch.tensor(embeddings, dtype=torch.float32)
        elif return_format == "mlx":
            return mx.array(embeddings, dtype=mx.float32)
        return embeddings
    else:
        result = func(query)
        if return_format == "numpy":
            return np.array(result, dtype=np.float32)
        elif return_format == "torch":
            return torch.tensor(result, dtype=torch.float32)
        elif return_format == "mlx":
            return mx.array(result, dtype=mx.float32)
        return result


def embed_chunks_parallel(chunk_texts: Union[str, List[str]], embed_model: Union[EmbedModelType, SentenceTransformer], batch_size: int = 32, show_progress: bool = False, return_format: Literal["list", "numpy"] = "list") -> Union[List[List[float]], np.ndarray]:
    """Embed chunks in batches to optimize performance."""
    start_time = time.time()
    if isinstance(chunk_texts, str):
        chunk_texts = [chunk_texts]
    logger.info("Embedding %d chunks in batches", len(chunk_texts))
    embedder = load_embed_model(embed_model) if isinstance(
        embed_model, str) else embed_model
    if not chunk_texts:
        logger.info("No chunks to embed, returning empty array")
        return [] if return_format == "list" else np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32)
    embeddings = []
    embed_chunks_iter = range(0, len(chunk_texts), batch_size)
    if show_progress:
        embed_chunks_iter = tqdm(embed_chunks_iter, desc="Embedding chunks")
    for i in embed_chunks_iter:
        batch = chunk_texts[i:i + batch_size]
        try:
            batch_embeddings = embedder.encode(
                batch, convert_to_numpy=True, batch_size=batch_size)
            batch_embeddings = np.ascontiguousarray(
                batch_embeddings.astype(np.float32))
            if return_format == "list":
                embeddings.extend([embedding.tolist()
                                  for embedding in batch_embeddings])
            else:
                embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error("Error embedding batch: %s", e)
            for _ in batch:
                embeddings.append(
                    [0.0] * embedder.get_sentence_embedding_dimension() if return_format == "list"
                    else np.zeros(embedder.get_sentence_embedding_dimension(), dtype=np.float32))
    duration = time.time() - start_time
    logger.info("Embedding completed in %.3f seconds", duration)
    return embeddings if return_format == "list" else np.vstack(embeddings)


def generate_embeddings(
    input_data: Union[str, List[str]],
    model: Union[SentenceTransformer,
                 EmbedModelType] = "static-retrieval-mrl-en-v1",
    batch_size: int = 32,
    show_progress: bool = False,
    return_format: Literal["list", "numpy"] = "list"
) -> Union[List[float], List[List[float]], np.ndarray]:
    """
    Generate embeddings for a single string or list of strings using SentenceTransformer.

    Args:
        input_data: A single string or list of strings to embed.
        model: Model name (EmbedModelType), SentenceTransformer instance to use for embedding.
        batch_size: Batch size for embedding multiple strings.
        show_progress: Whether to display a progress bar for batch processing.
        return_format: Format of the returned embeddings ("list" or "numpy").

    Returns:
        List[float] or np.ndarray for a single string input, or List[List[float]] or np.ndarray
        for a list of strings, based on return_format.
    """
    logger.info("Generating embeddings for input type: %s, show_progress: %s, return_format: %s",
                type(input_data), show_progress, return_format)

    if return_format not in ["list", "numpy"]:
        raise ValueError("return_format must be 'list' or 'numpy'")

    try:
        # Initialize SentenceTransformer
        embedder = load_embed_model(model) if isinstance(model, str) else model
        logger.debug(
            "Embedding model initialized with device: %s", embedder.device)

        if isinstance(input_data, str):
            # Handle single string input
            logger.debug("Processing single string input: %s", input_data[:50])
            embedding = embedder.encode(input_data, convert_to_numpy=True)
            embedding = np.ascontiguousarray(embedding.astype(np.float32))
            logger.debug("Generated embedding shape: %s", embedding.shape)
            return embedding if return_format == "numpy" else embedding.tolist()

        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            # Handle list of strings input
            logger.debug("Processing %d strings in batches of %d",
                         len(input_data), batch_size)
            if not input_data:
                logger.info(
                    "Empty input list, returning empty list of embeddings")
                return [] if return_format == "list" else np.array([])

            embeddings = []
            # Use tqdm for progress bar if show_progress is True
            iterator = range(0, len(input_data), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Embedding texts", total=len(
                    range(0, len(input_data), batch_size)))

            for i in iterator:
                batch = input_data[i:i + batch_size]
                logger.debug("Encoding batch %d-%d", i,
                             min(i + batch_size, len(input_data)))
                batch_embeddings = embedder.encode(
                    batch,
                    convert_to_numpy=True,
                    batch_size=batch_size
                )
                batch_embeddings = np.ascontiguousarray(
                    batch_embeddings.astype(np.float32))
                embeddings.extend(batch_embeddings.tolist())

            embeddings = np.array(embeddings, dtype=np.float32)
            logger.debug("Generated embeddings shape: %s", embeddings.shape)
            return embeddings if return_format == "numpy" else embeddings.tolist()

        else:
            logger.error(
                "Invalid input type: %s, expected str or List[str]", type(input_data))
            raise ValueError("Input must be a string or a list of strings")

    except Exception as e:
        logger.error("Failed to generate embeddings: %s", str(e))
        raise


def get_embedding_function(
    model_name: str,
    batch_size: int = 32,
    show_progress: bool = False,
    return_format: Literal["list", "numpy", "torch", "mlx"] = "list",
    model: Optional[Callable] = None,
    use_last_token_pool: bool = True
) -> Callable[[Union[str, List[str]]], EmbeddingOutput]:
    def embed_func(x): return generate_embeddings(
        x, model_name, batch_size=batch_size, show_progress=show_progress)
    return embed_func
