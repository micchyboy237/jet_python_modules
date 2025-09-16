from functools import lru_cache
import sys
import time
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from tokenizers import Tokenizer
from typing import Callable, Literal, TypeVar, Union, List, Optional, TypeAlias
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


# Global model caches
_embed_model_cache: dict[str, SentenceTransformer] = {}
_rerank_model_cache: dict[str, CrossEncoder] = {}


def load_embed_model(model: EmbedModelType, truncate_dim: Optional[int] = None) -> SentenceTransformer:
    return SentenceTransformerRegistry.load_model(model, truncate_dim)


def load_rerank_model(model: EmbedModelType) -> CrossEncoder:
    model_id = resolve_model_value(model)
    if model_id in _rerank_model_cache:
        return _rerank_model_cache[model_id]
    try:
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
    start_time = time.time()
    if isinstance(chunk_texts, str):
        chunk_texts = [chunk_texts]
    embedder = load_embed_model(embed_model) if isinstance(
        embed_model, str) else embed_model
    if not chunk_texts:
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
    return embeddings if return_format == "list" else np.vstack(embeddings)


# Define generic type variables
T = TypeVar('T', np.ndarray, List[List[float]])


def generate_embeddings(
    input_data: Union[str, List[str]],
    model: Union[SentenceTransformer,
                 EmbedModelType] = "static-retrieval-mrl-en-v1",
    batch_size: int = 32,
    show_progress: bool = False,
    return_format: Literal["list", "numpy"] = "numpy",
    truncate_dim: Optional[int] = None,
) -> T:
    if return_format not in ["list", "numpy"]:
        raise ValueError("return_format must be 'list' or 'numpy'")
    try:
        embedder = (
            SentenceTransformerRegistry.load_model(model)
            if isinstance(model, str)
            else model
        )

        logger.gray(f"\nGenerating embeddings...")
        logger.debug(
            f"truncate_dim: {embedder.get_sentence_embedding_dimension()}\nmax_seq_length: {embedder.max_seq_length}")

        if isinstance(input_data, str):
            embedding = embedder.encode(
                input_data, convert_to_numpy=True, show_progress_bar=False)
            embedding = np.ascontiguousarray(embedding.astype(np.float32))
            if truncate_dim is not None and embedding.shape[-1] > truncate_dim:
                embedding = embedding[:truncate_dim]
            return embedding if return_format == "numpy" else embedding.tolist()

        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            if not input_data:
                return [] if return_format == "list" else np.array([], dtype=np.float32)

            # Pre-allocate NumPy array for embeddings
            embedding_dim = embedder.get_sentence_embedding_dimension()
            if truncate_dim is not None and truncate_dim < embedding_dim:
                embedding_dim = truncate_dim
            embeddings = np.empty(
                (len(input_data), embedding_dim), dtype=np.float32)

            total_batches = math.ceil(len(input_data) / batch_size)
            if show_progress:
                from tqdm import tqdm  # Use tqdm for efficient progress bar
                progress_bar = tqdm(total=total_batches,
                                    desc="Processing batches", disable=False)

            for i in range(0, len(input_data), batch_size):
                batch = input_data[i:i + batch_size]
                batch_embeddings = embedder.encode(
                    batch,
                    convert_to_numpy=True,
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                batch_embeddings = np.ascontiguousarray(
                    batch_embeddings.astype(np.float32))
                if truncate_dim is not None and batch_embeddings.shape[-1] > truncate_dim:
                    batch_embeddings = batch_embeddings[:, :truncate_dim]

                # Directly assign to pre-allocated array
                embeddings[i:i + len(batch)] = batch_embeddings

                if show_progress:
                    progress_bar.update(1)

            if show_progress:
                progress_bar.close()

            return embeddings.tolist() if return_format == "list" else embeddings

        else:
            logger.error(
                "Invalid input type: %s, expected str or List[str]", type(input_data))
            raise ValueError("Input must be a string or a list of strings")

    except Exception as e:
        logger.error("Failed to generate embeddings: %s", str(e))
        raise


def get_embedding_function(
    model_name: EmbedModelType,
    batch_size: int = 32,
    show_progress: bool = False,
    return_format: Literal["list", "numpy"] = "numpy",
    truncate_dim: Optional[int] = None,
) -> Callable[[Union[str, List[str]]], EmbeddingOutput]:
    def embed_func(x): return generate_embeddings(
        x, model_name, batch_size=batch_size, show_progress=show_progress, return_format=return_format, truncate_dim=truncate_dim)
    return embed_func
