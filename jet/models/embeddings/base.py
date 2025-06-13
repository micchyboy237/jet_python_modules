import numpy as np
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from typing import Callable, Literal, Union, List, Optional, TypeAlias
import psutil
import math
from tqdm import tqdm
import torch
from jet.logger import logger
import mlx.core as mx
from jet.models.tokenizer.base import get_tokenizer, tokenize
from jet.models.tokenizer.utils import calculate_batch_size

EmbeddingOutput: TypeAlias = Union[List[int], List[List[int]],
                                   List[float], List[List[float]], np.ndarray, 'torch.Tensor', 'mx.array']


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


def generate_embeddings(
    input_data: Union[str, List[str]],
    model: str = "static-retrieval-mrl-en-v1",
    batch_size: int = 32,
    show_progress: bool = False
) -> Union[List[float], List[List[float]]]:
    """
    Generate embeddings for a single string or list of strings using SentenceTransformer.

    Args:
        input_data: A single string or list of strings to embed.
        model: Name of the SentenceTransformer model to use.
        batch_size: Batch size for embedding multiple strings.
        show_progress: Whether to display a progress bar for batch processing.

    Returns:
        List[float] for a single string input, or List[List[float]] for a list of strings.
    """
    logger.info("Generating embeddings for input type: %s, model: %s, show_progress: %s",
                type(input_data), model, show_progress)

    try:
        # Initialize SentenceTransformer with ONNX backend for Mac M1 compatibility
        embedder = SentenceTransformer(model, device="cpu", backend="onnx")
        logger.debug(
            "Embedding model initialized with device: %s", embedder.device)

        if isinstance(input_data, str):
            # Handle single string input
            logger.debug("Processing single string input: %s", input_data[:50])
            embedding = embedder.encode(input_data, convert_to_numpy=True)
            embedding = np.ascontiguousarray(embedding.astype(np.float32))
            logger.debug("Generated embedding shape: %s", embedding.shape)
            return embedding.tolist()

        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            # Handle list of strings input
            logger.debug("Processing %d strings in batches of %d",
                         len(input_data), batch_size)
            if not input_data:
                logger.info(
                    "Empty input list, returning empty list of embeddings")
                return []

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

            logger.debug("Generated embeddings shape: (%d, %d)", len(
                embeddings), len(embeddings[0]) if embeddings else 0)
            return embeddings

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
