import numpy as np
from tokenizers import Tokenizer
from typing import Callable, Literal, Union, List, Optional, TypeAlias
import psutil
import math
from tqdm import tqdm
import torch
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
    texts: Union[str, List[str]],
    model_name_or_tokenizer: Union[str, Tokenizer],
    batch_size: Optional[int] = None,
    show_progress: bool = False,
    return_format: Literal["list", "numpy", "torch", "mlx"] = "list",
    model: Optional[Callable] = None,
    use_last_token_pool: bool = True
) -> EmbeddingOutput:
    if return_format not in ["list", "numpy", "torch", "mlx"]:
        raise ValueError(
            "return_format must be 'list', 'numpy', 'torch', or 'mlx'")
    if use_last_token_pool and model is None:
        raise ValueError(
            "Model must be provided when use_last_token_pool is True")
    tokenizer = (
        get_tokenizer(model_name_or_tokenizer)
        if isinstance(model_name_or_tokenizer, str)
        else model_name_or_tokenizer
    )
    batch_size = (
        calculate_batch_size(texts)
        if not batch_size
        else batch_size
    )

    def process_batch(batch: Union[str, List[str]]) -> EmbeddingOutput:
        batch_list = [batch] if isinstance(batch, str) else batch
        if not model or not use_last_token_pool:
            embeddings = tokenize(batch_list, tokenizer)
        else:
            encodings = tokenizer.encode_batch(
                batch_list, add_special_tokens=True)
            input_ids_list = [encoding.ids for encoding in encodings]
            optimal_max_length = max(len(input_ids)
                                     for input_ids in input_ids_list)
            padded_inputs = []
            attention_masks = []
            for ids in tqdm(input_ids_list, desc="Padding Inputs", disable=not show_progress):
                if len(ids) > optimal_max_length:
                    ids = ids[:optimal_max_length]
                padded = ids + [0] * (optimal_max_length - len(ids))
                mask = [1] * len(ids) + [0] * (optimal_max_length - len(ids))
                padded_inputs.append(padded)
                attention_masks.append(mask)
            input_ids = mx.array(
                np.array(padded_inputs, dtype=np.int32), dtype=mx.int32)
            attention_mask = mx.array(
                np.array(attention_masks, dtype=np.int32), dtype=mx.int32)
            outputs = model(input_ids)
            embeddings = last_token_pool(outputs, attention_mask)
            embeddings = mx.concatenate([embeddings], axis=0)
            embeddings = embeddings / \
                mx.sqrt(mx.sum(embeddings * embeddings,
                        axis=1, keepdims=True) + 1e-8)
        if return_format == "numpy":
            return np.array(embeddings, dtype=np.float32)
        elif return_format == "torch":
            return torch.tensor(embeddings, dtype=torch.float32)
        elif return_format == "mlx":
            return mx.array(embeddings, dtype=mx.float32) if not use_last_token_pool else embeddings
        return embeddings
    embedding_batch_size = batch_size
    if isinstance(texts, str):
        return generate_multiple(texts, process_batch, embedding_batch_size, return_format)
    batch_iter = range(0, len(texts), batch_size)
    if show_progress:
        batch_iter = tqdm(batch_iter, total=math.ceil(
            len(texts) / batch_size), desc="Processing embeddings")
    embeddings = []
    for i in batch_iter:
        batch = texts[i:i + batch_size]
        batch_embeddings = process_batch(batch)
        if return_format in ["numpy", "torch", "mlx"]:
            embeddings.append(batch_embeddings)
        else:
            embeddings.extend(batch_embeddings)
    if return_format == "numpy":
        return np.concatenate(embeddings, axis=0) if embeddings else np.array([], dtype=np.float32)
    elif return_format == "torch":
        return torch.cat(embeddings, dim=0) if embeddings else torch.tensor([], dtype=torch.float32)
    elif return_format == "mlx":
        return mx.concatenate(embeddings, axis=0) if embeddings else mx.array([], dtype=mx.float32)
    return embeddings


def get_embedding_function(
    model_name: str,
    batch_size: Union[int, None] = None,
    show_progress: bool = False,
    return_format: Literal["list", "numpy", "torch", "mlx"] = "list",
    model: Optional[Callable] = None,
    use_last_token_pool: bool = True
) -> Callable[[Union[str, List[str]]], EmbeddingOutput]:
    tokenizer = Tokenizer.from_pretrained(model_name)

    def embed(texts: Union[str, List[str]]) -> EmbeddingOutput:
        optimal_batch_size = calculate_batch_size(texts, batch_size)
        return generate_embeddings(texts, tokenizer, optimal_batch_size, show_progress, return_format, model, use_last_token_pool)
    return embed
