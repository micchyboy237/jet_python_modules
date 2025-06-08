import numpy as np
import psutil
from tokenizers import Tokenizer
from typing import Dict, Optional, Union, List
from jet.llm.mlx.mlx_types import ModelType
from jet.models.tokenizer.base import count_tokens, get_tokenizer
from jet.models.utils import get_embedding_size


def calculate_batch_size(texts: Union[str, List[str]], batch_size: Optional[int] = None) -> int:
    """
    Calculate the optimal batch size for processing texts based on available memory.

    Args:
        texts (Union[str, List[str]]): A single text string or a list of text strings to process.
        batch_size (Union[int, None], optional): A fixed batch size to use. If None, calculates dynamically.

    Returns:
        int: The optimal batch size, either the provided batch_size or a calculated value based on
             available memory and average text length, capped at 128.

    Notes:
        - Uses 50% of available virtual memory to estimate the number of sequences that can be processed.
        - Assumes 0.001 MB per character as an estimate for memory usage per sequence.
    """
    if batch_size is not None:
        return batch_size
    available_memory = psutil.virtual_memory().available / (1024 * 1024)
    avg_text_length = len(texts) if isinstance(
        texts, str) else sum(len(t) for t in texts) / len(texts)
    estimated_memory_per_sequence = avg_text_length * 0.001
    max_sequences = max(1, int(available_memory * 0.5 /
                        estimated_memory_per_sequence))
    return min(max_sequences, 32)


def calculate_max_length(texts: Union[str, List[str]], model_name_or_tokenizer: Union[str, Tokenizer], max_length: Union[int, None] = None) -> int:
    """
    Calculate the optimal maximum sequence length for tokenization based on text lengths and memory.

    Args:
        texts (Union[str, List[str]]): A single text string or a list of text strings to process.
        model_name_or_tokenizer (Union[str, Tokenizer]): The tokenizer or model name used to encode the texts.
        max_length (Union[int, None], optional): A fixed maximum length. If None, calculates dynamically.

    Returns:
        int: The optimal maximum sequence length, either the provided max_length or a calculated value
             based on the 95th percentile of token lengths and available memory, bounded between 128 and 2048.

    Notes:
        - Uses the 95th percentile of token lengths to accommodate most texts without excessive truncation.
        - Limits the length based on 50% of available memory, assuming 0.0005 MB per token.
        - Returns 512 for empty text lists to provide a reasonable default.
    """
    if max_length is not None:
        return max_length
    tokenizer = (
        get_tokenizer(model_name_or_tokenizer)
        if isinstance(model_name_or_tokenizer, str)
        else model_name_or_tokenizer
    )
    texts_list = [texts] if isinstance(texts, str) else texts
    token_lengths = [len(encoding.ids)
                     for encoding in tokenizer.encode_batch(texts_list)]
    if not token_lengths:
        return 512  # Default max_length if no texts
    available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
    memory_per_token = 0.0005  # Estimated memory per token in MB
    max_tokens_by_memory = int(available_memory * 0.5 / memory_per_token)
    percentile_95 = int(np.percentile(token_lengths, 95)
                        ) if token_lengths else 512
    optimal_length = min(max(percentile_95, 128), max_tokens_by_memory, 2048)
    return max(128, min(optimal_length, 2048))


def calculate_n_ctx(model_name: ModelType, messages: str | List[str] | List[Dict]):
    tokenizer = get_tokenizer(model_name)
    if isinstance(messages, str):
        messages = [messages]
    token_counts: list[int] = count_tokens(
        tokenizer, messages, prevent_total=True)
    largest_size = max(token_counts)
    model_embedding_size = get_embedding_size(model_name)

    n_ctx = min(largest_size + 32, model_embedding_size)
    return n_ctx
