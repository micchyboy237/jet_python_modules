import numpy as np
from tokenizers import Tokenizer
from typing import Callable, Union, List
import psutil
import math


def get_embedding_function(
    model_name: str,
    batch_size: Union[int, None] = None,
) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
    """
    Returns a tokenize function optimized for batching performance for the specified model.

    Args:
        model_name: Name of the model (e.g., 'bert-base-cased', 'gpt2', 'sentence-transformers/all-MiniLM-L6-v2').
        batch_size: Optional batch size; if None, dynamically calculated.

    Returns:
        A function that tokenizes input text(s) into token IDs.
    """
    # Load tokenizer based on model name
    try:
        tokenizer = Tokenizer.from_pretrained(model_name)
    except Exception:
        # Fallback to Byte-Level BPE for unknown models
        tokenizer = Tokenizer.from_pretrained(
            "gpt2")  # Default to BPE for robustness

    # Determine optimal batch size if not provided
    def calculate_batch_size(texts: Union[str, List[str]]) -> int:
        if batch_size is not None:
            return batch_size
        # Heuristic: Estimate based on available memory and input size
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        avg_text_length = len(texts) if isinstance(
            texts, str) else sum(len(t) for t in texts) / len(texts)
        # Assume ~1KB per tokenized sequence, adjust based on memory
        estimated_memory_per_sequence = avg_text_length * 0.001
        max_sequences = max(
            1, int(available_memory * 0.5 / estimated_memory_per_sequence))
        return min(max_sequences, 128)  # Cap at 128 for practicality

    def tokenize(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Tokenize input text(s) into token IDs.

        Args:
            texts: Single string or list of strings to tokenize.

        Returns:
            Token IDs as list[float] for single input or list[list[float]] for batch.
        """
        # Convert single string to list for uniform processing
        is_single = isinstance(texts, str)
        texts_list = [texts] if is_single else texts

        # Calculate batch size
        optimal_batch_size = calculate_batch_size(texts_list)

        # Process in batches
        all_ids = []
        for i in range(0, len(texts_list), optimal_batch_size):
            batch = texts_list[i:i + optimal_batch_size]
            # Encode batch with padding for consistent length
            encodings = tokenizer.encode_batch(batch, add_special_tokens=True)
            batch_ids = [np.array(encoding.ids, dtype=np.float32).tolist()
                         for encoding in encodings]
            all_ids.extend(batch_ids)

        # Return single list for single input, else list of lists
        return all_ids[0] if is_single else all_ids

    return tokenize
