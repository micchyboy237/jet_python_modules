import numpy as np
from tokenizers import Tokenizer
from typing import Callable, Union, List
import psutil
import math
from tqdm import tqdm


def get_embedding_function(
    model_name: str,
    batch_size: Union[int, None] = None,
    show_progress: bool = False,
) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
    """
    Returns a tokenize function optimized for batching performance with optional progress tracking.

    Args:
        model_name: Name of the model (e.g., 'bert-base-cased', 'gpt2', 'sentence-transformers/all-MiniLM-L6-v2').
        batch_size: Optional batch size; if None, dynamically calculated.
        show_progress: If True, displays a progress bar for batch processing.

    Returns:
        A function that tokenizes input text(s) into token IDs.
    """
    # Load tokenizer based on model name
    try:
        tokenizer = Tokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = Tokenizer.from_pretrained("gpt2")  # Fallback to BPE

    def calculate_batch_size(texts: Union[str, List[str]]) -> int:
        if batch_size is not None:
            return batch_size
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        avg_text_length = len(texts) if isinstance(
            texts, str) else sum(len(t) for t in texts) / len(texts)
        estimated_memory_per_sequence = avg_text_length * 0.001
        max_sequences = max(
            1, int(available_memory * 0.5 / estimated_memory_per_sequence))
        return min(max_sequences, 128)  # Cap at 128

    def tokenize(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Tokenize input text(s) into token IDs with optional progress tracking.

        Args:
            texts: Single string or list of strings to tokenize.

        Returns:
            Token IDs as list[float] for single input or list[list[float]] for batch.
        """
        is_single = isinstance(texts, str)
        texts_list = [texts] if is_single else texts
        optimal_batch_size = calculate_batch_size(texts_list)

        all_ids = []
        # Use tqdm for progress tracking if enabled
        batch_iter = range(0, len(texts_list), optimal_batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, total=math.ceil(
                len(texts_list) / optimal_batch_size), desc="Tokenizing")

        for i in batch_iter:
            batch = texts_list[i:i + optimal_batch_size]
            encodings = tokenizer.encode_batch(batch, add_special_tokens=True)
            batch_ids = [np.array(encoding.ids, dtype=np.float32).tolist()
                         for encoding in encodings]
            all_ids.extend(batch_ids)

        return all_ids[0] if is_single else all_ids

    return tokenize
