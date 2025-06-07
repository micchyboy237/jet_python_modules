import numpy as np
from tokenizers import Tokenizer
from typing import Callable, Union, List
import psutil
import math
from tqdm import tqdm


def calculate_batch_size(texts: Union[str, List[str]], batch_size: Union[int, None] = None) -> int:
    if batch_size is not None:
        return batch_size
    available_memory = psutil.virtual_memory().available / (1024 * 1024)
    avg_text_length = len(texts) if isinstance(
        texts, str) else sum(len(t) for t in texts) / len(texts)
    estimated_memory_per_sequence = avg_text_length * 0.001
    max_sequences = max(1, int(available_memory * 0.5 /
                        estimated_memory_per_sequence))
    return min(max_sequences, 128)


def generate_embeddings(texts: Union[str, List[str]], tokenizer: Tokenizer, batch_size: int, show_progress: bool = False) -> Union[List[float], List[List[float]]]:
    is_single = isinstance(texts, str)
    texts_list = [texts] if is_single else texts
    all_ids = []
    batch_iter = range(0, len(texts_list), batch_size)
    if show_progress:
        batch_iter = tqdm(batch_iter, total=math.ceil(
            len(texts_list) / batch_size), desc="Tokenizing")
    for i in batch_iter:
        batch = texts_list[i:i + batch_size]
        encodings = tokenizer.encode_batch(batch, add_special_tokens=True)
        batch_ids = [np.array(encoding.ids, dtype=np.float32).tolist()
                     for encoding in encodings]
        all_ids.extend(batch_ids)
    return all_ids[0] if is_single else all_ids


def get_embedding_function(
    model_name: str,
    batch_size: Union[int, None] = None,
    show_progress: bool = False,
) -> Callable[[Union[str, List[str]]], Union[List[float], List[List[float]]]]:
    tokenizer = Tokenizer.from_pretrained(model_name)

    def tokenize_wrapper(texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        optimal_batch_size = calculate_batch_size(texts, batch_size)
        return generate_embeddings(texts, tokenizer, optimal_batch_size, show_progress)

    return tokenize_wrapper
