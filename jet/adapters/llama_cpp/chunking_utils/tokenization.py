# jet_python_modules/jet/adapters/llama_cpp/chunking_utils/tokenization.py
from typing import Callable

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.token_utils import get_tokenizer
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from tqdm import tqdm

LOCAL_BATCH_SIZE = 64


def _tokenize_for_size(text: str, model: str | LLAMACPP_KEYS = LLM_MODEL) -> list[int]:
    """Tokenize single text and return token IDs for size counting."""
    tokenizer = get_tokenizer(model)
    return tokenizer.encode(text, add_special_tokens=False)


def _tokenize_batch_for_size(
    texts: list[str],
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    show_progress: bool = False,
) -> list[list[int]]:
    """Tokenize multiple texts using the tokenizer's __call__ for efficiency."""
    if not texts:
        return []

    tokenizer = get_tokenizer(model)
    results: list[list[int]] = []

    text_iter = tqdm(
        range(0, len(texts), LOCAL_BATCH_SIZE),
        desc="Batch tokenizing",
        unit="batch",
        disable=not show_progress,
    )
    for i in text_iter:
        batch = texts[i : i + LOCAL_BATCH_SIZE]
        encoded = tokenizer(batch, add_special_tokens=False)
        if hasattr(encoded, "input_ids"):
            results.extend(encoded.input_ids)
        else:
            results.extend(encoded["input_ids"])

    return results


def _decode_tokens(tokens: list[int], model: str | LLAMACPP_KEYS = LLM_MODEL) -> str:
    """Decode token IDs back to text."""
    tokenizer = get_tokenizer(model)
    return tokenizer.decode(
        tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def _decode_tokens_batch(
    token_lists: list[list[int]],
    model: str | LLAMACPP_KEYS = LLM_MODEL,
    show_progress: bool = False,
) -> list[str]:
    """Batch decode multiple token lists to text."""
    if not token_lists:
        return []

    tokenizer = get_tokenizer(model)
    results: list[str] = []

    text_iter = tqdm(
        range(0, len(token_lists), LOCAL_BATCH_SIZE),
        desc="Batch decoding",
        unit="batch",
        disable=not show_progress,
    )
    for i in text_iter:
        batch = token_lists[i : i + LOCAL_BATCH_SIZE]
        batch_results = tokenizer.batch_decode(
            batch,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        results.extend(batch_results)

    return results


def _get_last_n_tokens_and_decode(
    text: str, n: int, model: str | LLAMACPP_KEYS = LLM_MODEL
) -> str:
    """Get the last n tokens from text and decode them back to string."""
    if n <= 0:
        return ""

    tokenizer = get_tokenizer(model)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    last_n = tokens[-n:] if len(tokens) >= n else tokens

    return tokenizer.decode(
        last_n,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def _get_size_fn(model: str | LLAMACPP_KEYS = LLM_MODEL) -> Callable:
    """Return a callable size_fn for chunking.

    Handles both single strings (returns list[int]) and
    lists of strings (returns list[list[int]]).
    Uses tokenizer.__call__() for batch compatibility across all backends.
    """
    tokenizer = get_tokenizer(model)

    def _fn(text, show_progress=False):
        if isinstance(text, list):
            if not text:
                return []

            results = []
            text_iter = tqdm(
                range(0, len(text), LOCAL_BATCH_SIZE),
                desc="Batch tokenizing (size_fn)",
                unit="batch",
                disable=not show_progress,
            )
            for i in text_iter:
                batch = text[i : i + LOCAL_BATCH_SIZE]
                encoded = tokenizer(batch, add_special_tokens=False)
                if hasattr(encoded, "input_ids"):
                    results.extend(encoded.input_ids)
                else:
                    results.extend(encoded["input_ids"])
            return results
        else:
            return tokenizer.encode(text, add_special_tokens=False)

    return _fn
