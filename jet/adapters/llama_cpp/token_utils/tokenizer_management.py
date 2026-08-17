from typing import Union

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.model_utils import get_model_hf_id
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.logger import logger
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

TOKENIZER_CACHE: dict[str, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = {}


def clear_tokenizer_cache() -> None:
    """Clear all cached tokenizers — useful in tests / REPL"""
    TOKENIZER_CACHE.clear()
    logger.debug("Tokenizer cache cleared")


def _normalize_model_key(model_name: str | None) -> str:
    """Create consistent cache key"""
    if model_name is None:
        return "__default__"
    return model_name.lower().strip()


def get_tokenizer(
    model_name: str | LLAMACPP_KEYS | None = None,
    cache: bool = True,
    verbose: bool = False,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    Get a HuggingFace tokenizer for a llama.cpp model key.
    Resolves the model key to a HuggingFace ID using `get_model_hf_id`,
    then loads the tokenizer via AutoTokenizer. Results are cached by default.
    """
    if model_name is None:
        model_name = LLM_MODEL
        logger.debug(f"No model specified, using default: {model_name}")
    key = _normalize_model_key(model_name)
    if cache and key in TOKENIZER_CACHE:
        if verbose:
            logger.debug(f"[tokenizer cache] HIT → {key}")
        return TOKENIZER_CACHE[key]
    if verbose:
        logger.debug(f"[tokenizer cache] MISS → loading {key}")
    try:
        hf_id = get_model_hf_id(model_name)
        logger.debug(f"Resolved '{model_name}' → '{hf_id}'")
    except ValueError:
        logger.debug(f"Model '{model_name}' not in mapping, trying as direct HF ID")
        hf_id = model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        logger.debug(f"Successfully loaded tokenizer for '{hf_id}'")
    except Exception as e:
        logger.error(f"Failed to load tokenizer for '{hf_id}': {e}")
        raise ValueError(
            f"Cannot load tokenizer for model '{model_name}' (HF: {hf_id})"
        ) from e
    if cache:
        TOKENIZER_CACHE[key] = tokenizer
    return tokenizer


def get_tokenizer_fn(
    model_name: str | LLAMACPP_KEYS | None = None,
    add_special_tokens: bool = False,
) -> callable:
    """
    Returns a callable that tokenizes text using the local tokenizer.
    """
    tokenizer = get_tokenizer(model_name)

    def _fn(text: str | list[str]) -> list[int] | list[list[int]]:
        if isinstance(text, str):
            return tokenizer.encode(text, add_special_tokens=add_special_tokens)
        else:
            return tokenizer.batch_encode_plus(
                text,
                add_special_tokens=add_special_tokens,
            )["input_ids"]

    return _fn


def get_detokenizer_fn(
    model_name: str | LLAMACPP_KEYS | None = None,
    skip_special_tokens: bool = True,
) -> callable:
    """
    Returns a callable that detokenizes token IDs using the local tokenizer.
    """
    tokenizer = get_tokenizer(model_name)

    def _fn(tokens: list[int] | list[list[int]]) -> str | list[str]:
        if tokens and isinstance(tokens[0], list):
            return tokenizer.batch_decode(
                tokens,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=True,
            )
        else:
            return tokenizer.decode(
                tokens,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=True,
            )

    return _fn
