from tokenizers import Tokenizer
from typing import Callable, Union, List
import numpy as np


def get_tokenizer(model_name: str) -> Tokenizer:
    """Initialize and return a tokenizer for the specified model."""
    return Tokenizer.from_pretrained(model_name)


def get_tokenizer_fn(
    model_name_or_tokenizer: Union[str, Tokenizer]
) -> Callable[[Union[str, List[str]]], Union[List[int], List[List[int]]]]:
    """Return a tokenizer function from a model name or a Tokenizer instance."""
    tokenizer = (
        get_tokenizer(model_name_or_tokenizer)
        if isinstance(model_name_or_tokenizer, str)
        else model_name_or_tokenizer
    )

    def tokenize_fn(texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        is_single = isinstance(texts, str)
        texts_list = [texts] if is_single else texts
        encodings = tokenizer.encode_batch(texts_list, add_special_tokens=True)
        token_ids = [np.array(encoding.ids, dtype=np.int32).tolist()
                     for encoding in encodings]
        return token_ids[0] if is_single else token_ids

    return tokenize_fn


def tokenize(
    texts: Union[str, List[str]],
    tokenizer: Union[str, Tokenizer]
) -> Union[List[int], List[List[int]]]:
    tokenizer = (
        get_tokenizer(tokenizer)
        if isinstance(tokenizer, str)
        else tokenizer
    )
    if isinstance(texts, str):
        encoding = tokenizer.encode(texts, add_special_tokens=True)
        return encoding.ids
    encodings = tokenizer.encode_batch(texts, add_special_tokens=True)
    return [encoding.ids for encoding in encodings]


def detokenize(
    token_ids: Union[List[int], List[List[int]]],
    tokenizer: Union[str, Tokenizer]
) -> Union[str, List[str]]:
    tokenizer = (
        get_tokenizer(tokenizer)
        if isinstance(tokenizer, str)
        else tokenizer
    )
    if isinstance(token_ids[0], int):
        return tokenizer.decode(token_ids)
    return [tokenizer.decode(ids) for ids in token_ids]
