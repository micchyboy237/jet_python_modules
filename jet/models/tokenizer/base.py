from tokenizers import Tokenizer
from typing import Callable, Union, List
import numpy as np


def get_tokenizer(model_name: str) -> Tokenizer:
    """Initialize and return a tokenizer for the specified model."""
    return Tokenizer.from_pretrained(model_name)


def get_tokenizer_fn(
    model_name: str
) -> Callable[[Union[str, List[str]]], Union[List[int], List[List[int]]]]:
    """Return a tokenizer function for the given model."""
    tokenizer = get_tokenizer(model_name)

    def tokenize(texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """Tokenize input texts using the initialized tokenizer."""
        is_single = isinstance(texts, str)
        texts_list = [texts] if is_single else texts

        encodings = tokenizer.encode_batch(texts_list, add_special_tokens=True)
        token_ids = [np.array(encoding.ids, dtype=np.int32).tolist()
                     for encoding in encodings]

        return token_ids[0] if is_single else token_ids

    return tokenize

# Note: The standalone tokenize function below is provided for direct use,
# but get_tokenizer_fn is preferred for reusability and modularity.


def tokenize(
    texts: Union[str, List[str]],
    model_name: str
) -> Union[List[int], List[List[int]]]:
    """Tokenize input texts using a tokenizer for the specified model."""
    tokenizer = get_tokenizer(model_name)
    is_single = isinstance(texts, str)
    texts_list = [texts] if is_single else texts

    encodings = tokenizer.encode_batch(texts_list, add_special_tokens=True)
    token_ids = [np.array(encoding.ids, dtype=np.int32).tolist()
                 for encoding in encodings]

    return token_ids[0] if is_single else token_ids
