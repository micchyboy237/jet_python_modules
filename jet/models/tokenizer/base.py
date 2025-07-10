from typing import Any, Callable, Dict, Iterator, Optional, TypedDict, Union, List
from pathlib import Path
import numpy as np
import logging
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from jet.logger import logger
from jet.models.config import MODELS_CACHE_DIR
from jet.models.utils import get_context_size, resolve_model_value
from jet.models.model_types import ModelType
from jet.wordnet.sentence import split_sentences

# Cache for storing loaded tokenizers
_tokenizer_cache: Dict[str, PreTrainedTokenizerBase] = {}


class EncodingWrapper:
    """Wraps encoding results to make .ids the default iterable while preserving other attributes."""

    def __init__(
        self,
        encoding: Union[List[int], List[List[int]]],
        tokenizer: PreTrainedTokenizerBase
    ):
        self._tokenizer = tokenizer
        self._ids = encoding

    def __iter__(self) -> Iterator[int]:
        """Make .ids the default iterable for single encodings."""
        if isinstance(self._ids[0], list):
            raise TypeError(
                "Cannot iterate over batch encoding directly; iterate over individual encodings")
        return iter(self._ids)

    def __len__(self) -> int:
        """Return length of .ids for single encodings."""
        if isinstance(self._ids[0], list):
            raise TypeError(
                "Cannot get length of batch encoding directly; access individual encodings")
        return len(self._ids)

    def __getitem__(self, index: int) -> Union[int, 'EncodingWrapper']:
        """Access token IDs or individual encodings for batch results."""
        if isinstance(self._ids[0], list):
            return EncodingWrapper(self._ids[index], self._tokenizer)
        return self._ids[index]

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the tokenizer if applicable."""
        if hasattr(self._tokenizer, name):
            return getattr(self._tokenizer, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")


class TokenizerWrapper:
    """A callable wrapper for PreTrainedTokenizerBase, leveraging built-in encode features."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        remove_pad_tokens: bool = False,
        add_special_tokens: bool = True,
        pad_token_id: Optional[int] = None,
        max_length: Optional[int] = None,
        truncation_side: str = "right"
    ):
        self.tokenizer = tokenizer
        self.remove_pad_tokens = remove_pad_tokens
        self.add_special_tokens = add_special_tokens
        self.pad_token_id = pad_token_id or (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else 0
        )
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, texts: Union[str, List[str]], **kwargs) -> Union[EncodingWrapper, List[EncodingWrapper]]:
        """Tokenize input texts, supporting single string or list of strings."""
        if isinstance(texts, str):
            return self.encode(texts, **kwargs)
        return self.encode_batch(texts, **kwargs)

    def encode(self, text: str, **kwargs) -> EncodingWrapper:
        """Encode a single text string into an EncodingWrapper using tokenizer's encode."""
        if not isinstance(text, str):
            logger.error(
                f"Invalid input type for encode: {type(text)}, expected str")
            raise TypeError("TextInputSequence must be str")

        encode_kwargs = {
            'add_special_tokens': self.add_special_tokens,
            'max_length': self.max_length,
            'truncation': True if self.max_length is not None else False,
            'return_tensors': None  # Ensure we get a list of token IDs
        }
        encode_kwargs.update(kwargs)

        token_ids = self.tokenizer.encode(text, **encode_kwargs)
        wrapped_encoding = EncodingWrapper(token_ids, self.tokenizer)
        if self.remove_pad_tokens:
            wrapped_encoding._ids = [
                tid for tid in wrapped_encoding._ids if tid != self.pad_token_id]
        return wrapped_encoding

    def encode_batch(self, texts: List[str], **kwargs) -> List[EncodingWrapper]:
        """Encode a batch of texts into a list of EncodingWrapper objects."""
        for text in texts:
            if not isinstance(text, str):
                logger.error(
                    f"Invalid input type in batch: {type(text)}, expected str")
                raise TypeError("TextInputSequence must be str")

        encode_kwargs = {
            'add_special_tokens': self.add_special_tokens,
            'max_length': self.max_length,
            'truncation': True if self.max_length is not None else False,
            'return_tensors': None
        }
        encode_kwargs.update(kwargs)

        token_ids_list = [
            self.tokenizer.encode(text, **encode_kwargs)
            for text in texts
        ]
        wrapped_encodings = [EncodingWrapper(
            ids, self.tokenizer) for ids in token_ids_list]
        if self.remove_pad_tokens:
            for wrapped in wrapped_encodings:
                wrapped._ids = [
                    tid for tid in wrapped._ids if tid != self.pad_token_id]
        return wrapped_encodings

    def decode(self, token_ids: Union[List[int], List[List[int]]], **kwargs) -> Union[str, List[str]]:
        """Decode token IDs back to text."""
        decode_kwargs = {'skip_special_tokens': self.add_special_tokens}
        decode_kwargs.update(kwargs)

        if isinstance(token_ids[0], int):
            return self.tokenizer.decode(token_ids, **decode_kwargs)
        return [
            self.tokenizer.decode(ids, **decode_kwargs)
            for ids in token_ids
        ]

    def convert_ids_to_tokens(
        self, token_ids: Union[List[int], List[List[int]]], **kwargs
    ) -> Union[List[str], List[List[str]]]:
        """Convert token IDs to their string representations."""
        convert_kwargs = {'skip_special_tokens': self.add_special_tokens}
        convert_kwargs.update(kwargs)

        if self.tokenizer.is_fast:
            if isinstance(token_ids[0], int):
                return self.tokenizer.convert_ids_to_tokens(token_ids, **convert_kwargs)
            return [
                self.tokenizer.convert_ids_to_tokens(ids, **convert_kwargs)
                for ids in token_ids
            ]
        else:
            if isinstance(token_ids[0], int):
                return self.tokenizer.convert_ids_to_tokens(token_ids, **convert_kwargs)
            return [
                self.tokenizer.convert_ids_to_tokens(ids, **convert_kwargs)
                for ids in token_ids
            ]


def get_tokenizer(
    model_name_or_tokenizer: Union[ModelType, PreTrainedTokenizerBase],
    local_cache_dir: Optional[str] = None,
    disable_cache: bool = False,
    pad_token_id: Optional[int] = None,
    max_length: Optional[int] = None,
    truncation_side: str = "right",
    documents: Optional[Union[str, List[str]]] = None,
) -> PreTrainedTokenizerBase:
    """
    Initialize and return a tokenizer for the specified model, with option to disable cache.

    Args:
        model_name_or_tokenizer: The model key (e.g., 'bge-large') or repository ID (e.g., 'BAAI/bge-large-en-v1.5').
        local_cache_dir: Optional local directory to load tokenizer from (defaults to Hugging Face cache).
        disable_cache: If True, bypasses the tokenizer cache and always loads fresh.
        pad_token_id: Optional padding token ID to set.
        max_length: Optional maximum length; if None and documents provided, set to max token count.
        truncation_side: Truncation side ("right" or "left").
        documents: Optional input texts to compute max_length dynamically.

    Returns:
        PreTrainedTokenizerBase: The loaded tokenizer.

    Raises:
        ValueError: If the tokenizer cannot be loaded from remote or local sources.
    """
    if isinstance(model_name_or_tokenizer, PreTrainedTokenizerBase):
        return model_name_or_tokenizer

    model_name = model_name_or_tokenizer
    model_path = resolve_model_value(model_name)
    logger.info(
        f"Attempting to load tokenizer for model_name: {model_name}, resolved to: {model_path}")

    if not disable_cache and model_path in _tokenizer_cache:
        logger.info(f"Using cached tokenizer for: {model_path}")
        tokenizer = _tokenizer_cache[model_path]
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, cache_dir=local_cache_dir)
            logger.info(
                f"Successfully loaded tokenizer from remote: {model_path}")
            if not disable_cache:
                _tokenizer_cache[model_path] = tokenizer
        except Exception as e:
            logger.error(
                f"Failed to load tokenizer for {model_path}: {str(e)}")
            raise ValueError(
                f"Could not load tokenizer for {model_path} from remote or local cache.")

    if pad_token_id is not None and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = pad_token_id
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_token_id)

    tokenizer.truncation_side = truncation_side

    if max_length is None and documents is not None:
        tokenizer_wrapper = TokenizerWrapper(
            tokenizer,
            remove_pad_tokens=False,
            add_special_tokens=True,
            pad_token_id=pad_token_id,
            max_length=None  # Temporarily set to None to get raw token counts
        )
        token_counts = [
            len(tokenizer_wrapper.encode(text, truncation=False)._ids)
            for text in ([documents] if isinstance(documents, str) else documents)
        ]
        max_length = max(
            token_counts) if token_counts else tokenizer.model_max_length
        logger.info(f"Computed max_length from documents: {max_length}")

    if max_length is not None:
        tokenizer.model_max_length = max_length

    return tokenizer


def get_tokenizer_fn(
    model_name_or_tokenizer: Union[ModelType, PreTrainedTokenizerBase, TokenizerWrapper],
    remove_pad_tokens: bool = False,
    add_special_tokens: bool = True,
    disable_cache: bool = False,
    documents: Optional[Union[str, List[str]]] = None,
    **kwargs,
) -> TokenizerWrapper:
    """Return a TokenizerWrapper instance from a model name or tokenizer instance."""
    if isinstance(model_name_or_tokenizer, TokenizerWrapper):
        return model_name_or_tokenizer
    elif isinstance(model_name_or_tokenizer, str):
        tokenizer = get_tokenizer(
            model_name_or_tokenizer,
            disable_cache=disable_cache,
            documents=documents,
            **kwargs
        )
    else:
        tokenizer = model_name_or_tokenizer
        # Default unset value
        if documents is not None and tokenizer.model_max_length == int(1e30):
            tokenizer_wrapper = TokenizerWrapper(
                tokenizer,
                remove_pad_tokens=False,
                add_special_tokens=True,
                pad_token_id=tokenizer.pad_token_id,
                max_length=None
            )
            token_counts = [
                len(tokenizer_wrapper.encode(text, truncation=False)._ids)
                for text in ([documents] if isinstance(documents, str) else documents)
            ]
            max_length = max(
                token_counts) if token_counts else tokenizer.model_max_length
            tokenizer.model_max_length = max_length
            logger.info(f"Set tokenizer max_length to: {max_length}")

    return TokenizerWrapper(
        tokenizer,
        remove_pad_tokens=remove_pad_tokens,
        add_special_tokens=add_special_tokens,
        pad_token_id=kwargs.get('pad_token_id', tokenizer.pad_token_id),
        max_length=kwargs.get('max_length', tokenizer.model_max_length),
        truncation_side=kwargs.get(
            'truncation_side', tokenizer.truncation_side)
    )


def get_detokenizer_fn(
    model_name_or_tokenizer: Union[ModelType, PreTrainedTokenizerBase, TokenizerWrapper],
    remove_pad_tokens: bool = False,
    add_special_tokens: bool = True,
    disable_cache: bool = False,
    documents: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    """Return a function that detokenizes token IDs or tokens back to text."""
    # Retrieve the TokenizerWrapper instance (same as get_tokenizer_fn)
    if isinstance(model_name_or_tokenizer, TokenizerWrapper):
        tokenizer_wrapper = model_name_or_tokenizer
    elif isinstance(model_name_or_tokenizer, str):
        tokenizer_wrapper = get_tokenizer(
            model_name_or_tokenizer,
            disable_cache=disable_cache,
            documents=documents,
            **kwargs
        )
    else:
        tokenizer = model_name_or_tokenizer
        if documents is not None and tokenizer.model_max_length == int(1e30):
            tokenizer_wrapper = TokenizerWrapper(
                tokenizer,
                remove_pad_tokens=False,
                add_special_tokens=True,
                pad_token_id=tokenizer.pad_token_id,
                max_length=None
            )
            token_counts = [
                len(tokenizer_wrapper.encode(text, truncation=False)._ids)
                for text in ([documents] if isinstance(documents, str) else documents)
            ]
            max_length = max(
                token_counts) if token_counts else tokenizer.model_max_length
            tokenizer.model_max_length = max_length
        else:
            tokenizer_wrapper = TokenizerWrapper(
                tokenizer,
                remove_pad_tokens=remove_pad_tokens,
                add_special_tokens=add_special_tokens,
                pad_token_id=kwargs.get(
                    'pad_token_id', tokenizer.pad_token_id),
                max_length=kwargs.get(
                    'max_length', tokenizer.model_max_length),
                truncation_side=kwargs.get(
                    'truncation_side', tokenizer.truncation_side)
            )

    def _detokenizer(
        input_data: Union[List[int], List[List[int]],
                          List[str], List[List[str]]]
    ) -> Union[str, List[str]]:
        """Detokenize token IDs or tokens back to text."""
        tokenizer = tokenizer_wrapper.tokenizer  # Access underlying tokenizer

        # Handle token strings (convert to IDs first)
        if isinstance(input_data, list) and all(isinstance(x, str) for x in input_data):
            token_ids = tokenizer.convert_tokens_to_ids(input_data)
            return tokenizer.decode(token_ids, skip_special_tokens=not add_special_tokens)
        elif isinstance(input_data, list) and all(isinstance(x, list) and all(isinstance(t, str) for t in x) for x in input_data):
            return [
                tokenizer.decode(tokenizer.convert_tokens_to_ids(
                    tokens), skip_special_tokens=not add_special_tokens)
                for tokens in input_data
            ]

        # Handle token IDs directly
        if isinstance(input_data, list) and all(isinstance(x, int) for x in input_data):
            return tokenizer.decode(input_data, skip_special_tokens=not add_special_tokens)
        elif isinstance(input_data, list) and all(isinstance(x, list) and all(isinstance(t, int) for t in x) for x in input_data):
            return [
                tokenizer.decode(
                    ids, skip_special_tokens=not add_special_tokens)
                for ids in input_data
            ]

        raise ValueError(
            "Input must be List[int], List[List[int]], List[str], or List[List[str]]")

    return _detokenizer


def get_string_tokenizer_fn(
    model_name: ModelType,
    disable_cache: bool = False,
    documents: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    tokenizer = get_tokenizer(
        model_name,
        disable_cache=disable_cache,
        documents=documents,
        **kwargs
    )

    def _tokenizer(text: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        if isinstance(text, str):
            token_ids = tokenizer.encode(
                text, add_special_tokens=False)
            return tokenizer.convert_ids_to_tokens(token_ids)
        else:
            token_ids_list = tokenizer.batch_encode_plus(
                text, add_special_tokens=False)["input_ids"]
            return [tokenizer.convert_ids_to_tokens(ids) for ids in token_ids_list]
    return _tokenizer


def get_string_detokenizer_fn(
    model_name: ModelType,
    disable_cache: bool = False,
    documents: Optional[Union[str, List[str]]] = None,
    **kwargs,
):
    # Retrieve the same tokenizer used for tokenization
    tokenizer = get_tokenizer(
        model_name,
        disable_cache=disable_cache,
        documents=documents,
        **kwargs
    )

    def _detokenizer(tokens: Union[List[str], List[List[str]]]) -> Union[str, List[str]]:
        if isinstance(tokens[0], str):  # Single list of tokens
            # Convert tokens back to token IDs
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            # Decode token IDs to string
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        else:  # List of lists of tokens
            # Process each list of tokens
            results = []
            for token_list in tokens:
                token_ids = tokenizer.convert_tokens_to_ids(token_list)
                decoded_text = tokenizer.decode(
                    token_ids, skip_special_tokens=True)
                results.append(decoded_text)
            return results

    return _detokenizer


# Example usage
if __name__ == "__main__":
    # Example tokenizer setup
    model_name = "bert-base-uncased"  # Replace with your model
    tokenizer_fn = get_string_tokenizer_fn(model_name)
    detokenizer_fn = get_string_detokenizer_fn(model_name)

    # Example text
    text = "Hello, world!"
    # Tokenize
    tokens = tokenizer_fn(text)  # e.g., ['hello', ',', 'world', '!']
    print("Tokens:", tokens)
    # Detokenize
    detokenized_text = detokenizer_fn(tokens)
    print("Detokenized:", detokenized_text)

    # Batch example
    texts = ["Hello, world!", "How are you?"]
    # e.g., [['hello', ',', 'world', '!'], ['how', 'are', 'you', '?']]
    batch_tokens = tokenizer_fn(texts)
    print("Batch Tokens:", batch_tokens)
    detokenized_texts = detokenizer_fn(batch_tokens)
    print("Detokenized Batch:", detokenized_texts)


def tokenize(
    texts: Union[str, List[str]],
    tokenizer: Union[ModelType, PreTrainedTokenizerBase],
    remove_pad_tokens: bool = False,
    add_special_tokens: bool = True,
    disable_cache: bool = False,
) -> Union[List[int], List[List[int]]]:
    """Tokenize texts using a TokenizerWrapper."""
    tokenizer_wrapper = get_tokenizer_fn(
        tokenizer,
        remove_pad_tokens=remove_pad_tokens,
        add_special_tokens=add_special_tokens,
        disable_cache=disable_cache,
    )
    result = tokenizer_wrapper(texts)
    if isinstance(texts, str):
        return result._ids
    return [wrapper._ids for wrapper in result]


def detokenize(
    token_ids: Union[List[int], List[List[int]]],
    tokenizer: Union[ModelType, PreTrainedTokenizerBase]
) -> Union[str, List[str]]:
    tokenizer = (
        get_tokenizer(tokenizer)
        if isinstance(tokenizer, str)
        else tokenizer
    )
    if isinstance(token_ids[0], int):
        return tokenizer.decode(token_ids)
    return [tokenizer.decode(ids) for ids in token_ids]


def count_tokens(
    model_name_or_tokenizer: Union[ModelType, PreTrainedTokenizerBase],
    messages: Union[str, List[str], List[Dict]],
    prevent_total: bool = False,
    remove_pad_tokens: bool = True,
    add_special_tokens: bool = False,
    disable_cache: bool = False
) -> Union[int, List[int]]:
    if not messages:
        return 0

    if isinstance(messages, list):
        messages = [str(t) for t in messages]

    tokenize = get_tokenizer_fn(
        model_name_or_tokenizer, remove_pad_tokens=remove_pad_tokens, add_special_tokens=add_special_tokens, disable_cache=disable_cache)
    tokenized = tokenize(messages)
    if isinstance(messages, str):
        return len(tokenized)
    else:
        token_counts = [len(item) for item in tokenized]
        return sum(token_counts) if not prevent_total else token_counts


def count_tokens_dim(
    model_name_or_tokenizer: Union[ModelType, PreTrainedTokenizerBase],
    messages: Union[str, List[str], List[Dict]],
) -> Union[int, List[int]]:
    if not messages:
        return 0

    if isinstance(messages, list):
        messages = [str(t) for t in messages]

    tokenize = get_tokenizer_fn(
        model_name_or_tokenizer, remove_pad_tokens=False, add_special_tokens=True)
    tokenized = tokenize(messages)
    if isinstance(messages, str):
        return len(tokenized)
    else:
        token_counts = [len(item) for item in tokenized]
        return token_counts[0]


def get_max_token_count(
    model_name_or_tokenizer: Union[ModelType, PreTrainedTokenizerBase],
    messages: Union[str, List[str], List[Dict]],
    buffer: int = 10,
    remove_pad_tokens: bool = True
) -> int:
    """
    Calculate the maximum number of tokens in the provided messages, adding a buffer and capping at 512.

    Args:
        model_name_or_tokenizer: The model name or tokenizer instance to use.
        messages: A string, list of strings, or list of dicts representing the input messages.
        buffer: Number of extra tokens to add as a safety margin (default: 10).
        remove_pad_tokens: Whether to remove padding tokens from the count (default: True).

    Returns:
        int: The maximum token count plus buffer, capped at 512.
    """
    token_counts: List[int] = count_tokens(
        model_name_or_tokenizer, messages, prevent_total=True, remove_pad_tokens=remove_pad_tokens
    )
    max_tokens = min(max(token_counts) + buffer, 512)  # Cap at 512
    logger.info(f"Max token count for {len(messages)} documents: {max_tokens}")
    return max_tokens


class MergeMetadata(TypedDict):
    texts_count: int
    is_truncated: bool
    total_tokens: int
    min_tokens: int
    max_tokens: int
    ave_tokens: int


class MergeResult(TypedDict):
    texts: List[str]
    token_counts: List[int]
    tokens: List[List[int]]
    token_strings: List[List[str]]
    decoded_tokens: List[List[str]]
    metadata: MergeMetadata


def merge_texts(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    skip_special_tokens: bool = True,
    max_length: Optional[int] = None,
    split_fn: Optional[Callable[[str], List[str]]] = None,
    remove_pad_tokens: bool = False
) -> MergeResult:
    # Encode the text into token IDs
    token_ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if remove_pad_tokens:
        token_ids = [tid for tid in token_ids if tid != pad_token_id]
    total_tokens: int = len(token_ids)

    # If max_length is None or greater than total tokens, no truncation needed
    if max_length is None or max_length >= total_tokens:
        token_strings: List[str] = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        decoded_tokens: List[str] = [
            dt for dt in tokenizer.batch_decode(
                [[tid] for tid in token_ids], skip_special_tokens=skip_special_tokens
            ) if dt
        ]

        return {
            "texts": [text] if text else [],
            "token_counts": [len(token_ids)],
            "tokens": [token_ids],
            "token_strings": [token_strings],
            "decoded_tokens": [decoded_tokens],
            "metadata": {
                "texts_count": 1,
                "is_truncated": False,
                "total_tokens": total_tokens,
                "min_tokens": total_tokens,
                "max_tokens": total_tokens,
                "ave_tokens": total_tokens,
            }
        }

    # Get the decoded text to find sentence boundaries
    decoded_text: str = tokenizer.decode(
        token_ids, skip_special_tokens=skip_special_tokens
    )

    # Split text into sentences using NLTK
    sentences: List[str] = split_fn(
        decoded_text) if split_fn else split_sentences(decoded_text)

    # Initialize variables for grouping texts
    grouped_texts: List[str] = []
    grouped_token_ids: List[List[int]] = []
    selected_token_ids: List[int] = []
    current_token_count: int = 0
    current_group: List[str] = []

    for i, sentence in enumerate(sentences):
        sentence_token_ids: List[int] = tokenizer.encode(
            sentence, add_special_tokens=False
        )
        if remove_pad_tokens:
            sentence_token_ids = [
                tid for tid in sentence_token_ids if tid != pad_token_id]
        sentence_token_count: int = len(sentence_token_ids)

        # If sentence token count > max_length, just add it
        if not max_length or sentence_token_count > max_length:
            grouped_texts.append(sentence)
            grouped_token_ids.append(sentence_token_ids)
        # Check if adding the sentence exceeds max_length
        elif current_token_count + sentence_token_count <= max_length:
            selected_token_ids.extend(sentence_token_ids)
            current_token_count += sentence_token_count
            current_group.append(sentence)
        else:
            # If there's a current group, add it to grouped_texts and clear it
            if current_group:
                grouped_texts.append(" ".join(current_group))
                grouped_token_ids.append(selected_token_ids)
                current_group = []
                current_token_count = 0
                selected_token_ids = []

            # Try merging with the next sentence if possible
            remaining_tokens: int = max_length - current_token_count
            if remaining_tokens > 0 and i + 1 < len(sentences):
                next_sentence: str = sentences[i + 1]
                merged_sentence: str = sentence + " " + next_sentence
                merged_token_ids: List[int] = tokenizer.encode(
                    merged_sentence, add_special_tokens=False
                )
                if remove_pad_tokens:
                    merged_token_ids = [
                        tid for tid in merged_token_ids if tid != pad_token_id]

                if len(merged_token_ids) <= max_length - current_token_count:
                    selected_token_ids.extend(merged_token_ids)
                    current_token_count += len(merged_token_ids)
                    current_group.append(merged_sentence)
                    # Skip the next sentence since it's merged
                    sentences[i + 1] = ""
                    continue

            # If we can't merge or no space left, start a new group
            if remaining_tokens >= sentence_token_count:
                current_group = [sentence]
                selected_token_ids.extend(sentence_token_ids)
                current_token_count = sentence_token_count
            else:
                break

    # Add the final group if it exists
    if current_group:
        grouped_texts.append(" ".join(current_group))
        grouped_token_ids.append(selected_token_ids)

    grouped_decoded_tokens: List[List[str]] = []
    grouped_token_strings: List[List[str]] = []
    token_counts: List[int] = []
    for token_ids in grouped_token_ids:
        token_counts.append(len(token_ids))
        token_strings = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        grouped_token_strings.append(token_strings)
        decoded_tokens = [
            dt for dt in tokenizer.batch_decode(
                [[tid] for tid in token_ids], skip_special_tokens=skip_special_tokens
            ) if dt
        ]
        grouped_decoded_tokens.append(decoded_tokens)

    # Prepare metadata
    metadata: MergeMetadata = {
        "texts_count": len(grouped_texts),
        "is_truncated": len(grouped_texts) > 1,
        "total_tokens": total_tokens,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "ave_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
    }

    return {
        "texts": grouped_texts,
        "token_counts": token_counts,
        "tokens": grouped_token_ids,
        "token_strings": grouped_token_strings,
        "decoded_tokens": grouped_decoded_tokens,
        "metadata": metadata
    }


__all__ = [
    "_tokenizer_cache",
    "get_tokenizer",
    "get_tokenizer_fn",
    "tokenize",
    "detokenize",
    "count_tokens",
    "get_max_token_count",
    "TokenizerWrapper",
]
