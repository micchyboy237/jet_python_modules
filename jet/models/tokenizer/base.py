import numpy as np
import logging

from typing import Callable, Dict, Optional, TypedDict, Union, List
from pathlib import Path
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerBase

from jet.logger import logger
from jet.models.utils import resolve_model_value
from jet.models.model_types import ModelType
from jet.wordnet.sentence import split_sentences


# Cache for storing loaded tokenizers
_tokenizer_cache: Dict[str, Tokenizer] = {}


def get_tokenizer(model_name: ModelType, local_cache_dir: Optional[str] = None) -> Tokenizer:
    """
    Initialize and return a tokenizer for the specified model, using a cache to prevent reloading.

    Args:
        model_name: The model key (e.g., 'bge-large') or repository ID (e.g., 'BAAI/bge-large-en-v1.5').
        local_cache_dir: Optional local directory to load tokenizer from (defaults to Hugging Face cache).

    Returns:
        Tokenizer: The loaded tokenizer.

    Raises:
        ValueError: If the tokenizer cannot be loaded from remote or local sources.
    """
    # Resolve model_name to full path if it's a key
    model_path = resolve_model_value(model_name)
    logger.info(
        f"Attempting to load tokenizer for model_name: {model_name}, resolved to: {model_path}")

    # Check cache first
    if model_path in _tokenizer_cache:
        logger.info(f"Using cached tokenizer for: {model_path}")
        return _tokenizer_cache[model_path]

    try:
        # Attempt to load from remote
        tokenizer = Tokenizer.from_pretrained(model_path)
        logger.info(f"Successfully loaded tokenizer from remote: {model_path}")
        _tokenizer_cache[model_path] = tokenizer
        return tokenizer
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from remote: {str(e)}")

        # Fallback to local cache
        if local_cache_dir is None:
            local_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        else:
            local_cache_dir = Path(local_cache_dir)

        # Construct snapshot directory path
        snapshot_dir = local_cache_dir / \
            f"models--{model_path.replace('/', '--')}" / "snapshots"
        if not snapshot_dir.exists():
            error_msg = f"Snapshot directory does not exist: {snapshot_dir}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Recursively search for tokenizer.json in snapshot directory
        tokenizer_files = list(snapshot_dir.rglob("tokenizer.json"))
        logger.debug(
            f"Found {len(tokenizer_files)} tokenizer.json files in {snapshot_dir}")

        for tokenizer_path in tokenizer_files:
            try:
                # Resolve symlinks to real path
                resolved_path = tokenizer_path.resolve()
                logger.debug(f"Resolved {tokenizer_path} to {resolved_path}")
                if not resolved_path.is_file():
                    logger.warning(
                        f"Resolved path is not a file: {resolved_path}")
                    continue

                tokenizer = Tokenizer.from_file(str(resolved_path))
                logger.info(
                    f"Successfully loaded tokenizer from local cache: {resolved_path}")
                _tokenizer_cache[model_path] = tokenizer
                return tokenizer
            except Exception as local_e:
                logger.error(
                    f"Failed to load tokenizer from {resolved_path}: {str(local_e)}")
                continue

        # If no valid tokenizer is found, raise an error
        error_msg = f"Could not load tokenizer for {model_path} from remote or local cache."
        logger.error(error_msg)
        raise ValueError(error_msg)


def get_tokenizer_fn(
    model_name_or_tokenizer: Union[ModelType, Tokenizer]
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
    tokenizer: Union[ModelType, Tokenizer]
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
    tokenizer: Union[ModelType, Tokenizer]
) -> Union[str, List[str]]:
    tokenizer = (
        get_tokenizer(tokenizer)
        if isinstance(tokenizer, str)
        else tokenizer
    )
    if isinstance(token_ids[0], int):
        return tokenizer.decode(token_ids)
    return [tokenizer.decode(ids) for ids in token_ids]


def count_tokens(model_name_or_tokenizer: Union[ModelType, Tokenizer], messages: str | List[str] | List[Dict], prevent_total: bool = False) -> int | list[int]:
    if not messages:
        return 0

    if isinstance(messages, list):
        messages = [str(t) for t in messages]

    tokenize = get_tokenizer_fn(model_name_or_tokenizer)
    tokenized = tokenize(messages)
    if isinstance(messages, str):
        return len(tokenized)
    else:
        token_counts = [len(item) for item in tokenized]
        return sum(token_counts) if not prevent_total else token_counts


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
    split_fn: Optional[Callable[[str], List[str]]] = None
) -> MergeResult:
    # Encode the text into token IDs
    token_ids: List[int] = tokenizer.encode(text, add_special_tokens=False)
    total_tokens: int = len(token_ids)

    # If max_length is None or greater than total tokens, no truncation needed
    if max_length is None or max_length >= total_tokens:
        token_strings: List[str] = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        # Use batch_decode to decode all token IDs at once
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
        # Convert selected token IDs to token strings and decoded tokens
        token_strings = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        grouped_token_strings.append(token_strings)
        # Use batch_decode to decode all selected token IDs at once
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
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
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
