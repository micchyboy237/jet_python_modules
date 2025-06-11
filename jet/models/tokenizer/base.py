import numpy as np
import logging

from typing import Callable, Dict, Optional, Union, List
from pathlib import Path
from tokenizers import Tokenizer

from jet.logger import logger
from jet.models.utils import resolve_model_value
from jet.models.model_types import ModelType


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_tokenizer(model_name: str, local_cache_dir: Optional[str] = None) -> Tokenizer:
    """
    Initialize and return a tokenizer for the specified model.

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

    try:
        # Attempt to load from remote
        tokenizer = Tokenizer.from_pretrained(model_path)
        logger.info(f"Successfully loaded tokenizer from remote: {model_path}")
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


def count_tokens(model_name_or_tokenizer: Union[str, Tokenizer], messages: str | List[str] | List[Dict], prevent_total: bool = False) -> int | list[int]:
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
