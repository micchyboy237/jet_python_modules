import zlib
import pickle
import os
import threading
import numpy as np
from typing import Dict, Union, Literal, List
from pathlib import Path
from jet.logger import logger
from jet.data.utils import hash_text
from jet.models.utils import resolve_model_value

CACHE_FILE = "embedding_cache.pkl"
CACHE_DIR = os.path.expanduser("~/.cache/jet_python_modules")
Path(CACHE_DIR).mkdir(exist_ok=True)
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_FILE)
_cache_lock = threading.Lock()
_memory_cache: Dict[str, Union[List[float], List[List[float]]]] = {}
MEMORY_CACHE_MAX_SIZE = 10000
MEMORY_CACHE_PRUNE_RATIO = 0.8

def load_cache() -> Dict[str, Union[List[float], List[List[float]]]]:
    """Load cache from file, return empty dict if file doesn't exist or is invalid."""
    global _memory_cache
    if _memory_cache:
        logger.debug(f"Using in-memory cache with {len(_memory_cache)} entries.")
        return _memory_cache
    try:
        with open(CACHE_PATH, 'rb') as f:
            compressed_data = f.read()
            data = zlib.decompress(compressed_data)
            cache = pickle.loads(data)
            logger.debug(f"Loaded cache from {CACHE_PATH} with {len(cache)} entries.")
            _memory_cache = cache
            return cache
    except (FileNotFoundError, pickle.PickleError, zlib.error, IOError) as e:
        logger.debug(f"Failed to load cache from {CACHE_PATH}: {e}. Starting with empty cache.")
        _memory_cache = {}
        return {}

def save_cache(cache: Dict[str, Union[List[float], List[List[float]]]]) -> None:
    """Save cache to file with compression and pruning."""
    try:
        if len(cache) > MEMORY_CACHE_MAX_SIZE:
            sorted_items = sorted(cache.items(), key=lambda x: x[0])
            pruned_size = int(MEMORY_CACHE_MAX_SIZE * MEMORY_CACHE_PRUNE_RATIO)
            pruned_cache = dict(sorted_items[:pruned_size])
            logger.info(f"Pruned cache to {len(pruned_cache)} entries to manage size.")
        else:
            pruned_cache = cache
        data = pickle.dumps(pruned_cache)
        compressed_data = zlib.compress(data, level=6)
        with open(CACHE_PATH, 'wb') as f:
            f.write(compressed_data)
        logger.debug(f"Saved cache to {CACHE_PATH} with {len(pruned_cache)} entries.")
    except (pickle.PickleError, zlib.error, IOError) as e:
        logger.error(f"Failed to save cache to {CACHE_PATH}: {e}")

def get_cached_embeddings(
    cache_key: str,
    return_format: Literal["numpy", "list"] = "numpy"
) -> Union[None, List[float], List[List[float]], np.ndarray]:
    """Retrieve embeddings from cache if available."""
    with _cache_lock:
        if cache_key in _memory_cache:
            cached_result = _memory_cache[cache_key]
            result = cached_result
            if return_format == "numpy":
                result = np.array(cached_result, dtype=np.float32)
            logger.debug(f"Memory cache hit for key: {cache_key}")
            return result
        cache = load_cache()
        if cache_key in cache:
            _memory_cache[cache_key] = cache[cache_key]
            result = cache[cache_key]
            if return_format == "numpy":
                result = np.array(cache[cache_key], dtype=np.float32)
            logger.debug(f"File cache hit for key: {cache_key}, file: {CACHE_PATH}")
            return result
    return None

def cache_embeddings(
    cache_key: str,
    embeddings: Union[List[float], List[List[float]], np.ndarray]
) -> None:
    """Store embeddings in memory and file cache."""
    with _cache_lock:
        cache = load_cache()
        embeddings_to_cache = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        _memory_cache[cache_key] = embeddings_to_cache
        cache[cache_key] = embeddings_to_cache
        save_cache(cache)
        logger.info(f"Cached embeddings for key: {cache_key}, file: {CACHE_PATH}")

def generate_cache_key(
    input_text: Union[str, List[str]],
    model_name: str,
    batch_size: int
) -> str:
    """Generate a cache key based on model name, batch size, and input text."""
    model_id = resolve_model_value(model_name)
    if isinstance(input_text, str):
        text_hash = hash_text(input_text)
    else:
        text_hash = hash_text("".join(sorted(input_text)))
    return f"embed:{model_id}:{batch_size}:{text_hash}"
