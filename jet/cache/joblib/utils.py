import os
import time
import threading
import joblib
from cachetools import TTLCache
from jet.transformers.object import make_serializable
from jet.logger import logger
from typing import Any, Type, Optional

from pydantic.main import BaseModel

import os
import time
import threading
import atexit
from cachetools import TTLCache
from jet.logger import logger


# ✅ Cache Configuration
CACHE_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/cache/joblib/.cache"
CACHE_FILE = f"{CACHE_DIR}/ttl_cache_1.pkl"
CACHE_TTL = 3600  # Time-to-live for TTLCache (seconds)
CACHE_SIZE = 10000  # Max number of items in TTLCache
CACHE_CLEANUP_INTERVAL = 600  # Cleanup every 10 minutes

# ✅ Initialize TTLCache
ttl_cache: TTLCache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)


def load_persistent_cache():
    """Load the embeddings cache from the persistent cache file."""
    if os.path.exists(CACHE_FILE):
        try:
            cache_data = joblib.load(CACHE_FILE)
            if isinstance(cache_data, dict):
                ttl_cache.update(cache_data)  # Load into TTL cache
                logger.success(
                    f"Loaded {len(cache_data)} embeddings from cache.")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")


def save_persistent_cache():
    """Save the current TTL cache to the persistent cache file."""
    try:
        joblib.dump(dict(ttl_cache), CACHE_FILE)
        logger.success(f"Saved {len(ttl_cache)} embeddings to {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")


def save_cache(file_path: str, data: Any) -> None:
    """Save data persistently using joblib and optionally to TTLCache."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(make_serializable(data), file_path)
    logger.success(f"Data saved successfully to {file_path}")

    # ✅ Store in TTLCache if enabled
    if ttl_cache is not None:
        ttl_cache[file_path] = data


def load_cache(file_path: str) -> Any:
    """Load data from TTLCache (if enabled) or from file using joblib."""

    # ✅ Check TTLCache first
    if ttl_cache is not None and file_path in ttl_cache:
        logger.info(f"Loading from TTL cache: {file_path}")
        return ttl_cache[file_path]

    # ✅ Otherwise, load from file
    if os.path.exists(file_path):
        file_age = time.time() - os.stat(file_path).st_mtime
        if file_age > CACHE_TTL:
            # os.remove(file_path)
            # ✅ Clear file instead of deleting
            open(file_path, 'w').close()
            logger.warning(f"Cleared expired cache file: {file_path}")
            return None  # File expired

        data = joblib.load(file_path)
        logger.orange(f"Data loaded successfully from {file_path}")

        # ✅ Store in TTLCache if enabled
        if ttl_cache is not None:
            ttl_cache[file_path] = data

        return data

    return None  # If file does not exist


def load_or_save_cache(
    file_path: str,
    data: Optional[Any | BaseModel] = None,
    model: Optional[Type[BaseModel]] = None
) -> Any:
    """
    Load data from or save data to a cache file, with optional TTLCache support.
    """

    # ✅ Check TTLCache first if enabled
    if ttl_cache is not None and file_path in ttl_cache:
        logger.info(f"Loaded from TTL cache: {file_path}")
        return ttl_cache[file_path]

    # ✅ Load from file if TTLCache is disabled or cache is missing
    if os.path.exists(file_path):
        loaded_data = load_cache(file_path)

        # ✅ Store in TTLCache if enabled
        if ttl_cache is not None:
            ttl_cache[file_path] = loaded_data

        return loaded_data

    # ✅ Save new data to file and TTLCache if provided
    elif data:
        save_cache(file_path, data)

        # ✅ Store in TTLCache if enabled
        if ttl_cache is not None:
            ttl_cache[file_path] = data

        return data


def load_from_cache_or_compute(func, *args, file_path: str = "", use_cache: bool = True, **kwargs):
    """
    Caches the result of a function to a .pkl file or computes it if cache doesn't exist.

    Parameters:
    - func: Function to compute the result.
    - args: Positional arguments for the function.
    - use_cache: Bool, whether to use the cache.
    - file_path: Path to the cache file.
    - kwargs: Keyword arguments for the function.

    Returns:
    - Cached or computed result.
    """
    if not file_path.endswith(".pkl"):
        raise ValueError("Cache file must have a .pkl extension.")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if use_cache and os.path.exists(file_path):
        logger.success(f"Cache hit! File: {file_path}")
        return joblib.load(file_path)

    # Compute the result and cache it
    logger.info(f"Cache miss! Computing result for: {file_path}")
    result = func(*args, **kwargs)
    joblib.dump(result, file_path)
    logger.success(f"Saved cache to: {file_path}")
    return result


# ✅ Cleanup Functions
def cleanup_persistent_cache():
    """Deletes expired cache files from disk."""
    logger.gray(CACHE_DIR)

    if not os.path.exists(CACHE_DIR):
        return

    now = time.time()
    for file in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, file)

        if file.endswith(".pkl"):
            file_age = now - os.stat(file_path).st_mtime
            if file_age > CACHE_TTL:
                # os.remove(file_path)
                # ✅ Clear file instead of deleting
                open(file_path, 'w').close()
                logger.warning(f"Cleared expired cache file: {file_path}")


def cleanup_ttl_cache():
    """Manually triggers cleanup for TTLCache."""
    if ttl_cache:
        logger.info("Running TTLCache cleanup...")
        ttl_cache.expire()  # ✅ Explicitly expire old entries


def background_cleanup():
    """Background thread to clean TTLCache & persistent cache periodically."""
    while True:
        logger.info("Running cache cleanup...")
        cleanup_ttl_cache()
        cleanup_persistent_cache()
        time.sleep(CACHE_CLEANUP_INTERVAL)  # Wait before running again


_cleanup_thread = None  # Global variable to track the cleanup thread


def start_cleanup_thread():
    """Starts a background cleanup thread if not already running."""
    global _cleanup_thread

    if _cleanup_thread and _cleanup_thread.is_alive():
        logger.info("Cleanup thread is already running.")
        return

    logger.info("Starting cache cleanup thread...")
    _cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
    _cleanup_thread.start()


def cleanup_on_exit():
    """Ensures cache cleanup when the program exits."""
    logger.info("Performing final cache cleanup before exit...")
    cleanup_ttl_cache()
    cleanup_persistent_cache()


# ✅ Register cleanup to run on exit
# atexit.register(cleanup_on_exit)

# ✅ Start cleanup thread safely
start_cleanup_thread()
# ✅ Load existing cache on startup
load_persistent_cache()


__all__ = [
    "save_cache",
    "load_cache",
    "load_or_save_cache",
    "load_from_cache_or_compute",
]


# Example Usage
if __name__ == "__main__":
    class MyCacheModel(BaseModel):
        key: str
        value: int

    cache_file = "generated/example.pkl"
    data = MyCacheModel(key="example", value=42)

    # Save data
    saved_data = load_or_save_cache(cache_file, data=data)
    logger.debug(saved_data)  # Output: key='example' value=42

    # Load data
    loaded_data = load_or_save_cache(cache_file, model=MyCacheModel)
    logger.success(loaded_data)  # Output: key='example' value=42
