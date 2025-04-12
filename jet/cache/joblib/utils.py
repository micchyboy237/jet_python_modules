import os
import time
import threading
import joblib
from cachetools import TTLCache
from jet.transformers.object import make_serializable
from jet.logger import logger
from typing import Any, Type, Optional, Union

from pydantic.main import BaseModel

import os
import time
import threading
import atexit
from cachetools import TTLCache
from jet.logger import logger


# ✅ Cache Configuration
CACHE_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/cache/joblib/.cache"
CACHE_TTL = 3600  # Time-to-live for TTLCache (seconds) (1 hour)
CACHE_SIZE = 10000  # Max number of items in TTLCache
CACHE_CLEANUP_INTERVAL = 600  # Cleanup every 10 minutes

MAX_CACHE_FILE_SIZE = 5 * 1024 * 1024  # 5 MB (example limit)

# ✅ Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# ✅ Initialize TTLCache
ttl_cache: TTLCache = TTLCache(maxsize=CACHE_SIZE, ttl=CACHE_TTL)

# ✅ Track loaded caches and original cache states
loaded_caches = set()
original_cache_data = {}


def load_persistent_cache(cache_files: Optional[Union[str, list[str]]] = None):
    """Load the key-value cache from one or more persistent cache files."""
    global loaded_caches, original_cache_data

    if cache_files is None:
        cache_files = [os.path.join(CACHE_DIR, f) for f in os.listdir(
            CACHE_DIR) if f.endswith(".pkl")]
    elif isinstance(cache_files, str):
        cache_files = [cache_files]

    for cache_file in cache_files:
        cache_key = cache_file
        cache_file = cache_file if os.path.isabs(
            cache_file) else os.path.join(CACHE_DIR, cache_file)

        if cache_key in loaded_caches:
            continue  # ✅ Skip if already loaded

        if os.path.exists(cache_file):
            try:
                cache_data = joblib.load(cache_file)
                if isinstance(cache_data, dict) and cache_data:
                    ttl_cache.update(cache_data)
                    loaded_caches.add(cache_key)
                    original_cache_data[cache_key] = cache_data.copy()
                    logger.success(
                        f"Loaded {cache_key} - {len(cache_data)} items from {cache_file}")
                else:
                    logger.warning(
                        f"Skipping {cache_file}: Not a valid dictionary or empty")
            except EOFError as e:
                logger.warning(f"Error loading {cache_file}: {e}")
            except Exception as e:
                logger.error(f"Error loading {cache_file}: {e}")


def save_persistent_cache(cache_files: Optional[Union[str, list[str]]] = None):
    """Save only modified loaded caches, or create new cache files if they don't exist."""
    if cache_files is None:
        cache_files = list(loaded_caches)
    elif isinstance(cache_files, str):
        cache_files = [cache_files]

    for cache_file in cache_files:
        cache_key = cache_file
        cache_file = cache_file if os.path.isabs(
            cache_file) else os.path.join(CACHE_DIR, cache_file)

        current_cache_data = dict(ttl_cache)

        try:
            # Check if cache file exceeds size limit
            cache_file_size = os.path.getsize(cache_file)
            if os.path.exists(cache_file) and cache_file_size > MAX_CACHE_FILE_SIZE:
                logger.warning(
                    f"Cache file {cache_file} ({cache_file_size}) exceeds size limit. Deleting it.")

                os.remove(cache_file)
        except FileNotFoundError:
            logger.warning(
                f"Tried to delete {cache_file}, but it was already missing.")

        # ✅ Save if cache was loaded and modified
        if cache_key in loaded_caches:
            if original_cache_data.get(cache_key) != current_cache_data.get(cache_key):
                try:
                    joblib.dump(current_cache_data, cache_file)
                    logger.success(
                        f"Saved {len(ttl_cache)} items to {cache_file}")
                except Exception as e:
                    logger.error(f"Error saving {cache_file}: {e}")

        # ✅ Create new cache file if it doesn't exist
        else:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                joblib.dump(current_cache_data, cache_file)
                loaded_caches.add(cache_key)
                logger.success(f"Created and saved new cache: {cache_file}")
            except Exception as e:
                logger.error(f"Error creating cache file {cache_file}: {e}")


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
        try:
            data = joblib.load(cache_file)
            logger.orange(f"Data loaded successfully from {file_path}")
        except EOFError:
            data = None

        # ✅ Store in TTLCache if enabled
        if data and ttl_cache is not None:
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
        cache_file = os.path.join(CACHE_DIR, file)

        try:
            # Check if cache file exceeds size limit
            cache_file_size = os.path.getsize(cache_file)
            if os.path.exists(cache_file) and cache_file_size > MAX_CACHE_FILE_SIZE:
                logger.warning(
                    f"Cache file {cache_file} ({cache_file_size}) exceeds size limit. Deleting it.")
                os.remove(cache_file)  # Delete the oversized file
            else:
                file_age = now - os.stat(cache_file).st_mtime
                if file_age > CACHE_TTL:
                    # os.remove(cache_file)
                    # ✅ Clear file instead of deleting
                    open(cache_file, 'w').close()
                    logger.warning(f"Cleared expired cache file: {cache_file}")
        except FileNotFoundError:
            logger.warning(
                f"Tried to delete {cache_file}, but it was already missing.")


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


# ✅ Start cleanup thread safely
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
# load_persistent_cache()


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
