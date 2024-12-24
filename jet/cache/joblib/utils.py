import os
import joblib
from jet.logger import logger


def load_from_cache_or_compute(func, *args, file_path: str = "", use_cache: bool = False, **kwargs):
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

    if use_cache and os.path.exists(file_path):
        logger.success(f"Cache hit! File: {file_path}")
        return joblib.load(file_path)

    # Compute the result and cache it
    logger.debug(f"Cache miss! Computing result for: {file_path}")
    result = func(*args, **kwargs)
    joblib.dump(result, file_path)
    logger.success(f"Saved cache to: {file_path}")
    return result


__all__ = [
    "load_from_cache_or_compute"
]
