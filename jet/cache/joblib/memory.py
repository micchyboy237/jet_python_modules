from joblib import Memory
from functools import wraps
from typing import Callable, Dict

_memory_instances: Dict[str, Memory] = {}


def memory(cache_dir: str):
    """
    Decorator factory that returns a joblib-based caching decorator
    for a given cache directory.
    """
    if cache_dir not in _memory_instances:
        _memory_instances[cache_dir] = Memory(location=cache_dir, verbose=1)

    memory = _memory_instances[cache_dir]

    def decorator(fn: Callable):
        cached_fn = memory.cache(fn)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return cached_fn(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "memory"
]
