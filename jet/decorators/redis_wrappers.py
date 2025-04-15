import functools
import hashlib
import json
from typing import Callable, Any

from jet.cache.redis.config import RedisConfig, DEFAULT_TTL_SECONDS

redis_config = RedisConfig()
redis_client = redis_config.get_client()


def _make_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    key_data = {
        "func": f"{func.__module__}.{func.__name__}",
        "args": args,
        "kwargs": kwargs,
    }
    key_json = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_json.encode()).hexdigest()


def redis_cache(ttl: int = DEFAULT_TTL_SECONDS):
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            key = _make_cache_key(func, args, kwargs)
            cached_value = redis_client.get(key)
            if cached_value is not None:
                return json.loads(cached_value)

            result = func(*args, **kwargs)
            redis_client.set(key, json.dumps(result),
                             ex=ttl or redis_config.default_ttl)
            return result
        return wrapper
    return decorator
