from jet.cache.redis import RedisConfigParams, RedisClient
import json

DEFAULT_CONFIG: RedisConfigParams = {
    "host": "localhost",
    "port": 6380,
    "db": 0,
    "max_connections": 100
}


class Cache:
    def __init__(self, config: RedisConfigParams = DEFAULT_CONFIG):
        self.redis_client = RedisClient(config=config)

    def get(self, key: str):
        """Retrieve the cached value from Redis."""
        cached_value = self.redis_client.get(key)
        if cached_value:
            return json.loads(cached_value.decode('utf-8'))
        return None

    def set(self, key: str, value: dict, ttl: int = 3600):
        """Store the result in cache."""
        self.redis_client.set(key, json.dumps(value), ttl)
