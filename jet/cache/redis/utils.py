import json
import traceback
from redis.exceptions import ConnectionError
from typing import Optional, List, Union
from jet.cache.redis import RedisConfig, RedisConfigParams, DEFAULT_CONFIG
from jet.logger import logger


class RedisClient:
    def __init__(self, config: Optional[RedisConfigParams] = {}):
        # Use the passed config or default values
        self.config = RedisConfig(**config)
        self.client = self.config.get_client()

    # Delegate methods to the underlying Redis client

    def __getattr__(self, name):
        """Delegate attribute access to the Redis client."""
        return getattr(self.client, name)

    def select_db(self, db: int):
        self.client.execute_command('SELECT', db)

    def get_all_keys(self, db: int, keys: Optional[List[str]] = None) -> List[bytes]:
        try:
            self.select_db(db)
        except ConnectionError as e:
            logger.error(f"Connection error on db: {db}")
            traceback.print_exc()
        db_keys = self.client.keys('*')
        if keys:
            db_keys = [key for key in db_keys if key in keys]
        return db_keys

    def get_key_value(self, key: str) -> Union[str, dict, list, set, str]:
        key_type = self.client.type(key).decode('utf-8')
        if key_type == 'string':
            return self.client.get(key).decode('utf-8')
        elif key_type == 'hash':
            return self.client.hgetall(key)
        elif key_type == 'list':
            return self.client.lrange(key, 0, -1)
        elif key_type == 'set':
            return self.client.smembers(key)
        else:
            return f"Other type of data: {key_type}"


class RedisCache:
    def __init__(self, config: RedisConfigParams = {}):
        config = {**DEFAULT_CONFIG, **config}
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

    def clear(self, key: Optional[str] = None):
        """Clear the cache. If key is provided, clear only that key. Otherwise, clear all keys."""
        try:
            if key:
                self.redis_client.delete(key)
            else:
                self.redis_client.flushdb()
        except ConnectionError as e:
            logger.error(f"Error clearing cache: {str(e)}")
            traceback.print_exc()
