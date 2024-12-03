from typing import TypedDict, Optional, List, Union
from .config import RedisConfig
import redis


# Define a TypedDict for Redis configuration
class RedisConfigParams(TypedDict, total=False):
    host: str
    port: int
    db: int
    max_connections: int


class RedisClient:
    def __init__(self, config: Optional[RedisConfigParams] = None):
        # Use the passed config or default values
        self.config = RedisConfig(**(config or {}))
        self.client = self.config.get_client()

    # Delegate methods to the underlying Redis client

    def __getattr__(self, name):
        """Delegate attribute access to the Redis client."""
        return getattr(self.client, name)

    def select_db(self, db: int):
        self.client.execute_command('SELECT', db)

    def get_all_keys(self, db: int, keys: Optional[List[str]] = None) -> List[bytes]:
        self.select_db(db)
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
