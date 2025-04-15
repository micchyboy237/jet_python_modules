import redis
from jet.cache.redis import RedisConfigParams

DEFAULT_CONFIG: RedisConfigParams = {
    "host": "localhost",
    "port": 6380,
    "db": 0,
    "max_connections": 100
}

# Default TTL set to 3 hours (10,800 seconds)
DEFAULT_TTL_SECONDS = 3 * 60 * 60  # 3 hours


class RedisConfig:
    def __init__(self, host='localhost', port=6380, db=0, max_connections=100, default_ttl=DEFAULT_TTL_SECONDS):
        self.host = host
        self.port = port
        self.db = db
        self.max_connections = max_connections
        self.default_ttl = default_ttl  # Store default TTL (3 hours)
        self.pool = redis.ConnectionPool(
            host=self.host, port=self.port, db=self.db, max_connections=self.max_connections)

    def get_client(self):
        """Return a Redis client with a default TTL applied to set operations."""
        client = redis.StrictRedis(connection_pool=self.pool)
        # Wrap the set method to enforce default TTL
        original_set = client.set

        def set_with_ttl(name, value, ex=None, px=None, nx=False, xx=False, keepttl=False, **kwargs):
            # Use default TTL if ex or px is not provided and keepttl is False
            if ex is None and px is None and not keepttl:
                ex = self.default_ttl
            return original_set(name, value, ex=ex, px=px, nx=nx, xx=xx, keepttl=keepttl, **kwargs)

        client.set = set_with_ttl
        return client
