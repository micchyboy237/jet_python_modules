import redis


class RedisConfig:
    def __init__(self, host='localhost', port=6380, db=0, max_connections=100):
        self.host = host
        self.port = port
        self.db = db
        self.max_connections = max_connections
        self.pool = redis.ConnectionPool(
            host=self.host, port=self.port, db=self.db, max_connections=self.max_connections)

    def get_client(self):
        return redis.StrictRedis(connection_pool=self.pool)
