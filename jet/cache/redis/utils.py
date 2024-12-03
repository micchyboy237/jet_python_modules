from .config import RedisConfig
from jet.logger import logger


class RedisClient:
    def __init__(self):
        self.redis_config = RedisConfig()
        self.client = self.redis_config.get_client()

    def select_db(self, db: int):
        self.client.execute_command('SELECT', db)

    def get_all_keys(self, db: int, keys: list = None):
        self.select_db(db)
        db_keys = self.client.keys('*')
        if keys:
            db_keys = [key for key in db_keys if key in keys]
        return db_keys

    def get_key_value(self, key: str):
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


def print_all_data(total_dbs=16):
    redis_client = RedisClient()
    for db in range(total_dbs):
        print_data(db, redis_client)


def print_data(db: int, redis_client: RedisClient, keys: list = None):
    db_keys = redis_client.get_all_keys(db, keys)
    if not db_keys:
        logger.log("Database", f"({db})", ":", "No data found", colors=[
                   "LOG", "DEBUG", "LOG", "WARNING"])
        return
    logger.log("Database:", db, colors=["LOG", "DEBUG"])
    for key in db_keys:
        key = key.decode('utf-8')
        if keys and key not in keys:
            continue
        print(f"  Key: {key}")
        value = redis_client.get_key_value(key)
        print(f"    Value: {value}")
