# jet/cache/redis/print_utils.py
from .utils import RedisClient
from jet.logger import logger


class RedisPrintUtils:
    def __init__(self) -> None:
        self.redis_client = RedisClient()

    def print_all_data(self, total_dbs=16):
        for db in range(total_dbs):
            self.print_data(db, self.redis_client)

    def print_data(self, db: int, keys: list = None):
        db_keys = self.redis_client.get_all_keys(db, keys)
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
            value = self.redis_client.get_key_value(key)
            print(f"    Value: {value}")
