from typing import List, Tuple, Optional
import time
import threading
from .utils import RedisClient


class RedisUseCases:
    def __init__(self) -> None:
        self.redis_client = RedisClient()
    # 1. Caching

    def get_product_details(self, product_id: int) -> str:
        cache_key = f"product:{product_id}"
        cached_data = self.redis_client.get(cache_key)
        if cached_data:
            return cached_data.decode('utf-8')

        # TODO: Should fetch details from DB
        product_details = f"Details for product {product_id}"
        self.redis_client.setex(cache_key, 3600, product_details)
        return product_details

    # 2. Session Management
    def manage_session(self, session_id: str, user_data: dict, ttl: int = 1800) -> Optional[str]:
        self.redis_client.setex(session_id, ttl, str(user_data))
        session = self.redis_client.get(session_id)
        return session.decode('utf-8') if session else None

    # 3. Publish/Subscribe Messaging
    def publish_message(self, channel: str, message: str):
        self.redis_client.publish(channel, message)

    def subscribe_to_channel(self, channel: str):
        def listener():
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(channel)
            for message in pubsub.listen():
                if message['type'] == 'message':
                    print(f"Received message: {
                          message['data'].decode('utf-8')}")

        threading.Thread(target=listener, daemon=True).start()

    # 4. Rate Limiting
    def check_rate_limit(self, user_id: str, limit: int = 100, ttl: int = 60) -> bool:
        rate_limit_key = f"rate_limit:{user_id}"
        if self.redis_client.incr(rate_limit_key) == 1:
            self.redis_client.expire(rate_limit_key, ttl)
        current_count = int(self.redis_client.get(rate_limit_key))
        return current_count <= limit

    # 5. Leaderboards
    def update_leaderboard(self, leaderboard: str, scores: dict):
        self.redis_client.zadd(leaderboard, scores)

    def get_leaderboard(self, leaderboard: str, top: int = 10) -> List[Tuple[str, float]]:
        return self.redis_client.zrevrange(leaderboard, 0, top - 1, withscores=True)

    # 6. Geospatial Indexing
    def add_to_geospatial(self, index: str, locations: List[Tuple[float, float, str]]):
        self.redis_client.geoadd(index, *locations)

    def find_nearby(self, index: str, longitude: float, latitude: float, radius: float, unit: str = 'km') -> List[Tuple[str, float]]:
        return self.redis_client.georadius(index, longitude, latitude, radius, unit=unit, withdist=True)

    # 7. Real-Time Analytics
    def increment_page_view(self, page: str) -> int:
        return self.redis_client.incr(page)

    def get_page_views(self, page: str) -> int:
        return int(self.redis_client.get(page) or 0)

    # 8. Distributed Locking
    def acquire_lock(self, lock_key: str, ttl: int = 10) -> bool:
        return bool(self.redis_client.set(lock_key, "locked", nx=True, ex=ttl))

    def release_lock(self, lock_key: str):
        self.redis_client.delete(lock_key)

    # 9. Message Queue
    def add_task_to_queue(self, queue_name: str, task: str):
        self.redis_client.lpush(queue_name, task)

    def get_task_from_queue(self, queue_name: str) -> Optional[str]:
        task = self.redis_client.rpop(queue_name)
        return task.decode('utf-8') if task else None

    # 10. Time-Series Data
    def add_sensor_reading(self, key: str, value: str):
        timestamp = int(time.time())
        self.redis_client.zadd(key, {value: timestamp})

    def get_sensor_readings(self, key: str, start_time: int, end_time: int) -> List[str]:
        return self.redis_client.zrangebyscore(key, start_time, end_time)


# Instructions
# 1. Install Redis Python client: `pip install redis`
# 2. Start a Redis server on localhost.
# 3. Use the class methods to interact with Redis based on use case.

# Example Usage:
# redis_use_cases = RedisUseCases()
# print(redis_use_cases.get_product_details(1))
