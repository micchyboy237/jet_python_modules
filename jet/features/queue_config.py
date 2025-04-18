# queue_config.py
import redis
from rq import Queue

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
QUEUE_NAME = 'default'

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
eval_queue = Queue(QUEUE_NAME, connection=r)
