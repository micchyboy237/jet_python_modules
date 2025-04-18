from rq import Worker, Queue
from redis import Redis

redis_conn = Redis(host='localhost', port=6379, db=0)
eval_queue = Queue(connection=redis_conn)


def run_worker():
    """Run the RQ worker with Redis connection."""

    # Initialize Redis connection
    worker = Worker([eval_queue])
    worker.work()
