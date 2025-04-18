# jet_python_modules/jet/features/eval_worker.py
import time
import subprocess
import os
import redis
# Import the queue and constants
from jet.features.queue_config import QUEUE_NAME, REDIS_HOST, REDIS_PORT

MAX_WORKERS = 4  # Default for M1 with 8 cores (4 performance)
MIN_WORKERS = 1
SCALE_UP_THRESHOLD = 2
SCALE_DOWN_THRESHOLD = 0
SCALE_DOWN_IDLE_TIME = 30  # seconds
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
worker_pids = set()
last_job_processed_time = time.time()


def get_queue_length(queue_name):
    return r.llen(f'rq:queue:{queue_name}')


def start_worker(queue_name):
    if len(worker_pids) < MAX_WORKERS:
        process = subprocess.Popen(['rq', 'worker', queue_name])
        worker_pids.add(process.pid)
        print(f"Started worker with PID: {process.pid}")


def stop_worker(pid):
    if len(worker_pids) > MIN_WORKERS and pid in worker_pids:
        try:
            os.kill(pid, 15)
            worker_pids.remove(pid)
            print(f"Stopped worker with PID: {pid}")
        except OSError:
            print(f"Error stopping worker with PID: {pid}")


if __name__ == "__main__":
    while True:
        queue_length = get_queue_length(QUEUE_NAME)
        print(
            f"Queue length: {queue_length}, Active workers: {len(worker_pids)}")
        if queue_length > SCALE_UP_THRESHOLD and len(worker_pids) < MAX_WORKERS:
            start_worker(QUEUE_NAME)
        if queue_length < SCALE_DOWN_THRESHOLD and len(worker_pids) > MIN_WORKERS and (time.time() - last_job_processed_time > SCALE_DOWN_IDLE_TIME if worker_pids else True):
            if worker_pids:
                pid_to_stop = next(iter(worker_pids))
                stop_worker(pid_to_stop)
        for pid in list(worker_pids):
            if not os.path.exists(f"/proc/{pid}"):
                worker_pids.remove(pid)
                print(f"Worker with PID {pid} seems to have died.")
        time.sleep(10)
