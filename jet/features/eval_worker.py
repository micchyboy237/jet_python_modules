import time
import subprocess
import redis
import os

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
QUEUE_NAME = 'default'
MAX_WORKERS = 5
MIN_WORKERS = 1
SCALE_UP_THRESHOLD = 5  # If queue length exceeds this, add a worker
SCALE_DOWN_THRESHOLD = 1  # If queue length is below this, consider removing a worker
SCALE_DOWN_IDLE_TIME = 60  # Seconds of idle time before considering scale down

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
            os.kill(pid, 15)  # SIGTERM
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
            # Find the oldest worker and stop it (simplistic approach)
            if worker_pids:
                pid_to_stop = next(iter(worker_pids))
                stop_worker(pid_to_stop)

        # Basic check for worker liveness (you'd need more robust monitoring)
        for pid in list(worker_pids):
            if not os.path.exists(f"/proc/{pid}"):
                worker_pids.remove(pid)
                print(f"Worker with PID {pid} seems to have died.")

        # Check queue length and adjust workers every 10 seconds
        time.sleep(10)
