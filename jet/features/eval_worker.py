import time
import subprocess
import os
from jet.logger import CustomLogger
import redis
import psutil
from jet.features.queue_config import QUEUE_NAME, REDIS_HOST, REDIS_PORT


MAX_WORKERS = 4
MIN_WORKERS = 1
SCALE_UP_THRESHOLD = 0
SCALE_DOWN_THRESHOLD = 0
SCALE_DOWN_IDLE_TIME = 30
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
worker_pids = set()
last_job_processed_time = time.time()


LOGGER_OUTPUT_DIR = os.path.dirname(__file__)
logger = CustomLogger(log_file=os.path.join(
    LOGGER_OUTPUT_DIR, 'eval_worker.log'))


def get_queue_length(queue_name):
    return r.llen(f'rq:queue:{queue_name}')


def is_process_running(pid):
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def start_worker(queue_name):
    if len(worker_pids) < MAX_WORKERS:
        process = subprocess.Popen(
            ['rq', 'worker', queue_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        worker_pids.add(process.pid)
        logger.info(
            f"Started worker with PID: {process.pid} (using default RQ Worker)")
        time.sleep(1)
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(
                f"Worker with PID {process.pid} failed to start. Exit code: {process.returncode}")
            logger.error(f"stdout: {stdout}")
            logger.error(f"stderr: {stderr}")
            worker_pids.remove(process.pid)


def stop_worker(pid):
    if len(worker_pids) > MIN_WORKERS and pid in worker_pids:
        try:
            os.kill(pid, 15)
            worker_pids.remove(pid)
            logger.info(f"Stopped worker with PID: {pid}")
        except OSError as e:
            logger.error(f"Error stopping worker with PID: {pid}: {e}")


if __name__ == "__main__":
    while True:
        queue_length = get_queue_length(QUEUE_NAME)
        logger.info(
            f"Queue length: {queue_length}, Active workers: {len(worker_pids)}")
        if queue_length > SCALE_UP_THRESHOLD and len(worker_pids) < MAX_WORKERS:
            start_worker(QUEUE_NAME)
        if queue_length < SCALE_DOWN_THRESHOLD and len(worker_pids) > MIN_WORKERS and (
            time.time() - last_job_processed_time > SCALE_DOWN_IDLE_TIME if worker_pids else True
        ):
            if worker_pids:
                pid_to_stop = next(iter(worker_pids))
                stop_worker(pid_to_stop)
        for pid in list(worker_pids):
            if not is_process_running(pid):
                worker_pids.remove(pid)
                logger.warning(f"Worker with PID {pid} seems to have died.")
        time.sleep(10)
