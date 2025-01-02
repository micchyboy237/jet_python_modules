import threading
import time
from jet.logger import logger, sleep_countdown


class WorkManager:
    def __init__(self, total_work, num_threads=2):
        self.total_work = total_work
        self.progress = 0  # Total progress shared between threads
        self.lock = threading.Lock()
        self.num_threads = num_threads

        # Create pause events dynamically based on the number of threads
        self.pause_events = [threading.Event() for _ in range(num_threads)]
        for pause_event in self.pause_events:
            pause_event.set()  # Initially set to allow execution

        # Create a stop countdown callbacks to stop all sleep countdowns once work is complete
        self.stop_countdowns = []

        # Create threads dynamically based on num_threads
        self.threads = []
        for i in range(num_threads):
            thread = threading.Thread(
                target=self.log_elapsed_time, args=(i + 1,), daemon=False)
            self.threads.append(thread)

    def log_elapsed_time(self, thread_num, throttle: float = 0.1):
        while True:
            if not self.pause_events[thread_num - 1].is_set():
                continue
            time.sleep(throttle)

            with self.lock:  # Ensure thread-safe updates to shared progress
                self.progress += 1
                print(f"Thread {thread_num} Progress updated: {self.progress}/{self.total_work} "
                      f"({(self.progress / self.total_work) * 100:.2f}%)")

            if self.progress >= self.total_work:
                break

        self.handle_work_complete()

    def start_threads(self):
        for thread in self.threads:
            thread.start()

    def clean_up(self):
        logger.info(f"Stopping sleep countdowns: {len(self.stop_countdowns)}")
        for stop_countdown in self.stop_countdowns:
            stop_countdown()
        logger.debug("Done clean up.")

    def pause_thread(self, thread_num, timeout: float = None, delay: float = None):
        if delay:
            stop_countdown = sleep_countdown(
                delay, f"Thread {thread_num} pausing in")
            self.stop_countdowns.append(stop_countdown)

        logger.warning(f"\nPausing thread {thread_num}...")
        self.pause_events[thread_num - 1].clear()  # Pause the thread

        stop_countdown = sleep_countdown(
            timeout, f"Thread {thread_num} resuming in")
        self.stop_countdowns.append(stop_countdown)

        logger.debug(f"\nResuming thread {thread_num}...")
        self.pause_events[thread_num - 1].set()  # Resume the thread

    def handle_work_complete(self):
        for idx, event in enumerate(self.pause_events):
            event.clear()
            logger.log(f"Stopped event {
                       idx + 1}:", event.__repr__(), colors=["DEBUG", "SUCCESS"])
        self.clean_up()
        logger.success(f"\nWork complete!")


# Example usage
if __name__ == '__main__':
    work_manager = WorkManager(total_work=100, num_threads=2)
    work_manager.start_threads()

    work_manager.pause_thread(1, timeout=5, delay=3)
    work_manager.pause_thread(2, timeout=8, delay=10)
