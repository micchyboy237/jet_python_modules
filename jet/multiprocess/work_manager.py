import sys
import threading
import time
from jet.logger import logger, asleep_countdown


class WorkManager:
    def __init__(self, work_callbacks, num_threads=2):
        self.work_callbacks = work_callbacks  # List of callbacks for work items
        self.progress = 0  # Total progress shared between threads
        self.lock = threading.Lock()
        self.num_threads = num_threads

        # Create pause events dynamically based on the number of threads
        self.pause_events = [threading.Event() for _ in range(num_threads)]
        for pause_event in self.pause_events:
            pause_event.set()  # Initially set to allow execution

        # Create stop countdown callbacks to stop all sleep countdowns once work is complete
        self.stop_countdowns = []

        # Create threads dynamically based on num_threads
        self.threads = []
        for i in range(num_threads):
            thread = threading.Thread(
                target=self.process_work, args=(i + 1,), daemon=False)
            self.threads.append(thread)

    def process_work(self, thread_num, throttle: float = 0.1):
        while self.work_callbacks:
            if not self.pause_events[thread_num - 1].is_set():
                continue

            time.sleep(throttle)

            with self.lock:
                if not self.work_callbacks:
                    break

                work_callback = self.work_callbacks.pop(0)
                try:
                    work_callback()  # Execute the work callback
                except Exception as e:
                    logger.error(f"Error in work callback: {e}")

                self.progress += 1
                print(f"Thread {thread_num} Progress updated: {self.progress}/{len(self.work_callbacks) + self.progress} "
                      f"({(self.progress / (self.progress + len(self.work_callbacks))) * 100:.2f}%)")

        self.handle_work_complete()

    def start_threads(self):
        for thread in self.threads:
            thread.start()

    def clean_up(self):
        logger.info(f"Stopping sleep countdowns: {len(self.stop_countdowns)}")
        for stop_countdown in self.stop_countdowns:
            stop_countdown()
        logger.success(f"Done cleaning up countdowns: {
                       len(self.stop_countdowns)}")

        # Clear all pause events to make sure no threads are stuck
        for event in self.pause_events:
            event.clear()
        logger.debug("Cleared all pause events.")

    def pause_thread(self, thread_num, timeout: float = None, delay: float = None):
        if delay:
            start = asleep_countdown(
                delay, f"Thread {thread_num} pausing in")
            self.stop_countdowns.append(start())

        logger.warning(f"\nPausing thread {thread_num}...")
        self.pause_events[thread_num - 1].clear()  # Pause the thread

        start = asleep_countdown(
            timeout, f"Thread {thread_num} resuming in")
        self.stop_countdowns.append(start())

        logger.debug(f"\nResuming thread {thread_num}...")
        self.pause_events[thread_num - 1].set()  # Resume the thread

    def handle_work_complete(self):
        logger.info(f"Work complete. Stopping threads and cleaning up.")
        for idx, event in enumerate(self.pause_events):
            event.clear()
            logger.log(
                f"Stopped event {idx + 1}:", event.__repr__(), colors=["DEBUG", "SUCCESS"])

        # Clean up any remaining countdowns
        self.clean_up()
        logger.success(f"\nWork complete!")
