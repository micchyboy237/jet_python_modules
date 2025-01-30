from typing import Callable, Literal, Optional
import asyncio
import time
import threading
from typing import Callable
from functools import wraps
from jet.logger import logger


def time_it(_func=None, *, hide_params=False, function_name=None):
    """
    Decorator to time a function and log the duration, as well as incrementally logging the elapsed time.
    Optionally hides the function's parameters if hide_params is True.
    Can be used as a regular decorator or inline.

    :param _func: The function to be wrapped. Default is None, which allows decorator to be used with parameters.
    :param hide_params: If True, function parameters are not logged. Default is False.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal function_name
            if not function_name:
                if func.__name__ == "__init__":
                    # Extract class name from __qualname__ for __init__ methods
                    class_name = func.__qualname__.split(
                        '.<locals>', 1)[0].rsplit('.', 1)[0]
                    function_name = f"{class_name} __init__"
                else:
                    function_name = func.__name__

            # Log parameters if requested
            if not hide_params:
                args_without_function = args[1:]  # Remove function self/cls
                # Process args to log type name for non-primitive values, specific dictionaries, or arrays directly
                processed_args = ""
                for arg in args_without_function:
                    if isinstance(arg, (int, float, str, bool, list, tuple)):
                        processed_args += f"{arg}, "
                    elif isinstance(arg, dict) and set(arg.keys()) == {"beam_size", "best_of"}:
                        processed_args += f"{arg}, "
                    else:
                        processed_args += f"{type(arg).__name__}, "

                # Process kwargs similarly, for readability and logging specific dictionaries or arrays directly
                processed_kwargs = {
                    k: v if isinstance(v, (int, float, str, bool, list, tuple)) or
                    (isinstance(v, dict) and set(v.keys()) == {"beam_size", "best_of"}) else type(v).__name__
                    for k, v in kwargs.items()
                }

                if processed_args:
                    # limit processed_args string length to 80 characters
                    processed_args = processed_args[:80]
                    # logger.info(
                    #     f"Arguments for {function_name}: {processed_args}")
                if processed_kwargs:
                    # limit processed_kwargs string length to 80 characters
                    processed_kwargs = str(processed_kwargs)[:80]
                    # logger.info(
                    #     f"Keyword Arguments for {function_name}: {processed_kwargs}")

            start_time = time.time()
            stop_logging = threading.Event()

            # Function to log elapsed time
            def log_elapsed_time():
                while not stop_logging.is_set():
                    time.sleep(1)  # Log every second
                    elapsed = time.time() - start_time
                    logger.log(
                        f"{function_name}:",
                        f"{int(elapsed)}s\r",
                        colors=["GRAY", "DEBUG"],
                        flush=True
                    )

            # Start the background thread for logging elapsed time
            logger_thread = threading.Thread(
                target=log_elapsed_time, daemon=True)
            logger_thread.start()

            result = func(*args, **kwargs)  # Call the main function

            # Once the main function is done, signal the logger thread to stop
            stop_logging.set()
            logger_thread.join()

            # Log total duration
            duration = time.time() - start_time
            logger.log(
                f"\n{function_name}",
                "took",
                f"{int(duration)}s\n",
                colors=["INFO", "WHITE", "SUCCESS"],
            )

            return result
        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def sleep_countdown(count: int, message: str = "Sleep"):
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        remaining = count - elapsed
        if remaining <= 0:
            break
        # Apply ceiling on the remaining time to display whole seconds
        # Ceiling of remaining time
        remaining_ceiling = int(-(-remaining // 1))
        logger.log(
            f"{message}:",
            f"{remaining_ceiling}s\r",
            colors=["GRAY", "DEBUG"],
            flush=True
        )
        # Sleep for 1 second or the remaining time if less than 1 second.
        time.sleep(min(1, remaining))
    logger.log(
        f"\n{message}",
        "took",
        f"{count}s\n",
        colors=["INFO", "WHITE", "SUCCESS"],
    )


def asleep_countdown(count: int, message: str = "Sleep") -> Callable[[], None]:
    stop_event = threading.Event()  # Event to control the countdown
    start_time = time.time()

    def stop_countdown() -> None:
        stop_event.set()  # This will trigger the event to stop the countdown
        logger.info("STOPPED COUNTDOWN:", stop_event.is_set())

    def start():
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            remaining = count - elapsed
            if remaining <= 0 or stop_event.is_set():
                break
            # Apply ceiling on the remaining time to display whole seconds
            # Ceiling of remaining time
            remaining_ceiling = int(-(-remaining // 1))
            logger.log(
                f"{message}:",
                f"{remaining_ceiling}s\r",
                colors=["GRAY", "DEBUG"],
                flush=True
            )
            # Sleep for 1 second or the remaining time if less than 1 second.
            time.sleep(min(1, remaining))
        logger.log(
            f"\n{message}",
            "took",
            f"{count}s\n",
            colors=["INFO", "WHITE", "SUCCESS"],
        )

        return stop_countdown

    return start  # Return stop function to allow external stopping


# Define the status literal type
SleepStatus = Literal['pending', 'completed', 'cancelled']


class SleepInterruptible:
    def __init__(self, sleep_duration: float, on_complete: Optional[Callable[[SleepStatus, float, float], None]] = None):
        """
        :param sleep_duration: Duration in seconds for the sleep
        :param on_complete: Callback function to be called after sleep is completed or canceled
        """
        self.sleep_duration: float = sleep_duration
        self._thread: threading.Thread | None = None
        self._cancel_flag: threading.Event = threading.Event()
        self._done_flag: threading.Event = threading.Event()  # To track when sleep finishes
        # Callback function
        self._on_complete: Optional[Callable[[
            SleepStatus, float, float], None]] = on_complete
        # Possible values: 'pending', 'completed', 'cancelled'
        self._status: SleepStatus = 'pending'
        self.total_elapsed = 0.0
        self.restart_elapsed = 0.0

    def _sleep_task(self):
        """Helper function to simulate sleep and update status"""
        elapsed_time = 0.0
        while elapsed_time < self.sleep_duration and not self._cancel_flag.is_set():
            time.sleep(0.1)  # Sleep in short intervals to allow cancellation
            elapsed_time += 0.1
            self.total_elapsed += 0.1

        # Update status based on whether the sleep was interrupted or completed
        if self._cancel_flag.is_set():
            self._status = 'cancelled'  # If cancelled during sleep
        else:
            self._status = 'completed'  # If completed successfully

        self._done_flag.set()  # Set done flag when sleep finishes

        # Trigger the callback (if provided) after completion
        if self._on_complete and self._status == "completed" and self.total_elapsed - self.restart_elapsed > self.sleep_duration:
            self._on_complete(self._status, self.total_elapsed,
                              self.restart_elapsed)

    def start_sleep(self):
        """Start the sleep operation in a new thread"""
        self.total_elapsed = 0.0
        self._cancel_flag.clear()  # Ensure cancel flag is reset
        self._done_flag.clear()  # Reset done flag
        self._status = 'pending'  # Reset status to 'pending'
        self._thread = threading.Thread(target=self._sleep_task)
        self._thread.start()

    def cancel_sleep(self):
        """Cancel the ongoing sleep"""
        self._cancel_flag.set()
        if self._thread:
            self._thread.join()
        self._status = 'cancelled'  # Mark status as cancelled

    def restart_sleep(self):
        """Restart the sleep countdown"""
        self.restart_elapsed = self.total_elapsed
        # Cancel any ongoing sleep first
        self.cancel_sleep()

        # Reset the status to 'pending' and start sleep again
        self._cancel_flag.clear()  # Ensure cancel flag is reset
        self._done_flag.clear()  # Reset done flag
        self._status = 'pending'  # Reset status to 'pending'
        self._thread = threading.Thread(target=self._sleep_task)
        self._thread.start()

    def get_status(self) -> SleepStatus:
        """Get the current status of the sleep"""
        return self._status
