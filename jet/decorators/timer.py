import threading
import time
import signal
import sys
from functools import wraps


def timeout(seconds: int):
    """
    Decorator to raise TimeoutError if a function takes too long and show countdown inline.

    Args:
        seconds: Timeout in seconds.
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            print(
                f"\rFunction {func.__name__} timed out after {seconds} seconds", flush=True)
            raise TimeoutError(
                f"Function {func.__name__} timed out after {seconds} seconds")

        @wraps(func)
        def wrapper(*args, **kwargs):
            stop_event = threading.Event()

            def countdown():
                for i in range(seconds, 0, -1):
                    if stop_event.is_set():
                        return
                    sys.stdout.write(f"\r⏳ Timeout in {i} second(s)...")
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\r")  # Clear line on completion
                sys.stdout.flush()

            countdown_thread = threading.Thread(target=countdown, daemon=True)
            countdown_thread.start()

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)

            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                stop_event.set()
                countdown_thread.join()
                sys.stdout.write("\r")  # Clear countdown line
                sys.stdout.flush()
        return wrapper
    return decorator
