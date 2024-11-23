import sys
import threading
import time
from jet.logger import logger


def time_it(func):
    """
    Decorator to time a function and log the duration, 
    as well as incrementally logging the elapsed time.
    """
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}...")

        start_time = time.time()
        stop_logging = threading.Event()

        # Function to log elapsed time
        def log_elapsed_time():
            while not stop_logging.is_set():
                time.sleep(1)  # Log every second
                elapsed = time.time() - start_time
                # Overwrite the line with the new elapsed time
                sys.stdout.write(f"Elapsed time (s): {elapsed:.4f}\r")
                sys.stdout.flush()  # Ensure the content is immediately printed

        # Start the background thread for logging elapsed time
        logger_thread = threading.Thread(target=log_elapsed_time, daemon=True)
        logger_thread.start()

        result = func(*args, **kwargs)  # Call the main function

        # Once the main function is done, signal the logger thread to stop
        stop_logging.set()
        logger_thread.join()

        # Move to the next line after logging elapsed time
        sys.stdout.write('\n')
        sys.stdout.flush()

        # Log total duration
        duration = time.time() - start_time
        # Additional new line added here
        logger.info(f"{func.__name__} took {duration:.4f} seconds\n")

        return result
    return wrapper
